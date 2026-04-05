"""
Pulse Engine - the heartbeat loop.

This is the main orchestrator. It:
1. Runs a heartbeat timer (free-think ticks)
2. Checks for due scheduled tasks
3. Builds context and prompts the companion via llama-server
4. Dispatches actions to channels (toast, Telegram, etc.)
"""

import asyncio
import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from core.llm import LLMClient, PulseResponse
from core.llm_anthropic import AnthropicClient
from core.context import ContextManager
from core.scheduler import ScheduleManager

logger = logging.getLogger(__name__)


class PulseEngine:
    """Main engine that drives the heartbeat loop."""

    def __init__(self, config: dict, channels: dict, skill_registry=None,
                 llm_endpoint: str = None, api_key: str = ""):
        """
        Args:
            config: Parsed config.yaml
            channels: Dict of channel_name -> Channel instance
            skill_registry: Optional SkillRegistry for tool-calling in conversations
            llm_endpoint: OpenAI-compatible API endpoint (from LlamaServer or cloud)
            api_key: API key for cloud providers (empty for local)
        """
        self.config = config
        self.channels = channels
        self.skill_registry = skill_registry

        # Initialize core components
        model_config = config.get("model", {})
        provider_config = config.get("provider", {})
        provider_type = provider_config.get("type", "local")
        server_config = config.get("server", {})
        endpoint = llm_endpoint or f"http://{server_config.get('host', '127.0.0.1')}:{server_config.get('port', 8012)}/v1"

        # Model name: from provider config for API, from model config for local
        if provider_type != "local" and provider_config.get("model"):
            model_name = provider_config["model"]
        else:
            model_name = model_config.get("model_file", "default")

        # Token usage tracker (only meaningful for cloud APIs, but always safe)
        from core.usage import UsageTracker
        usage_tracker = UsageTracker(
            config.get("paths", {}).get("usage", "data/usage.json")
        )

        if provider_type == "anthropic":
            self.llm = AnthropicClient(
                endpoint=endpoint,
                model_name=model_name,
                temperature=model_config.get("temperature", 0.7),
                max_tokens=model_config.get("max_response_tokens", 1024),
                api_key=api_key,
                usage_tracker=usage_tracker,
            )
        else:
            self.llm = LLMClient(
                endpoint=endpoint,
                model_name=model_name,
                temperature=model_config.get("temperature", 0.7),
                max_tokens=model_config.get("max_response_tokens", 1024),
                frequency_penalty=model_config.get("frequency_penalty", 0.0),
                presence_penalty=model_config.get("presence_penalty", 0.0),
                api_key=api_key,
                usage_tracker=usage_tracker,
                reasoning=model_config.get("reasoning", False),
                provider_type=provider_type,
            )
        self.llm._provider_name = provider_type

        self.max_tool_rounds = model_config.get("max_tool_rounds", 5)
        self.context = ContextManager(config)
        self.scheduler = ScheduleManager(
            config.get("paths", {}).get("schedules", "data/schedules.json")
        )

        # Wire journal skill into context manager (if available)
        if skill_registry:
            journal_skill = skill_registry.get_skill("journal")
            if journal_skill:
                self.context.set_journal_skill(journal_skill)

            # Wire context-injectable skills (config: context.inject_skills)
            inject_names = config.get("context", {}).get("inject_skills", [])
            if inject_names:
                inject_skills = []
                for name in inject_names:
                    skill = skill_registry.get_skill(name)
                    if skill:
                        inject_skills.append(skill)
                        logger.info(f"Context injection: {name}")
                    else:
                        logger.warning(f"Context inject_skills: '{name}' not found, skipping")
                if inject_skills:
                    self.context.set_inject_skills(inject_skills)

        heartbeat_config = config.get("heartbeat", {})
        self._heartbeat_randomize = heartbeat_config.get("randomize", False)
        self._heartbeat_min = heartbeat_config.get("interval_min_minutes", 20) * 60
        self._heartbeat_max = heartbeat_config.get("interval_max_minutes", 45) * 60
        fixed_interval = heartbeat_config.get("interval_minutes", 120) * 60
        self.interval = self._roll_heartbeat_interval(fixed_interval)
        self.quiet_start = heartbeat_config.get("quiet_hours_start", 23)
        self.quiet_end = heartbeat_config.get("quiet_hours_end", 8)
        self.startup_checkin = heartbeat_config.get("startup_checkin", True)

        # Dev tick — autonomous self-improvement
        dev_config = config.get("dev_tick", {})
        self.dev_tick_enabled = dev_config.get("enabled", False)
        self.dev_tick_interval = dev_config.get("interval_minutes", 720) * 60  # default 12h
        self.dev_tick_max_rounds = dev_config.get("max_rounds", 8)
        self.dev_tick_schedule = dev_config.get("schedule_time", "")  # e.g. "20:00"
        self._pulse_root = str(Path(config.get("_pulse_root", ".")).resolve())

        self._running = False
        self._stop_event = asyncio.Event()

        # Activity tracking — heartbeat only fires after silence
        self._last_activity = 0.0   # timestamp of last message exchange

        # Rate limiting — prevent the companion from flooding channels
        self._last_notify = 0       # timestamp of last notification
        self._notify_cooldown = heartbeat_config.get("notify_cooldown_minutes", 60) * 60  # default 1 hour

        # Action log — tracks what the companion does each heartbeat
        self._action_log_path = Path(
            config.get("paths", {}).get("action_log", "data/action_log.json")
        )
        self._action_log_max = 20  # keep last 20 actions

    def _roll_heartbeat_interval(self, fixed: int = 0) -> int:
        """Pick the next heartbeat interval in seconds.

        If randomize is enabled, returns a random value between min and max.
        Otherwise returns the fixed interval.
        """
        if self._heartbeat_randomize:
            interval = random.randint(self._heartbeat_min, self._heartbeat_max)
            logger.info(f"Next heartbeat interval: {interval // 60} minutes")
            return interval
        return fixed or self.interval

    def _log_action(self, action: str, tools_used: list[str] = None, summary: str = ""):
        """Append an entry to the heartbeat action log (ring buffer)."""
        log = []
        if self._action_log_path.exists():
            try:
                with open(self._action_log_path, "r", encoding="utf-8") as f:
                    log = json.load(f)
            except (json.JSONDecodeError, IOError):
                log = []

        log.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "action": action,
            "tools": tools_used or [],
            "summary": summary[:120],
        })

        # Keep only the most recent entries
        log = log[-self._action_log_max:]

        try:
            self._action_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._action_log_path, "w", encoding="utf-8") as f:
                json.dump(log, f, indent=2)
        except IOError as e:
            logger.warning(f"Failed to write action log: {e}")

    async def _interruptible_sleep(self, seconds: int):
        """Sleep that can be interrupted instantly by stop()."""
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass  # Normal — timeout means we slept the full duration

    def _in_quiet_hours(self) -> bool:
        """Check if we're in quiet hours."""
        hour = datetime.now().hour
        if self.quiet_start > self.quiet_end:
            # Wraps around midnight (e.g., 23-8)
            return hour >= self.quiet_start or hour < self.quiet_end
        else:
            return self.quiet_start <= hour < self.quiet_end

    def _clear_pending_skill_output(self):
        """Clear pending output (sources, images) from skills after non-conversation use.

        Heartbeat and dev ticks can trigger tools like web_search, but their
        output goes through _dispatch, not the Telegram message handler.
        Without clearing, leftover pending data leaks into the next user message.
        """
        if not self.skill_registry:
            return
        web_skill = self.skill_registry.get_skill("web_search")
        if web_skill:
            web_skill.pending_sources.clear()
            web_skill.pending_images.clear()

    async def _dispatch(self, response: PulseResponse, is_scheduled_task: bool = False):
        """Route a response to the appropriate channel.

        Rate limits apply to free-think ticks but NOT to scheduled tasks
        (those were explicitly requested, so they should always go through).
        """
        now = time.time()

        if response.action == "silent":
            logger.info("Companion chose to stay silent.")
            return

        if response.action == "notify":
            # Rate limit notifications (unless it's a scheduled task)
            if not is_scheduled_task and (now - self._last_notify) < self._notify_cooldown:
                mins_left = int((self._notify_cooldown - (now - self._last_notify)) / 60)
                logger.info(f"Notification rate-limited ({mins_left}m cooldown remaining). Skipping.")
                return
            # Send via toast AND Telegram (if available)
            sent = False
            telegram = self.channels.get("telegram")
            if telegram:
                await telegram.send(response.message)
                sent = True
            toast = self.channels.get("toast")
            if toast:
                await toast.send(response.message)
                sent = True
            if sent:
                self._last_notify = now
                logger.info(f"Notification sent: {response.message[:50]}...")
            else:
                logger.info(f"[No notification channels] {response.message[:80]}...")

        elif response.action == "schedule":
            if response.schedule_task and response.schedule_when:
                try:
                    entry = self.scheduler.add(
                        task=response.schedule_task,
                        created_by="nova-self",
                        when=response.schedule_when
                    )
                    logger.info(f"Self-scheduled: {entry['id']} — {response.schedule_task}")
                except ValueError as e:
                    logger.warning(f"Tried to self-schedule without a valid time: {e}")

        # Handle combined actions (e.g., notify AND schedule)
        if response.schedule_task and response.schedule_when and response.action != "schedule":
            try:
                entry = self.scheduler.add(
                    task=response.schedule_task,
                    created_by="nova-self",
                    when=response.schedule_when
                )
                logger.info(f"Also scheduled: {entry['id']} — {response.schedule_task}")
            except ValueError as e:
                logger.warning(f"Tried to schedule without a valid time: {e}")

    async def handle_message(self, message: str, source: str = "telegram",
                            image_url: str = None) -> tuple[str, list[str]]:
        """Handle an incoming message and return the companion's response.

        This is for bidirectional chat (Telegram, future voice, etc.).
        Unlike heartbeat ticks, the companion responds naturally without JSON structure.

        All LLM calls run via asyncio.to_thread() so they don't block the
        event loop — this keeps Telegram polling alive during inference.

        Args:
            message: The user's message text
            source: Where the message came from ("telegram", "voice", etc.)
            image_url: Optional base64 data URI for an image (vision models)

        Returns:
            Tuple of (companion's response text, list of tools used). Text is None if LLM unavailable.
        """
        # Reset heartbeat timer — conversation is active, re-roll interval
        self._last_activity = time.time()
        self.interval = self._roll_heartbeat_interval()

        # is_available() is a sync HTTP call — run in thread to not block
        available = await asyncio.to_thread(self.llm.is_available)
        if not available:
            logger.warning(f"LLM server not available — can't respond to {source} message.")
            return None, []

        # Load conversation history
        history = self.context._load_conversation()

        # Build conversation prompt (with optional image)
        messages = self.context.build_conversation_prompt(message, history, image_url=image_url)
        logger.info(f"Sending {len(messages)} messages to LLM for {source} reply"
                     f"{' (with image)' if image_url else ''}...")

        # Get response — run in thread so we don't block the event loop!
        # If skills are available, use tool-calling mode; otherwise plain chat.
        tools_used = []
        tools = self.skill_registry.get_all_tools() if self.skill_registry else []
        t0 = time.time()
        if tools:
            logger.info(f"Tool-calling mode: {len(tools)} tools available")
            reply, tools_used = await asyncio.to_thread(
                self.llm.chat_with_tools, messages, tools, self.skill_registry,
                max_rounds=self.max_tool_rounds
            )
        else:
            response = await asyncio.to_thread(self.llm.chat, messages)
            reply = response.raw.strip() if response and response.raw else None
        elapsed = time.time() - t0
        logger.info(f"LLM {source} reply took {elapsed:.1f}s (tools: {', '.join(tools_used) if tools_used else 'none'})")

        # Safety net: strip <think> tags that reasoning models may include
        if reply:
            from core.llm import strip_think_tags
            reply = strip_think_tags(reply)

        # Preserve spoken text in conversation history so the companion
        # remembers what it said aloud.  When both voice and text exist,
        # prepend the voice transcript so nothing is lost.
        voice_text = None
        if self.skill_registry:
            tts_skill = self.skill_registry.get_skill("tts")
            if tts_skill and tts_skill.pending_voice_text:
                voice_text = f"🔊 {tts_skill.pending_voice_text}"

        if voice_text and reply:
            history_reply = f"{voice_text}\n{reply}"
        else:
            history_reply = reply or voice_text
        if not history_reply:
            return None, tools_used

        # Update conversation history (store text only — images are too large to persist)
        # Timestamps help the companion understand temporal flow
        timestamp = datetime.now().strftime("[%b %d, %I:%M %p]")
        image_note = " [sent an image]" if image_url else ""
        history.append({"role": "user", "content": f"{timestamp} {message}{image_note}"})
        history.append({"role": "assistant", "content": history_reply})

        # Auto-summarize when conversation approaches its token budget (~85%)
        conv_budget_tokens = self.config.get("context_budget", {}).get("conversation", 4000)
        max_chars = int(conv_budget_tokens * 4 * 0.85)  # 4 chars/token, trigger at 85%
        total_chars = sum(len(m.get("content", "")) if isinstance(m.get("content"), str) else 100
                         for m in history)
        if total_chars > max_chars:
            logger.info(f"Conversation at ~{int(total_chars / (conv_budget_tokens * 4) * 100)}% of budget — summarizing...")
            summary = await self._summarize_conversation(history)
            if summary:
                # Archive the full conversation before it's replaced by the summary
                try:
                    self.context.archive_conversation(history, summary=summary)
                except Exception as e:
                    logger.warning(f"Conversation archive failed (non-fatal): {e}")
                # Save to persistent memory for cross-session continuity
                try:
                    self.context.save_to_memory(summary)
                except Exception as e:
                    logger.warning(f"Memory persistence failed (non-fatal): {e}")
                # Keep last 2 exchanges so the companion doesn't lose the
                # most recent context right after summarization
                recent_tail = history[-4:]
                history = [{"role": "user", "content": f"[Context] Previous conversation summary: {summary}"},
                           {"role": "assistant", "content": "Got it, I remember. Let's continue."}
                ] + recent_tail
            else:
                # Summarization failed — trim history to prevent unbounded growth
                logger.warning("Summarization failed — trimming conversation to last 10 messages.")
                history = history[-10:]

        self.context.save_conversation(history)
        log_reply = reply or voice_text or "(voice-only)"
        logger.info(f"Replied to {source}: {log_reply[:50]}...")
        return reply, tools_used

    async def _summarize_conversation(self, history: list[dict]) -> str:
        """Summarize the conversation so far (for context compression)."""
        from core.llm import strip_think_tags

        ai_name = self.context.ai_name
        user_name = self.context.user_name

        conversation_text = "\n".join(
            f"{user_name if m['role'] == 'user' else ai_name}: {m['content']}"
            for m in history if m['role'] in ('user', 'assistant')
        )

        # Scale summary length to conversation size
        msg_count = len([m for m in history if m['role'] in ('user', 'assistant')])
        if msg_count <= 10:
            length_guide = "2-3 sentences"
        elif msg_count <= 30:
            length_guide = "4-6 sentences"
        else:
            length_guide = "a thorough paragraph (8-10 sentences)"

        summary_prompt = [
            {"role": "system", "content": (
                "You are a summarizer. Do NOT continue the conversation. "
                f"Do NOT respond as {ai_name}. Do NOT use emoji or roleplay. "
                f"Write a plain summary in {length_guide}."
            )},
            {"role": "user", "content": (
                f"Summarize this conversation in {length_guide}. "
                f"Focus on: topics discussed, decisions made, emotional tone, "
                f"and anything {ai_name} should remember for next time.\n\n"
                f"{conversation_text}"
            )}
        ]
        # Run in thread — this is also a sync HTTP call
        # Don't pass tools — this is a utility call, not a conversation
        response = await asyncio.to_thread(self.llm.chat, summary_prompt)
        if response and response.raw:
            summary = strip_think_tags(response.raw).strip()
            if summary:
                logger.info(f"Conversation summarized: {summary[:80]}...")
                return summary
        return None

    async def _check_due_tasks(self) -> bool:
        """Check for and execute any due scheduled tasks.

        Returns True if any tasks were executed, False otherwise.
        This runs frequently (every 60s) so reminders fire on time.
        """
        due_tasks = self.scheduler.get_due_tasks()
        if not due_tasks:
            return False

        available = await asyncio.to_thread(self.llm.is_available)
        if not available:
            logger.warning("LLM server not available — can't execute due tasks.")
            return False

        tools = self.skill_registry.get_all_tools() if self.skill_registry else []

        for task in due_tasks:
            logger.info(f"Executing due task: {task['task']}")
            messages = self.context.build_task_prompt(task, has_tools=bool(tools))

            response = None
            if tools:
                text, tools_used = await asyncio.to_thread(
                    self.llm.chat_with_tools, messages, tools, self.skill_registry,
                    max_rounds=self.max_tool_rounds
                )
                if tools_used:
                    logger.info(f"Task tools used: {', '.join(tools_used)}")
                if text:
                    response = PulseResponse.from_llm_output(text)
                    await self._dispatch(response, is_scheduled_task=True)
            else:
                response = await asyncio.to_thread(self.llm.chat, messages)
                if response:
                    await self._dispatch(response, is_scheduled_task=True)

            # Log the notification into conversation history so the companion
            # remembers what it sent (and replies make sense in context)
            if response and response.action == "notify" and response.message:
                history = self.context._load_conversation()
                history.append({
                    "role": "assistant",
                    "content": f"[Scheduled reminder] {response.message}"
                })
                self.context.save_conversation(history)

            self.scheduler.mark_completed(task["id"])

        return True

    async def _do_heartbeat(self):
        """Execute a single heartbeat tick (free-think).

        If skills are available, the companion can use tools (search memory,
        write journal, etc.) before deciding on an action. The final response
        is still parsed as JSON (PulseResponse) for dispatch.
        """
        logger.info("--- Heartbeat tick ---")

        if self._in_quiet_hours():
            logger.info("Quiet hours — skipping heartbeat.")
            return

        available = await asyncio.to_thread(self.llm.is_available)
        if not available:
            logger.warning("LLM server is not available — skipping tick.")
            return

        # Free-think tick — companion decides what to do
        tools = self.skill_registry.get_all_tools() if self.skill_registry else []
        skill_summary = self.skill_registry.get_skill_summary() if self.skill_registry else []
        messages = self.context.build_heartbeat_prompt(
            has_tools=bool(tools), skill_summary=skill_summary
        )

        t0 = time.time()
        if tools:
            # Tool-calling mode: companion can use tools, then gives JSON decision
            logger.info(f"Heartbeat with {len(tools)} tools available")
            text, tools_used = await asyncio.to_thread(
                self.llm.chat_with_tools, messages, tools, self.skill_registry,
                max_rounds=self.max_tool_rounds
            )
            elapsed = time.time() - t0
            if tools_used:
                logger.info(f"Heartbeat tools used: {', '.join(tools_used)}")
            logger.info(f"Heartbeat tick completed in {elapsed:.1f}s")
            # Clear any pending skill output (web search sources/images etc.)
            # so it doesn't leak into the next Telegram message handler
            self._clear_pending_skill_output()
            if text:
                response = PulseResponse.from_llm_output(text)
                await self._dispatch(response)
                self._log_action(response.action, tools_used, response.message)
            else:
                self._log_action("silent", tools_used)
        else:
            # Plain mode: no tools, just JSON response
            response = await asyncio.to_thread(self.llm.chat, messages)
            elapsed = time.time() - t0
            logger.info(f"Heartbeat tick completed in {elapsed:.1f}s")
            if response:
                await self._dispatch(response)
                self._log_action(response.action, summary=response.message)

    async def _do_dev_tick(self):
        """Execute a dev tick — autonomous self-improvement session.

        The companion gets agentic tools (read code, search, write skills) and
        works on a git branch. Changes are validated and committed, then the
        human is pinged for review. All writes are sandboxed to skills/ and
        persona.json.
        """
        import subprocess

        logger.info("=== Dev tick starting ===")

        # Dev ticks are silent background work — they bypass quiet hours.
        # Notifications (the approval ping) are sent regardless.

        available = await asyncio.to_thread(self.llm.is_available)
        if not available:
            logger.warning("LLM server not available — skipping dev tick.")
            return

        # Only use dev tools during dev ticks (not the full skill registry)
        dev_skill = self.skill_registry.get_skill("dev") if self.skill_registry else None
        if not dev_skill:
            logger.warning("Dev skill not available — skipping dev tick.")
            return

        # Create a git branch for isolation
        branch_name = f"nova/dev-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        try:
            # Check for uncommitted changes first
            status = subprocess.run(
                ["git", "status", "--porcelain", "--", "skills/", "persona.json"],
                capture_output=True, text=True, cwd=self._pulse_root
            )
            if status.stdout.strip():
                logger.warning("Uncommitted changes in skills/ — skipping dev tick to avoid conflicts.")
                return

            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                capture_output=True, text=True, cwd=self._pulse_root, check=True
            )
            logger.info(f"Created dev branch: {branch_name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create dev branch: {e}")
            return

        try:
            # Build dev tools (only the dev skill's tools)
            dev_tools = dev_skill.get_tools()

            # Build a dev-specific skill registry that only has dev tools
            class _DevRegistry:
                """Minimal registry that only routes to the dev skill."""
                def get_all_tools(self):
                    return dev_tools
                def execute(self, tool_name, arguments):
                    return dev_skill.execute(tool_name, arguments)

            dev_registry = _DevRegistry()

            # Build the dev tick prompt
            messages = self.context.build_dev_prompt()

            # Run the agentic loop with dev tools only
            logger.info(f"Dev tick: {len(dev_tools)} tools available, max {self.dev_tick_max_rounds} rounds")
            text, tools_used = await asyncio.to_thread(
                self.llm.chat_with_tools, messages, dev_tools, dev_registry,
                max_rounds=self.dev_tick_max_rounds,
            )

            if tools_used:
                logger.info(f"Dev tick tools used: {', '.join(tools_used)}")

            # Check if any files were actually written
            diff_result = subprocess.run(
                ["git", "diff", "--name-only"],
                capture_output=True, text=True, cwd=self._pulse_root
            )
            changed_files = [f for f in diff_result.stdout.strip().split("\n") if f]

            # Also check for new untracked files in skills/
            untracked = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard", "skills/"],
                capture_output=True, text=True, cwd=self._pulse_root
            )
            new_files = [f for f in untracked.stdout.strip().split("\n") if f]

            all_changed = changed_files + new_files

            if all_changed:
                # Commit changes on the dev branch
                subprocess.run(
                    ["git", "add"] + all_changed,
                    cwd=self._pulse_root, check=True
                )

                commit_msg = f"[nova-dev] {text[:100] if text else 'Dev tick changes'}"
                subprocess.run(
                    ["git", "commit", "-m", commit_msg],
                    capture_output=True, text=True, cwd=self._pulse_root, check=True
                )
                logger.info(f"Dev tick committed {len(all_changed)} file(s) on {branch_name}")

                # Notify human for review
                summary = (
                    f"Hey! I did some coding on branch `{branch_name}`.\n"
                    f"Changed files: {', '.join(all_changed)}\n"
                    f"Tools used: {', '.join(set(tools_used)) if tools_used else 'none'}\n"
                    f"Have a look when you get a chance?"
                )
                if text:
                    summary += f"\n\nMy notes: {text[:300]}"

                telegram = self.channels.get("telegram")
                if telegram:
                    await telegram.send(summary)
                toast = self.channels.get("toast")
                if toast:
                    await toast.send(f"Dev tick: {len(all_changed)} file(s) on {branch_name}")

                self._log_action("dev_tick", tools_used, f"branch: {branch_name}, files: {len(all_changed)}")
            else:
                logger.info("Dev tick: no files changed.")
                self._log_action("dev_tick_noop", tools_used, "no changes")

        except Exception as e:
            logger.error(f"Dev tick failed: {e}", exc_info=True)
            self._log_action("dev_tick_error", summary=str(e)[:120])
        finally:
            # Always switch back to main, regardless of what happened
            try:
                subprocess.run(
                    ["git", "checkout", "main"],
                    capture_output=True, text=True, cwd=self._pulse_root, check=True
                )
                logger.info("Switched back to main branch.")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to switch back to main: {e}")

    async def run(self):
        """Main loop — runs forever until stopped.

        Three loops in one:
        - Fast loop: checks for due scheduled tasks every 60s (so reminders fire on time)
        - Slow loop: free-think heartbeat ticks on the configured interval (e.g., every 30m)
        - Dev loop: autonomous self-improvement ticks (e.g., every 12h)
        """
        self._running = True
        self._stop_event.clear()
        logger.info("Pulse engine starting...")
        if self._heartbeat_randomize:
            logger.info(f"Heartbeat interval: {self._heartbeat_min // 60}-{self._heartbeat_max // 60} minutes (randomized)")
        else:
            logger.info(f"Heartbeat interval: {self.interval // 60} minutes")
        logger.info(f"Quiet hours: {self.quiet_start}:00 - {self.quiet_end}:00")

        # Check LLM server availability
        if self.llm.is_available():
            logger.info("LLM server is reachable!")
        else:
            logger.warning("LLM server is not available — will retry on each tick.")

        # Start the silence clock — heartbeat fires after interval_minutes of no activity
        self._last_activity = time.time()

        # Optional startup check-in
        if self.startup_checkin:
            logger.info("Running startup check-in...")
            await self._check_due_tasks()
            await self._do_heartbeat()

        # Main loop — wake up every 60s to check schedules,
        # heartbeat fires only after enough silence (no messages exchanged)
        scheduler_interval = 60  # seconds — check for due tasks every minute
        ticks_since_dev = 0
        # Dev tick scheduling: either at a specific time or on an interval
        dev_ticks = 0
        dev_tick_ran_today = False
        if self.dev_tick_enabled:
            if self.dev_tick_schedule:
                logger.info(f"Dev ticks enabled: daily at {self.dev_tick_schedule}")
            else:
                dev_ticks = self.dev_tick_interval // scheduler_interval
                logger.info(f"Dev ticks enabled: every {self.dev_tick_interval // 60} minutes")

        while self._running:
            await self._interruptible_sleep(scheduler_interval)

            if not self._running:
                break

            if dev_ticks:
                ticks_since_dev += 1

            try:
                # Always check for due tasks (fast — just reads a file)
                await self._check_due_tasks()

                # Heartbeat fires only after enough silence since last activity
                silence = time.time() - self._last_activity
                if silence >= self.interval:
                    self._last_activity = time.time()  # reset so we don't fire every 60s
                    await self._do_heartbeat()
                    self.interval = self._roll_heartbeat_interval()

                # Dev tick — scheduled time or interval
                if self.dev_tick_enabled:
                    if self.dev_tick_schedule:
                        # Clock-based: fire once when the scheduled minute arrives
                        now = datetime.now()
                        try:
                            h, m = self.dev_tick_schedule.split(":")
                            if now.hour == int(h) and now.minute == int(m):
                                if not dev_tick_ran_today:
                                    dev_tick_ran_today = True
                                    await self._do_dev_tick()
                            else:
                                # Reset the flag once we've moved past the scheduled minute
                                if dev_tick_ran_today and now.hour != int(h):
                                    dev_tick_ran_today = False
                        except ValueError:
                            logger.warning(f"Invalid dev_tick schedule_time: {self.dev_tick_schedule}")
                    elif dev_ticks and ticks_since_dev >= dev_ticks:
                        # Interval-based: fire every N ticks
                        ticks_since_dev = 0
                        await self._do_dev_tick()
            except Exception as e:
                logger.error(f"Tick failed: {e}", exc_info=True)

    def stop(self):
        """Signal the engine to stop — exits immediately."""
        self._running = False
        self._stop_event.set()  # Wake up from any sleep instantly
        logger.info("Pulse engine stopping...")
