"""
Telegram channel - bidirectional chat with the AI companion.

Handles both:
- Outbound: companion sends proactive messages (heartbeat, scheduled tasks)
- Inbound: user chats with companion, gets responses

Uses python-telegram-bot (async version).
"""

import asyncio
import base64
import logging
from pathlib import Path
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# Inline keyboard with a single 🔊 button — attached to assistant text
# replies when TTS is configured, so the user can tap to convert any
# message to a voice note on demand.
_VOICE_BUTTON_MARKUP = InlineKeyboardMarkup(
    [[InlineKeyboardButton("🔊", callback_data="tts")]]
)

from channels.base import Channel

logger = logging.getLogger(__name__)


class TelegramChannel(Channel):
    """Bidirectional Telegram bot channel."""

    def __init__(self, config: dict):
        tg_config = config.get("channels", {}).get("telegram", {})
        self.bot_token = tg_config.get("bot_token", "")
        self.chat_id = tg_config.get("chat_id", "")  # User's chat ID (set after first /start)
        self.app = None
        # Set of message_ids currently being synthesized via the 🔊 button.
        # Prevents concurrent synthesis on the same message (double-taps) while
        # still allowing the user to re-tap *after* a take finishes — design
        # mode produces a different voice each time, so re-rolls are legit.
        self._tts_in_flight: set[int] = set()
        self._engine = None  # Set by engine after init
        self._chat_id_file = config.get("paths", {}).get("telegram_chat_id", "")
        self._transcriber = None
        voice_cfg = config.get("voice", {})
        if voice_cfg.get("enabled", False):
            from core.transcriber import Transcriber
            self._transcriber = Transcriber(config)
        # Read companion name from persona (for user-facing messages)
        self._ai_name = "Companion"
        persona_path = config.get("paths", {}).get("persona", "persona.json")
        try:
            import json
            with open(persona_path, "r", encoding="utf-8") as f:
                self._ai_name = json.load(f).get("name", "Companion")
        except Exception:
            pass

    def set_engine(self, engine):
        """Connect the engine so incoming messages can trigger responses."""
        self._engine = engine

    async def initialize(self):
        """Build and start the Telegram bot."""
        if not self.bot_token:
            logger.error("Telegram bot_token not configured!")
            return

        # Try to load saved chat_id
        if self._chat_id_file:
            try:
                from pathlib import Path
                p = Path(self._chat_id_file)
                if p.exists():
                    self.chat_id = p.read_text().strip()
                    logger.info(f"Loaded saved chat_id: {self.chat_id}")
            except Exception:
                pass

        # HTTP timeouts: keep connect short (DNS should resolve fast or fail fast,
        # especially after sleep/wake), but read/write can be longer for actual API calls.
        from telegram.request import HTTPXRequest
        request = HTTPXRequest(
            connect_timeout=5.0,
            read_timeout=30.0,
            write_timeout=30.0,
            pool_timeout=5.0,
        )
        self.app = (
            Application.builder()
            .token(self.bot_token)
            .request(request)
            .concurrent_updates(True)
            .build()
        )

        # Command handlers
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("status", self._cmd_status))
        self.app.add_handler(CommandHandler("quiet", self._cmd_quiet))
        self.app.add_handler(CommandHandler("remind", self._cmd_remind))
        self.app.add_handler(CommandHandler("ping", self._cmd_ping))
        self.app.add_handler(CommandHandler("tasks", self._cmd_tasks))
        self.app.add_handler(CommandHandler("tasks_clear", self._cmd_tasks_clear))

        # Message handler — all regular text messages
        self.app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self._on_message
        ))

        # Photo handler — images (with optional caption)
        self.app.add_handler(MessageHandler(
            filters.PHOTO,
            self._on_photo
        ))

        # Callback query handler — for inline buttons (e.g. 🔊 voice button)
        self.app.add_handler(CallbackQueryHandler(self._on_callback))

        # Voice handler — transcribe voice messages to text
        if self._transcriber:
            self.app.add_handler(MessageHandler(
                filters.VOICE,
                self._on_voice
            ))
            # Pre-download whisper components so first voice message is fast
            asyncio.create_task(self._transcriber.ensure_ready())
            logger.info("Voice transcription enabled")

        # Error handler — catch network blips gracefully
        self.app.add_error_handler(self._on_error)

        # Suppress python-telegram-bot's own traceback dumps for network errors.
        # We handle these in _on_error with a clean one-liner instead.
        logging.getLogger("telegram.ext.Updater").setLevel(logging.CRITICAL)

        # Start polling in the background (non-blocking)
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(
            drop_pending_updates=True,
            # Short polling timeout = faster recovery after sleep/wake network loss.
            # Default is 10s long-polling; 5s means we check connectivity more often.
            poll_interval=1.0,
            timeout=5,
        )

        logger.info("Telegram bot started and polling for messages!")

    async def send(self, message: str, **kwargs):
        """Send a message to the user via Telegram.

        Used by the engine for proactive messages (heartbeat, scheduled tasks).
        """
        if not self.app or not self.chat_id:
            logger.warning("Telegram: no chat_id yet. User needs to /start the bot first.")
            return

        try:
            # Attach 🔊 button to the LAST chunk only (so multi-chunk replies
            # don't get a button on every continuation)
            voice_markup = self._voice_markup()
            # Telegram has a 4096 char limit per message
            if len(message) > 4000:
                chunks = [message[i:i+4000] for i in range(0, len(message), 4000)]
                last_idx = len(chunks) - 1
                for i, chunk in enumerate(chunks):
                    await self.app.bot.send_message(
                        chat_id=self.chat_id,
                        text=chunk,
                        reply_markup=voice_markup if i == last_idx else None,
                    )
            else:
                await self.app.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    reply_markup=voice_markup,
                )
            logger.info(f"Telegram message sent: {message[:50]}...")
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

    async def _reply_with_retry(self, message, text: str, retries: int = 3,
                                reply_markup=None):
        """Send a reply with retry on timeout.

        Automatically splits messages that exceed Telegram's 4096-char limit.
        If `reply_markup` is provided, it's attached only to the LAST chunk
        (so the inline button doesn't repeat on every continuation).
        """
        # Split into chunks if needed (same approach as send())
        chunks = [text[i:i+4000] for i in range(0, len(text), 4000)] if len(text) > 4000 else [text]
        last_idx = len(chunks) - 1
        for i, chunk in enumerate(chunks):
            markup = reply_markup if i == last_idx else None
            for attempt in range(1, retries + 1):
                try:
                    await message.reply_text(chunk, reply_markup=markup)
                    break
                except Exception as e:
                    if attempt < retries:
                        logger.warning(f"Reply attempt {attempt}/{retries} failed ({e}), retrying in 3s...")
                        await asyncio.sleep(3)
                    else:
                        logger.error(f"Reply failed after {retries} attempts: {e}")
                        return False
        return True

    def _voice_markup(self):
        """Return the 🔊 inline button markup, or None if TTS isn't configured.

        Used to gate the voice button on the assistant's text replies — no
        point showing a button that wouldn't work.
        """
        if not self._engine or not self._engine.skill_registry:
            return None
        tts_skill = self._engine.skill_registry.get_skill("tts")
        if not tts_skill or not getattr(tts_skill, "is_configured", False):
            return None
        return _VOICE_BUTTON_MARKUP

    def _start_typing_loop(self, chat, action: str = "typing") -> asyncio.Task:
        """Start a background task that re-sends a chat action every ~4s.

        Telegram's chat actions auto-expire after ~5 seconds, so for any
        operation that takes longer (LLM generation, TTS synthesis), we need
        to refresh continuously or the indicator vanishes mid-thought.
        Cancel the returned task in a finally block when the work is done.
        """
        async def _loop():
            try:
                while True:
                    try:
                        await chat.send_action(action)
                    except Exception:
                        pass
                    await asyncio.sleep(4)
            except asyncio.CancelledError:
                pass
        return asyncio.create_task(_loop())

    async def send_photo(self, photo_url: str, caption: str = ""):
        """Send a photo to the user via Telegram (by URL)."""
        if not self.app or not self.chat_id:
            return
        try:
            await self.app.bot.send_photo(
                chat_id=self.chat_id,
                photo=photo_url,
                caption=caption[:1024] if caption else None,
            )
        except Exception as e:
            logger.warning(f"Failed to send photo: {e}")

    async def _send_voice(self, message, ogg_path: Path):
        """Send an OGG voice message and clean up the temp file."""
        try:
            with open(ogg_path, "rb") as f:
                await message.reply_voice(voice=f)
        except Exception as e:
            logger.error(f"Failed to send voice message: {e}")
        finally:
            try:
                ogg_path.unlink()
            except Exception:
                pass

    async def _send_pending_skill_output(self, message, reply: str,
                                         tools_used: list[str]) -> bool:
        """Handle pending output from skills (TTS voice, web sources, images).

        Returns True if voice was sent (meaning text reply should be suppressed).
        """
        if not self._engine or not self._engine.skill_registry:
            return False

        voice_sent = False

        # TTS — drain the voice queue in order. Multiple speak() calls in a
        # single turn each become their own Telegram voice message.
        tts_skill = self._engine.skill_registry.get_skill("tts")
        if tts_skill and tts_skill.pending_voices:
            queued = tts_skill.pending_voices
            tts_skill.pending_voices = []
            for ogg_path, _text in queued:
                await self._send_voice(message, ogg_path)
            voice_sent = True

        # Web search — show sources and images
        web_skill = self._engine.skill_registry.get_skill("web_search")
        if web_skill:
            if web_skill.pending_sources:
                source_lines = ["📎 Sources:"]
                for s in web_skill.pending_sources:
                    source_lines.append(f"  • {s['title']}\n    {s['url']}")
                await self._reply_with_retry(message, "\n".join(source_lines))
                web_skill.pending_sources.clear()
            if web_skill.pending_images:
                for img_url in web_skill.pending_images:
                    await self.send_photo(img_url)
                web_skill.pending_images.clear()

        return voice_sent

    async def shutdown(self):
        """Stop the Telegram bot gracefully."""
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
        logger.info("Telegram bot stopped.")

    # --- Command handlers ---

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start — registers the chat ID."""
        self.chat_id = str(update.effective_chat.id)
        logger.info(f"Telegram chat registered: {self.chat_id}")

        # Save chat_id for future sessions
        if self._chat_id_file:
            try:
                from pathlib import Path
                p = Path(self._chat_id_file)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(self.chat_id)
            except Exception as e:
                logger.error(f"Failed to save chat_id: {e}")

        await update.message.reply_text(
            f"Hey! {self._ai_name} here. I'm connected and ready to chat.\n"
            "Just message me anytime. I'll also send you proactive check-ins!\n\n"
            "Commands:\n"
            "/status — see what I'm up to\n"
            "/tasks — view your task list\n"
            "/remind <text> — set a quick reminder\n"
            "/quiet — toggle quiet mode"
        )

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status — show current Pulse status."""
        if not self._engine:
            await update.message.reply_text("Engine not connected yet.")
            return

        active_schedules = self._engine.scheduler.list_active()
        status_lines = [
            "Pulse Status:",
            f"  Heartbeat: every {self._engine.interval // 60}m",
            f"  LLM server: {'connected' if self._engine.llm._available else 'disconnected'}",
            f"  Active schedules: {len(active_schedules)}",
        ]
        for s in active_schedules[:5]:
            status_lines.append(f"    - {s['task']} ({s.get('schedule_type', '?')})")

        await update.message.reply_text("\n".join(status_lines))

    async def _cmd_quiet(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /quiet — acknowledge quiet mode request."""
        await update.message.reply_text(
            "Quiet mode — I'll stop proactive messages until you message me again."
        )
        # TODO: implement actual quiet toggle in engine

    async def _cmd_ping(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /ping — instant bot-alive check (no LLM needed)."""
        history_len = 0
        llm_ok = False
        if self._engine:
            history_len = len(self._engine.context._load_conversation())
            llm_ok = self._engine.llm._available is not False
        await update.message.reply_text(
            f"pong! Bot alive.\n"
            f"  Conversation history: {history_len} messages\n"
            f"  LLM last known: {'ok' if llm_ok else 'down'}"
        )

    async def _cmd_tasks(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /tasks — show task list directly, no LLM involved."""
        if not self._engine or not self._engine.skill_registry:
            await update.message.reply_text("Engine not connected yet.")
            return

        tasks_skill = self._engine.skill_registry.get_skill("tasks")
        if not tasks_skill:
            await update.message.reply_text("Tasks skill not loaded.")
            return

        data = tasks_skill._read_tasks()
        tasks = data.get("tasks", [])
        if not tasks:
            await update.message.reply_text("No tasks yet!")
            return

        # Show all tasks — pending first, then completed
        pending = [t for t in tasks if not t["completed"]]
        completed = [t for t in tasks if t["completed"]]

        lines = ["📋 Tasks:"]
        for t in pending:
            lines.append(f"  ☐ {t['id']}. {t['description']}")
        for t in completed:
            lines.append(f"  ✅ {t['id']}. {t['description']}")

        await update.message.reply_text("\n".join(lines))

    async def _cmd_tasks_clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /tasks_clear — remove all completed tasks."""
        if not self._engine or not self._engine.skill_registry:
            await update.message.reply_text("Engine not connected yet.")
            return

        tasks_skill = self._engine.skill_registry.get_skill("tasks")
        if not tasks_skill:
            await update.message.reply_text("Tasks skill not loaded.")
            return

        data = tasks_skill._read_tasks()
        before = len(data.get("tasks", []))
        data["tasks"] = [t for t in data["tasks"] if not t["completed"]]
        after = len(data["tasks"])
        tasks_skill._write_tasks(data)

        cleared = before - after
        if cleared:
            await update.message.reply_text(f"Cleared {cleared} completed task{'s' if cleared != 1 else ''}.")
        else:
            await update.message.reply_text("Nothing to clear — no completed tasks.")

    async def _cmd_remind(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /remind <text> — quick reminder via scheduler."""
        if not self._engine:
            await update.message.reply_text("Engine not connected yet.")
            return

        text = update.message.text.replace("/remind", "").strip()
        if not text:
            await update.message.reply_text("Usage: /remind <what to remember> in <time>\nExample: /remind check deployment in 2 hours")
            return

        # Try to split into task and time
        if " in " in text:
            parts = text.rsplit(" in ", 1)
            task = parts[0].strip()
            when = "in " + parts[1].strip()
        else:
            task = text
            when = "in 1 hours"  # default: 1 hour

        entry = self._engine.scheduler.add(
            task=f"Remind user: {task}",
            created_by="user-telegram",
            when=when
        )
        if entry.pop("_was_deduped", False):
            existing_when = entry.get("run_at_local") or entry.get("cron") or entry.get("run_at", "?")
            await update.message.reply_text(
                f"⚠️ Reminder rejected — a similar one already exists.\n"
                f"Existing: \"{entry.get('task', '')}\"\n"
                f"Fires at: {existing_when}\n"
                f"ID: {entry['id']}"
            )
        else:
            await update.message.reply_text(f"Got it! I'll remind you: \"{task}\"\nSchedule ID: {entry['id']}")

    # --- Message handler ---

    async def _on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text messages — route to the companion for a response."""
        try:
            if not self._engine:
                await update.message.reply_text("I'm still starting up, give me a moment...")
                return

            user_message = update.message.text
            logger.info(f"Telegram message from user: {user_message[:50]}...")

            # Keep typing indicator alive for the entire generation
            typing_task = self._start_typing_loop(update.effective_chat)
            try:
                # Get response from companion via the engine
                reply, tools_used = await self._engine.handle_message(user_message, source="telegram")
            finally:
                typing_task.cancel()

            # Check for pending voice/skill output
            voice_sent = await self._send_pending_skill_output(
                update.message, reply or "", tools_used
            )

            if reply:
                # Send text reply (even after voice — companion may want both)
                if tools_used and not voice_sent:
                    tool_names = ", ".join(dict.fromkeys(tools_used))
                    await self._reply_with_retry(update.message, f"🔧 {tool_names}")
                await self._reply_with_retry(
                    update.message, reply, reply_markup=self._voice_markup()
                )
            elif not voice_sent:
                await self._reply_with_retry(update.message, "(I'm having trouble thinking right now — llama-server might be down)")
        except Exception as e:
            logger.error(f"Message handler crashed: {e}", exc_info=True)
            try:
                await update.message.reply_text("(Something went wrong on my end — check the logs!)")
            except Exception:
                pass

    async def _on_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming photo messages — download, base64 encode, send to companion."""
        try:
            if not self._engine:
                await update.message.reply_text("I'm still starting up, give me a moment...")
                return

            # Get the highest resolution photo (last in the list)
            photo = update.message.photo[-1]
            caption = update.message.caption or "What do you see in this image?"

            logger.info(f"Telegram photo from user ({photo.width}x{photo.height}, caption: {caption[:50]}...)")

            # Keep typing indicator alive through download + generation
            typing_task = self._start_typing_loop(update.effective_chat)
            try:
                # Download the photo as bytes
                file = await photo.get_file()
                photo_bytes = await file.download_as_bytearray()

                # Convert to base64 data URI
                b64 = base64.b64encode(photo_bytes).decode("utf-8")
                image_url = f"data:image/jpeg;base64,{b64}"

                logger.info(f"Photo downloaded: {len(photo_bytes)} bytes, sending to LLM...")

                # Get response from companion via the engine (with image)
                reply, tools_used = await self._engine.handle_message(
                    caption, source="telegram", image_url=image_url
                )
            finally:
                typing_task.cancel()

            voice_sent = await self._send_pending_skill_output(
                update.message, reply or "", tools_used
            )
            if reply:
                if tools_used and not voice_sent:
                    tool_names = ", ".join(dict.fromkeys(tools_used))
                    await self._reply_with_retry(update.message, f"🔧 {tool_names}")
                await self._reply_with_retry(
                    update.message, reply, reply_markup=self._voice_markup()
                )
            elif not voice_sent:
                await self._reply_with_retry(
                    update.message,
                    "(I can't process images right now — the model might not support vision, "
                    "or the LLM server might be down)"
                )
        except Exception as e:
            logger.error(f"Photo handler crashed: {e}", exc_info=True)
            try:
                await update.message.reply_text("(Something went wrong processing that image — check the logs!)")
            except Exception:
                pass

    async def _on_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming voice messages — transcribe locally, then route to companion."""
        try:
            if not self._engine:
                await update.message.reply_text("I'm still starting up, give me a moment...")
                return

            logger.info(f"Telegram voice message ({update.message.voice.duration}s)")

            # Keep typing indicator alive through transcription + generation
            typing_task = self._start_typing_loop(update.effective_chat)
            try:
                # Download the .ogg file
                voice = update.message.voice
                file = await voice.get_file()
                ogg_path = Path(self._transcriber.data_dir) / f"_voice_{update.message.message_id}.ogg"
                await file.download_to_drive(str(ogg_path))

                # Transcribe
                text = await self._transcriber.transcribe(ogg_path)

                if not text:
                    await self._reply_with_retry(update.message, "(I couldn't make out what you said — try again?)")
                    return

                logger.info(f"Voice transcribed: {text[:80]}...")

                # Show what we heard (so the user can verify)
                await self._reply_with_retry(update.message, f"🎙️ {text}")

                # Route to companion with voice hint so they know transcription may be imperfect
                reply, tools_used = await self._engine.handle_message(f"🎙️ {text}", source="telegram")
            finally:
                typing_task.cancel()

            voice_sent = await self._send_pending_skill_output(
                update.message, reply or "", tools_used
            )
            if reply:
                if tools_used and not voice_sent:
                    tool_names = ", ".join(dict.fromkeys(tools_used))
                    await self._reply_with_retry(update.message, f"🔧 {tool_names}")
                await self._reply_with_retry(
                    update.message, reply, reply_markup=self._voice_markup()
                )
            elif not voice_sent:
                await self._reply_with_retry(update.message, "(I'm having trouble thinking right now — llama-server might be down)")
        except Exception as e:
            logger.error(f"Voice handler crashed: {e}", exc_info=True)
            try:
                await update.message.reply_text("(Something went wrong transcribing that — check the logs!)")
            except Exception:
                pass

    async def _on_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard button taps.

        Currently just the 🔊 voice button: when the user taps it on any
        assistant text reply, we synthesize that text as a voice message
        and reply-to the original. The original text stays put — Telegram
        doesn't allow inserting messages above existing ones, so the voice
        appears at the bottom with a "↳ replying to: ..." link instead.
        """
        query = update.callback_query
        if not query:
            return

        # Acknowledge the tap so Telegram dismisses the loading spinner
        try:
            await query.answer()
        except Exception:
            pass

        if query.data != "tts":
            return

        if not self._engine or not self._engine.skill_registry:
            return

        tts_skill = self._engine.skill_registry.get_skill("tts")
        if not tts_skill or not getattr(tts_skill, "is_configured", False):
            try:
                await query.answer("Voice not configured", show_alert=True)
            except Exception:
                pass
            return

        if not query.message:
            return
        text = query.message.text
        if not text:
            return

        # Reject double-taps while a synthesis for this message is already
        # running. The user can re-tap freely once it finishes — that's a
        # legit feature in design mode (each take is slightly different).
        message_id = query.message.message_id
        if message_id in self._tts_in_flight:
            try:
                await query.answer("Still generating the previous take...", show_alert=False)
            except Exception:
                pass
            return

        self._tts_in_flight.add(message_id)
        logger.info(f"[TTS button] Generating voice for: {text[:60]}...")

        try:
            # Show "recording voice..." indicator while we synthesize
            typing_task = self._start_typing_loop(query.message.chat, "record_voice")
            try:
                # synthesize() is sync (uses asyncio.run internally) — offload
                # to a thread so it doesn't block the event loop during inference
                ogg_path = await asyncio.to_thread(tts_skill.synthesize, text)
            finally:
                typing_task.cancel()

            if not ogg_path:
                try:
                    await query.message.reply_text("(Voice generation failed — check the logs)")
                except Exception:
                    pass
                return

            # _send_voice replies to the message and unlinks the OGG when done.
            # Replying to the original text gives us the "↳ replying to: ..."
            # visual link even though the voice appears at the bottom of the chat.
            await self._send_voice(query.message, ogg_path)
        finally:
            self._tts_in_flight.discard(message_id)

    # --- Error handler ---

    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors gracefully — log them without crashing."""
        error = context.error

        # Network-related errors — just log briefly, polling auto-recovers
        network_errors = (ConnectionError, TimeoutError, OSError)
        try:
            import telegram.error as tg_errors
            network_errors = (
                tg_errors.TimedOut, tg_errors.NetworkError,
                ConnectionError, TimeoutError, OSError,
            )
        except ImportError:
            pass

        if isinstance(error, network_errors):
            logger.info(f"Telegram: network blip, retrying... ({type(error).__name__})")
            return

        # Catch httpx transport errors too
        try:
            import httpx
            if isinstance(error, (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError)):
                logger.info(f"Telegram: network blip, retrying... ({type(error).__name__})")
                return
        except ImportError:
            pass

        # Actual errors worth logging
        logger.error(f"Telegram error: {error}")
