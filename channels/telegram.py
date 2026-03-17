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
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
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
        self._engine = None  # Set by engine after init
        self._chat_id_file = config.get("paths", {}).get("telegram_chat_id", "")
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
            .build()
        )

        # Command handlers
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("status", self._cmd_status))
        self.app.add_handler(CommandHandler("quiet", self._cmd_quiet))
        self.app.add_handler(CommandHandler("remind", self._cmd_remind))
        self.app.add_handler(CommandHandler("ping", self._cmd_ping))

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
            # Telegram has a 4096 char limit per message
            if len(message) > 4000:
                # Split into chunks
                for i in range(0, len(message), 4000):
                    await self.app.bot.send_message(
                        chat_id=self.chat_id,
                        text=message[i:i+4000]
                    )
            else:
                await self.app.bot.send_message(
                    chat_id=self.chat_id,
                    text=message
                )
            logger.info(f"Telegram message sent: {message[:50]}...")
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

    async def _reply_with_retry(self, message, text: str, retries: int = 3):
        """Send a reply with retry on timeout."""
        for attempt in range(1, retries + 1):
            try:
                await message.reply_text(text)
                return True
            except Exception as e:
                if attempt < retries:
                    logger.warning(f"Reply attempt {attempt}/{retries} failed ({e}), retrying in 3s...")
                    await asyncio.sleep(3)
                else:
                    logger.error(f"Reply failed after {retries} attempts: {e}")
                    return False

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

            # Show typing indicator
            await update.effective_chat.send_action("typing")

            # Get response from companion via the engine
            reply, tools_used = await self._engine.handle_message(user_message, source="telegram")

            if reply:
                # Show which tools were used (transparency!)
                if tools_used:
                    tool_names = ", ".join(dict.fromkeys(tools_used))  # dedupe, preserve order
                    await self._reply_with_retry(update.message, f"🔧 {tool_names}")
                await self._reply_with_retry(update.message, reply)

                # Send any pending media/sources from web search
                if self._engine and self._engine.skill_registry:
                    web_skill = self._engine.skill_registry.get_skill("web_search")
                    if web_skill:
                        # Show sources so the user can verify
                        if web_skill.pending_sources:
                            source_lines = ["📎 Sources:"]
                            for s in web_skill.pending_sources:
                                source_lines.append(f"  • {s['title']}\n    {s['url']}")
                            await self._reply_with_retry(update.message, "\n".join(source_lines))
                            web_skill.pending_sources.clear()
                        # Send images inline
                        if web_skill.pending_images:
                            for img_url in web_skill.pending_images:
                                await self.send_photo(img_url)
                            web_skill.pending_images.clear()
            else:
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

            # Show typing indicator
            await update.effective_chat.send_action("typing")

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

            if reply:
                if tools_used:
                    tool_names = ", ".join(dict.fromkeys(tools_used))
                    await self._reply_with_retry(update.message, f"🔧 {tool_names}")
                await self._reply_with_retry(update.message, reply)
            else:
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
