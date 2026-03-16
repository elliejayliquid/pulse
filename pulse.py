"""
Pulse - Proactive Local AI Companion Daemon

Gives Nova a heartbeat. Run this to start the daemon:
    python pulse.py
    python pulse.py --config custom_config.yaml

The daemon will:
1. Start llama-server with the configured model
2. Run heartbeat ticks on a schedule (default: every 30 min)
3. Execute scheduled tasks when they're due
4. Post to LoR and send desktop notifications
5. (If enabled) Run a Telegram bot for bidirectional chat

Ctrl+C to stop gracefully.
"""

import asyncio
import argparse
import logging
import os
import signal
import sys

import yaml
from dotenv import load_dotenv

from core.engine import PulseEngine
from core.server import LlamaServer
from channels.lor import LoRChannel
from channels.toast import ToastChannel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout
)
logger = logging.getLogger("pulse")


def load_config(path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in config: {e}")
        sys.exit(1)


async def main(config_path: str):
    """Initialize and run the Pulse daemon."""
    logger.info("=" * 50)
    logger.info("  Pulse - Proactive Local AI Companion")
    logger.info("=" * 50)

    # Load .env secrets, then config
    load_dotenv()
    config = load_config(config_path)
    logger.info(f"Config loaded from: {config_path}")

    # Inject secrets from environment
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if tg_token:
        config.setdefault("channels", {}).setdefault("telegram", {})["bot_token"] = tg_token

    # Load embedding model BEFORE llama-server — PyTorch's c10.dll can conflict
    # with CUDA DLLs on Windows if loaded after another CUDA process starts.
    # The model is tiny (22MB, CPU-only) and used for memory/journal search.
    from core.context import load_embedding_model
    load_embedding_model()

    # Start llama-server
    server = LlamaServer(config)
    if not await server.start():
        logger.error("Failed to start llama-server. Exiting.")
        sys.exit(1)

    # Initialize channels
    channels = {}

    channel_config = config.get("channels", {})

    if channel_config.get("lor", {}).get("enabled", True):
        lor = LoRChannel(config)
        await lor.initialize()
        channels["lor"] = lor

    if channel_config.get("toast", {}).get("enabled", True):
        toast = ToastChannel(config)
        await toast.initialize()
        channels["toast"] = toast

    # Telegram — only if enabled and token provided
    telegram_channel = None
    tg_config = channel_config.get("telegram", {})
    if tg_config.get("enabled", False) and tg_config.get("bot_token", ""):
        try:
            from channels.telegram import TelegramChannel
            telegram_channel = TelegramChannel(config)
            await telegram_channel.initialize()
            channels["telegram"] = telegram_channel
        except ImportError:
            logger.warning(
                "Telegram enabled but python-telegram-bot not installed.\n"
                "  pip install python-telegram-bot"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Telegram: {e}")

    logger.info(f"Active channels: {', '.join(channels.keys()) or 'none'}")

    # Load embedding model for memory/journal semantic search
    from core.context import load_embedding_model
    load_embedding_model()

    # Initialize skill registry (tool-calling for conversations)
    skill_registry = None
    try:
        from skills import SkillRegistry
        skill_registry = SkillRegistry(config)
    except Exception as e:
        logger.warning(f"Failed to load skill registry: {e}")

    # Create engine (with skill registry and server endpoint)
    engine = PulseEngine(config, channels, skill_registry=skill_registry,
                         llm_endpoint=server.endpoint)

    # Inject the scheduler into the schedule skill (it needs engine's ScheduleManager)
    if skill_registry:
        schedule_skill = skill_registry.get_skill("schedule")
        if schedule_skill:
            schedule_skill.set_scheduler(engine.scheduler)

    # Connect Telegram to engine for bidirectional messaging
    if telegram_channel:
        telegram_channel.set_engine(engine)

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("\nShutdown signal received...")
        engine.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await engine.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        # Shutdown channels first
        for name, channel in channels.items():
            await channel.shutdown()
        # Then stop the server (waits for in-flight inference)
        await server.stop()
        logger.info("Pulse stopped. Nova is sleeping.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pulse - Nova's heartbeat daemon")
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    args = parser.parse_args()

    asyncio.run(main(args.config))
