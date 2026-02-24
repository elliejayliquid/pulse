"""
Pulse - Proactive Local AI Companion Daemon

Gives Nova a heartbeat. Run this to start the daemon:
    python pulse.py
    python pulse.py --config custom_config.yaml

The daemon will:
1. Connect to LM Studio's OpenAI-compatible API
2. Run heartbeat ticks on a schedule (default: every 2 hours)
3. Execute scheduled tasks when they're due
4. Post to LoR and send desktop notifications

Ctrl+C to stop gracefully.
"""

import asyncio
import argparse
import logging
import signal
import sys

import yaml

from core.engine import PulseEngine
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

    # Load config
    config = load_config(config_path)
    logger.info(f"Config loaded from: {config_path}")

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

    logger.info(f"Active channels: {', '.join(channels.keys()) or 'none'}")

    # Create and run engine
    engine = PulseEngine(config, channels)

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
        # Shutdown channels
        for name, channel in channels.items():
            await channel.shutdown()
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
