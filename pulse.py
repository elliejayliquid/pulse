"""
Pulse - Proactive Local AI Companion Daemon

Gives your AI companion a heartbeat. Run this to start the daemon:
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
from pathlib import Path

import yaml
from dotenv import load_dotenv

from core.engine import PulseEngine
from core.server import LlamaServer
from channels.toast import ToastChannel

# Configure logging — console + rotating file
_log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_log_datefmt = "%H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=_log_format,
    datefmt=_log_datefmt,
    stream=sys.stdout,
)

# File handler — set up after persona is resolved (see _setup_persona_logging)
from logging.handlers import RotatingFileHandler
_log_dir = Path(__file__).parent / "logs"
_log_dir.mkdir(exist_ok=True)

# Start with a shared pulse.log for pre-persona startup messages
_file_handler = RotatingFileHandler(
    _log_dir / "pulse.log", maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_file_handler.setFormatter(logging.Formatter(_log_format, datefmt=_log_datefmt))
_file_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(_file_handler)


def _setup_persona_logging(persona_name: str):
    """Switch file logging to a per-persona log file (e.g. logs/nova.log)."""
    global _file_handler
    root = logging.getLogger()
    root.removeHandler(_file_handler)
    _file_handler.close()

    _file_handler = RotatingFileHandler(
        _log_dir / f"{persona_name}.log",
        maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    _file_handler.setFormatter(logging.Formatter(_log_format, datefmt=_log_datefmt))
    _file_handler.setLevel(logging.INFO)
    root.addHandler(_file_handler)
    logger.info(f"Logging to logs/{persona_name}.log")

logger = logging.getLogger("pulse")

# Silence noisy HTTP request logs (httpx logs every Telegram poll)
logging.getLogger("httpx").setLevel(logging.WARNING)


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


def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base. Overlay values win on conflict.

    Dicts merge recursively; everything else (strings, lists, numbers)
    is replaced outright by the overlay value.
    """
    merged = base.copy()
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_persona(config: dict, persona_name: str | None, pulse_root: Path) -> dict:
    """Load and merge a persona's config overlay if one is active.

    Resolution order:
    1. --persona CLI flag (explicit)
    2. active_persona key in base config (default)
    3. None — no persona, use base config as-is

    Returns the merged config.
    """
    name = persona_name or config.get("active_persona")
    if not name:
        return config

    persona_dir = pulse_root / "personas" / name
    if not persona_dir.is_dir():
        logger.error(f"Persona directory not found: {persona_dir}")
        sys.exit(1)

    logger.info(f"Active persona: {name} ({persona_dir})")

    # Load persona's .env if it exists (persona-specific API keys, etc.)
    persona_env = persona_dir / ".env"
    if persona_env.exists():
        load_dotenv(persona_env, override=True)
        logger.info(f"  Loaded persona .env: {persona_env}")

    # Load and merge persona config overlay
    persona_config_path = persona_dir / "config.yaml"
    persona_config = {}
    if persona_config_path.exists():
        persona_config = load_config(str(persona_config_path))
        config = deep_merge(config, persona_config)
        logger.info(f"  Merged persona config: {persona_config_path}")

    # Auto-default data paths to persona directory.
    # Any path not explicitly set in the persona's config.yaml gets
    # pointed to personas/<name>/data/ — zero config for new personas.
    persona_data = persona_dir / "data"
    persona_paths = persona_config.get("paths", {})
    default_paths = {
        "memories": str(persona_data / "memories"),
        "journal": str(persona_data / "journal"),
        "tasks": str(persona_data / "tasks.json"),
        "dev_journal": str(persona_data / "dev_journal.json"),
        "schedules": str(persona_data / "schedules.json"),
        "conversation": str(persona_data / "conversation.json"),
        "conversation_archive": str(persona_data / "conversation_archive.jsonl"),
        "telegram_chat_id": str(persona_data / "telegram_chat_id.txt"),
        "action_log": str(persona_data / "action_log.json"),
        "usage": str(persona_data / "usage.json"),
        "paintings": str(persona_data / "paintings"),
    }
    config.setdefault("paths", {})
    for key, default in default_paths.items():
        if key not in persona_paths:
            config["paths"][key] = default

    # Auto-set persona identity path (.yaml preferred over .json)
    persona_yaml = persona_dir / "persona.yaml"
    persona_json = persona_dir / "persona.json"
    persona_identity = persona_yaml if persona_yaml.exists() else persona_json
    if persona_identity.exists():
        config["paths"]["persona"] = str(persona_identity)
        logger.info(f"  Using persona identity: {persona_identity}")

    # Store persona metadata for skills/context that need it
    config["_persona_name"] = name
    config["_persona_dir"] = str(persona_dir)

    return config


async def main(config_path: str, persona_name: str | None = None):
    """Initialize and run the Pulse daemon."""
    logger.info("=" * 50)
    logger.info("  Pulse - Proactive Local AI Companion")
    logger.info("=" * 50)

    # Load .env secrets, then config
    load_dotenv()
    config = load_config(config_path)
    logger.info(f"Config loaded from: {config_path}")

    # Inject pulse root directory for dev skill and engine
    pulse_root = Path(__file__).parent.resolve()
    config["_pulse_root"] = str(pulse_root)

    # Resolve persona overlay (merges persona config over base)
    config = resolve_persona(config, persona_name, pulse_root)

    # Switch to per-persona log file now that we know who we are
    active_persona = config.get("_persona_name")
    if active_persona:
        _setup_persona_logging(active_persona)

    # Inject secrets from environment (after persona .env may have loaded)
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if tg_token:
        config.setdefault("channels", {}).setdefault("telegram", {})["bot_token"] = tg_token

    # Load embedding model BEFORE llama-server — PyTorch's c10.dll can conflict
    # with CUDA DLLs on Windows if loaded after another CUDA process starts.
    # The model is tiny (22MB, CPU-only) and used for memory/journal search.
    # Always local regardless of provider — embeddings power memory dedup & search.
    from core.context import load_embedding_model
    load_embedding_model()

    # Resolve LLM provider
    provider_config = config.get("provider", {})
    provider_type = provider_config.get("type", "local")
    server = None
    endpoint = None
    api_key = ""

    if provider_type == "local":
        # Local llama.cpp — start and manage the server process
        server = LlamaServer(config)
        if not await server.start():
            logger.error("Failed to start llama-server. Exiting.")
            sys.exit(1)
        endpoint = server.endpoint
    else:
        # Cloud API — resolve endpoint and key, no server needed
        api_key_env = provider_config.get("api_key_env", "")
        api_key = os.getenv(api_key_env, "") if api_key_env else ""
        if not api_key:
            logger.error(
                f"Provider '{provider_type}' requires an API key. "
                f"Set {api_key_env or 'provider.api_key_env'} in .env"
            )
            sys.exit(1)

        # Default base URLs per provider (override with provider.base_url)
        default_urls = {
            "openai": "https://api.openai.com/v1",
            "openrouter": "https://openrouter.ai/api/v1",
            "anthropic": "https://api.anthropic.com/v1",
        }
        endpoint = (
            provider_config.get("base_url")
            or default_urls.get(provider_type, "")
        )
        if not endpoint:
            logger.error(
                f"No base_url for provider '{provider_type}'. "
                "Set provider.base_url in config.yaml"
            )
            sys.exit(1)

        model_name = provider_config.get("model", "")
        logger.info(f"Using cloud provider: {provider_type}")
        logger.info(f"  Endpoint: {endpoint}")
        logger.info(f"  Model: {model_name or '(default)'}")

    # Initialize channels
    channels = {}

    channel_config = config.get("channels", {})

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
            # Retry up to 3 times — Telegram API can timeout on first try
            for attempt in range(1, 4):
                try:
                    await telegram_channel.initialize()
                    channels["telegram"] = telegram_channel
                    break
                except Exception as e:
                    if attempt < 3:
                        logger.warning(f"Telegram init attempt {attempt}/3 failed ({e}), retrying in 5s...")
                        await asyncio.sleep(5)
                    else:
                        logger.error(f"Telegram init failed after 3 attempts: {e}")
        except ImportError:
            logger.warning(
                "Telegram enabled but python-telegram-bot not installed.\n"
                "  pip install python-telegram-bot"
            )

    logger.info(f"Active channels: {', '.join(channels.keys()) or 'none'}")

    # Initialize skill registry (tool-calling for conversations)
    skill_registry = None
    try:
        from skills import SkillRegistry
        skill_registry = SkillRegistry(config)
    except Exception as e:
        logger.warning(f"Failed to load skill registry: {e}")

    # Create engine (with skill registry and provider details)
    engine = PulseEngine(config, channels, skill_registry=skill_registry,
                         llm_endpoint=endpoint, api_key=api_key)

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
        # Free TTS model VRAM before llama-server finishes winding down,
        # so the LLM has extra headroom for in-flight inference.
        if engine.skill_registry:
            tts_skill = engine.skill_registry.get_skill("tts")
            if tts_skill and hasattr(tts_skill, "shutdown"):
                try:
                    tts_skill.shutdown()
                except Exception as e:
                    logger.warning(f"TTS shutdown failed: {e}")
        # Stop the server if we started one (waits for in-flight inference)
        if server:
            await server.stop()
        logger.info("Pulse stopped. Companion is sleeping.")


def pick_persona(pulse_root: Path) -> str | None:
    """Interactive persona picker — shown when no persona is specified."""
    personas_dir = pulse_root / "personas"
    if not personas_dir.is_dir():
        return None

    # Find persona directories (skip _template and hidden dirs)
    personas = []
    for d in sorted(personas_dir.iterdir()):
        if d.is_dir() and not d.name.startswith(("_", ".")):
            # Try to read the persona's display name (.yaml preferred)
            display_name = d.name
            persona_yaml = d / "persona.yaml"
            persona_json = d / "persona.json"
            persona_file = persona_yaml if persona_yaml.exists() else persona_json
            if persona_file.exists():
                try:
                    if persona_file.suffix in (".yaml", ".yml"):
                        import yaml
                        with open(persona_file, "r", encoding="utf-8") as f:
                            display_name = yaml.safe_load(f).get("name", d.name)
                    else:
                        import json
                        with open(persona_file, "r", encoding="utf-8") as f:
                            display_name = json.load(f).get("name", d.name)
                except Exception:
                    pass
            personas.append((d.name, display_name))

    if not personas:
        return None

    if len(personas) == 1:
        # Only one persona — use it automatically
        name, display = personas[0]
        print(f"  Found persona: {display} ({name})")
        return name

    print("  Available personas:\n")
    for i, (name, display) in enumerate(personas, 1):
        label = f"{display} ({name})" if display != name else name
        print(f"    {i}. {label}")
    print(f"    0. No persona (use base config)\n")

    while True:
        try:
            choice = input("  Pick a persona [1]: ").strip()
            if choice == "":
                return personas[0][0]  # default to first
            num = int(choice)
            if num == 0:
                return None
            if 1 <= num <= len(personas):
                return personas[num - 1][0]
        except (ValueError, EOFError):
            pass
        print(f"  Please enter 0-{len(personas)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pulse - AI companion heartbeat daemon")
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    parser.add_argument(
        "--persona", "-p",
        default=None,
        help="Persona to activate (e.g. 'nova'). Overrides active_persona in config."
    )
    args = parser.parse_args()

    persona = args.persona
    if persona is None:
        # Check if active_persona is set in config before prompting
        config_check = load_config(args.config)
        if not config_check.get("active_persona"):
            print()
            picked = pick_persona(Path(__file__).parent.resolve())
            if picked:
                persona = picked
                print()

    asyncio.run(main(args.config, persona_name=persona))
