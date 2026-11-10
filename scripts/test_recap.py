"""Run-once test for the auto-recap task.

Bootstraps Pulse just enough to invoke ``RecapTask.maybe_run()`` against the
current persona's conversation history, with the inactivity and cooldown gates
relaxed so it fires immediately. Prints the generated recap (or the reason it
didn't fire) without affecting the live engine.

Usage:
    python scripts/test_recap.py                # use default persona
    python scripts/test_recap.py --persona ada  # specific persona
    python scripts/test_recap.py --dry-run      # generate but don't save

Notes:
- Uses whichever LLM provider the persona/config points at. If it's a local
  llama-server provider, that server needs to already be running (e.g. start
  Pulse normally in another terminal first, OR the test will fail at the
  availability check).
- Reads from the same DBs Pulse normally writes to. With ``--dry-run`` the
  recap is generated but not saved. Without it, the recap IS persisted just
  like a real auto-recap would be.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# Add pulse root to path so `from core.engine import PulseEngine` works
PULSE_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PULSE_ROOT))

import yaml
from dotenv import load_dotenv


def _load_config(config_path: Path) -> dict:
    """Load and parse YAML config — minimal version of pulse.py's loader."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Merge overlay over base recursively (same behavior as pulse.py)."""
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_persona(config: dict, persona_name: str | None, pulse_root: Path) -> dict:
    """Apply persona overlay if specified.

    MUST mirror pulse.py's resolve_persona — particularly the per-persona
    database path setting. An earlier version of this function skipped that,
    causing the test to read the default persona's DB even when a different
    --persona was passed. Don't repeat that mistake.
    """
    if not persona_name:
        return config
    persona_dir = pulse_root / "personas" / persona_name
    if not persona_dir.exists():
        print(f"[!] Persona dir not found: {persona_dir}")
        return config

    # Load persona's .env if it exists
    persona_env = persona_dir / ".env"
    if persona_env.exists():
        load_dotenv(persona_env, override=True)

    # Load and merge persona config.yaml
    persona_config_path = persona_dir / "config.yaml"
    persona_config: dict = {}
    if persona_config_path.exists():
        persona_config = _load_config(persona_config_path)
        config = _deep_merge(config, persona_config)

    # Mirror pulse.py: auto-default any unset path to the persona's data dir
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
        "garden": str(persona_data / "garden"),
    }
    config.setdefault("paths", {})
    for key, default in default_paths.items():
        if key not in persona_paths:
            config["paths"][key] = default

    # Persona identity file
    persona_yaml = persona_dir / "persona.yaml"
    persona_json = persona_dir / "persona.json"
    persona_identity = persona_yaml if persona_yaml.exists() else persona_json
    if persona_identity.exists():
        config["paths"]["persona"] = str(persona_identity)

    config["_persona_name"] = persona_name
    config["_persona_dir"] = str(persona_dir)

    # Per-persona database path — THIS is what was missing before
    config["paths"]["database"] = str(persona_data / f"{persona_name}.db")

    # Shared database — only when persona's memories path is absolute and outside
    # its own data dir, AND a shared.db exists in that location
    memories_path = Path(config["paths"].get("memories", ""))
    if memories_path.is_absolute() and not str(memories_path).startswith(str(persona_data)):
        shared_db = memories_path / "shared.db"
        if shared_db.exists():
            config["paths"]["shared_database"] = str(shared_db)

    return config


async def main(persona_name: str | None, dry_run: bool):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # Quiet down chatty libs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    print("=" * 60)
    print("  Auto-recap test — run-once invocation")
    print("=" * 60)

    # 1. Load config and persona
    load_dotenv()
    config = _load_config(PULSE_ROOT / "config.yaml")
    config["_pulse_root"] = str(PULSE_ROOT)
    config = _resolve_persona(config, persona_name, PULSE_ROOT)

    active_persona = config.get("_persona_name") or "(default)"
    print(f"\n[persona] {active_persona}")
    provider_type = config.get("provider", {}).get("type", "local")
    print(f"[provider] {provider_type}")

    # 2. Open DBs (same as pulse.py)
    db = None
    shared_db = None
    db_path = config.get("paths", {}).get("database")
    if db_path:
        from core.db import PulseDatabase
        db = PulseDatabase(Path(db_path))
        config["_db"] = db
        shared_db_path = config.get("paths", {}).get("shared_database")
        if shared_db_path:
            shared_db = PulseDatabase(Path(shared_db_path))
            config["_shared_db"] = shared_db
        else:
            config["_shared_db"] = db
    print(f"[db] {db_path or '(no db configured)'}")

    # 3. Load embedding model (for recap embedding generation)
    try:
        from core.context import load_embedding_model
        load_embedding_model()
    except Exception as e:
        print(f"[!] Could not load embedding model: {e}")

    # 4. Resolve provider endpoint (same logic as pulse.py)
    provider_config = config.get("provider", {})
    endpoint = ""
    api_key = ""
    if provider_type == "local":
        # We don't start llama-server here — assume it's already running
        # (or warn the user if it isn't reachable).
        local_url = provider_config.get("local_url", "http://localhost:8080/v1")
        endpoint = local_url
        print(f"[endpoint] {endpoint} (local — assumed already running)")
    else:
        api_key_env = provider_config.get("api_key_env", "")
        api_key = os.getenv(api_key_env, "") if api_key_env else ""
        if not api_key:
            print(f"[!] Missing API key. Set {api_key_env} in .env")
            sys.exit(1)
        default_urls = {
            "openai": "https://api.openai.com/v1",
            "openrouter": "https://openrouter.ai/api/v1",
            "anthropic": "https://api.anthropic.com/v1",
        }
        endpoint = (
            provider_config.get("base_url")
            or default_urls.get(provider_type, "")
        )
        print(f"[endpoint] {endpoint}")

    # 5. Construct the engine (no channels, no skill registry — recap doesn't need them)
    from core.engine import PulseEngine
    engine = PulseEngine(
        config,
        channels={},
        skill_registry=None,
        llm_endpoint=endpoint,
        api_key=api_key,
    )

    # 6. Force the recap task gates open for the test
    engine._last_activity = time.time() - (3 * 3600)  # pretend 3 hours of silence
    engine.recap_task.cooldown_seconds = 0  # bypass cooldown
    engine.recap_task.min_messages = 1  # don't gate on conversation length

    # 7. Inspect current state
    print("\n--- Pre-run state ---")
    history = engine.context._load_conversation()
    msg_count = sum(1 for m in history if m.get("role") in ("user", "assistant"))
    print(f"Messages in current conversation: {msg_count}")
    if msg_count == 0:
        print("[!] No conversation history found — recap would be empty.")
        print("    Have a chat with this persona first, then run this test.")
        sys.exit(0)

    last_recap_ts = engine.recap_task._get_last_recap_timestamp()
    if last_recap_ts:
        from datetime import datetime as _dt
        print(f"Last recap saved: {_dt.fromtimestamp(last_recap_ts).isoformat()}")
    else:
        print("Last recap saved: (none yet)")

    # 8. Run the recap. If dry-run, intercept the save call to print instead.
    print("\n--- Generating recap ---")
    if dry_run:
        # Monkey-patch the save method so nothing is persisted
        original_save = engine.recap_task._save_recap

        def _print_instead_of_save(text: str) -> bool:
            print("\n=== RECAP (dry-run, not saved) ===")
            print(text)
            print("=== end recap ===\n")
            return True
        engine.recap_task._save_recap = _print_instead_of_save  # type: ignore

    fired = await engine.recap_task.maybe_run()

    # 9. Report what happened
    print("\n--- Result ---")
    if fired:
        print("[OK] Recap was generated.")
        if not dry_run:
            # Pull the recap we just saved and show it
            db_to_use = shared_db or db
            if db_to_use:
                memories = db_to_use.get_all_memories()
                recaps = sorted(
                    [m for m in memories if m.get("type") == "recap"],
                    key=lambda m: m.get("date") or "",
                    reverse=True,
                )
                if recaps:
                    latest = recaps[0]
                    print("\n=== Latest recap (saved) ===")
                    print(latest.get("text", ""))
                    print("=== end recap ===")
                    print(f"\n[id] {latest.get('id')}")
                    print(f"[date] {latest.get('date')}")
                    print(f"[tags] {latest.get('tags')}")
    else:
        print("[!] Recap did NOT fire. Check the logs above for the reason.")
        print("    Common causes: LLM not reachable, conversation empty, or")
        print("    LLM returned empty text.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test auto-recap generation.")
    parser.add_argument(
        "--persona",
        default=None,
        help="Persona name (e.g. 'ada'). Uses default config if not set.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate recap but don't save it to the DB.",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.persona, args.dry_run))
    except KeyboardInterrupt:
        print("\n[!] Interrupted.")
        sys.exit(1)
