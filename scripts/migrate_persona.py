"""
Migration helper — set up a persona directory from existing Pulse data.

Usage:
    python scripts/migrate_persona.py nova

This will:
1. Create personas/<name>/ with config.yaml and persona.json
2. COPY (not move) data files into personas/<name>/data/
3. Print instructions for updating the base config

Safe to run multiple times — skips files that already exist.
"""

import json
import shutil
import sys
from pathlib import Path

PULSE_ROOT = Path(__file__).parent.parent.resolve()


def migrate(persona_name: str):
    persona_dir = PULSE_ROOT / "personas" / persona_name
    data_dir = persona_dir / "data"

    print(f"Migrating to: {persona_dir}")
    print()

    # Create directory structure
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "journal").mkdir(exist_ok=True)

    # --- Copy persona.json ---
    src_persona = PULSE_ROOT / "persona.json"
    dst_persona = persona_dir / "persona.json"
    _copy_file(src_persona, dst_persona)

    # --- Copy .env (persona-specific secrets) ---
    src_env = PULSE_ROOT / ".env"
    dst_env = persona_dir / ".env"
    _copy_file(src_env, dst_env)

    # --- Copy data files ---
    data_files = [
        "conversation.json",
        "tasks.json",
        "dev_journal.json",
        "schedules.json",
        "action_log.json",
        "usage.json",
        "telegram_chat_id.txt",
    ]
    for filename in data_files:
        src = PULSE_ROOT / "data" / filename
        dst = data_dir / filename
        _copy_file(src, dst)

    # --- Copy journal directory (entries + identity + latest.md) ---
    src_journal = PULSE_ROOT / "data" / "journal"
    dst_journal = data_dir / "journal"
    if src_journal.is_dir():
        # Copy subdirectories (entries/, identity/)
        for subdir_name in ("entries", "identity"):
            src_sub = src_journal / subdir_name
            dst_sub = dst_journal / subdir_name
            if src_sub.is_dir():
                dst_sub.mkdir(exist_ok=True)
                count = 0
                for f in src_sub.iterdir():
                    if f.is_file():
                        _copy_file(f, dst_sub / f.name)
                        count += 1
                print(f"  journal/{subdir_name}/: {count} files")

        # Copy top-level journal files (latest.md, old entry_*.json backups)
        for f in src_journal.iterdir():
            if f.is_file():
                _copy_file(f, dst_journal / f.name)

    # --- Generate persona config.yaml ---
    config_path = persona_dir / "config.yaml"
    if config_path.exists():
        print(f"  SKIP {config_path.name} (already exists)")
    else:
        # Read base config to extract current paths
        import yaml
        base_config = yaml.safe_load(
            (PULSE_ROOT / "config.yaml").read_text(encoding="utf-8")
        )
        current_paths = base_config.get("paths", {})
        memory_path = current_paths.get("memories", "")

        persona_config = {
            "paths": {
                "memories": memory_path,
                "journal": str(data_dir / "journal"),
                "tasks": str(data_dir / "tasks.json"),
                "dev_journal": str(data_dir / "dev_journal.json"),
                "schedules": str(data_dir / "schedules.json"),
                "conversation": str(data_dir / "conversation.json"),
                "persona": str(dst_persona),
                "telegram_chat_id": str(data_dir / "telegram_chat_id.txt"),
            },
        }
        config_path.write_text(
            "# Persona config overlay for: " + persona_name + "\n"
            "# Values here override the base config.yaml.\n\n"
            + yaml.dump(persona_config, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        print(f"  WROTE {config_path.name}")

    # --- Summary ---
    print()
    print("=" * 50)
    print(f"Done! Persona '{persona_name}' is ready.")
    print()
    print("To activate, either:")
    print(f'  1. Set active_persona: "{persona_name}" in config.yaml')
    print(f"  2. Run: python pulse.py --persona {persona_name}")
    print()
    print("Optional cleanup (once you've verified it works):")
    print("  - Genericize base config.yaml paths to relative defaults")
    print("  - The original data/ files are untouched (safe to keep or remove)")
    print("=" * 50)


def _copy_file(src: Path, dst: Path):
    """Copy a file if source exists and destination doesn't."""
    if not src.exists():
        return
    if dst.exists():
        print(f"  SKIP {dst.name} (already exists)")
        return
    shutil.copy2(src, dst)
    print(f"  COPY {src.name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/migrate_persona.py <persona_name>")
        print("Example: python scripts/migrate_persona.py nova")
        sys.exit(1)

    name = sys.argv[1].strip().lower()
    if name.startswith("_"):
        print("Error: persona names starting with _ are reserved (e.g. _template)")
        sys.exit(1)

    migrate(name)
