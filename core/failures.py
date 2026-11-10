"""Failure log: one JSON line per thing that went wrong quietly.

Lives at data/failures.jsonl in the persona folder. Tool crashes, heartbeat or
scheduled-task replies that couldn't be parsed, and startup warnings land here,
so they are visible in one place instead of only in the persona's logfile.
Only what failed and why is stored: never tool argument values or reply text.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def failure_log_path(config: dict) -> Path:
    """data/failures.jsonl in the persona folder (or ./data without one)."""
    persona_dir = config.get("_persona_dir")
    return (Path(persona_dir) if persona_dir else Path(".")) / "data" / "failures.jsonl"


def record_failure(path: Path, persona: str, source: str, error: str,
                   arg_keys=()) -> None:
    """Append one entry. Best-effort: logging a failure never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "persona": persona,
                "tool": source,
                "error": str(error)[:300],
                "arg_keys": sorted(arg_keys),
            }, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug(f"Could not record failure: {e}")
