"""
Token usage tracker — logs API token consumption per day.

Keeps a simple JSON log (data/usage.json) with one entry per day.
Used when provider != "local" to help track costs.
Local inference is free, so usage is only logged for API calls.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class UsageTracker:
    """Tracks daily token usage for cloud API providers."""

    def __init__(self, config: dict):
        self.config = config
        self._db = config.get("_db")
        self._shared_db = config.get("_shared_db")
        
        path = config.get("paths", {}).get("usage", "data/usage.json")
        self.path = Path(path)
        if not self._db:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, prompt_tokens: int, completion_tokens: int,
               provider: str = "", model: str = ""):
        """Record token usage for the current day.

        Accumulates into the existing entry for today, or creates a new one.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        
        db = self._shared_db or self._db
        if db:
            try:
                db.record_usage(
                    date=today,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    calls=1,
                    provider=provider,
                    model=model
                )
                return
            except Exception as e:
                logger.warning(f"Failed to record usage to DB: {e}")

        # JSON Fallback
        log = self._load()

        # Find or create today's entry
        entry = None
        for item in log:
            if item.get("date") == today:
                entry = item
                break

        if entry is None:
            entry = {
                "date": today,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "calls": 0,
                "provider": provider,
                "model": model,
            }
            log.append(entry)

        entry["prompt_tokens"] += prompt_tokens
        entry["completion_tokens"] += completion_tokens
        entry["calls"] += 1
        # Update provider/model in case it changed mid-day
        if provider:
            entry["provider"] = provider
        if model:
            entry["model"] = model

        self._save(log)
        logger.debug(
            f"Usage (JSON): +{prompt_tokens}p/{completion_tokens}c tokens "
            f"(today total: {entry['prompt_tokens']}p/{entry['completion_tokens']}c, "
            f"{entry['calls']} calls)"
        )

    def get_today(self) -> dict:
        """Get today's usage summary."""
        today = datetime.now().strftime("%Y-%m-%d")
        
        db = self._shared_db or self._db
        if db:
            rows = db.get_usage_today(today)
            if rows:
                # Merge multiple model entries for the "today summary"
                summary = {
                    "date": today,
                    "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
                    "completion_tokens": sum(r["completion_tokens"] for r in rows),
                    "calls": sum(r["calls"] for r in rows),
                }
                return summary
            return {"date": today, "prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

        # JSON Fallback
        for item in self._load():
            if item.get("date") == today:
                return item
        return {"date": today, "prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def get_recent(self, days: int = 7) -> list[dict]:
        """Get usage for the last N days."""
        db = self._shared_db or self._db
        if db:
            return db.get_usage_recent(days)
            
        # JSON Fallback
        log = self._load()
        return log[-days:] if len(log) > days else log

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def _save(self, log: list[dict]):
        # Keep last 90 days max to prevent unbounded growth
        if len(log) > 90:
            log = log[-90:]
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(log, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error(f"Failed to save usage log: {e}")
