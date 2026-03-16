"""
Schedule manager - handles recurring and one-time tasks.

Manages schedules.json and evaluates which tasks are due.
Supports both cron-like recurring schedules and one-time reminders.
"""

import json
import hashlib
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def generate_schedule_id() -> str:
    """Generate a unique schedule ID."""
    raw = f"{time.time()}-{os.urandom(4).hex()}"
    return "sch_" + hashlib.sha256(raw.encode()).hexdigest()[:8]


def parse_simple_cron(cron: str, now: datetime) -> bool:
    """Check if a simple cron expression matches the current time.

    Supports a subset of cron:
        "0 8 * * *"     = daily at 8:00
        "0 */2 * * *"   = every 2 hours
        "30 9 * * 1"    = Monday at 9:30
        "0 8 1 * *"     = 1st of month at 8:00

    Format: minute hour day_of_month month day_of_week
    """
    parts = cron.strip().split()
    if len(parts) != 5:
        logger.warning(f"Invalid cron format: {cron}")
        return False

    minute, hour, dom, month, dow = parts

    def matches(field: str, value: int, max_val: int) -> bool:
        if field == "*":
            return True
        if field.startswith("*/"):
            step = int(field[2:])
            return value % step == 0
        if "," in field:
            return value in [int(x) for x in field.split(",")]
        return value == int(field)

    return (
        matches(minute, now.minute, 59) and
        matches(hour, now.hour, 23) and
        matches(dom, now.day, 31) and
        matches(month, now.month, 12) and
        matches(dow, now.isoweekday() % 7, 6)  # 0=Sunday for cron
    )


def parse_human_time(when: str, now: datetime = None) -> Optional[datetime]:
    """Parse human-friendly time expressions into datetime.

    Supports:
        "in 2 hours"
        "in 30 minutes"
        "tomorrow 9:00"
        "friday 3pm"
        "daily 8:00" -> returns a cron string instead (special case)
        "2026-03-01 15:00" -> ISO format
    """
    if now is None:
        now = datetime.now(timezone.utc)

    when = when.strip().lower()

    # "in X hours/minutes"
    if when.startswith("in "):
        parts = when[3:].split()
        if len(parts) >= 2:
            try:
                amount = int(parts[0])
                unit = parts[1]
                if "hour" in unit:
                    return now + timedelta(hours=amount)
                elif "minute" in unit or "min" in unit:
                    return now + timedelta(minutes=amount)
                elif "day" in unit:
                    return now + timedelta(days=amount)
            except ValueError:
                pass

    # ISO format "2026-03-01 15:00"
    try:
        dt = datetime.fromisoformat(when)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass

    logger.warning(f"Could not parse time expression: {when}")
    return None


class ScheduleManager:
    """Manages scheduled tasks and reminders."""

    def __init__(self, schedules_path: str):
        self.path = Path(schedules_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._save([])

    def _load(self) -> list[dict]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def _save(self, schedules: list[dict]):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(schedules, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error(f"Failed to save schedules: {e}")

    def add(self, task: str, created_by: str,
            cron: str = "", run_at: str = "", when: str = "") -> dict:
        """Add a new schedule.

        Args:
            task: What to do
            created_by: Who created it (author_id, "nova-self", etc.)
            cron: Cron expression for recurring tasks
            run_at: ISO datetime for one-time tasks
            when: Human-friendly time (parsed into run_at or cron)
        """
        schedules = self._load()

        entry = {
            "id": generate_schedule_id(),
            "task": task,
            "created_by": created_by,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "enabled": True,
        }

        # Parse human time if provided
        if when and not cron and not run_at:
            if "daily" in when.lower():
                # Convert "daily 8:00" to cron
                time_part = when.lower().replace("daily", "").strip()
                try:
                    parts = time_part.split(":")
                    h = int(parts[0])
                    m = int(parts[1]) if len(parts) > 1 else 0
                    cron = f"{m} {h} * * *"
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse daily time: {when}")
            else:
                parsed = parse_human_time(when)
                if parsed:
                    run_at = parsed.isoformat()

        if cron:
            entry["schedule_type"] = "recurring"
            entry["cron"] = cron
            entry["last_run"] = None
        else:
            entry["schedule_type"] = "once"
            entry["run_at"] = run_at
            entry["completed"] = False

        schedules.append(entry)
        self._save(schedules)

        logger.info(f"Schedule added: {entry['id']} — {task}")
        return entry

    def get_due_tasks(self) -> list[dict]:
        """Get all tasks that are due right now."""
        schedules = self._load()
        now = datetime.now(timezone.utc)
        local_now = datetime.now()
        due = []

        for s in schedules:
            if not s.get("enabled", True):
                continue

            if s.get("schedule_type") == "recurring":
                # Check cron match
                if parse_simple_cron(s.get("cron", ""), local_now):
                    # Don't fire twice in the same minute
                    last_run = s.get("last_run")
                    if last_run:
                        last_dt = datetime.fromisoformat(last_run)
                        if last_dt.tzinfo is None:
                            last_dt = last_dt.replace(tzinfo=timezone.utc)
                        if (now - last_dt).total_seconds() < 60:
                            continue
                    due.append(s)

            elif s.get("schedule_type") == "once":
                if s.get("completed"):
                    continue
                run_at = s.get("run_at", "")
                if run_at:
                    run_dt = datetime.fromisoformat(run_at)
                    if run_dt.tzinfo is None:
                        run_dt = run_dt.replace(tzinfo=timezone.utc)
                    if now >= run_dt:
                        due.append(s)

        return due

    def mark_completed(self, schedule_id: str):
        """Mark a one-time task as completed, or update last_run for recurring.

        Also auto-purges old completed one-time tasks (keeps last 5 max)
        to prevent stale tasks from polluting Nova's context.
        """
        schedules = self._load()
        for s in schedules:
            if s["id"] == schedule_id:
                if s.get("schedule_type") == "once":
                    s["completed"] = True
                s["last_run"] = datetime.now(timezone.utc).isoformat()
                break

        # Auto-purge: remove old completed one-time tasks (keep only the 5 most recent)
        completed = [s for s in schedules if s.get("schedule_type") == "once" and s.get("completed")]
        if len(completed) > 5:
            # Sort by completion time, keep newest 5
            completed.sort(key=lambda s: s.get("last_run", ""), reverse=True)
            old_ids = {s["id"] for s in completed[5:]}
            schedules = [s for s in schedules if s["id"] not in old_ids]
            logger.info(f"Purged {len(old_ids)} old completed schedule(s)")

        self._save(schedules)

    def remove(self, schedule_id: str) -> bool:
        """Remove a schedule entirely."""
        schedules = self._load()
        original_len = len(schedules)
        schedules = [s for s in schedules if s["id"] != schedule_id]
        if len(schedules) < original_len:
            self._save(schedules)
            return True
        return False

    def list_all(self) -> list[dict]:
        """List all schedules."""
        return self._load()

    def list_active(self) -> list[dict]:
        """List only active (enabled, not completed) schedules."""
        return [
            s for s in self._load()
            if s.get("enabled", True) and not s.get("completed", False)
        ]
