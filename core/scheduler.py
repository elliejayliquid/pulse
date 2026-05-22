"""
Schedule manager - handles recurring and one-time tasks.

Manages schedules.json and evaluates which tasks are due.
Supports both cron-like recurring schedules and one-time reminders.
"""

import json
import hashlib
import logging
import os
import re
import time
from difflib import SequenceMatcher
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


WEEKDAYS = {
    "sunday": 0,
    "sun": 0,
    "monday": 1,
    "mon": 1,
    "tuesday": 2,
    "tue": 2,
    "tues": 2,
    "wednesday": 3,
    "wed": 3,
    "thursday": 4,
    "thu": 4,
    "thur": 4,
    "thurs": 4,
    "friday": 5,
    "fri": 5,
    "saturday": 6,
    "sat": 6,
}

MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


def _normalize_task(text: str) -> str:
    """Normalize task text for dedup comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)  # strip punctuation/emoji
    text = re.sub(r"\s+", " ", text)
    # Strip meta-framing prefixes ("remind lena about X" → "X")
    text = re.sub(
        r"^(remind|tell|ask|nudge|check with|ping|poke)\s+\w+\s+"
        r"(about|to|that|regarding|re)\s+",
        "", text
    )
    return text


def _cosine_similarity(a, b) -> float:
    """Cosine similarity between two vectors."""
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _tasks_similar(a: str, b: str, threshold: float = 0.7) -> bool:
    """Check if two task descriptions are similar enough to be duplicates.

    Uses semantic similarity (sentence embeddings) when available,
    falls back to SequenceMatcher for string-level comparison.
    """
    na, nb = _normalize_task(a), _normalize_task(b)
    if na == nb:
        return True
    # Check containment (one is a substring of the other)
    if na in nb or nb in na:
        return True

    # Try semantic similarity via the embedding model (already in memory)
    try:
        from core.context import _get_embedding_model
        model = _get_embedding_model()
        if model is not None:
            embeddings = model.encode([a, b])
            sim = _cosine_similarity(embeddings[0], embeddings[1])
            logger.debug(f"Semantic similarity: {sim:.3f} — '{a[:50]}' vs '{b[:50]}'")
            return sim >= 0.75
    except Exception as e:
        logger.debug(f"Embedding similarity failed, falling back to string match: {e}")

    # Fallback: string-level similarity
    return SequenceMatcher(None, na, nb).ratio() >= threshold


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


def _strip_ordinal_suffix(text: str) -> str:
    """Normalize ordinal dates: 1st -> 1, 22nd -> 22."""
    return re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", text)


def _parse_day_of_month(text: str) -> Optional[int]:
    """Parse a day-of-month token."""
    text = _strip_ordinal_suffix(text.strip().lower())
    m = re.search(r"\b(\d{1,2})\b", text)
    if not m:
        return None
    day = int(m.group(1))
    if 1 <= day <= 31:
        return day
    return None


def _find_time_of_day(text: str) -> Optional[tuple[int, int]]:
    """Find a time-of-day anywhere in a larger phrase."""
    patterns = [
        r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b",
        r"\b\d{1,2}\s*(?:am|pm)\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            parsed = _parse_time_of_day(m.group(0))
            if parsed:
                return parsed
    return None


def parse_recurring_time(when: str) -> Optional[tuple[str, str]]:
    """Parse human recurring reminders into a cron expression and label.

    Supports examples like:
        daily 8:00
        weekly monday 9:00 / every monday at 9:00
        monthly 15th 8:30pm
        yearly jan 1 9:00 / annually 05-18 9:00
    """
    local_now = datetime.now().astimezone()
    text = when.strip().lower()
    text = _strip_ordinal_suffix(text)
    normalized = re.sub(r"\bat\b", " ", text)
    normalized = re.sub(r"\bon\b", " ", normalized)
    normalized = re.sub(r"\bthe\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    parsed_time = _find_time_of_day(normalized)
    if not parsed_time:
        return None
    hour, minute = parsed_time
    without_time = re.sub(r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b", " ", normalized)
    without_time = re.sub(r"\b\d{1,2}\s*(?:am|pm)\b", " ", without_time)
    without_time = re.sub(r"\s+", " ", without_time).strip()

    if re.search(r"\b(daily|every day|each day)\b", normalized):
        return f"{minute} {hour} * * *", f"daily at {hour:02d}:{minute:02d}"

    weekly = re.search(r"\b(weekly|every week|each week)\b", normalized)
    weekday = None
    for name, value in WEEKDAYS.items():
        if re.search(rf"\b{name}s?\b", normalized):
            weekday = value
            break
    if weekly or weekday is not None:
        if weekday is None:
            weekday = local_now.isoweekday() % 7
        weekday_name = [
            "Sunday", "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday"
        ][weekday]
        return (
            f"{minute} {hour} * * {weekday}",
            f"weekly on {weekday_name} at {hour:02d}:{minute:02d}",
        )

    if re.search(r"\b(monthly|every month|each month)\b", normalized):
        day = _parse_day_of_month(without_time)
        if day is None:
            day = local_now.day
        return (
            f"{minute} {hour} {day} * *",
            f"monthly on day {day} at {hour:02d}:{minute:02d}",
        )

    if re.search(r"\b(yearly|annually|annual|every year|each year)\b", normalized):
        month = None
        day = None

        md = re.search(r"\b(\d{1,2})[/-](\d{1,2})\b", without_time)
        if md:
            month = int(md.group(1))
            day = int(md.group(2))
        else:
            for name, value in MONTHS.items():
                if re.search(rf"\b{name}\b", without_time):
                    month = value
                    after_month = without_time.split(name, 1)[1]
                    day = _parse_day_of_month(after_month)
                    break

        if month is None:
            month = local_now.month
        if day is None:
            day = local_now.day
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return None

        return (
            f"{minute} {hour} {day} {month} *",
            f"yearly on {month:02d}-{day:02d} at {hour:02d}:{minute:02d}",
        )

    return None


def describe_cron(cron: str) -> str:
    """Return a human-readable description for the supported cron subset."""
    parts = cron.strip().split()
    if len(parts) != 5:
        return cron

    minute, hour, dom, month, dow = parts
    try:
        time_str = f"{int(hour):02d}:{int(minute):02d}"
    except ValueError:
        return cron

    if dom == "*" and month == "*" and dow == "*":
        return f"daily at {time_str}"
    if dom == "*" and month == "*" and dow != "*":
        try:
            weekday_name = [
                "Sunday", "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday"
            ][int(dow)]
            return f"weekly on {weekday_name} at {time_str}"
        except (ValueError, IndexError):
            return cron
    if dom != "*" and month == "*" and dow == "*":
        return f"monthly on day {dom} at {time_str}"
    if dom != "*" and month != "*" and dow == "*":
        try:
            return f"yearly on {int(month):02d}-{int(dom):02d} at {time_str}"
        except ValueError:
            return cron
    return cron


def parse_human_time(when: str, now: datetime = None) -> Optional[datetime]:
    """Parse human-friendly time expressions into datetime.

    Supports:
        "in 2 hours"
        "in 30 minutes"
        "tomorrow 9:00"
        "friday 3pm"
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

    # "today 3pm", "today 15:00", "tomorrow 3pm", "tomorrow 15:00"
    today_tomorrow = re.match(r"(today|tomorrow)\s+(.+)", when)
    if today_tomorrow:
        day_word, time_part = today_tomorrow.groups()
        local_now = datetime.now().astimezone()
        base_date = local_now.date()
        if day_word == "tomorrow":
            base_date += timedelta(days=1)
        parsed_time = _parse_time_of_day(time_part)
        if parsed_time:
            h, m = parsed_time
            dt = datetime(base_date.year, base_date.month, base_date.day, h, m,
                          tzinfo=local_now.tzinfo)
            return dt.astimezone(timezone.utc)

    # Bare time: "3:00 PM", "3pm", "15:00", "3:30pm"
    bare_time = _parse_time_of_day(when)
    if bare_time:
        h, m = bare_time
        local_now = datetime.now().astimezone()
        dt = datetime(local_now.year, local_now.month, local_now.day, h, m,
                      tzinfo=local_now.tzinfo)
        # If the time has already passed today, assume tomorrow
        if dt <= local_now:
            dt += timedelta(days=1)
        return dt.astimezone(timezone.utc)

    # ISO format "2026-03-01 15:00"
    try:
        dt = datetime.fromisoformat(when)
        if dt.tzinfo is None:
            # Treat naive datetimes as LOCAL time (what the LLM intends),
            # then convert to UTC for storage.
            dt = dt.astimezone()  # attach local tz
            dt = dt.astimezone(timezone.utc)  # convert to UTC
        return dt
    except ValueError:
        pass

    logger.warning(f"Could not parse time expression: {when}")
    return None


def _parse_time_of_day(text: str) -> Optional[tuple[int, int]]:
    """Parse a time-of-day string into (hour, minute).

    Supports: "3pm", "3:00 PM", "15:00", "3:30pm", "3:30 pm", etc.
    """
    text = text.strip().lower()

    # "3:00 pm", "3:30pm", "11:45 am"
    m = re.match(r"(\d{1,2}):(\d{2})\s*(am|pm)?$", text)
    if m:
        hour, minute, ampm = int(m.group(1)), int(m.group(2)), m.group(3)
        if ampm == "pm" and hour < 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return (hour, minute)

    # "3pm", "12am"
    m = re.match(r"(\d{1,2})\s*(am|pm)$", text)
    if m:
        hour, ampm = int(m.group(1)), m.group(2)
        if ampm == "pm" and hour < 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        if 0 <= hour <= 23:
            return (hour, 0)

    return None


class ScheduleManager:
    """Manages scheduled tasks and reminders."""

    def __init__(self, config: dict):
        self.config = config
        self._db = config.get("_db")
        self._shared_db = config.get("_shared_db")
        
        # Resolve legacy path for fallback
        schedules_path = config.get("paths", {}).get("schedules", "data/schedules.json")
        self.path = Path(schedules_path)
        
        if not self._db:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                self._save_json([])

    def _load(self) -> list[dict]:
        db = self._shared_db or self._db
        if db:
            return db.get_schedules(enabled_only=False)
        return self._load_json()

    def _load_json(self) -> list[dict]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def _save_json(self, schedules: list[dict]):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(schedules, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error(f"Failed to save schedules: {e}")

    def add(self, task: str, created_by: str,
            cron: str = "", run_at: str = "", when: str = "",
            origin: str = "companion",
            priority: str = "routine") -> dict:
        """Add a new schedule.

        Args:
            task: What to do
            created_by: Who created it (author_id, "<persona>-self", etc.)
            cron: Cron expression for recurring tasks
            run_at: ISO datetime for one-time tasks
            when: Human-friendly time (parsed into run_at or cron)
            origin: "user" (explicitly requested, always fires) or
                    "companion" (self-scheduled, can be skipped if redundant)
            priority: "urgent", "routine", or "creative" (stored for context)
        """
        schedules = self._load()

        entry = {
            "id": generate_schedule_id(),
            "task": task,
            "created_by": created_by,
            "origin": origin,
            "priority": priority,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_at_local": datetime.now().strftime("%A %b %d, %I:%M %p"),
            "enabled": True,
        }

        # Parse human time if provided
        if when and not cron and not run_at:
            recurring = parse_recurring_time(when)
            if recurring:
                cron, _label = recurring
            else:
                parsed = parse_human_time(when)
                if parsed:
                    run_at = parsed.isoformat()

        if cron:
            # Validate cron format before saving
            parts = cron.strip().split()
            if len(parts) != 5:
                raise ValueError(
                    f"Invalid cron format '{cron}'. "
                    "Use 'daily HH:MM' for recurring reminders."
                )
            entry["schedule_type"] = "recurring"
            entry["cron"] = cron
            entry["last_run"] = None
        else:
            if not run_at:
                raise ValueError(
                    "Cannot create a one-time schedule without a time. "
                    "Use 'in 30 minutes', 'in 2 hours', or an ISO datetime."
                )
            entry["schedule_type"] = "once"
            entry["run_at"] = run_at
            entry["completed"] = False
            # Human-readable local time for the JSON file
            if run_at:
                try:
                    dt = datetime.fromisoformat(run_at)
                    entry["run_at_local"] = dt.astimezone().strftime("%A %b %d, %I:%M %p")
                except ValueError:
                    pass

        # Dedup: skip if an active schedule with a similar task already exists
        active = [s for s in schedules
                  if s.get("enabled", True) and not s.get("completed", False)]
        for existing in active:
            if _tasks_similar(task, existing.get("task", "")):
                logger.info(
                    f"Dedup: skipping '{task}' — similar to existing '{existing['task']}' "
                    f"({existing['id']})"
                )
                # Return a shallow copy with a transient flag so callers (e.g. the
                # schedule skill) can tell the entry was deduped vs newly created.
                # The flag is not persisted — only the copy carries it.
                deduped = dict(existing)
                deduped["_was_deduped"] = True
                return deduped

        schedules.append(entry)

        db = self._shared_db or self._db
        if db:
            db.save_schedule(entry)
        else:
            self._save_json(schedules)

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
                # Check cron match — skip if cron is missing/empty
                if not s.get("cron"):
                    continue
                if parse_simple_cron(s["cron"], local_now):
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

    def count_active(self) -> int:
        """Count active schedules without running due-task checks."""
        db = self._shared_db or self._db
        if db:
            return len([
                s for s in db.get_schedules(enabled_only=True)
                if not s.get("completed", False)
            ])
        return len([
            s for s in self._load_json()
            if s.get("enabled", True) and not s.get("completed", False)
        ])

    def mark_completed(self, schedule_id: str):
        """Mark a one-time task as completed, or update last_run for recurring.

        Also auto-purges old completed one-time tasks (keeps last 5 max)
        to prevent stale tasks from polluting the companion's context.
        """
        schedules = self._load()
        # Update the entry in the list and persistence
        db = self._shared_db or self._db
        for s in schedules:
            if s["id"] == schedule_id:
                if s.get("schedule_type") == "once":
                    s["completed"] = True
                now_iso = datetime.now(timezone.utc).isoformat()
                s["last_run"] = now_iso
                
                if db:
                    if s.get("schedule_type") == "once":
                        db.mark_schedule_completed(schedule_id)
                    db.update_schedule_last_run(schedule_id)
                break

        if not db:
            # Auto-purge: remove old completed one-time tasks (keep only the 5 most recent)
            completed = [s for s in schedules if s.get("schedule_type") == "once" and s.get("completed")]
            if len(completed) > 5:
                # Sort by completion time, keep newest 5
                completed.sort(key=lambda s: s.get("last_run", ""), reverse=True)
                old_ids = {s["id"] for s in completed[5:]}
                schedules = [s for s in schedules if s["id"] not in old_ids]
                logger.info(f"Purged {len(old_ids)} old completed schedule(s)")
            self._save_json(schedules)

    def remove(self, schedule_id: str) -> bool:
        """Remove a schedule entirely."""
        db = self._shared_db or self._db
        if db:
            return db.delete_schedule(schedule_id)

        # Legacy JSON Fallback
        schedules = self._load_json()
        original_len = len(schedules)
        schedules = [s for s in schedules if s["id"] != schedule_id]
        
        if len(schedules) < original_len:
            self._save_json(schedules)
            return True
        return False

    def list_all(self) -> list[dict]:
        """List all schedules."""
        return self._load()

    def update(self, schedule_id: str, **kwargs) -> Optional[dict]:
        """Update fields on an existing schedule.

        Accepts keyword args: task, when, priority, enabled.
        'when' is re-parsed into run_at (one-time) or cron (recurring).
        Returns the updated entry, or None if not found.
        """
        schedules = self._load()
        entry = None
        for s in schedules:
            if s["id"] == schedule_id:
                entry = s
                break
        if entry is None:
            return None

        if "task" in kwargs and kwargs["task"]:
            entry["task"] = kwargs["task"].strip()

        if "priority" in kwargs and kwargs["priority"]:
            entry["priority"] = kwargs["priority"]

        if "enabled" in kwargs:
            entry["enabled"] = kwargs["enabled"]

        if "when" in kwargs and kwargs["when"]:
            when = kwargs["when"].strip()
            recurring = parse_recurring_time(when)
            if recurring:
                cron, _label = recurring
                entry["schedule_type"] = "recurring"
                entry["cron"] = cron
                entry.pop("run_at", None)
                entry.pop("run_at_local", None)
                entry.pop("completed", None)
            else:
                parsed = parse_human_time(when)
                if parsed:
                    entry["schedule_type"] = "once"
                    entry["run_at"] = parsed.isoformat()
                    entry["completed"] = False
                    entry.pop("cron", None)
                    try:
                        dt = datetime.fromisoformat(entry["run_at"])
                        entry["run_at_local"] = dt.astimezone().strftime(
                            "%A %b %d, %I:%M %p"
                        )
                    except ValueError:
                        pass

        db = self._shared_db or self._db
        if db:
            db.save_schedule(entry)
        else:
            self._save_json(schedules)
            
        logger.info(f"Schedule updated: {entry['id']} — {entry.get('task', '')}")
        return entry

    def list_active(self) -> list[dict]:
        """List only active (enabled, not completed) schedules."""
        return [
            s for s in self._load()
            if s.get("enabled", True) and not s.get("completed", False)
        ]
