from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OpenRouterMessage:
    role: str
    content: str
    timestamp: str


@dataclass(frozen=True)
class OpenRouterImport:
    title: str
    session_id: str
    messages: list[OpenRouterMessage]
    skipped_reasoning: int


def parse_openrouter_chat(path: Path) -> OpenRouterImport:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse OpenRouter export: {e}") from e

    if payload.get("version") != "orpg.3.0":
        raise ValueError("Unsupported OpenRouter export format.")
    raw_messages = payload.get("messages")
    raw_items = payload.get("items")
    if not isinstance(raw_messages, dict) or not isinstance(raw_items, dict):
        raise ValueError("OpenRouter export is missing messages/items maps.")

    title = _clean_title(payload.get("title") or "OpenRouter Chat")
    imported: list[tuple[str, str, datetime | None]] = []
    skipped_reasoning = 0

    for message in raw_messages.values():
        if not isinstance(message, dict):
            continue
        role = message.get("type")
        if role not in {"user", "assistant"}:
            continue

        visible_parts: list[str] = []
        for item_id in _message_item_ids(message):
            item = raw_items.get(item_id)
            if not isinstance(item, dict):
                continue
            data = item.get("data")
            if not isinstance(data, dict):
                continue
            item_type = data.get("type")
            if item_type == "reasoning":
                skipped_reasoning += 1
                continue
            if item_type != "message":
                continue
            if data.get("role") and data.get("role") != role:
                continue
            text = _content_text(data.get("content")).strip()
            if text:
                visible_parts.append(text)

        content = "\n\n".join(visible_parts).strip()
        if content:
            imported.append((role, content, _parse_timestamp(message.get("createdAt"))))

    if not imported:
        raise ValueError("No visible user/assistant messages found in OpenRouter export.")

    messages = _assign_timestamps(imported)
    return OpenRouterImport(
        title=title,
        session_id=_session_id(title, messages[0].timestamp),
        messages=messages,
        skipped_reasoning=skipped_reasoning,
    )


def _message_item_ids(message: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for item in message.get("items") or []:
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, dict) and isinstance(item.get("id"), str):
            ids.append(item["id"])
    return ids


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for chunk in content:
        if isinstance(chunk, str):
            parts.append(chunk)
        elif isinstance(chunk, dict):
            for key in ("text", "content", "value"):
                value = chunk.get(key)
                if isinstance(value, str):
                    parts.append(value)
                    break
    return "\n".join(part for part in parts if part)


def _clean_title(value: Any) -> str:
    title = str(value or "").strip()
    title = re.sub(r"\s+", " ", title)
    return title[:120] or "OpenRouter Chat"


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _assign_timestamps(imported: list[tuple[str, str, datetime | None]]) -> list[OpenRouterMessage]:
    first_valid = next((dt for _, _, dt in imported if dt is not None), None)
    fallback = (first_valid - timedelta(seconds=len(imported) + 1)) if first_valid else datetime.now(timezone.utc)
    last_dt: datetime | None = None
    messages: list[OpenRouterMessage] = []

    for role, content, dt in imported:
        if dt is None:
            if last_dt is None:
                fallback = fallback + timedelta(seconds=1)
                dt = fallback
            else:
                dt = last_dt + timedelta(microseconds=1)
        elif last_dt is not None and dt <= last_dt:
            dt = last_dt + timedelta(microseconds=1)
        messages.append(OpenRouterMessage(role=role, content=content, timestamp=dt.isoformat()))
        last_dt = dt

    return messages


def _session_id(title: str, first_timestamp: str) -> str:
    try:
        dt = datetime.fromisoformat(first_timestamp)
        stamp = dt.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")
    except ValueError:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "chat"
    return f"openrouter_{stamp}_{slug[:40]}"
