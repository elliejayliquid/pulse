from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "you": "user",
    "assistant": "assistant",
    "model": "assistant",
    "ai": "assistant",
}


@dataclass
class PreviewMessage:
    id: str
    role: str
    content: str
    created_at: str = ""
    source_role: str = ""
    model: str = ""
    skipped: bool = False
    skip_reason: str = ""
    attachment_placeholder: bool = False
    excluded: bool = False

    def to_dict(self, include_content: bool = True) -> dict:
        data = {
            "id": self.id,
            "role": self.role,
            "created_at": self.created_at,
            "source_role": self.source_role,
            "model": self.model,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "attachment_placeholder": self.attachment_placeholder,
            "excluded": self.excluded,
            "char_count": len(self.content),
        }
        if include_content:
            data["content"] = self.content
        return data


@dataclass
class PreviewConversation:
    id: str
    source_id: str
    title: str
    created_at: str = ""
    updated_at: str = ""
    model: str = ""
    messages: list[PreviewMessage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_summary_dict(self) -> dict:
        visible_messages = [msg for msg in self.messages if not msg.skipped]
        skipped_count = len(self.messages) - len(visible_messages)
        review_issues = conversation_review_issues(self.messages)
        duplicate_clusters = duplicate_resend_clusters(self.messages)
        user_count = sum(1 for msg in visible_messages if msg.role == "user")
        assistant_count = sum(1 for msg in visible_messages if msg.role == "assistant")
        preview = next((msg.content for msg in visible_messages if msg.content.strip()), "")
        warnings = list(self.warnings)
        if skipped_count:
            warnings.append(f"{skipped_count} empty/skipped message{'s' if skipped_count != 1 else ''}.")
        if not visible_messages:
            warnings.append("No visible human/companion text.")
        return {
            "id": self.id,
            "source_id": self.source_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "model": self.model,
            "message_count": len(visible_messages),
            "skipped_message_count": skipped_count,
            "review_issue_count": len(review_issues),
            "duplicate_cluster_count": len(duplicate_clusters),
            "default_choice": "exclude" if not visible_messages else "keep",
            "user_count": user_count,
            "assistant_count": assistant_count,
            "estimated_tokens": estimate_tokens("\n".join(msg.content for msg in visible_messages)),
            "preview": collapse_ws(preview)[:220],
            "warnings": warnings,
        }

    def to_detail_dict(self, message_offset: int = 0, message_limit: int = 160) -> dict:
        total_messages = len(self.messages)
        limit = clamp_int(message_limit, default=160, minimum=1, maximum=500)
        offset = clamp_int(message_offset, default=0, minimum=0, maximum=max(total_messages - 1, 0))
        if total_messages and offset >= total_messages:
            offset = max(0, total_messages - limit)
        end = min(total_messages, offset + limit)
        messages = []
        duplicate_map = duplicate_resend_map(self.messages)
        for index, msg in enumerate(self.messages[offset:end], start=offset):
            data = msg.to_dict()
            data["index"] = index
            if index in duplicate_map:
                data.update(duplicate_map[index])
            messages.append(data)
        review_issues = conversation_review_issues(self.messages)
        return {
            **self.to_summary_dict(),
            "messages": messages,
            "messages_truncated": total_messages > len(messages),
            "shown_message_count": len(messages),
            "total_message_count": total_messages,
            "message_offset": offset,
            "message_end": end,
            "message_limit": limit,
            "has_earlier": offset > 0,
            "has_later": end < total_messages,
            "review_issues": review_issues,
            "review_issue_count": len(review_issues),
        }


@dataclass
class PreviewBatch:
    source: str
    source_version: str
    file_name: str
    file_size: int
    conversations: list[PreviewConversation]
    warnings: list[str] = field(default_factory=list)
    skipped_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        message_count = sum(
            len([msg for msg in conv.messages if not msg.skipped])
            for conv in self.conversations
        )
        skipped_message_count = sum(
            len([msg for msg in conv.messages if msg.skipped])
            for conv in self.conversations
        )
        conversations = sort_conversations_recent_first(self.conversations)
        models = sorted({conv.model for conv in conversations if conv.model})
        summaries = [conv.to_summary_dict() for conv in conversations]
        return {
            "source": self.source,
            "source_version": self.source_version,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "conversation_count": len(self.conversations),
            "message_count": message_count,
            "skipped_message_count": skipped_message_count,
            "estimated_tokens": sum(summary["estimated_tokens"] for summary in summaries),
            "models": models,
            "warnings": self.warnings,
            "skipped_counts": self.skipped_counts,
            "conversations": summaries,
        }


def preview_import_file(path: Path) -> PreviewBatch:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Conversation export file not found.")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _preview_json(path)
    if suffix in {".md", ".markdown"}:
        return _preview_markdown(path)
    raise ValueError("Unsupported import preview file type. Choose a JSON or Markdown export.")


def _preview_json(path: Path) -> PreviewBatch:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON export: {e}") from e

    file_size = path.stat().st_size
    if isinstance(payload, dict) and payload.get("version") == "orpg.3.0":
        return _preview_openrouter(path.name, file_size, payload)
    if is_grok_export(payload):
        return _preview_grok(path.name, file_size, payload)

    if isinstance(payload, list) and payload:
        first = next((item for item in payload if isinstance(item, dict)), {})
        if isinstance(first, dict) and "chat_messages" in first:
            return _preview_claude(path.name, file_size, payload)
        if isinstance(first, dict) and "mapping" in first:
            return _preview_chatgpt(path.name, file_size, payload)
        generic = _preview_generic_list(path.name, file_size, payload)
        if generic.conversations:
            return generic

    if isinstance(payload, dict):
        generic = _preview_generic_dict(path.name, file_size, payload)
        if generic.conversations:
            return generic

    raise ValueError("Could not recognize a supported conversation export shape.")


def _preview_claude(file_name: str, file_size: int, payload: list[Any]) -> PreviewBatch:
    conversations: list[PreviewConversation] = []
    skipped_empty = 0
    for index, conv in enumerate(payload):
        if not isinstance(conv, dict):
            continue
        messages: list[PreviewMessage] = []
        raw_messages = conv.get("chat_messages") or []
        for msg_index, msg in enumerate(raw_messages):
            if not isinstance(msg, dict):
                continue
            role = normalize_role(msg.get("sender"))
            text = sanitize_claude_text(extract_claude_text(msg))
            if not role:
                if not text:
                    skipped_empty += 1
                continue
            attachment_placeholder = claude_attachment_placeholder(msg) if role == "user" else ""
            if not text and attachment_placeholder:
                messages.append(PreviewMessage(
                    id=str(msg.get("uuid") or f"m{msg_index}"),
                    role=role,
                    source_role=str(msg.get("sender") or ""),
                    content=attachment_placeholder,
                    created_at=normalize_timestamp(msg.get("created_at")),
                    model=extract_model_hint(msg),
                    attachment_placeholder=True,
                ))
                continue
            if not text:
                skipped_empty += 1
                messages.append(make_skipped_message(
                    id=str(msg.get("uuid") or f"m{msg_index}"),
                    role=role,
                    source_role=str(msg.get("sender") or ""),
                    created_at=normalize_timestamp(msg.get("created_at")),
                    model=extract_model_hint(msg),
                    reason="Empty message",
                ))
                continue
            messages.append(PreviewMessage(
                id=str(msg.get("uuid") or f"m{msg_index}"),
                role=role,
                source_role=str(msg.get("sender") or ""),
                content=text,
                created_at=normalize_timestamp(msg.get("created_at")),
                model=extract_model_hint(msg),
            ))
        warnings = []
        if not raw_messages:
            skipped_empty += 1
            warnings.append("No messages found.")
        title = clean_title(conv.get("name"), fallback=f"Untitled Claude Conversation {index + 1}")
        if not str(conv.get("name") or "").strip():
            warnings.append("Untitled conversation.")
        conversations.append(PreviewConversation(
            id=f"claude:{conv.get('uuid') or index}",
            source_id=str(conv.get("uuid") or index),
            title=title,
            created_at=normalize_timestamp(conv.get("created_at")),
            updated_at=normalize_timestamp(conv.get("updated_at")),
            model=extract_model_hint(conv),
            messages=messages,
            warnings=warnings,
        ))
    return PreviewBatch(
        source="claude",
        source_version="conversations.json",
        file_name=file_name,
        file_size=file_size,
        conversations=conversations,
        skipped_counts={"empty_messages_or_conversations": skipped_empty},
    )


def _preview_chatgpt(file_name: str, file_size: int, payload: list[Any]) -> PreviewBatch:
    conversations: list[PreviewConversation] = []
    skipped_internal = 0
    skipped_empty = 0
    skipped_branches = 0
    skipped_hidden = 0
    skipped_empty_companion = 0
    for index, conv in enumerate(payload):
        if not isinstance(conv, dict):
            continue
        walked, branch_count = walk_chatgpt_messages(
            conv.get("mapping") or {},
            str(conv.get("current_node") or ""),
        )
        skipped_branches += branch_count
        messages = []
        for msg_index, msg in enumerate(walked):
            role = normalize_role(msg.get("role"))
            text = msg.get("content", "").strip()
            recipient = msg.get("recipient", "")
            attachment_kinds = msg.get("attachment_kinds") or []
            has_image = any("image" in kind.lower() for kind in attachment_kinds)
            if msg.get("hidden"):
                # Platform context injections (memory blobs, custom
                # instructions) are marked visually hidden — they were never
                # part of the visible dialogue and must not import as the
                # human's words.
                skipped_hidden += 1
                continue
            if str(msg.get("role") or "") == "tool" and has_image:
                # Image generations surface through tool-role messages
                # (dalle.text2im and friends). Show them as companion
                # attachment placeholders so image exchanges stay coherent.
                messages.append(PreviewMessage(
                    id=str(msg.get("id") or f"m{msg_index}"),
                    role="assistant",
                    source_role=str(msg.get("role") or ""),
                    content="[Companion sent an image. Image content was not included in this export.]",
                    created_at=normalize_timestamp(msg.get("created_at")),
                    model=msg.get("model", ""),
                    attachment_placeholder=True,
                ))
                continue
            if not role:
                skipped_internal += 1
                continue
            if recipient and recipient != "all":
                # Tool-directed traffic (dalle.text2im, python, browser, ...)
                skipped_internal += 1
                continue
            if not text:
                if is_chatgpt_internal_content(msg.get("content_type")):
                    skipped_internal += 1
                    continue
                if attachment_kinds:
                    # Image/attachment-only turns (human uploads, companion
                    # image generations) become placeholders, not skipped
                    # empties — the exchange stays coherent for review and
                    # for any future landing.
                    sender = "Human" if role == "user" else "Companion"
                    names = msg.get("attachment_names") or []
                    if names:
                        placeholder = (
                            f"[{sender} sent an attachment: {', '.join(names[:3])}. "
                            "File content was not included in this export.]"
                        )
                    elif has_image:
                        placeholder = f"[{sender} sent an image. Image content was not included in this export.]"
                    else:
                        placeholder = f"[{sender} sent an attachment. Attachment content was not included in this export.]"
                    messages.append(PreviewMessage(
                        id=str(msg.get("id") or f"m{msg_index}"),
                        role=role,
                        source_role=str(msg.get("role") or ""),
                        content=placeholder,
                        created_at=normalize_timestamp(msg.get("created_at")),
                        model=msg.get("model", ""),
                        attachment_placeholder=True,
                    ))
                    continue
                if role == "assistant":
                    # Empty companion text turns are pipeline artifacts
                    # (tool dispatch wrappers, thinking-flow segments) the
                    # ChatGPT UI never displayed. Count them, don't show them.
                    skipped_empty_companion += 1
                    continue
                skipped_empty += 1
                messages.append(make_skipped_message(
                    id=str(msg.get("id") or f"m{msg_index}"),
                    role=role,
                    source_role=str(msg.get("role") or ""),
                    created_at=normalize_timestamp(msg.get("created_at")),
                    model=msg.get("model", ""),
                    reason="Empty message",
                ))
                continue
            messages.append(PreviewMessage(
                id=str(msg.get("id") or f"m{msg_index}"),
                role=role,
                source_role=str(msg.get("role") or ""),
                content=text,
                created_at=normalize_timestamp(msg.get("created_at")),
                model=msg.get("model", ""),
            ))
        if not messages:
            skipped_empty += 1
        warnings = []
        if branch_count:
            warnings.append(
                f"{branch_count} message{'s' if branch_count != 1 else ''} from edited/regenerated branches left out."
            )
        source_id = conv.get("id") or conv.get("conversation_id") or index
        created_at = normalize_timestamp(conv.get("create_time"), from_epoch=True)
        conversations.append(PreviewConversation(
            id=f"chatgpt:{source_id}",
            source_id=str(source_id),
            title=clean_title(conv.get("title"), fallback=f"Untitled ChatGPT Conversation {index + 1}"),
            created_at=created_at,
            updated_at=normalize_timestamp(conv.get("update_time"), from_epoch=True),
            model=extract_model_hint(conv),
            messages=messages,
            warnings=warnings,
        ))
    return PreviewBatch(
        source="chatgpt",
        source_version="conversations.json",
        file_name=file_name,
        file_size=file_size,
        conversations=conversations,
        skipped_counts={
            "internal_or_unknown_messages": skipped_internal,
            "empty_messages_or_conversations": skipped_empty,
            "edited_branch_messages": skipped_branches,
            "hidden_context_messages": skipped_hidden,
            "empty_companion_turns": skipped_empty_companion,
        },
    )


def _preview_grok(file_name: str, file_size: int, payload: dict[str, Any]) -> PreviewBatch:
    conversations: list[PreviewConversation] = []
    skipped_branches = 0
    skipped_empty = 0
    partial_responses = 0
    error_responses = 0
    attachment_placeholders = 0
    citation_render_blocks = 0

    for index, entry in enumerate(payload.get("conversations") or []):
        if not isinstance(entry, dict):
            continue
        metadata = entry.get("conversation")
        if not isinstance(metadata, dict):
            continue
        walked, branch_count = walk_grok_responses(
            entry.get("responses") or [],
            metadata.get("leaf_response_id"),
        )
        skipped_branches += branch_count
        messages: list[PreviewMessage] = []
        conversation_partials = 0
        conversation_errors = 0
        conversation_citations = 0
        for response_index, response in enumerate(walked):
            role = normalize_role(response.get("sender"))
            if not role:
                continue
            if response.get("partial"):
                partial_responses += 1
                conversation_partials += 1
            if response.get("error"):
                error_responses += 1
                conversation_errors += 1

            text = str(response.get("message") or "").strip()
            if role == "assistant" and text:
                text, removed_citations = sanitize_grok_text(text)
                citation_render_blocks += removed_citations
                conversation_citations += removed_citations
            attachment_placeholder = grok_attachment_placeholder(response, role)
            if not text and attachment_placeholder:
                attachment_placeholders += 1
                messages.append(PreviewMessage(
                    id=str(response.get("_id") or f"m{response_index}"),
                    role=role,
                    source_role=str(response.get("sender") or ""),
                    content=attachment_placeholder,
                    created_at=normalize_grok_timestamp(response.get("create_time")),
                    model=str(response.get("model") or ""),
                    attachment_placeholder=True,
                ))
                continue
            if not text:
                skipped_empty += 1
                # Grok exports empty interrupted/error response wrappers.
                # They are transport state, not visible dialogue.
                continue
            messages.append(PreviewMessage(
                id=str(response.get("_id") or f"m{response_index}"),
                role=role,
                source_role=str(response.get("sender") or ""),
                content=text,
                created_at=normalize_grok_timestamp(response.get("create_time")),
                model=str(response.get("model") or ""),
            ))

        warnings = []
        if branch_count:
            warnings.append(
                f"{branch_count} response{'s' if branch_count != 1 else ''} from alternate branches left out."
            )
        if conversation_partials:
            warnings.append(
                f"{conversation_partials} interrupted/partial response"
                f"{'s' if conversation_partials != 1 else ''} found."
            )
        if conversation_errors:
            warnings.append(
                f"{conversation_errors} error response"
                f"{'s' if conversation_errors != 1 else ''} found."
            )
        if conversation_citations:
            warnings.append(
                f"{conversation_citations} exported citation marker"
                f"{'s' if conversation_citations != 1 else ''} cleaned from companion text."
            )
        if not messages:
            skipped_empty += 1

        source_id = metadata.get("id") or metadata.get("_id") or index
        model = next(
            (message.model for message in reversed(messages) if message.model),
            "",
        )
        conversations.append(PreviewConversation(
            id=f"grok:{source_id}",
            source_id=str(source_id),
            title=clean_title(metadata.get("title"), fallback=f"Untitled Grok Conversation {index + 1}"),
            created_at=normalize_grok_timestamp(metadata.get("create_time")),
            updated_at=normalize_grok_timestamp(metadata.get("modify_time")),
            model=model,
            messages=messages,
            warnings=warnings,
        ))

    projects = [item for item in (payload.get("projects") or []) if isinstance(item, dict)]
    project_ids = {
        str(item.get("workspace_id"))
        for item in projects
        if item.get("workspace_id") not in (None, "")
    }
    linked_project_conversations = sum(
        1
        for item in (payload.get("conversations") or [])
        if isinstance(item, dict)
        and isinstance(item.get("conversation"), dict)
        and str(item["conversation"].get("workspace_id") or "") in project_ids
    )
    batch_warnings = []
    if project_ids and not linked_project_conversations:
        batch_warnings.append(
            "Grok project metadata is present, but this export contains no project-linked conversations. "
            "Project chats may be missing from the source export."
        )

    return PreviewBatch(
        source="grok",
        source_version="prod-grok-backend",
        file_name=file_name,
        file_size=file_size,
        conversations=conversations,
        warnings=batch_warnings,
        skipped_counts={
            "empty_or_transport_responses": skipped_empty,
            "alternate_branch_responses": skipped_branches,
            "partial_responses": partial_responses,
            "error_responses": error_responses,
            "attachment_placeholders": attachment_placeholders,
            "citation_render_blocks": citation_render_blocks,
        },
    )


def _preview_openrouter(file_name: str, file_size: int, payload: dict[str, Any]) -> PreviewBatch:
    raw_messages = payload.get("messages")
    raw_items = payload.get("items")
    if not isinstance(raw_messages, dict) or not isinstance(raw_items, dict):
        raise ValueError("OpenRouter export is missing messages/items maps.")

    messages: list[PreviewMessage] = []
    skipped_reasoning = 0
    skipped_empty = 0
    for index, message in enumerate(raw_messages.values()):
        if not isinstance(message, dict):
            continue
        role = normalize_role(message.get("type"))
        if role not in {"user", "assistant"}:
            continue
        visible_parts: list[str] = []
        for item_id in openrouter_item_ids(message):
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
            if data.get("role") and normalize_role(data.get("role")) != role:
                continue
            text = content_text(data.get("content")).strip()
            if text:
                visible_parts.append(text)
        content = "\n\n".join(visible_parts).strip()
        if not content:
            skipped_empty += 1
            messages.append(make_skipped_message(
                id=str(message.get("id") or f"m{index}"),
                role=role,
                source_role=str(message.get("type") or ""),
                created_at=normalize_timestamp(message.get("createdAt")),
                model=extract_model_hint(message),
                reason="Empty visible message",
            ))
            continue
        messages.append(PreviewMessage(
            id=str(message.get("id") or f"m{index}"),
            role=role,
            source_role=str(message.get("type") or ""),
            content=content,
            created_at=normalize_timestamp(message.get("createdAt")),
            model=extract_model_hint(message),
        ))

    if not messages:
        raise ValueError("No visible human/companion messages found in OpenRouter export.")

    title = clean_title(payload.get("title"), fallback="OpenRouter Chat")
    conv = PreviewConversation(
        id=f"openrouter:{slug_id(title)}",
        source_id=str(payload.get("id") or title),
        title=title,
        created_at=messages[0].created_at,
        updated_at=messages[-1].created_at,
        model=extract_model_hint(payload),
        messages=messages,
        warnings=(["Hidden reasoning blocks were skipped."] if skipped_reasoning else []),
    )
    return PreviewBatch(
        source="openrouter",
        source_version=str(payload.get("version") or ""),
        file_name=file_name,
        file_size=file_size,
        conversations=[conv],
        skipped_counts={
            "reasoning_blocks": skipped_reasoning,
            "empty_messages": skipped_empty,
        },
    )


def _preview_generic_list(file_name: str, file_size: int, payload: list[Any]) -> PreviewBatch:
    conversations: list[PreviewConversation] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        conv = generic_conversation_from_dict(item, index, "generic")
        if conv:
            conversations.append(conv)
    return PreviewBatch(
        source="generic",
        source_version="json-list",
        file_name=file_name,
        file_size=file_size,
        conversations=conversations,
        warnings=(["Generic parser used. Review role/date mapping carefully."] if conversations else []),
    )


def _preview_generic_dict(file_name: str, file_size: int, payload: dict[str, Any]) -> PreviewBatch:
    conv = generic_conversation_from_dict(payload, 0, "generic")
    conversations = [conv] if conv else []
    return PreviewBatch(
        source="generic",
        source_version="json-object",
        file_name=file_name,
        file_size=file_size,
        conversations=conversations,
        warnings=(["Generic parser used. Review role/date mapping carefully."] if conversations else []),
    )


def _preview_markdown(path: Path) -> PreviewBatch:
    text = path.read_text(encoding="utf-8")
    metadata, body = split_frontmatter(text)
    messages = []
    for index, (role, content) in enumerate(parse_markdown_messages(body)):
        normalized = normalize_role(role)
        if not normalized:
            continue
        if not content.strip():
            messages.append(make_skipped_message(
                id=f"m{index}",
                role=normalized,
                source_role=role,
                created_at=normalize_timestamp(metadata.get("date")),
                model=str(metadata.get("model") or metadata.get("model_slug") or "").strip(),
                reason="Empty message",
            ))
            continue
        messages.append(PreviewMessage(
            id=f"m{index}",
            role=normalized,
            source_role=role,
            content=content.strip(),
            created_at=normalize_timestamp(metadata.get("date")),
            model=str(metadata.get("model") or metadata.get("model_slug") or "").strip(),
        ))
    if not messages:
        raise ValueError("No visible human/companion messages found in Markdown export.")
    title = clean_title(metadata.get("title"), fallback=path.stem)
    conv = PreviewConversation(
        id=f"markdown:{slug_id(path.stem)}",
        source_id=str(metadata.get("conversation_id") or path.stem),
        title=title,
        created_at=normalize_timestamp(metadata.get("date")),
        model=str(metadata.get("model") or metadata.get("model_slug") or "").strip(),
        messages=messages,
    )
    return PreviewBatch(
        source=str(metadata.get("source") or "markdown"),
        source_version="markdown",
        file_name=path.name,
        file_size=path.stat().st_size,
        conversations=[conv],
    )


def generic_conversation_from_dict(item: dict[str, Any], index: int, source: str) -> PreviewConversation | None:
    raw_messages = item.get("messages") or item.get("chat_messages") or item.get("conversation")
    if not isinstance(raw_messages, list):
        return None
    messages = []
    for msg_index, raw in enumerate(raw_messages):
        if not isinstance(raw, dict):
            continue
        source_role = raw.get("role") or raw.get("sender") or raw.get("author")
        role = normalize_role(source_role)
        content = (
            raw.get("content")
            or raw.get("text")
            or raw.get("message")
            or raw.get("body")
            or ""
        )
        if isinstance(content, list):
            content = content_text(content)
        if not role:
            continue
        if not str(content).strip():
            messages.append(make_skipped_message(
                id=str(raw.get("id") or raw.get("uuid") or f"m{msg_index}"),
                role=role,
                source_role=str(source_role or ""),
                created_at=normalize_timestamp(raw.get("created_at") or raw.get("timestamp") or raw.get("createdAt")),
                model=extract_model_hint(raw),
                reason="Empty message",
            ))
            continue
        messages.append(PreviewMessage(
            id=str(raw.get("id") or raw.get("uuid") or f"m{msg_index}"),
            role=role,
            source_role=str(source_role or ""),
            content=str(content).strip(),
            created_at=normalize_timestamp(raw.get("created_at") or raw.get("timestamp") or raw.get("createdAt")),
            model=extract_model_hint(raw),
        ))
    if not messages:
        return None
    return PreviewConversation(
        id=f"{source}:{item.get('id') or item.get('uuid') or index}",
        source_id=str(item.get("id") or item.get("uuid") or index),
        title=clean_title(item.get("title") or item.get("name"), fallback=f"Imported Conversation {index + 1}"),
        created_at=normalize_timestamp(item.get("created_at") or item.get("createdAt") or item.get("create_time")),
        updated_at=normalize_timestamp(item.get("updated_at") or item.get("updatedAt") or item.get("update_time")),
        model=extract_model_hint(item),
        messages=messages,
        warnings=["Generic parser used."],
    )


def make_skipped_message(
    id: str,
    role: str,
    source_role: str = "",
    created_at: str = "",
    model: str = "",
    reason: str = "Skipped message",
) -> PreviewMessage:
    return PreviewMessage(
        id=id,
        role=role,
        source_role=source_role,
        content="",
        created_at=created_at,
        model=model,
        skipped=True,
        skip_reason=reason,
    )


def conversation_review_issues(messages: list[PreviewMessage]) -> list[dict[str, Any]]:
    issues = []
    duplicate_clusters = duplicate_resend_clusters(messages)
    duplicate_indexes = {
        index
        for cluster in duplicate_clusters
        for index in range(cluster["start_index"], cluster["end_index"] + 1)
    }
    for cluster in duplicate_clusters:
        count = cluster["count"]
        role_label = "Companion" if cluster["role"] == "assistant" else "Human"
        issues.append({
            "id": f"duplicate-{cluster['start_index']}",
            "exchange_index": len(issues) + 1,
            "message_index": cluster["start_index"],
            "message_end_index": cluster["end_index"],
            "warnings": [f"Duplicate {role_label} resend cluster ({count} copies)."],
            "label": f"Duplicate {role_label} resend cluster ({count} copies).",
        })
    for exchange_index, exchange in enumerate(build_review_exchanges(messages), start=1):
        if all(index in duplicate_indexes for index in range(exchange["start_index"], exchange["end_index"] + 1)):
            continue
        warnings = exchange_review_warnings(exchange["messages"])
        if not warnings:
            continue
        issues.append({
            "id": f"issue-{exchange['start_index']}",
            "exchange_index": exchange_index,
            "message_index": exchange["start_index"],
            "message_end_index": exchange["end_index"],
            "warnings": warnings,
            "label": "; ".join(warnings),
        })
    return issues


def duplicate_resend_clusters(messages: list[PreviewMessage]) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    run_start = 0
    run_key = duplicate_resend_key(messages[0]) if messages else ""

    def finish_run(end_index: int) -> None:
        if not run_key or end_index - run_start + 1 < 2:
            return
        first = messages[run_start]
        clusters.append({
            "id": f"dup-{run_start}",
            "role": first.role,
            "content_key": run_key,
            "start_index": run_start,
            "end_index": end_index,
            "count": end_index - run_start + 1,
        })

    for index in range(1, len(messages)):
        key = duplicate_resend_key(messages[index])
        if key and key == run_key:
            continue
        finish_run(index - 1)
        run_start = index
        run_key = key
    if messages:
        finish_run(len(messages) - 1)
    return clusters


def duplicate_resend_map(messages: list[PreviewMessage]) -> dict[int, dict[str, Any]]:
    mapped: dict[int, dict[str, Any]] = {}
    for cluster in duplicate_resend_clusters(messages):
        for index in range(cluster["start_index"], cluster["end_index"] + 1):
            mapped[index] = {
                "duplicate_cluster_id": cluster["id"],
                "duplicate_cluster_count": cluster["count"],
                "duplicate_cluster_start": cluster["start_index"],
                "duplicate_cluster_end": cluster["end_index"],
                "duplicate_index": index - cluster["start_index"] + 1,
                "duplicate_representative": index == cluster["start_index"],
            }
    return mapped


def duplicate_resend_key(message: PreviewMessage) -> str:
    if message.skipped or message.attachment_placeholder:
        return ""
    text = normalize_duplicate_text(message.content)
    if not text:
        return ""
    return f"{message.role}:{text}"


def normalize_duplicate_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def build_review_exchanges(messages: list[PreviewMessage]) -> list[dict[str, Any]]:
    exchanges: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def finish() -> None:
        nonlocal current
        if not current or not current["messages"]:
            return
        exchanges.append(current)
        current = None

    for index, message in enumerate(messages):
        role = "assistant" if message.role == "assistant" else "user"
        if not current:
            current = {
                "start_index": index,
                "end_index": index,
                "messages": [message],
            }
            continue

        # One exchange = a run of human messages followed by a run of
        # companion replies. Double-texting stays inside the exchange; a new
        # human message after a companion reply starts the next one.
        has_assistant = any(item.role == "assistant" for item in current["messages"])
        if role == "user" and has_assistant:
            finish()
            current = {
                "start_index": index,
                "end_index": index,
                "messages": [message],
            }
            continue
        current["messages"].append(message)
        current["end_index"] = index

    finish()
    return exchanges


def exchange_review_warnings(messages: list[PreviewMessage]) -> list[str]:
    warnings = []
    roles = ["assistant" if message.role == "assistant" else "user" for message in messages]
    if any(message.skipped for message in messages):
        warnings.append("Contains skipped empty message.")
    if "user" not in roles:
        warnings.append("No human message in this exchange.")
    if "assistant" not in roles:
        warnings.append("No companion reply in this exchange.")
    return warnings


def extract_claude_text(msg: dict[str, Any]) -> str:
    text = str(msg.get("text") or "").strip()
    if text:
        return text
    parts = []
    for block in msg.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
            parts.append(str(block["text"]).strip())
    return "\n\n".join(part for part in parts if part)


def sanitize_claude_text(text: str) -> str:
    unsupported = r"```[\s\n]*This block is not supported on your current device yet\.?[\s\n]*```"
    text = re.sub(unsupported, "", text, flags=re.IGNORECASE)
    text = text.replace("This block is not supported on your current device yet.", "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def claude_attachment_placeholder(msg: dict[str, Any]) -> str:
    files = [item for item in (msg.get("files") or []) if isinstance(item, dict)]
    attachments = [item for item in (msg.get("attachments") or []) if isinstance(item, dict)]
    content_blocks = [item for item in (msg.get("content") or []) if isinstance(item, dict)]
    non_text_blocks = [
        str(item.get("type") or "attachment")
        for item in content_blocks
        if str(item.get("type") or "").lower() not in {"", "text"}
    ]
    if not files and not attachments and not non_text_blocks:
        return ""

    names = [
        str(item.get("file_name") or item.get("name") or item.get("filename") or "").strip()
        for item in [*files, *attachments]
    ]
    names = [name for name in names if name]
    if names:
        return f"[Human sent an attachment: {', '.join(names[:3])}. File content was not included in this export.]"
    if any("image" in kind.lower() for kind in non_text_blocks):
        return "[Human sent an image. Image content was not included in this export.]"
    return "[Human sent an attachment. Attachment content was not included in this export.]"


def is_grok_export(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    conversations = payload.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return False
    first = next((item for item in conversations if isinstance(item, dict)), None)
    return (
        isinstance(first, dict)
        and isinstance(first.get("conversation"), dict)
        and isinstance(first.get("responses"), list)
    )


def walk_grok_responses(
    wrapped_responses: Any,
    declared_leaf_id: Any = None,
) -> tuple[list[dict[str, Any]], int]:
    """Return Grok's canonical response chain and alternate-branch count."""
    if not isinstance(wrapped_responses, list):
        return [], 0
    responses = [
        wrapper.get("response")
        for wrapper in wrapped_responses
        if isinstance(wrapper, dict) and isinstance(wrapper.get("response"), dict)
    ]
    by_id = {
        str(response.get("_id")): response
        for response in responses
        if response.get("_id") not in (None, "")
    }
    if not by_id:
        return responses, 0

    leaf_id = str(declared_leaf_id or "")
    if leaf_id not in by_id:
        parent_ids = {
            str(response.get("parent_response_id"))
            for response in responses
            if response.get("parent_response_id") not in (None, "")
        }
        terminal_ids = [
            str(response.get("_id"))
            for response in responses
            if response.get("_id") not in (None, "")
            and str(response.get("_id")) not in parent_ids
        ]
        # Backend exports are ordered; if several regenerated branches are
        # terminal, the last terminal is the current visible branch.
        leaf_id = terminal_ids[-1] if terminal_ids else str(responses[-1].get("_id") or "")

    path: list[dict[str, Any]] = []
    seen: set[str] = set()
    while leaf_id and leaf_id in by_id and leaf_id not in seen:
        seen.add(leaf_id)
        response = by_id[leaf_id]
        path.append(response)
        leaf_id = str(response.get("parent_response_id") or "")
    path.reverse()
    if not path:
        return responses, 0
    return path, max(0, len(responses) - len(path))


def grok_attachment_placeholder(response: dict[str, Any], role: str) -> str:
    files = response.get("file_attachments")
    generated_images = response.get("generated_image_urls")
    image_edit = response.get("image_edit_uri")
    if not files and not generated_images and not image_edit:
        return ""
    sender = "Human" if role == "user" else "Companion"
    if generated_images or image_edit:
        return f"[{sender} sent an image. Image content was not included in this export.]"
    return f"[{sender} sent an attachment. Attachment content was not included in this export.]"


def sanitize_grok_text(text: str) -> tuple[str, int]:
    """Remove Grok UI citation cards while preserving ordinary markup/code."""
    cleaned, removed = re.subn(
        r"<grok:render\b[^>]*>.*?</grok:render\s*>",
        "",
        str(text or ""),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not removed:
        return cleaned.strip(), 0
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip(), removed


def walk_chatgpt_messages(mapping: dict[str, Any], current_node: str = "") -> tuple[list[dict[str, Any]], int]:
    """Return (messages, skipped_branch_count) for the canonical thread.

    ChatGPT stores every edit/regeneration as a sibling branch in the mapping
    tree; the live thread is the parent chain above ``current_node``. Walking
    anything else imports ghost messages from abandoned branches.
    """
    if not isinstance(mapping, dict):
        return [], 0
    node_ids = chatgpt_canonical_node_ids(mapping, current_node)
    if not node_ids:
        node_ids = chatgpt_bfs_node_ids(mapping)
    messages = []
    for node_id in node_ids:
        node = mapping.get(node_id)
        if not isinstance(node, dict):
            continue
        msg = node.get("message")
        if not isinstance(msg, dict):
            continue
        author = msg.get("author") or {}
        metadata = msg.get("metadata") or {}
        role = author.get("role")
        raw_content = msg.get("content") or {}
        content = chatgpt_content_text(raw_content)
        created_at = normalize_timestamp(msg.get("create_time"), from_epoch=True)
        # File/document uploads live in metadata.attachments, not in the
        # content parts — an "empty" human message often carried a file.
        meta_attachments = [
            item for item in (metadata.get("attachments") or []) if isinstance(item, dict)
        ]
        attachment_kinds = chatgpt_attachment_kinds(raw_content) + [
            str(item.get("mimeType") or item.get("mime_type") or "attachment")
            for item in meta_attachments
        ]
        messages.append({
            "id": msg.get("id") or node_id,
            "role": role,
            "content": content,
            "content_type": raw_content.get("content_type") if isinstance(raw_content, dict) else "",
            "recipient": str(msg.get("recipient") or ""),
            "attachment_kinds": attachment_kinds,
            "attachment_names": [
                str(item.get("name") or "").strip()
                for item in meta_attachments
                if str(item.get("name") or "").strip()
            ],
            "hidden": bool(metadata.get("is_visually_hidden_from_conversation")),
            "model": str(metadata.get("model_slug") or ""),
            "created_at": created_at,
        })
    total_message_nodes = sum(
        1 for node in mapping.values()
        if isinstance(node, dict) and isinstance(node.get("message"), dict)
    )
    skipped_branches = max(0, total_message_nodes - len(messages))
    return messages, skipped_branches


def chatgpt_canonical_node_ids(mapping: dict[str, Any], current_node: str) -> list[str]:
    node_id = str(current_node or "")
    if node_id not in mapping:
        return []
    path: list[str] = []
    seen: set[str] = set()
    while node_id and node_id in mapping and node_id not in seen:
        seen.add(node_id)
        path.append(node_id)
        node = mapping.get(node_id)
        parent = node.get("parent") if isinstance(node, dict) else None
        node_id = parent if isinstance(parent, str) else ""
    path.reverse()
    return path


def chatgpt_bfs_node_ids(mapping: dict[str, Any]) -> list[str]:
    all_children: set[str] = set()
    for node in mapping.values():
        if isinstance(node, dict):
            all_children.update(child for child in node.get("children", []) if isinstance(child, str))
    roots = [key for key in mapping if key not in all_children] or list(mapping.keys())
    queue = list(roots)
    seen: set[str] = set()
    ordered = []
    while queue:
        node_id = queue.pop(0)
        if node_id in seen:
            continue
        seen.add(node_id)
        node = mapping.get(node_id)
        if not isinstance(node, dict):
            continue
        ordered.append(node_id)
        queue.extend(child for child in node.get("children", []) if isinstance(child, str))
    return ordered


def is_chatgpt_internal_content(content_type: Any) -> bool:
    # code/execution_output is code-interpreter and image-generation tool
    # traffic; thoughts/reasoning_recap are thinking-model UI chrome
    # ("Thought for 34s"). None of it is dialogue — never surface it as
    # transcript text.
    return str(content_type or "") in {
        "system_error",
        "user_editable_context",
        "thoughts",
        "reasoning_recap",
        "code",
        "execution_output",
    }


def chatgpt_content_text(content: Any) -> str:
    if not isinstance(content, dict):
        return ""
    if is_chatgpt_internal_content(content.get("content_type")):
        return ""
    parts = []
    for part in content.get("parts") or []:
        if isinstance(part, str):
            if part.strip():
                parts.append(part.strip())
        elif isinstance(part, dict):
            # audio_transcription and similar parts carry their text inline;
            # image_asset_pointer and other binary parts do not.
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n\n".join(parts)


def chatgpt_attachment_kinds(content: Any) -> list[str]:
    """Content-part types that carry no inline text (images, files, audio)."""
    if not isinstance(content, dict):
        return []
    kinds = []
    for part in content.get("parts") or []:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            continue
        kinds.append(str(part.get("content_type") or "attachment"))
    return kinds


def openrouter_item_ids(message: dict[str, Any]) -> list[str]:
    ids = []
    for item in message.get("items") or []:
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, dict) and isinstance(item.get("id"), str):
            ids.append(item["id"])
    return ids


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
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


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    metadata: dict[str, Any] = {}
    for raw in parts[1].splitlines():
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"').strip("'")
    return metadata, parts[2]


def parse_markdown_messages(body: str) -> list[tuple[str, str]]:
    blocks = re.split(r"^> \*\*([^*]+):\*\*", body, flags=re.MULTILINE)
    messages = []
    for index in range(1, len(blocks), 2):
        role = blocks[index].strip()
        content_block = blocks[index + 1] if index + 1 < len(blocks) else ""
        lines = []
        for line in content_block.splitlines():
            while line.startswith("> "):
                line = line[2:]
            while line.startswith(">"):
                line = line[1:]
            lines.append(line)
        content = "\n".join(lines).strip()
        messages.append((role, content))
    return messages


def normalize_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    return ROLE_ALIASES.get(role, "")


def normalize_timestamp(value: Any, from_epoch: bool = False) -> str:
    if value in (None, ""):
        return ""
    try:
        if from_epoch or isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
        raw = str(value).strip()
        if not raw:
            return ""
        if re.fullmatch(r"\d+(\.\d+)?", raw):
            return datetime.fromtimestamp(float(raw), tz=timezone.utc).isoformat()
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return ""


def normalize_grok_timestamp(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("$date", value)
        if isinstance(value, dict):
            value = value.get("$numberLong", "")
        try:
            # Mongo Extended JSON stores epoch milliseconds.
            return normalize_timestamp(float(value) / 1000.0, from_epoch=True)
        except (TypeError, ValueError):
            return ""
    return normalize_timestamp(value)


def sort_conversations_recent_first(conversations: list[PreviewConversation]) -> list[PreviewConversation]:
    return sorted(
        conversations,
        key=lambda conv: timestamp_sort_value(
            conv.updated_at
            or conv.created_at
            or (conv.messages[-1].created_at if conv.messages else "")
        ),
        reverse=True,
    )


def timestamp_sort_value(value: str) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def extract_model_hint(value: Any, depth: int = 0) -> str:
    if depth > 3:
        return ""
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if "model" in key_text and isinstance(item, str):
                cleaned = collapse_ws(item)
                if 1 <= len(cleaned) <= 120:
                    return cleaned
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in {"text", "content", "messages", "chat_messages", "mapping"}:
                continue
            found = extract_model_hint(item, depth + 1)
            if found:
                return found
    elif isinstance(value, list):
        for item in value[:20]:
            found = extract_model_hint(item, depth + 1)
            if found:
                return found
    return ""


def clean_title(value: Any, fallback: str) -> str:
    title = collapse_ws(str(value or ""))
    return title[:140] or fallback


def collapse_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def slug_id(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug[:48] or "conversation"
