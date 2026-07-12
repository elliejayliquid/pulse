"""
Documents skill — read files from the persona's inbox.

Files sent over Telegram (PDFs, text files, code) are saved to
personas/<p>/data/inbox/. Small documents are injected inline into the
conversation by the channel; this skill lets the companion list the inbox
and read longer documents in slices.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from core.documents import extract_text, inbox_dir, safe_filename
from skills.base import BaseSkill

logger = logging.getLogger(__name__)

SLICE_CHARS = 6000


class DocumentsSkill(BaseSkill):
    name = "documents"
    description = "Read documents and files your human sent (inbox)"
    search_summary = "List and read files your human sent: PDFs, text, code"
    search_examples = [
        "read the pdf", "open the document", "what files do I have",
        "read the attachment",
    ]
    aliases = ["document", "file", "pdf", "attachment", "inbox", "read file"]
    categories = ["files", "reading"]
    always_loaded = False
    workflow = (
        "Use list_documents to see what your human has sent, then "
        "read_document to read one. Long documents come in slices — the "
        "footer tells you the offset for the next slice. Quote or summarize "
        "faithfully; say so if a document could not be read."
    )

    def __init__(self, config: dict):
        super().__init__(config)
        self._inbox = inbox_dir(config)

    def get_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_documents",
                    "description": (
                        "List files in your inbox — documents your human sent "
                        "you (via Telegram or placed there directly)."
                    ),
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_document",
                    "description": (
                        "Read a document from your inbox by filename. Long "
                        "documents are returned in slices; pass the offset "
                        "from the previous slice's footer to continue reading."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "Exact filename from list_documents.",
                            },
                            "offset": {
                                "type": "integer",
                                "description": "Character offset to start from (default 0).",
                            },
                        },
                        "required": ["filename"],
                    },
                },
            },
        ]

    def execute(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "list_documents":
            return self._list()
        if tool_name == "read_document":
            return self._read(
                str(arguments.get("filename", "") or ""),
                int(arguments.get("offset", 0) or 0),
            )
        return f"Unknown tool: {tool_name}"

    def _list(self) -> str:
        if not self._inbox.is_dir():
            return "Your inbox is empty — no documents have been sent yet."
        files = sorted(
            (p for p in self._inbox.iterdir() if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not files:
            return "Your inbox is empty — no documents have been sent yet."
        lines = [f"Inbox ({len(files)} file{'s' if len(files) != 1 else ''}):"]
        for path in files[:50]:
            stat = path.stat()
            received = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            size_kb = max(1, stat.st_size // 1024)
            lines.append(f"- {path.name} ({size_kb} KB, received {received})")
        if len(files) > 50:
            lines.append(f"...and {len(files) - 50} older file(s).")
        return "\n".join(lines)

    def _read(self, filename: str, offset: int) -> str:
        if not filename:
            return "Which document? Use list_documents to see the inbox."
        # Sanitize and resolve inside the inbox only — filenames come from
        # model output.
        path = (self._inbox / safe_filename(filename)).resolve()
        try:
            path.relative_to(self._inbox.resolve())
        except ValueError:
            return "Documents can only be read from the inbox."
        if not path.is_file():
            return f"No document named '{filename}' in the inbox. Use list_documents."

        text, note = extract_text(path)
        if not text:
            return f"Could not read '{path.name}': {note}"

        total = len(text)
        offset = max(0, min(offset, total))
        chunk = text[offset:offset + SLICE_CHARS]
        end = offset + len(chunk)
        header = f"[{path.name} — chars {offset}-{end} of {total}]\n"
        footer = (
            f"\n[Document continues. Call read_document(filename=\"{path.name}\", "
            f"offset={end}) for the next part.]"
            if end < total else "\n[End of document.]"
        )
        return header + chunk + footer
