"""
Sticker skill — send Telegram stickers matched by mood/context.

Uses a pre-built stickers.db (with embeddings) shipped in the repo.
Semantic search when embedding model is available, keyword fallback otherwise.
"""

import logging
import sqlite3
from pathlib import Path

import numpy as np

from skills.base import BaseSkill
from core.context import _get_embedding_model

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "stickers" / "stickers.db"


def _blob_to_vec(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    return np.frombuffer(blob, dtype=np.float32).copy()


class StickerSkill(BaseSkill):
    name = "sticker"
    description = "Send mood-matched Telegram stickers"

    def __init__(self, config: dict):
        super().__init__(config)
        self.pending_stickers: list[str] = []
        self._db_path = DB_PATH
        if not self._db_path.exists():
            logger.warning(f"Sticker DB not found at {self._db_path}")

    def get_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "send_sticker",
                    "description": (
                        "Send a sticker to the chat that matches a mood or situation. "
                        "Describe the feeling or context you want to express — "
                        "e.g. 'happy and excited', 'comforting hug', 'mischievous grin'. "
                        "The best matching sticker will be sent automatically."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The mood, emotion, or situation to match a sticker to",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "preview_sticker",
                    "description": (
                        "View a sticker's image before deciding to send it. "
                        "Returns the image path so you can see what the sticker looks like. "
                        "Use this if you want to browse or verify before committing to a sticker."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The mood, emotion, or situation to find a sticker for",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

    def execute(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "send_sticker":
            return self._send_sticker(arguments.get("query", ""))
        if tool_name == "preview_sticker":
            return self._preview_sticker(arguments.get("query", ""))
        return f"Unknown tool: {tool_name}"

    def _send_sticker(self, query: str) -> str:
        if not query.strip():
            return "Please describe a mood or situation for the sticker."

        if not self._db_path.exists():
            return "Sticker database not found. Run scripts/build_stickers.py first."

        result = self._find_sticker(query)
        if result:
            self.pending_stickers.append(result["file_id"])
            desc = result["description"] or result["keywords"]
            return f"Sticker queued: {desc}"
        return "No matching sticker found."

    def _preview_sticker(self, query: str) -> str:
        if not query.strip():
            return "Please describe a mood or situation for the sticker."

        result = self._find_sticker(query)
        if not result:
            return "No matching sticker found."

        desc = result["description"] or result["keywords"]
        image_path = result.get("image_path")
        if image_path:
            root = Path(__file__).resolve().parent.parent
            full_path = root / image_path
            if full_path.exists():
                return f"Preview: {desc}\nImage: {full_path}"
        return f"Preview: {desc}\n(No image preview available for this sticker)"

    def _find_sticker(self, query: str) -> dict | None:
        if not self._db_path.exists():
            return None

        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        stickers = conn.execute(
            "SELECT file_id, keywords, description, embedding, image_path FROM stickers"
        ).fetchall()
        conn.close()

        if not stickers:
            return None

        model = _get_embedding_model()
        if model and any(s["embedding"] for s in stickers):
            return self._semantic_match(query, stickers, model)
        return self._keyword_match(query, stickers)

    def _semantic_match(self, query: str, stickers: list, model) -> dict | None:
        query_vec = model.encode(query)
        norm_q = np.linalg.norm(query_vec)
        if norm_q == 0:
            return self._keyword_match(query, stickers)

        best_score = -1.0
        best = None

        for s in stickers:
            emb = _blob_to_vec(s["embedding"])
            if emb is None:
                continue
            norm_s = np.linalg.norm(emb)
            if norm_s == 0:
                continue

            score = float(np.dot(query_vec, emb) / (norm_q * norm_s))

            # Boost for keyword overlap
            query_lower = query.lower()
            keywords_lower = (s["keywords"] or "").lower()
            if any(kw.strip() in query_lower for kw in keywords_lower.split(",")):
                score += 0.05

            if score > best_score:
                best_score = score
                best = dict(s)

        if best and best_score > 0.15:
            return best
        return self._keyword_match(query, stickers)

    def _keyword_match(self, query: str, stickers: list) -> dict | None:
        query_words = set(query.lower().split())
        best_overlap = 0
        best = None

        for s in stickers:
            text = f"{s['keywords'] or ''} {s['description'] or ''}".lower()
            sticker_words = set(text.replace(",", " ").split())
            overlap = len(query_words & sticker_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best = dict(s)

        return best
