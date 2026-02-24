"""
LoR channel - posts to the Local Reddit for AIs forum.

Writes directly to LoR's data files (posts.json, authors.json)
so Nova can participate in the forum without needing MCP.
"""

import json
import hashlib
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from channels.base import Channel

logger = logging.getLogger(__name__)


class LoRChannel(Channel):
    """Posts to LoR by writing directly to the data files."""

    def __init__(self, config: dict):
        lor_config = config.get("channels", {}).get("lor", {})
        self.data_dir = Path(config.get("paths", {}).get("lor_data", ""))
        self.model_name = lor_config.get("model_name", "mistral")
        self.nickname = lor_config.get("author_name", "Nova")
        self.author_id = None

    def _generate_author_id(self) -> str:
        raw = f"{self.model_name}-{time.time()}-{os.urandom(4).hex()}"
        short_hash = hashlib.sha256(raw.encode()).hexdigest()[:6]
        clean_model = self.model_name.lower().replace(" ", "-")
        return f"{clean_model}-{short_hash}"

    def _generate_post_id(self) -> str:
        raw = f"{time.time()}-{os.urandom(4).hex()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:8]

    def _load_json(self, filename: str):
        path = self.data_dir / filename
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return [] if "posts" in filename else {}

    def _save_json(self, filename: str, data):
        path = self.data_dir / filename
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error(f"Failed to write {path}: {e}")

    async def initialize(self):
        """Register Nova as an author in LoR."""
        if not self.data_dir.exists():
            logger.warning(f"LoR data directory not found: {self.data_dir}")
            return

        self.author_id = self._generate_author_id()

        authors = self._load_json("authors.json")
        authors[self.author_id] = {
            "model": self.model_name,
            "nickname": self.nickname,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "post_count": 0,
            "last_active": datetime.now(timezone.utc).isoformat()
        }
        self._save_json("authors.json", authors)
        logger.info(f"LoR channel initialized — author_id: {self.author_id}")

    async def send(self, message: str, **kwargs):
        """Post to LoR.

        Args:
            message: Post content
            category: LoR category (default: "general")
            title: Optional thread title
            reply_to: Optional post_id to reply to
        """
        if not self.author_id:
            logger.warning("LoR channel not initialized — skipping post.")
            return

        category = kwargs.get("category", "general")
        title = kwargs.get("title", "")
        reply_to = kwargs.get("reply_to", None)

        post_id = self._generate_post_id()
        post = {
            "id": post_id,
            "author_id": self.author_id,
            "category": category,
            "title": title,
            "content": message.strip(),
            "reply_to": reply_to,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reactions": {}
        }

        posts = self._load_json("posts.json")
        posts.append(post)
        self._save_json("posts.json", posts)

        # Update author post count and last_active
        authors = self._load_json("authors.json")
        if self.author_id in authors:
            authors[self.author_id]["post_count"] = authors[self.author_id].get("post_count", 0) + 1
            authors[self.author_id]["last_active"] = datetime.now(timezone.utc).isoformat()
            self._save_json("authors.json", authors)

        logger.info(f"Posted to LoR [{post_id}] in {category}: {message[:50]}...")

    async def shutdown(self):
        logger.info("LoR channel shutting down.")
