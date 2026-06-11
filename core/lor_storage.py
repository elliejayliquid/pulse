"""Shared LoR storage helpers."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


LOR_STORAGE_DEFAULTS = {
    "posts.json": [],
    "authors.json": {},
    "embeddings.json": {},
}


def ensure_lor_storage(data_dir: Path) -> bool:
    """Create the LoR data directory and empty storage files if needed."""
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        for filename, default in LOR_STORAGE_DEFAULTS.items():
            path = data_dir / filename
            if not path.exists():
                path.write_text(
                    json.dumps(default, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                logger.info(f"LoR storage initialized: {path}")
        return True
    except OSError as e:
        logger.error(f"Failed to initialize LoR storage at {data_dir}: {e}")
        return False
