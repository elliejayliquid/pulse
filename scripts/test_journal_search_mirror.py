"""Verify journal search-summary mirrors stay compact and stable."""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.db import PulseDatabase
import skills.journal as journal_module
from skills.journal import JournalSkill


class FakeEmbeddingModel:
    def __init__(self):
        self.texts = []

    def encode(self, text, *args, **kwargs):
        self.texts.append(text)
        return np.array([len(text), 1.0, 0.5], dtype=np.float32)


def test_journal_search_mirror():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        db = PulseDatabase(root / "demo.db")
        model = FakeEmbeddingModel()
        original_get_model = journal_module._get_embedding_model
        journal_module._get_embedding_model = lambda: model
        try:
            skill = JournalSkill({
                "_db": db,
                "_persona_name": "Demo",
                "_user_name": "Lena",
                "paths": {
                    "journal": str(root / "journal"),
                    "memories": str(root / "memories"),
                },
            })

            result = skill._write_entry(
                content=(
                    "A reflective opening about the day.\n\n"
                    "The important searchable fact is that Piper is Lena's girl cat."
                ),
                entry_type="reflection",
                why_it_mattered="Future Demo should not misgender Piper.",
                search_summary="Piper is Lena's girl cat; remember she is female.",
                tags="piper,cat",
                force=True,
            )
            assert "Journal entry saved" in result

            entry = db.get_journal_entry("001")
            assert entry["search_summary"].startswith("Piper is Lena")
            assert entry["summary_needs_review"] is False

            journal_memories = [
                mem for mem in db.get_all_memories(include_superseded=True)
                if mem.get("type") == "journal"
            ]
            assert len(journal_memories) == 1
            mirror = journal_memories[0]
            mirror_id = mirror["id"]
            assert "Search summary:" in mirror["text"]
            assert "Piper is Lena's girl cat" in mirror["text"]
            assert "Journal entry" not in model.texts[-1]
            assert "Search summary:" not in model.texts[-1]
            assert "Piper is Lena's girl cat" in model.texts[-1]

            update = skill._update_entry(
                "001",
                "Updated full entry. Piper is still Lena's girl cat.",
                search_summary="Piper is Lena's female cat; do not call her a boy.",
            )
            assert "updated" in update
            updated_entry = db.get_journal_entry("001")
            assert updated_entry["summary_needs_review"] is False
            assert "female cat" in updated_entry["search_summary"]

            updated_mirror = [
                mem for mem in db.get_all_memories(include_superseded=True)
                if mem.get("type") == "journal"
            ][0]
            assert updated_mirror["id"] == mirror_id
            assert "female cat" in updated_mirror["text"]

            skill._update_entry("001", "Content changed without touching the summary.")
            drift_entry = db.get_journal_entry("001")
            assert drift_entry["summary_needs_review"] is True
        finally:
            journal_module._get_embedding_model = original_get_model
            db.close()


if __name__ == "__main__":
    test_journal_search_mirror()
    print("[OK] Journal search mirror")
