import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.append(os.getcwd())

from gui.api import PulseAPI
from gui.backup import BackupManager


def test_gui_backup_creates_config_only_backup():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        persona_dir = root / "personas" / "demo"
        data_dir = persona_dir / "data"
        data_dir.mkdir(parents=True)

        (persona_dir / "config.yaml").write_text("provider:\n  type: openrouter\n", encoding="utf-8")
        (persona_dir / "persona.yaml").write_text("name: Demo\n", encoding="utf-8")
        (persona_dir / ".env").write_text("SECRET=value\n", encoding="utf-8")
        (data_dir / "demo.db").write_text("runtime data", encoding="utf-8")
        (data_dir / "status.json").write_text("{}", encoding="utf-8")

        api = PulseAPI(root)
        result = api.create_backup("demo", reason="pre-edit")
        assert result["ok"] is True

        backup = result["backup"]
        backup_dir = root / backup["path"]
        assert backup_dir.is_dir()
        assert (backup_dir / "config.yaml").exists()
        assert (backup_dir / "persona.yaml").exists()
        assert not (backup_dir / ".env").exists()
        assert not (backup_dir / "demo.db").exists()
        assert not (backup_dir / "status.json").exists()

        metadata = json.loads((backup_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["persona"] == "demo"
        assert metadata["reason"] == "pre-edit"
        assert metadata["files"] == ["config.yaml", "persona.yaml"]
        assert metadata["source_paths"]["config.yaml"] == "personas/demo/config.yaml"

        listed = api.list_backups("demo")
        assert listed["ok"] is True
        assert len(listed["backups"]) == 1
        assert listed["backups"][0]["path"] == backup["path"]


def test_gui_backup_prunes_per_persona():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        persona_dir = root / "personas" / "demo"
        persona_dir.mkdir(parents=True)
        (persona_dir / "config.yaml").write_text("provider:\n  type: local\n", encoding="utf-8")

        manager = BackupManager(root, keep_per_persona=2, max_age_days=30)
        manager.create_backup("demo", reason="one")
        manager.create_backup("demo", reason="two")
        manager.create_backup("demo", reason="three")

        backups = manager.list_backups("demo")
        assert len(backups) == 2
        assert [b["reason"] for b in backups] == ["three", "two"]


if __name__ == "__main__":
    test_gui_backup_creates_config_only_backup()
    test_gui_backup_prunes_per_persona()
    print("[OK] GUI backup")
