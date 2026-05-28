import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.append(os.getcwd())

from gui.api import PulseAPI
from gui.backup import BackupManager


def test_gui_backup_creates_user_authored_backup():
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
        assert (backup_dir / ".env").exists()
        assert not (backup_dir / "demo.db").exists()
        assert not (backup_dir / "status.json").exists()

        metadata = json.loads((backup_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["persona"] == "demo"
        assert metadata["reason"] == "pre-edit"
        assert metadata["files"] == ["config.yaml", ".env", "persona.yaml"]
        assert metadata["source_paths"]["config.yaml"] == "personas/demo/config.yaml"
        assert metadata["source_paths"][".env"] == "personas/demo/.env"

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


def test_gui_backup_restore_creates_safety_backup_and_restores_files():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        persona_dir = root / "personas" / "demo"
        persona_dir.mkdir(parents=True)
        config = persona_dir / "config.yaml"
        identity = persona_dir / "persona.yaml"
        config.write_text("tts:\n  voice_description: old\n", encoding="utf-8")
        identity.write_text("name: Demo\nmodel: Old\n", encoding="utf-8")

        api = PulseAPI(root)
        backup = api.create_backup("demo", reason="manual")["backup"]
        config.write_text("tts:\n  voice_description: new\n", encoding="utf-8")
        identity.write_text("name: Demo\nmodel: New\n", encoding="utf-8")

        restored = api.restore_backup("demo", backup["path"])
        assert restored["ok"] is True
        assert restored["changed"] is True
        assert restored["safety_backup"]["reason"] == "pre-restore"
        assert config.read_text(encoding="utf-8") == "tts:\n  voice_description: old\n"
        assert identity.read_text(encoding="utf-8") == "name: Demo\nmodel: Old\n"
        assert len(api.list_backups("demo")["backups"]) == 2


def test_gui_backup_restore_rejects_path_traversal():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        persona_dir = root / "personas" / "demo"
        persona_dir.mkdir(parents=True)
        (persona_dir / "config.yaml").write_text("tts: {}\n", encoding="utf-8")

        api = PulseAPI(root)
        result = api.restore_backup("demo", "../other/20260522_120000")
        assert result["ok"] is False


def test_gui_backup_restore_rejects_nested_destination_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        persona_dir = root / "personas" / "demo"
        persona_dir.mkdir(parents=True)
        (persona_dir / "config.yaml").write_text("tts: {}\n", encoding="utf-8")

        api = PulseAPI(root)
        backup = api.create_backup("demo", reason="manual")["backup"]
        metadata_path = root / backup["path"] / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["source_paths"]["config.yaml"] = "personas/demo/data/config.yaml"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        result = api.restore_backup("demo", backup["path"])
        assert result["ok"] is False
        assert "top-level persona file" in result["error"]


if __name__ == "__main__":
    test_gui_backup_creates_user_authored_backup()
    test_gui_backup_prunes_per_persona()
    test_gui_backup_restore_creates_safety_backup_and_restores_files()
    test_gui_backup_restore_rejects_path_traversal()
    test_gui_backup_restore_rejects_nested_destination_metadata()
    print("[OK] GUI backup")
