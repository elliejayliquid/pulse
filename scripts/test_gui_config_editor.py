import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.append(os.getcwd())

from gui.api import PulseAPI


def test_config_editor_preview_and_save():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        persona_dir = root / "personas" / "demo"
        persona_dir.mkdir(parents=True)

        (persona_dir / "persona.yaml").write_text(
            """name: Demo
user_name: Lena
model: "Old Model"

system_prompt: |
  Keep this block.

voice_notes: |
  Old voice notes.
""",
            encoding="utf-8",
        )
        (persona_dir / "config.yaml").write_text(
            """# Persona config overlay
provider:
  type: "openrouter"

tts:
  voice_description: "Old voice"
  voice_sample: ""
  voice_sample_text: ""
""",
            encoding="utf-8",
        )
        (persona_dir / "data").mkdir()
        (persona_dir / "data" / "status.json").write_text("{}", encoding="utf-8")

        api = PulseAPI(root)
        changes = {
            "identity": {
                "model": "New Model",
                "voice_notes": "Line one.\nLine two.",
            },
            "tts": {
                "voice_description": "New voice",
                "voice_sample": "personas/demo/data/tts/ref.ogg",
                "voice_sample_text": "Reference text",
            },
        }
        preview = api.preview_persona_save("demo", changes)
        assert preview["ok"] is True
        assert preview["preview"]["has_changes"] is True
        assert "Old Model" in preview["preview"]["diff"]
        assert "New Model" in preview["preview"]["diff"]

        result = api.save_persona("demo", changes)
        assert result["ok"] is True
        assert result["changed"] is True
        assert result["backup"]

        persona_text = (persona_dir / "persona.yaml").read_text(encoding="utf-8")
        assert 'model: "New Model"' in persona_text
        assert "system_prompt: |" in persona_text
        assert "  Keep this block." in persona_text
        assert "voice_notes: |" in persona_text
        assert "  Line one." in persona_text

        config_text = (persona_dir / "config.yaml").read_text(encoding="utf-8")
        assert "# Persona config overlay" in config_text
        assert 'voice_description: "New voice"' in config_text
        assert 'voice_sample: "personas/demo/data/tts/ref.ogg"' in config_text
        assert 'voice_sample_text: "Reference text"' in config_text

        backup_path = root / result["backup"]["path"]
        assert (backup_path / "config.yaml").exists()
        assert (backup_path / "persona.yaml").exists()
        assert not (backup_path / "status.json").exists()
        metadata = json.loads((backup_path / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["reason"] == "pre-edit"


def test_config_editor_rejects_unknown_fields():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        persona_dir = root / "personas" / "demo"
        persona_dir.mkdir(parents=True)
        (persona_dir / "persona.yaml").write_text("name: Demo\n", encoding="utf-8")
        (persona_dir / "config.yaml").write_text("tts: {}\n", encoding="utf-8")

        api = PulseAPI(root)
        result = api.preview_persona_save("demo", {
            "identity": {"system_prompt": "Nope."},
        })
        assert result["ok"] is False
        assert "Unsupported field" in result["error"]


if __name__ == "__main__":
    test_config_editor_preview_and_save()
    test_config_editor_rejects_unknown_fields()
    print("[OK] GUI config editor")
