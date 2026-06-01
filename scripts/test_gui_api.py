import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.append(os.getcwd())

from gui.api import PulseAPI


def test_gui_api_read_only_persona_loading():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "personas" / "demo").mkdir(parents=True)
        (root / "logs").mkdir()

        (root / "config.yaml").write_text(
            """
provider:
  type: local
model:
  max_context: 16384
  max_response_tokens: 1024
skills:
  memory:
    enabled: true
channels:
  telegram:
    enabled: false
  toast:
    app_name: Demo
""",
            encoding="utf-8",
        )
        (root / "persona.yaml").write_text(
            "name: Base\nuser_name: User\nmodel: Base Model\n",
            encoding="utf-8",
        )
        (root / "personas" / "demo" / "config.yaml").write_text(
            """
provider:
  type: openrouter
  model: demo/model
  api_key_env: DEMO_KEY
  max_context: 32768
skills:
  tts:
    enabled: true
""",
            encoding="utf-8",
        )
        (root / "personas" / "demo" / "persona.yaml").write_text(
            """
name: Demo
user_name: Lena
model: Demo Model
system_prompt: Hello.
traits:
  - careful
voice_notes: Warm.
""",
            encoding="utf-8",
        )
        (root / "personas" / "demo" / ".env").write_text(
            "DEMO_KEY=secret\n",
            encoding="utf-8",
        )

        api = PulseAPI(root)
        personas = api.list_personas()
        assert [p["name"] for p in personas] == ["__base__", "demo"]

        demo = api.load_persona("demo")
        assert demo["display_name"] == "Demo"
        assert demo["summary"]["provider_type"] == "openrouter"
        assert demo["summary"]["provider_model"] == "demo/model"
        assert demo["summary"]["max_context"] == 32768
        assert demo["summary"]["provider_max_context"] == 32768
        assert demo["summary"]["local_max_context"] == 16384
        assert demo["summary"]["server"]["host"] == "127.0.0.1"
        assert demo["summary"]["server"]["port"] == 8012
        assert demo["summary"]["model_file"] == ""
        assert demo["summary"]["mmproj_file"] == ""
        assert demo["key_status"]["provider_type"] == "openrouter"
        assert demo["key_status"]["api_key_env"] == "DEMO_KEY"
        assert demo["key_status"]["expected_api_key_env"] == "OPENROUTER_API_KEY"
        assert demo["key_status"]["api_key_set"] is True
        assert demo["key_status"]["provider_key_status"]["openrouter"] is False
        assert demo["key_status"]["telegram_set"] is False
        assert demo["key_status"]["telegram_enabled"] is False
        channels = {channel["name"]: channel["enabled"] for channel in demo["channels"]}
        assert channels["telegram"] is False
        assert channels["toast"] is True

        preview_result = api.preview_persona_save("demo", {
            "identity": {"model": "Demo Model 2"},
            "tts": {"voice_description": "Warm demo voice."},
        })
        assert preview_result["ok"] is True
        assert preview_result["preview"]["has_changes"] is True

        stop_result = api.stop_pulse("demo")
        assert stop_result["ok"] is True
        assert (root / "personas" / "demo" / "data" / "shutdown_requested").exists()

        missing_stop = api.stop_pulse("missing")
        assert missing_stop["ok"] is False

        status_path = root / "personas" / "demo" / "data" / "status.json"
        status_path.write_text(
            """{
  "persona": "demo",
  "phase": "running",
  "running": true,
  "updated_at": "%s"
}""" % (datetime.now(timezone.utc) - timedelta(seconds=130)).isoformat(),
            encoding="utf-8",
        )
        assert api.get_status("demo")["stale"] is True


def test_gui_api_standard_provider_env_fallback():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "personas" / "demo").mkdir(parents=True)
        (root / "logs").mkdir()

        (root / "config.yaml").write_text(
            """
provider:
  type: openai
channels:
  telegram:
    enabled: true
""",
            encoding="utf-8",
        )
        (root / "persona.yaml").write_text("name: Base\n", encoding="utf-8")
        (root / ".env").write_text(
            "OPENAI_API_KEY=secret\nOPENROUTER_API_KEY=other\nTELEGRAM_BOT_TOKEN=token\n",
            encoding="utf-8",
        )
        (root / "personas" / "demo" / "config.yaml").write_text("", encoding="utf-8")
        (root / "personas" / "demo" / "persona.yaml").write_text("name: Demo\n", encoding="utf-8")

        api = PulseAPI(root)
        status = api.load_persona("demo")["key_status"]
        assert status["provider_type"] == "openai"
        assert status["api_key_env"] == ""
        assert status["expected_api_key_env"] == "OPENAI_API_KEY"
        assert status["api_key_set"] is True
        assert status["provider_key_status"]["openai"] is True
        assert status["provider_key_status"]["openrouter"] is True
        assert status["provider_key_status"]["anthropic"] is False
        assert status["telegram_enabled"] is True
        assert status["telegram_set"] is True


def test_gui_api_get_lantern_read_only():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        persona_dir = root / "personas" / "demo"
        data_dir = persona_dir / "data"
        data_dir.mkdir(parents=True)
        (root / "logs").mkdir()
        (root / "config.yaml").write_text("", encoding="utf-8")
        (root / "persona.yaml").write_text("name: Base\n", encoding="utf-8")
        (persona_dir / "config.yaml").write_text("", encoding="utf-8")
        (persona_dir / "persona.yaml").write_text("name: Demo\n", encoding="utf-8")

        api = PulseAPI(root)
        missing = api.get_lantern("demo")
        assert missing["ok"] is True
        assert missing["exists"] is False

        db_path = data_dir / "demo.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                """
                CREATE TABLE resident_lanterns (
                    resident_id TEXT PRIMARY KEY,
                    mode TEXT,
                    mood TEXT,
                    focus TEXT,
                    note TEXT,
                    open_thread TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "INSERT INTO resident_lanterns "
                "(resident_id, mode, mood, focus, note, open_thread, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "demo",
                    "quiet company",
                    "settled",
                    "continuity UI",
                    "testing lantern",
                    "wire read-only view",
                    (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

        lantern = api.get_lantern("demo")
        assert lantern["ok"] is True
        assert lantern["exists"] is True
        assert lantern["state"] == "stale"
        assert lantern["stale"] is True
        assert lantern["expired"] is False
        assert lantern["fields"]["mode"] == "quiet company"
        assert lantern["fields"]["open_thread"] == "wire read-only view"


def test_gui_api_model_file_path_validation():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        models_dir = root / "models"
        nested_dir = models_dir / "vision"
        outside_dir = root / "elsewhere"
        nested_dir.mkdir(parents=True)
        outside_dir.mkdir()
        model_file = models_dir / "main.gguf"
        projector_file = nested_dir / "mmproj.gguf"
        outside_file = outside_dir / "main.gguf"
        wrong_file = models_dir / "notes.txt"
        model_file.write_text("", encoding="utf-8")
        projector_file.write_text("", encoding="utf-8")
        outside_file.write_text("", encoding="utf-8")
        wrong_file.write_text("", encoding="utf-8")

        api = PulseAPI(root)
        assert api._relative_model_file(str(models_dir), str(model_file)) == "main.gguf"
        assert api._relative_model_file(str(models_dir), str(projector_file)) == "vision/mmproj.gguf"

        for chosen in (outside_file, wrong_file):
            try:
                api._relative_model_file(str(models_dir), str(chosen))
                assert False, "Expected invalid model file path to be rejected"
            except ValueError:
                pass


def test_gui_api_secrets_preserve_env_and_validate_keys():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        persona_dir = root / "personas" / "demo"
        persona_dir.mkdir(parents=True)
        (root / "logs").mkdir()

        (root / "config.yaml").write_text(
            """
provider:
  type: openrouter
channels:
  telegram:
    enabled: true
""",
            encoding="utf-8",
        )
        (root / ".env").write_text(
            "OPENROUTER_API_KEY=root-openrouter\nTELEGRAM_BOT_TOKEN=root-telegram\n",
            encoding="utf-8",
        )
        (persona_dir / "config.yaml").write_text("", encoding="utf-8")
        (persona_dir / "persona.yaml").write_text("name: Demo\n", encoding="utf-8")
        (persona_dir / ".env").write_text(
            "# demo secrets\nOTHER_KEY=keep-me\nTELEGRAM_BOT_TOKEN=persona-telegram\n",
            encoding="utf-8",
        )

        api = PulseAPI(root)
        secrets = api.get_secrets("demo")
        by_key = {item["key"]: item for item in secrets["secrets"]}
        assert by_key["OPENROUTER_API_KEY"]["source"] == "inherited"
        assert by_key["OPENROUTER_API_KEY"]["masked"].endswith("uter")
        assert by_key["TELEGRAM_BOT_TOKEN"]["source"] == "persona"
        assert by_key["TELEGRAM_BOT_TOKEN"]["masked"].endswith("gram")
        assert "root-openrouter" not in str(secrets)
        assert api.reveal_secret("demo", "OPENROUTER_API_KEY")["value"] == "root-openrouter"

        rejected = api.save_secrets("demo", {"NOT_ALLOWED": "nope"})
        assert rejected["ok"] is False

        saved = api.save_secrets("demo", {
            "OPENROUTER_API_KEY": "persona-openrouter",
            "TELEGRAM_BOT_TOKEN": None,
        })
        assert saved["ok"] is True
        assert saved["changed"] is True

        env_text = (persona_dir / ".env").read_text(encoding="utf-8")
        assert "# demo secrets" in env_text
        assert "OTHER_KEY=keep-me" in env_text
        assert "OPENROUTER_API_KEY=persona-openrouter" in env_text
        assert "TELEGRAM_BOT_TOKEN" not in env_text

        status = api.get_key_status("demo")
        assert status["api_key_source"] == "persona"
        assert status["telegram_source"] == "inherited"

        backup = api.list_backups("demo")["backups"][0]
        assert ".env" in backup["files"]


if __name__ == "__main__":
    test_gui_api_read_only_persona_loading()
    test_gui_api_standard_provider_env_fallback()
    test_gui_api_get_lantern_read_only()
    test_gui_api_model_file_path_validation()
    test_gui_api_secrets_preserve_env_and_validate_keys()
    print("[OK] GUI API")
