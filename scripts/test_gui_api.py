import os
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


if __name__ == "__main__":
    test_gui_api_read_only_persona_loading()
    test_gui_api_standard_provider_env_fallback()
    print("[OK] GUI API")
