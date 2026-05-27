import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.append(os.getcwd())

from gui.api import PulseAPI
from gui.config_editor import (
    _assert_no_duplicate_keys,
    _format_field,
    _set_deep_nested_field,
    _set_nested_field,
    _set_top_level_field,
)


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
server:
  llama_cpp_dir: "C:\\llama-cpp"
  models_dir: "C:\\llama-cpp\\models"
  host: "127.0.0.1"
  port: 8012
  gpu_layers: -1
  flash_attention: true
  parallel: 1

provider:
  type: "openrouter"
  model: "old/provider"
  max_context: 131072

model:
  model_file: "old.gguf"
  mmproj_file: ""
  max_context: 16384
  temperature: 0.7
  max_response_tokens: 2048
  frequency_penalty: 0.3
  presence_penalty: 0.2
  top_p: 1.0
  reasoning: true
  reasoning_effort: "high"
  show_reasoning: false
  max_tool_rounds: 8

context_budget:
  recent_tail_exchanges: 4

tts:
  voice_description: "Old voice"
  voice_sample: ""
  voice_sample_text: ""

skills:
  tts:
    enabled: true
    voice: "warm"
  lor:
    enabled: true
  lantern:
    notes: "existing config"

heartbeat:
  interval_minutes: 30
  randomize: true
  interval_min_minutes: 30
  interval_max_minutes: 60
  quiet_hours_start: 23
  quiet_hours_end: 8
  debug: true

channels:
  lor:
    enabled: true
    author_name: "Demo"
  toast:
    app_name: "Demo"
  telegram:
    enabled: true
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
            "provider": {
                "type": "openai",
                "model": "gpt-5.1",
                "max_context": 262144,
            },
            "server": {
                "llama_cpp_dir": "D:/llama.cpp",
                "models_dir": "D:/models",
                "host": "127.0.0.1",
                "port": 8013,
                "gpu_layers": 35,
                "flash_attention": False,
                "parallel": 2,
            },
            "model": {
                "model_file": "new-model.gguf",
                "mmproj_file": "mmproj.gguf",
                "max_context": 32768,
                "temperature": 0.85,
                "max_response_tokens": 4096,
                "frequency_penalty": 0.35,
                "presence_penalty": 0.15,
                "top_p": 0.95,
                "reasoning": False,
                "reasoning_effort": "medium",
                "show_reasoning": True,
                "max_tool_rounds": 12,
            },
            "context_budget": {
                "recent_tail_exchanges": 6,
            },
            "heartbeat": {
                "interval_minutes": 45,
                "randomize": False,
                "interval_min_minutes": 20,
                "interval_max_minutes": 50,
                "quiet_hours_start": 22,
                "quiet_hours_end": 7,
                "debug": False,
            },
            "channels": {
                "telegram": False,
                "toast": False,
            },
            "skills": {
                "tts": False,
                "lantern": False,
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
        assert 'type: "openai"' in config_text
        assert 'model: "gpt-5.1"' in config_text
        assert "max_context: 262144" in config_text
        assert 'llama_cpp_dir: "D:/llama.cpp"' in config_text
        assert 'models_dir: "D:/models"' in config_text
        assert 'host: "127.0.0.1"' in config_text
        assert "port: 8013" in config_text
        assert "gpu_layers: 35" in config_text
        assert "flash_attention: false" in config_text
        assert "parallel: 2" in config_text
        assert 'model_file: "new-model.gguf"' in config_text
        assert 'mmproj_file: "mmproj.gguf"' in config_text
        assert "max_context: 32768" in config_text
        assert "temperature: 0.85" in config_text
        assert "max_response_tokens: 4096" in config_text
        assert "frequency_penalty: 0.35" in config_text
        assert "presence_penalty: 0.15" in config_text
        assert "top_p: 0.95" in config_text
        assert "reasoning: false" in config_text
        assert 'reasoning_effort: "medium"' in config_text
        assert "show_reasoning: true" in config_text
        assert "max_tool_rounds: 12" in config_text
        assert "recent_tail_exchanges: 6" in config_text
        assert "interval_minutes: 45" in config_text
        assert "randomize: false" in config_text
        assert "interval_min_minutes: 20" in config_text
        assert "interval_max_minutes: 50" in config_text
        assert "quiet_hours_start: 22" in config_text
        assert "quiet_hours_end: 7" in config_text
        assert "debug: false" in config_text
        assert 'interval_minutes: "45"' not in config_text
        assert "telegram:\n    enabled: false" in config_text
        assert "toast:\n    app_name: \"Demo\"\n    enabled: false" in config_text
        assert 'enabled: "false"' not in config_text
        assert "tts:\n    enabled: false\n    voice: \"warm\"" in config_text
        assert "lantern:" in config_text
        assert "notes: \"existing config\"" in config_text
        assert "enabled: false" in config_text

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
        (persona_dir / "config.yaml").write_text("tts: {}\nskills:\n  tts:\n    enabled: true\n", encoding="utf-8")

        api = PulseAPI(root)
        result = api.preview_persona_save("demo", {
            "identity": {"system_prompt": "Nope."},
        })
        assert result["ok"] is False
        assert "Unsupported field" in result["error"]

        result = api.preview_persona_save("demo", {
            "heartbeat": {"quiet_hours_start": 24},
        })
        assert result["ok"] is False
        assert "between 0 and 23" in result["error"]

        result = api.preview_persona_save("demo", {
            "heartbeat": {"randomize": "true"},
        })
        assert result["ok"] is False
        assert "true or false" in result["error"]

        result = api.preview_persona_save("demo", {
            "channels": {"email": True},
        })
        assert result["ok"] is False
        assert "Unsupported field" in result["error"]

        result = api.preview_persona_save("demo", {
            "channels": {"telegram": "false"},
        })
        assert result["ok"] is False
        assert "true or false" in result["error"]

        result = api.preview_persona_save("demo", {
            "skills": {"definitely_not_a_skill": True},
        })
        assert result["ok"] is False
        assert "Unsupported field" in result["error"]

        result = api.preview_persona_save("demo", {
            "skills": {"tts": "false"},
        })
        assert result["ok"] is False
        assert "true or false" in result["error"]

        result = api.preview_persona_save("demo", {
            "provider": {"type": "madeup"},
        })
        assert result["ok"] is False
        assert "Provider Type" in result["error"]

        for provider_type in ("local", "openrouter", "openai", "anthropic", "custom"):
            result = api.preview_persona_save("demo", {
                "provider": {"type": provider_type},
            })
            assert result["ok"] is True

        result = api.preview_persona_save("demo", {
            "model": {"temperature": -0.1},
        })
        assert result["ok"] is False
        assert "Temperature" in result["error"]

        result = api.preview_persona_save("demo", {
            "model": {"top_p": 1.5},
        })
        assert result["ok"] is False
        assert "Top P" in result["error"]

        result = api.preview_persona_save("demo", {
            "model": {"frequency_penalty": "0.5"},
        })
        assert result["ok"] is False
        assert "Frequency Penalty" in result["error"]

        result = api.preview_persona_save("demo", {
            "model": {"reasoning_effort": "extreme"},
        })
        assert result["ok"] is False
        assert "Reasoning Effort" in result["error"]

        for effort in ("", "low", "medium", "high"):
            result = api.preview_persona_save("demo", {
                "model": {"reasoning_effort": effort},
            })
            assert result["ok"] is True

        result = api.preview_persona_save("demo", {
            "server": {"port": 80},
        })
        assert result["ok"] is False
        assert "Server Port" in result["error"]

        result = api.preview_persona_save("demo", {
            "server": {"port": 70000},
        })
        assert result["ok"] is False
        assert "Server Port" in result["error"]

        result = api.preview_persona_save("demo", {
            "server": {"port": 8012},
        })
        assert result["ok"] is True

        result = api.preview_persona_save("demo", {
            "server": {"gpu_layers": -2},
        })
        assert result["ok"] is False
        assert "GPU Layers" in result["error"]

        for layers in (-1, 99):
            result = api.preview_persona_save("demo", {
                "server": {"gpu_layers": layers},
            })
            assert result["ok"] is True

        result = api.preview_persona_save("demo", {
            "server": {"parallel": 9},
        })
        assert result["ok"] is False
        assert "Parallel Slots" in result["error"]

        result = api.preview_persona_save("demo", {
            "server": {"flash_attention": "false"},
        })
        assert result["ok"] is False
        assert "Flash Attention" in result["error"]

        result = api.preview_persona_save("demo", {
            "server": {"flash_attention": False},
        })
        assert result["ok"] is True

        result = api.preview_persona_save("demo", {
            "model": {"model_file": "my-model.gguf"},
        })
        assert result["ok"] is True

        result = api.preview_persona_save("demo", {
            "model": {"model_file": "bad\x00model.gguf"},
        })
        assert result["ok"] is False
        assert "null bytes" in result["error"]


def test_writer_sets_new_top_level_field():
    assert _set_top_level_field("", "model", "Grok") == 'model: "Grok"\n'


def test_writer_updates_existing_top_level_field():
    text = 'name: Demo\nmodel: "Old"\nuser_name: Lena\n'
    updated = _set_top_level_field(text, "model", "New")
    assert updated == 'name: Demo\nmodel: "New"\nuser_name: Lena\n'


def test_writer_sets_nested_field_parent_exists():
    text = 'tts:\n  voice_description: "Old"\nmodel:\n  reasoning: true\n'
    updated = _set_nested_field(text, "tts", "voice_description", "New")
    assert 'tts:\n  voice_description: "New"\nmodel:' in updated


def test_writer_sets_nested_field_parent_missing():
    updated = _set_nested_field('provider:\n  type: "openrouter"\n', "tts", "voice_description", "Warm")
    assert updated.endswith('\ntts:\n  voice_description: "Warm"\n')


def test_writer_sets_deep_nested_field_parent_exists():
    text = 'channels:\n  telegram:\n    enabled: true\n  lor:\n    enabled: true\n'
    updated = _set_deep_nested_field(text, ["channels", "telegram", "enabled"], False)
    assert "telegram:\n    enabled: false" in updated
    assert "lor:\n    enabled: true" in updated


def test_writer_sets_deep_nested_field_subblock_missing():
    text = 'channels:\n  toast:\n    app_name: "Demo"\n'
    updated = _set_deep_nested_field(text, ["channels", "toast", "enabled"], False)
    assert 'toast:\n    app_name: "Demo"\n    enabled: false' in updated


def test_writer_sets_deep_nested_field_all_blocks_missing():
    updated = _set_deep_nested_field('provider:\n  type: "openrouter"\n', ["channels", "telegram", "enabled"], True)
    assert updated.endswith('\nchannels:\n  telegram:\n    enabled: true\n')


def test_writer_deep_nested_field_preserves_siblings():
    text = 'channels:\n  lor:\n    author_name: "Nova"\n    model_name: "sonnet"\n'
    updated = _set_deep_nested_field(text, ["channels", "lor", "enabled"], True)
    assert 'author_name: "Nova"' in updated
    assert 'model_name: "sonnet"' in updated
    assert "enabled: true" in updated


def test_writer_sets_skill_enabled_without_touching_config():
    text = 'skills:\n  tts:\n    enabled: true\n    voice: "warm"\n'
    updated = _set_deep_nested_field(text, ["skills", "tts", "enabled"], False)
    assert 'tts:\n    enabled: false\n    voice: "warm"' in updated


def test_writer_multiline_value():
    updated = _set_top_level_field("name: Demo\n", "voice_notes", "Line one.\nLine two.")
    assert "voice_notes: |\n  Line one.\n  Line two.\n" in updated


def test_writer_preserves_folded_block_style():
    text = "voice_notes: >\n  Old line.\nname: Demo\n"
    updated = _set_top_level_field(text, "voice_notes", "New line.\nStill folded.")
    assert updated.startswith("voice_notes: >\n  New line.\n  Still folded.\n")


def test_writer_ignores_commented_out_keys():
    text = 'provider:\n  #model: "old/commented"\n  model: "real"\n'
    updated = _set_nested_field(text, "provider", "model", "new")
    assert '  #model: "old/commented"' in updated
    assert '  model: "new"' in updated
    assert updated.count("model:") == 2


def test_writer_preserves_commented_out_sibling_keys():
    text = """provider:
  #type: "openrouter"
  type: "openai"
  #api_key_env: "OPENROUTER_API_KEY"
  api_key_env: "OPENAI_API_KEY"
"""
    updated = _set_nested_field(text, "provider", "type", "local")
    assert '#type: "openrouter"' in updated
    assert '#api_key_env: "OPENROUTER_API_KEY"' in updated
    assert 'type: "local"' in updated
    assert 'type: "openai"' not in updated


def test_writer_preserves_four_space_indent():
    text = 'tts:\n    voice_description: "Old"\n    voice_sample: ""\n'
    updated = _set_nested_field(text, "tts", "voice_description", "New")
    assert '    voice_description: "New"' in updated
    assert '\n  voice_description' not in updated


def test_writer_handles_comments_with_different_indent():
    text = 'tts:\n    # design voice\n    voice_description: "Old"\n  # odd comment\nmodel:\n  reasoning: true\n'
    updated = _set_nested_field(text, "tts", "voice_sample", "ref.ogg")
    assert '    voice_sample: "ref.ogg"' in updated
    assert "model:\n  reasoning: true" in updated


def test_writer_inserts_before_top_level_divider_comment():
    text = 'model:\n  model_file: "main.gguf"\n\n# --- Paths ---\npaths:\n  database: demo.db\n'
    updated = _set_nested_field(text, "model", "mmproj_file", "mmproj.gguf")
    assert 'model_file: "main.gguf"\n  mmproj_file: "mmproj.gguf"\n\n# --- Paths ---' in updated


def test_writer_rejects_duplicate_keys():
    try:
        _assert_no_duplicate_keys('tts:\n  voice_description: "a"\n  voice_description: "b"\n')
    except ValueError as e:
        assert "Duplicate key 'voice_description'" in str(e)
    else:
        raise AssertionError("Expected duplicate key rejection")


def test_writer_empty_string_value():
    assert _format_field("voice_sample", "", 2) == '  voice_sample: ""'


def test_writer_formats_numbers_and_booleans():
    assert _format_field("interval_minutes", 30, 2) == "  interval_minutes: 30"
    assert _format_field("randomize", True, 2) == "  randomize: true"
    assert _format_field("randomize", False, 2) == "  randomize: false"


def test_writer_formats_floats():
    assert _format_field("temperature", 0.85, 2) == "  temperature: 0.85"
    assert _format_field("top_p", 1.0, 2) == "  top_p: 1.0"
    assert _format_field("frequency_penalty", 0.3, 2) == "  frequency_penalty: 0.3"
    assert '"0.85"' not in _format_field("temperature", 0.85, 2)


def test_writer_special_chars_json_encoded():
    updated = _set_top_level_field("", "model", 'A "quoted": value # not comment')
    assert updated == 'model: "A \\"quoted\\": value # not comment"\n'


def test_writer_trailing_newline_stable():
    once = _set_top_level_field('model: "A"\n', "model", "B")
    twice = _set_top_level_field(once, "model", "B")
    assert once == twice


def test_writer_same_value_round_trip_no_diff():
    text = 'model: "Grok"\nvoice_notes: |\n  Hi\n'
    updated = _set_top_level_field(text, "model", "Grok")
    assert updated == text


def test_preview_diff_handles_missing_final_newline():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        persona_dir = root / "personas" / "demo"
        persona_dir.mkdir(parents=True)
        (persona_dir / "persona.yaml").write_text("name: Demo\n", encoding="utf-8")
        (persona_dir / "config.yaml").write_text('lor_identity:\n  nickname: "Demo"', encoding="utf-8")

        api = PulseAPI(root)
        result = api.preview_persona_save("demo", {
            "server": {"models_dir": "D:/models"},
        })
        assert result["ok"] is True
        diff = result["preview"]["diff"]
        assert '-  nickname: "Demo"+  nickname: "Demo"' not in diff
        assert '+server:' in diff


if __name__ == "__main__":
    test_config_editor_preview_and_save()
    test_config_editor_rejects_unknown_fields()
    test_writer_sets_new_top_level_field()
    test_writer_updates_existing_top_level_field()
    test_writer_sets_nested_field_parent_exists()
    test_writer_sets_nested_field_parent_missing()
    test_writer_sets_deep_nested_field_parent_exists()
    test_writer_sets_deep_nested_field_subblock_missing()
    test_writer_sets_deep_nested_field_all_blocks_missing()
    test_writer_deep_nested_field_preserves_siblings()
    test_writer_sets_skill_enabled_without_touching_config()
    test_writer_multiline_value()
    test_writer_preserves_folded_block_style()
    test_writer_ignores_commented_out_keys()
    test_writer_preserves_commented_out_sibling_keys()
    test_writer_preserves_four_space_indent()
    test_writer_handles_comments_with_different_indent()
    test_writer_inserts_before_top_level_divider_comment()
    test_writer_rejects_duplicate_keys()
    test_writer_empty_string_value()
    test_writer_formats_numbers_and_booleans()
    test_writer_formats_floats()
    test_writer_special_chars_json_encoded()
    test_writer_trailing_newline_stable()
    test_writer_same_value_round_trip_no_diff()
    test_preview_diff_handles_missing_final_newline()
    print("[OK] GUI config editor")
