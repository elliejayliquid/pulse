import os
import json
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


def test_gui_api_lantern_update_and_dim_write_safely():
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
        updated = api.set_lantern("demo", {
            "mode": "focused",
            "mood": "curious",
            "focus": "safe GUI writes",
            "open_thread": "lantern update",
            "note": "first GUI write",
        })
        assert updated["ok"] is True
        lantern = api.get_lantern("demo")
        assert lantern["exists"] is True
        assert lantern["fields"]["mode"] == "focused"
        assert lantern["fields"]["note"] == "first GUI write"

        dimmed = api.dim_lantern("demo", "Resting after validation.")
        assert dimmed["ok"] is True
        lantern = api.get_lantern("demo")
        assert lantern["fields"]["mode"] == ""
        assert lantern["fields"]["focus"] == ""
        assert lantern["fields"]["open_thread"] == ""
        assert lantern["fields"]["note"] == "Resting after validation."

        backups = list((root / "gui_data" / "db_backups" / "demo").glob("*/*.json"))
        assert len(backups) == 2
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in backups]
        assert {payload["action"] for payload in payloads} == {"lantern-update", "lantern-dim"}
        dim_backup = next(payload for payload in payloads if payload["action"] == "lantern-dim")
        assert dim_backup["before"]["mode"] == "focused"

        unchanged_dim = api.dim_lantern("demo", "Resting after validation.")
        assert unchanged_dim["ok"] is True
        assert unchanged_dim["changed"] is False
        backups_after_noop_dim = list((root / "gui_data" / "db_backups" / "demo").glob("*/*.json"))
        assert len(backups_after_noop_dim) == 2

        unchanged_update = api.set_lantern("demo", {
            "mode": "",
            "mood": "",
            "focus": "",
            "open_thread": "",
            "note": "Resting after validation.",
        })
        assert unchanged_update["ok"] is True
        assert unchanged_update["changed"] is False
        backups_after_noop_update = list((root / "gui_data" / "db_backups" / "demo").glob("*/*.json"))
        assert len(backups_after_noop_update) == 2

        rejected = api.set_lantern("demo", {"mode": "\x00"})
        assert rejected["ok"] is False


def test_gui_api_list_memories_read_only_with_history_views():
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

        db_path = data_dir / "demo.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                """
                CREATE TABLE memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '[]',
                    type TEXT NOT NULL DEFAULT 'fact',
                    importance INTEGER NOT NULL DEFAULT 5,
                    retrieval_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed TEXT,
                    supersedes INTEGER,
                    journal_file TEXT,
                    date TEXT NOT NULL,
                    status TEXT,
                    confidence TEXT,
                    source TEXT,
                    last_confirmed TEXT,
                    time_sensitive INTEGER,
                    valid_until TEXT,
                    embedding BLOB
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE journal_entries (
                    id TEXT PRIMARY KEY,
                    author TEXT NOT NULL DEFAULT 'Pulse',
                    title TEXT,
                    entry_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    why_it_mattered TEXT,
                    tags TEXT NOT NULL DEFAULT '[]',
                    importance INTEGER NOT NULL DEFAULT 5,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    resolved INTEGER,
                    date TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "INSERT INTO memories "
                "(id, text, tags, type, importance, date, status, confidence, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (1, "Lena likes blue", json.dumps(["preference"]), "fact", 5,
                 "2026-01-01T00:00:00+00:00", "superseded", "medium", "model_extracted"),
            )
            conn.execute(
                "INSERT INTO memories "
                "(id, text, tags, type, importance, date, status, confidence, source, supersedes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (2, "Lena likes purple", json.dumps(["preference"]), "fact", 6,
                 "2026-02-01T00:00:00+00:00", "current", "high", "user_defined", 1),
            )
            conn.execute(
                "INSERT INTO memories "
                "(id, text, tags, type, importance, date, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (3, "Old archived memory", "[]", "fact", 1,
                 "2026-03-01T00:00:00+00:00", "archived"),
            )
            conn.execute(
                "INSERT INTO memories "
                "(id, text, tags, type, importance, date, status, journal_file) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (4, "Journal preview", json.dumps(["journal"]), "journal", 5,
                 "2026-04-01T00:00:00+00:00", "current", "entries/001.md"),
            )
            conn.execute(
                "INSERT INTO journal_entries "
                "(id, title, entry_type, content, why_it_mattered, tags, date) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "001",
                    "Full journal entry",
                    "reflection",
                    "This is the full journal entry, not the companion memory preview.",
                    "It gives the detail view the whole entry.",
                    "[]",
                    "2026-04-01T00:00:00+00:00",
                ),
            )
            conn.execute(
                "INSERT INTO memories "
                "(id, text, tags, type, importance, date, status, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (5, "Session summary", "[]", "session_log", 4,
                 "2026-05-01T00:00:00+00:00", "historical", "system"),
            )
            conn.commit()
        finally:
            conn.close()

        api = PulseAPI(root)
        active = api.list_memories("demo", "active", page=1, page_size=25)
        assert active["ok"] is True
        assert active["total"] == 1
        assert active["items"][0]["id"] == 2
        assert active["items"][0]["has_history"] is True
        assert active["items"][0]["version_count"] == 2

        all_active = api.list_memories("demo", "active", "all", page=1, page_size=25)
        assert all_active["ok"] is True
        assert all_active["total"] == 3

        journal = api.list_memories("demo", "active", "journal", page=1, page_size=25)
        assert journal["ok"] is True
        assert journal["total"] == 1
        assert journal["items"][0]["id"] == 4

        journal_detail = api.get_memory_detail("demo", 4)
        assert journal_detail["ok"] is True
        assert "Full journal entry" in journal_detail["versions"][0]["text"]
        assert "not the companion memory preview" in journal_detail["versions"][0]["text"]
        assert journal_detail["versions"][0]["detail_source"] == "journal_entry"

        session_logs = api.list_memories("demo", "active", "session_log", page=1, page_size=25)
        assert session_logs["ok"] is True
        assert session_logs["total"] == 1
        assert session_logs["items"][0]["id"] == 5

        archived = api.list_memories("demo", "archived", page=1, page_size=25)
        assert archived["ok"] is True
        assert archived["total"] == 1
        assert archived["items"][0]["id"] == 3

        detail = api.get_memory_detail("demo", 1)
        assert detail["ok"] is True
        assert detail["current_id"] == 2
        assert [item["id"] for item in detail["versions"]] == [2, 1]


def test_gui_api_list_journal_entries_read_only_views():
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

        db_path = data_dir / "demo.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                """
                CREATE TABLE journal_entries (
                    id TEXT PRIMARY KEY,
                    author TEXT NOT NULL DEFAULT 'Pulse',
                    title TEXT,
                    entry_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    why_it_mattered TEXT,
                    tags TEXT NOT NULL DEFAULT '[]',
                    importance INTEGER NOT NULL DEFAULT 5,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    resolved INTEGER,
                    date TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "INSERT INTO journal_entries "
                "(id, author, title, entry_type, content, why_it_mattered, tags, importance, pinned, resolved, date) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "001", "Demo", "Cyclone thread", "open_thread",
                    "A stale open thread that should stay visible until resolved.",
                    "Future Demo should know whether this still matters.",
                    json.dumps(["weather", "stale"]),
                    6, 0, 0, "2026-04-01T00:00:00+00:00",
                ),
            )
            conn.execute(
                "INSERT INTO journal_entries "
                "(id, author, title, entry_type, content, why_it_mattered, tags, importance, pinned, resolved, date) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "002", "Demo", "Resolved coffee", "follow_up",
                    "The coffee follow-up was completed.",
                    "It was resolved cleanly.",
                    json.dumps(["coffee"]),
                    4, 0, 1, "2026-04-02T00:00:00+00:00",
                ),
            )
            conn.execute(
                "INSERT INTO journal_entries "
                "(id, author, title, entry_type, content, why_it_mattered, tags, importance, pinned, resolved, date) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "003", "Demo", "Quiet reflection", "reflection",
                    "A normal reflection without a resolved lifecycle.",
                    "It captures a useful internal state.",
                    json.dumps(["tone"]),
                    5, 1, None, "2026-04-03T00:00:00+00:00",
                ),
            )
            conn.commit()
        finally:
            conn.close()

        api = PulseAPI(root)
        active = api.list_journal_entries("demo", "active", "all", page=1, page_size=25)
        assert active["ok"] is True
        assert active["total"] == 2
        assert [item["id"] for item in active["items"]] == ["003", "001"]
        assert active["items"][0]["status"] == "reference"
        assert active["items"][0]["pinned"] is True

        open_threads = api.list_journal_entries("demo", "active", "open_thread", page=1, page_size=25)
        assert open_threads["ok"] is True
        assert open_threads["total"] == 1
        assert open_threads["items"][0]["id"] == "001"

        resolved = api.list_journal_entries("demo", "resolved", "all", page=1, page_size=25)
        assert resolved["ok"] is True
        assert resolved["total"] == 1
        assert resolved["items"][0]["id"] == "002"

        detail = api.get_journal_entry("demo", "001")
        assert detail["ok"] is True
        assert detail["entry"]["content"].startswith("A stale open thread")
        assert detail["entry"]["why_it_mattered"].startswith("Future Demo")
        assert detail["entry"]["tags"] == ["weather", "stale"]


def test_gui_api_get_core_anchor_read_only():
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

        db_path = data_dir / "demo.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                """
                CREATE TABLE identity (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    sections TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    last_updated TEXT
                )
                """
            )
            conn.execute(
                "INSERT INTO identity (id, title, sections, created_at, last_updated) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    "_user",
                    "About My Human",
                    json.dumps({
                        "who_they_are": "Lena is building Pulse.",
                        "what_theyre_like": "Curious and careful.",
                        "extra_notes": "",
                    }),
                    "2026-04-01T00:00:00+00:00",
                    "2026-04-02T00:00:00+00:00",
                ),
            )
            conn.commit()
        finally:
            conn.close()

        api = PulseAPI(root)
        anchor = api.get_core_anchor("demo", "_user")
        assert anchor["ok"] is True
        assert anchor["anchor"]["title"] == "About My Human"
        assert anchor["anchor"]["empty"] is False
        assert anchor["anchor"]["sections"][0]["label"] == "Who They Are"
        assert anchor["anchor"]["sections"][0]["value"] == "Lena is building Pulse."
        assert anchor["anchor"]["sections"][1]["label"] == "What They're Like"

        loaded = api.load_persona("demo")
        assert loaded["core_anchors"]["_user"]["has_content"] is True
        assert loaded["core_anchors"]["_self"]["has_content"] is False

        missing = api.get_core_anchor("demo", "_self")
        assert missing["ok"] is True
        assert missing["anchor"]["empty"] is True
        assert missing["anchor"]["title"] == "Who I Am"

        rejected = api.get_core_anchor("demo", "persona_yaml")
        assert rejected["ok"] is False


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
    test_gui_api_lantern_update_and_dim_write_safely()
    test_gui_api_list_memories_read_only_with_history_views()
    test_gui_api_list_journal_entries_read_only_views()
    test_gui_api_get_core_anchor_read_only()
    test_gui_api_model_file_path_validation()
    test_gui_api_secrets_preserve_env_and_validate_keys()
    print("[OK] GUI API")
