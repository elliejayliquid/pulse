import os
import json
import sqlite3
import sys
import tempfile
from contextlib import closing
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


def test_gui_api_garden_summary_reads_grid_and_memory_tooltips():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        persona_dir = root / "personas" / "demo"
        data_dir = persona_dir / "data"
        data_dir.mkdir(parents=True)
        (root / "config.yaml").write_text("provider:\n  type: local\n", encoding="utf-8")
        (root / "persona.yaml").write_text("name: Base\n", encoding="utf-8")
        (persona_dir / "config.yaml").write_text("", encoding="utf-8")
        (persona_dir / "persona.yaml").write_text("name: Demo\n", encoding="utf-8")

        db_path = data_dir / "demo.db"
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, text TEXT)")
            conn.execute(
                """
                CREATE TABLE garden_plants (
                    id INTEGER PRIMARY KEY,
                    x INTEGER NOT NULL,
                    y INTEGER NOT NULL,
                    memory_id INTEGER,
                    species TEXT,
                    name TEXT,
                    growth REAL NOT NULL DEFAULT 0.0,
                    health REAL NOT NULL DEFAULT 1.0,
                    bloom_emoji TEXT,
                    planted_at TEXT NOT NULL DEFAULT (datetime('now')),
                    last_tended TEXT NOT NULL DEFAULT (datetime('now')),
                    last_watered TEXT
                )
                """
            )
            conn.execute("INSERT INTO memories (id, text) VALUES (42, ?)", ("Lena loves tiny practical gardens.",))
            conn.execute(
                """
                INSERT INTO garden_plants
                    (x, y, memory_id, species, name, growth, health, bloom_emoji, last_watered)
                VALUES
                    (2, 3, 42, 'personal', 'Sprouty', 1.2, 1.0, NULL, NULL),
                    (5, 1, NULL, 'wildflower', NULL, 2.4, 0.2, '🌷', datetime('now'))
                """
            )
            conn.commit()

        api = PulseAPI(root)
        result = api.get_garden_summary("demo")
        assert result["ok"] is True
        assert result["plant_count"] == 2
        assert result["wilted_count"] == 1
        assert result["needs_water_count"] == 1
        sprout = result["grid"][3][2]
        assert sprout["name"] == "Sprouty"
        assert sprout["stage"] == "Sprout"
        assert "Lena loves tiny practical gardens" in sprout["tooltip"]
        assert result["grid"][0][0]["empty"] is True


def test_gui_api_paint_gallery_reads_recent_index_safely():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        persona_dir = root / "personas" / "demo"
        paintings_dir = persona_dir / "data" / "paintings"
        paintings_dir.mkdir(parents=True)
        (root / "config.yaml").write_text("provider:\n  type: local\n", encoding="utf-8")
        (root / "persona.yaml").write_text("name: Base\n", encoding="utf-8")
        (persona_dir / "config.yaml").write_text("", encoding="utf-8")
        (persona_dir / "persona.yaml").write_text("name: Demo\n", encoding="utf-8")

        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
            b"\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\xf8\x0f\x00"
            b"\x01\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        (paintings_dir / "tiny@4x.png").write_bytes(png_bytes)
        (paintings_dir / "paintings.json").write_text(
            json.dumps([
                {
                    "id": "tiny",
                    "title": "Tiny Sun",
                    "caption": "A small warm square.",
                    "date": "2026-06-10T10:00:00",
                    "intent": "test art",
                    "width": 16,
                    "height": 16,
                    "upscaled_path": "tiny@4x.png",
                    "true_path": "../outside.png",
                }
            ]),
            encoding="utf-8",
        )

        api = PulseAPI(root)
        result = api.get_paint_gallery("demo")
        assert result["ok"] is True
        assert result["total"] == 1
        assert result["items"][0]["title"] == "Tiny Sun"
        assert result["items"][0]["image"].startswith("data:image/png;base64,")
        assert result["items"][0]["true_path"] == ""
        assert "test art" in result["items"][0]["tooltip"]


def test_gui_api_skill_status_summaries_are_read_only():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        persona_dir = root / "personas" / "demo"
        data_dir = persona_dir / "data"
        stickers_dir = root / "stickers"
        data_dir.mkdir(parents=True)
        stickers_dir.mkdir()
        (root / "config.yaml").write_text("provider:\n  type: local\n", encoding="utf-8")
        (root / "persona.yaml").write_text("name: Base\n", encoding="utf-8")
        (persona_dir / "config.yaml").write_text("", encoding="utf-8")
        (persona_dir / "persona.yaml").write_text("name: Demo\n", encoding="utf-8")

        db_path = data_dir / "demo.db"
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    description TEXT NOT NULL,
                    list TEXT NOT NULL DEFAULT 'Daily',
                    completed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    completed_at TEXT
                )
                """
            )
            conn.execute("INSERT INTO tasks (description, list) VALUES (?, ?)", ("Check the importer", "Pulse"))
            conn.execute("INSERT INTO tasks (description, completed) VALUES (?, 1)", ("Done thing",))
            conn.commit()

        with closing(sqlite3.connect(stickers_dir / "stickers.db")) as conn:
            conn.execute(
                """
                CREATE TABLE packs (
                    id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    title TEXT,
                    added_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE stickers (
                    id INTEGER PRIMARY KEY,
                    pack_id TEXT,
                    file_id TEXT,
                    keywords TEXT,
                    description TEXT,
                    embedding BLOB,
                    image_path TEXT
                )
                """
            )
            conn.execute("INSERT INTO packs (id, name, title) VALUES (1, 'cherry', 'Cherry Pack')")
            conn.execute(
                "INSERT INTO stickers (pack_id, file_id, keywords, description, embedding) VALUES (?, ?, ?, ?, ?)",
                (1, "file-id", "hug,warm", "Warm hug", b"1234"),
            )
            conn.commit()

        api = PulseAPI(root)
        web = api.get_web_search_status("demo")
        assert web["ok"] is True
        assert {tool["name"] for tool in web["tools"]} == {"web_search", "image_search", "fetch_url"}

        stickers = api.get_sticker_summary("demo")
        assert stickers["ok"] is True
        assert stickers["ready"] is True
        assert stickers["count"] == 1
        assert stickers["with_embeddings"] == 1
        assert stickers["packs"] == ["Cherry Pack"]

        tasks = api.get_tasks_summary("demo")
        assert tasks["ok"] is True
        assert tasks["pending"] == 1
        assert tasks["completed"] == 1
        assert tasks["recent"][0]["list"] == "Pulse"
        assert "importer" in tasks["recent"][0]["short_description"]

        listed = api.list_tasks("demo")
        assert listed["ok"] is True
        assert len(listed["tasks"]) == 2

        added = api.add_task("demo", {"description": "Review the task editor", "list": "GUI"})
        assert added["ok"] is True
        task_id = added["task_id"]

        updated = api.update_task("demo", task_id, {"description": "Review the polished task editor"})
        assert updated["ok"] is True
        assert updated["task"]["description"] == "Review the polished task editor"

        completed = api.update_task("demo", task_id, {"completed": True})
        assert completed["ok"] is True
        assert completed["task"]["completed"] is True

        deleted = api.delete_task("demo", task_id)
        assert deleted["ok"] is True
        assert api.restore_db_before_image("demo", deleted["undo_stamp"])["ok"] is True
        restored = api.get_task("demo", task_id)
        assert restored["ok"] is True
        assert restored["task"]["description"] == "Review the polished task editor"
        assert restored["task"]["completed"] is True

        staged = api.save_tasks("demo", [
            {"id": 1, "description": "Check the importer after staging", "list": "Pulse", "completed": False},
            {"description": "A brand new staged task", "list": "GUI", "completed": False},
        ])
        assert staged["ok"] is True
        assert staged["changed"] is True
        saved_tasks = api.list_tasks("demo")["tasks"]
        assert len(saved_tasks) == 2
        assert any(task["description"] == "A brand new staged task" for task in saved_tasks)
        assert api.restore_db_before_image("demo", staged["undo_stamp"])["ok"] is True
        undone_tasks = api.list_tasks("demo")["tasks"]
        assert len(undone_tasks) == 3
        assert any(task["description"] == "Review the polished task editor" for task in undone_tasks)


def test_gui_api_create_persona_from_template_initializes_db():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        template_dir = root / "personas" / "_template"
        (template_dir / "data").mkdir(parents=True)
        (root / "logs").mkdir()
        (root / "config.yaml").write_text(
            """
provider:
  type: local
model:
  max_context: 16384
""",
            encoding="utf-8",
        )
        (root / "persona.yaml").write_text("name: Base\nuser_name: User\n", encoding="utf-8")
        (template_dir / "config.yaml").write_text(
            "# Persona config overlay\n# Inherits from base.\n",
            encoding="utf-8",
        )
        (template_dir / "persona.yaml").write_text(
            """name: Your Persona Name
user_name: Your Name
model: ""

system_prompt: |
  You are {name}, a local AI companion living on {user_name}'s machine.

traits:
  - trait one
""",
            encoding="utf-8",
        )

        api = PulseAPI(root)
        created = api.create_persona({
            "display_name": "New Claude",
            "slug": "new_claude",
            "user_name": "Lena",
        })
        assert created["ok"] is True
        assert created["name"] == "new_claude"
        persona_dir = root / "personas" / "new_claude"
        assert persona_dir.is_dir()
        assert (persona_dir / "config.yaml").read_text(encoding="utf-8").startswith("# Persona config overlay")
        assert not (persona_dir / ".env").exists()

        identity = (persona_dir / "persona.yaml").read_text(encoding="utf-8")
        assert "name: New Claude" in identity
        assert "user_name: Lena" in identity
        assert "system_prompt: |" in identity
        assert "trait one" in identity

        db_path = persona_dir / "data" / "new_claude.db"
        assert db_path.exists()
        with closing(sqlite3.connect(db_path)) as conn:
            tables = {
                row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            assert {"messages", "sessions", "memories", "journal_entries", "identity"}.issubset(tables)

        personas = api.list_personas()
        assert "new_claude" in [p["name"] for p in personas]

        duplicate = api.create_persona({
            "display_name": "New Claude",
            "slug": "new_claude",
            "user_name": "Lena",
        })
        assert duplicate["ok"] is False
        assert "already exists" in duplicate["error"]

        reserved = api.create_persona({
            "display_name": "Base",
            "slug": "__base__",
            "user_name": "Lena",
        })
        assert reserved["ok"] is False

        leading_dot = api.create_persona({
            "display_name": "Dot",
            "slug": ".dot",
            "user_name": "Lena",
        })
        assert leading_dot["ok"] is False


def test_gui_api_import_openrouter_chat_visible_messages_only():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        template_dir = root / "personas" / "_template"
        (template_dir / "data").mkdir(parents=True)
        (root / "logs").mkdir()
        (root / "config.yaml").write_text("provider:\n  type: local\n", encoding="utf-8")
        (root / "persona.yaml").write_text("name: Base\nuser_name: User\n", encoding="utf-8")
        (template_dir / "config.yaml").write_text("", encoding="utf-8")
        (template_dir / "persona.yaml").write_text(
            "name: Your Persona Name\nuser_name: Your Name\nsystem_prompt: Hi.\n",
            encoding="utf-8",
        )

        export_path = root / "openrouter.json"
        export_path.write_text(json.dumps({
            "version": "orpg.3.0",
            "title": "Hi.",
            "messages": {
                "u1": {
                    "type": "user",
                    "createdAt": "2026-06-03T01:02:03.000Z",
                    "items": [{"id": "item-u1", "type": "message"}],
                },
                "a1": {
                    "type": "assistant",
                    "createdAt": "2026-06-03T01:03:04.000Z",
                    "items": [
                        {"id": "rs-a1", "type": "reasoning"},
                        {"id": "msg-a1", "type": "message"},
                    ],
                },
                "u2": {
                    "type": "user",
                    "items": [{"id": "item-u2", "type": "message"}],
                },
            },
            "items": {
                "item-u1": {
                    "messageId": "u1",
                    "data": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Hi Claude."}],
                    },
                },
                "rs-a1": {
                    "messageId": "a1",
                    "data": {
                        "type": "reasoning",
                        "content": [{"type": "reasoning_text", "text": "Private duplicated thought. Private duplicated thought."}],
                    },
                },
                "msg-a1": {
                    "messageId": "a1",
                    "data": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hello Lena."}],
                    },
                },
                "item-u2": {
                    "messageId": "u2",
                    "data": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "One more thing."}],
                    },
                },
            },
        }), encoding="utf-8")

        api = PulseAPI(root)
        created = api.create_persona({
            "display_name": "Claude",
            "slug": "claude",
            "user_name": "Lena",
        })
        assert created["ok"] is True

        imported = api.import_openrouter_chat("claude", "openrouter.json")
        assert imported["ok"] is True
        assert imported["title"] == "Hi."
        assert imported["imported_messages"] == 3
        assert imported["skipped_reasoning"] == 1
        assert imported["session_id"].startswith("openrouter_20260603_010203")

        db_path = root / "personas" / "claude" / "data" / "claude.db"
        with closing(sqlite3.connect(db_path)) as conn:
            session = conn.execute(
                "SELECT id, title FROM sessions WHERE id = ?",
                (imported["session_id"],),
            ).fetchone()
            assert session[1] == "Hi."
            rows = conn.execute(
                "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp, id",
                (imported["session_id"],),
            ).fetchall()
            assert [(row[0], row[1]) for row in rows] == [
                ("user", "Hi Claude."),
                ("assistant", "Hello Lena."),
                ("user", "One more thing."),
            ]
            assert rows[0][2] == "2026-06-03T01:02:03+00:00"
            assert rows[2][2] == "2026-06-03T01:03:04.000001+00:00"
            assert "Private duplicated thought" not in "\n".join(row[1] for row in rows)

        imported_again = api.import_openrouter_chat("claude", str(export_path))
        assert imported_again["ok"] is True
        assert imported_again["session_id"].endswith("_2")

        bad_path = root / "bad.json"
        bad_path.write_text(json.dumps({"version": "not-orpg"}), encoding="utf-8")
        rejected = api.import_openrouter_chat("claude", str(bad_path))
        assert rejected["ok"] is False
        assert "Unsupported" in rejected["error"]


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


def test_gui_api_add_memory_creates_table_and_undoes_insert():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        persona_dir = root / "personas" / "demo"
        persona_dir.mkdir(parents=True)
        (root / "logs").mkdir()
        (root / "config.yaml").write_text("", encoding="utf-8")
        (root / "persona.yaml").write_text("name: Base\n", encoding="utf-8")
        (persona_dir / "config.yaml").write_text("", encoding="utf-8")
        (persona_dir / "persona.yaml").write_text("name: Demo\n", encoding="utf-8")

        db_path = root / "personas" / "demo" / "data" / "demo.db"
        db_path.parent.mkdir(parents=True)
        with closing(sqlite3.connect(db_path)) as conn:
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
                    embedding BLOB
                )
                """
            )
            conn.commit()

        api = PulseAPI(root)
        added = api.add_memory("demo", {
            "text": "Piper is Lena's girl cat.",
            "tags": "piper, cat, correction",
            "importance": 7,
            "date": "2026-04-20",
            "time_sensitive": False,
        })
        assert added["ok"] is True
        assert added["changed"] is True
        assert added["memory_id"] == 1

        with closing(sqlite3.connect(db_path)) as conn:
            row = conn.execute(
                "SELECT text, tags, type, importance, status, confidence, source, embedding, date "
                "FROM memories WHERE id = 1"
            ).fetchone()
            assert row[0] == "Piper is Lena's girl cat."
            assert json.loads(row[1]) == ["piper", "cat", "correction"]
            assert row[2] == "fact"
            assert row[3] == 7
            assert row[4] == "current"
            assert row[5] == "medium"
            assert row[6] == "user_defined"
            assert row[7] is None
            assert row[8] == "2026-04-20T00:00:00+00:00"

        payload_path = next((root / "gui_data" / "db_backups" / "demo" / added["undo_stamp"]).glob("*.json"))
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        assert payload["action"] == "memory-add"
        assert payload["before"] is None
        assert payload["delete_where"] == {"id": 1}

        restored = api.restore_db_before_image("demo", added["undo_stamp"])
        assert restored["ok"] is True
        with closing(sqlite3.connect(db_path)) as conn:
            assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0

        rejected = api.add_memory("demo", {"text": "   "})
        assert rejected["ok"] is False
        rejected_status = api.add_memory("demo", {
            "text": "This should not be born orphaned.",
            "status": "superseded",
        })
        assert rejected_status["ok"] is False
        rejected_date = api.add_memory("demo", {
            "text": "This date is nonsense.",
            "date": "not-a-date",
        })
        assert rejected_date["ok"] is False


def test_gui_api_memory_mutations_backup_restore_and_relink():
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
            rows = [
                (1, "Piper is Lena's boy cat.", "[]", "fact", 5, None, "2026-01-01T00:00:00+00:00", "current", b"old-embed"),
                (2, "Middle memory", "[]", "fact", 5, 1, "2026-01-02T00:00:00+00:00", "current", b"mid"),
                (3, "Tail memory", "[]", "fact", 5, 2, "2026-01-03T00:00:00+00:00", "current", b"tail"),
                (10, "Head memory", "[]", "fact", 5, None, "2026-01-04T00:00:00+00:00", "current", b"head"),
                (11, "Head child", "[]", "fact", 5, 10, "2026-01-05T00:00:00+00:00", "current", b"child"),
                (20, "Tail only", "[]", "fact", 5, 19, "2026-01-06T00:00:00+00:00", "current", b"tail-only"),
            ]
            conn.executemany(
                "INSERT INTO memories "
                "(id, text, tags, type, importance, supersedes, date, status, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        finally:
            conn.close()

        api = PulseAPI(root)
        update = api.update_memory("demo", 1, {
            "text": "Piper is Lena's girl cat.",
            "tags": ["piper", "cat"],
            "confidence": "high",
        })
        assert update["ok"] is True
        with closing(sqlite3.connect(db_path)) as conn:
            row = conn.execute("SELECT text, tags, confidence, embedding FROM memories WHERE id = 1").fetchone()
            assert row[0] == "Piper is Lena's girl cat."
            assert json.loads(row[1]) == ["piper", "cat"]
            assert row[2] == "high"
            assert row[3] is None

        payload_path = next((root / "gui_data" / "db_backups" / "demo" / update["undo_stamp"]).glob("*.json"))
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        assert payload["before"]["embedding"]["__pulse_bytes_b64"]

        restored = api.restore_db_before_image("demo", update["undo_stamp"])
        assert restored["ok"] is True
        with closing(sqlite3.connect(db_path)) as conn:
            row = conn.execute("SELECT text, tags, confidence, embedding FROM memories WHERE id = 1").fetchone()
            assert row[0] == "Piper is Lena's boy cat."
            assert json.loads(row[1]) == []
            assert row[2] is None
            assert row[3] == b"old-embed"

        middle = api.delete_memory("demo", 2)
        assert middle["ok"] is True
        assert middle["relinked"] == 1
        with closing(sqlite3.connect(db_path)) as conn:
            assert conn.execute("SELECT COUNT(*) FROM memories WHERE id = 2").fetchone()[0] == 0
            assert conn.execute("SELECT supersedes FROM memories WHERE id = 3").fetchone()[0] == 1
        assert api.restore_db_before_image("demo", middle["undo_stamp"])["ok"] is True
        with closing(sqlite3.connect(db_path)) as conn:
            assert conn.execute("SELECT supersedes FROM memories WHERE id = 3").fetchone()[0] == 2

        head = api.delete_memory("demo", 10)
        assert head["ok"] is True
        with closing(sqlite3.connect(db_path)) as conn:
            assert conn.execute("SELECT supersedes FROM memories WHERE id = 11").fetchone()[0] is None
        assert api.restore_db_before_image("demo", head["undo_stamp"])["ok"] is True

        tail = api.delete_memory("demo", 20)
        assert tail["ok"] is True
        assert tail["relinked"] == 0
        with closing(sqlite3.connect(db_path)) as conn:
            assert conn.execute("SELECT COUNT(*) FROM memories WHERE id = 20").fetchone()[0] == 0


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


def test_gui_api_journal_mutations_update_mirror_and_restore():
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
                    search_summary TEXT,
                    summary_needs_review INTEGER NOT NULL DEFAULT 0,
                    tags TEXT NOT NULL DEFAULT '[]',
                    importance INTEGER NOT NULL DEFAULT 5,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    resolved INTEGER,
                    date TEXT NOT NULL
                )
                """
            )
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
                "INSERT INTO journal_entries "
                "(id, author, title, entry_type, content, why_it_mattered, search_summary, "
                "summary_needs_review, tags, importance, pinned, resolved, date) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "001", "Demo", "Piper note", "reflection",
                    "Piper was mentioned as a cat.",
                    "This matters for pet continuity.",
                    "Piper is Lena's cat.",
                    0,
                    json.dumps(["piper"]),
                    5, 0, None, "2026-04-01T00:00:00+00:00",
                ),
            )
            conn.execute(
                "INSERT INTO memories "
                "(id, text, tags, type, importance, date, status, confidence, source, journal_file, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    42,
                    "Old journal mirror",
                    json.dumps(["journal", "piper"]),
                    "journal",
                    5,
                    "2026-04-01T00:00:00+00:00",
                    "current",
                    "medium",
                    "model_extracted",
                    "entries/001.md",
                    b"journal-embed",
                ),
            )
            conn.commit()
        finally:
            conn.close()

        api = PulseAPI(root)
        updated = api.update_journal_entry("demo", "001", {
            "content": "Piper is Lena's girl cat, not a boy.",
            "search_summary": "Piper is Lena's female cat; never call her a boy.",
            "tags": ["piper", "cat"],
        })
        assert updated["ok"] is True
        with closing(sqlite3.connect(db_path)) as conn:
            entry = conn.execute(
                "SELECT content, search_summary, summary_needs_review, tags FROM journal_entries WHERE id = '001'"
            ).fetchone()
            assert entry[0] == "Piper is Lena's girl cat, not a boy."
            assert "female cat" in entry[1]
            assert entry[2] == 0
            assert json.loads(entry[3]) == ["piper", "cat"]
            mirror = conn.execute(
                "SELECT text, tags, embedding FROM memories WHERE id = 42"
            ).fetchone()
            assert "Search summary:" in mirror[0]
            assert "female cat" in mirror[0]
            assert json.loads(mirror[1]) == ["journal", "piper", "cat"]
            assert mirror[2] is None

        assert api.restore_db_before_image("demo", updated["undo_stamp"])["ok"] is True
        with closing(sqlite3.connect(db_path)) as conn:
            entry = conn.execute(
                "SELECT content, search_summary, tags FROM journal_entries WHERE id = '001'"
            ).fetchone()
            assert entry[0] == "Piper was mentioned as a cat."
            assert entry[1] == "Piper is Lena's cat."
            assert json.loads(entry[2]) == ["piper"]
            mirror = conn.execute("SELECT text, embedding FROM memories WHERE id = 42").fetchone()
            assert mirror[0] == "Old journal mirror"
            assert mirror[1] == b"journal-embed"

        deleted = api.delete_journal_entry("demo", "001")
        assert deleted["ok"] is True
        assert deleted["deleted_mirrors"] == 1
        with closing(sqlite3.connect(db_path)) as conn:
            assert conn.execute("SELECT COUNT(*) FROM journal_entries").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
        assert api.restore_db_before_image("demo", deleted["undo_stamp"])["ok"] is True
        with closing(sqlite3.connect(db_path)) as conn:
            assert conn.execute("SELECT COUNT(*) FROM journal_entries").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1

        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute(
                "INSERT INTO journal_entries "
                "(id, author, title, entry_type, content, why_it_mattered, search_summary, "
                "summary_needs_review, tags, importance, pinned, resolved, date) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "002", "Demo", "Missing mirror", "reflection",
                    "Entry starts without a companion memory mirror.",
                    "Undo should remove any mirror created by editing.",
                    "Entry has no mirror yet.",
                    0,
                    json.dumps(["mirror"]),
                    5, 0, None, "2026-04-02T00:00:00+00:00",
                ),
            )
            conn.commit()

        created = api.update_journal_entry("demo", "002", {
            "search_summary": "This edit creates a missing journal mirror."
        })
        assert created["ok"] is True
        with closing(sqlite3.connect(db_path)) as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM memories WHERE type = 'journal' AND journal_file = 'entries/002.md'"
            ).fetchone()[0] == 1
        assert api.restore_db_before_image("demo", created["undo_stamp"])["ok"] is True
        with closing(sqlite3.connect(db_path)) as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM memories WHERE type = 'journal' AND journal_file = 'entries/002.md'"
            ).fetchone()[0] == 0


def test_gui_api_core_anchor_read_and_write():
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

        updated = api.set_core_anchor(
            "demo",
            "_user",
            {
                "who_they_are": "Lena is building Pulse.",
                "what_theyre_like": "Curious, careful, and stubborn in a useful way.",
                "extra_notes": "Piper is a girl.",
            },
        )
        assert updated["ok"] is True
        assert updated["changed"] is True
        assert updated["anchor"]["sections"][1]["value"].startswith("Curious, careful")
        backups = list((root / "gui_data" / "db_backups" / "demo").glob("*/*.json"))
        assert len(backups) == 1
        payload = json.loads(backups[0].read_text(encoding="utf-8"))
        assert payload["table"] == "identity"
        assert payload["before"]["id"] == "_user"

        no_change = api.set_core_anchor(
            "demo",
            "_user",
            {
                "who_they_are": "Lena is building Pulse.",
                "what_theyre_like": "Curious, careful, and stubborn in a useful way.",
                "extra_notes": "Piper is a girl.",
            },
        )
        assert no_change["ok"] is True
        assert no_change["changed"] is False
        assert len(list((root / "gui_data" / "db_backups" / "demo").glob("*/*.json"))) == 1

        created = api.set_core_anchor(
            "demo",
            "_self",
            {
                "who_i_am": "I am Demo.",
                "extra_notes": "Freshly created from the GUI.",
            },
        )
        assert created["ok"] is True
        assert created["changed"] is True
        assert created["anchor"]["title"] == "Who I Am"
        assert created["anchor"]["empty"] is False

        invalid = api.set_core_anchor("demo", "_self", {"system_prompt": "nope"})
        assert invalid["ok"] is False


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
        assert {"OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN"}.issubset(by_key)
        assert by_key["OPENROUTER_API_KEY"]["source"] == "inherited"
        assert by_key["OPENROUTER_API_KEY"]["masked"].endswith("uter")
        assert by_key["ANTHROPIC_API_KEY"]["source"] == "missing"
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
    test_gui_api_garden_summary_reads_grid_and_memory_tooltips()
    test_gui_api_paint_gallery_reads_recent_index_safely()
    test_gui_api_skill_status_summaries_are_read_only()
    test_gui_api_create_persona_from_template_initializes_db()
    test_gui_api_import_openrouter_chat_visible_messages_only()
    test_gui_api_standard_provider_env_fallback()
    test_gui_api_get_lantern_read_only()
    test_gui_api_lantern_update_and_dim_write_safely()
    test_gui_api_list_memories_read_only_with_history_views()
    test_gui_api_add_memory_creates_table_and_undoes_insert()
    test_gui_api_memory_mutations_backup_restore_and_relink()
    test_gui_api_list_journal_entries_read_only_views()
    test_gui_api_journal_mutations_update_mirror_and_restore()
    test_gui_api_core_anchor_read_and_write()
    test_gui_api_model_file_path_validation()
    test_gui_api_secrets_preserve_env_and_validate_keys()
    print("[OK] GUI API")
