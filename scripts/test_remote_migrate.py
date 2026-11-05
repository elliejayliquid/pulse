"""Migration helper + remote guard (designs/remote_host.md §0, §8).

    .venv/Scripts/python.exe scripts/test_remote_migrate.py
"""
import json
import os
import sqlite3
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.append(os.getcwd())

import scripts.remote as remote

PATH_MAP = {"D:/Claude/memories": "/srv/claude-memories", "D:/Claude/LoR/lor_data": "/srv/lor"}

SHARED_CONFIG = '''# Valentine-style overlay
provider:
  type: "anthropic"

# --- Shared data ---
paths:
  memories: "D:/Claude/memories"
  shared_database: "D:/Claude/memories/shared.db"
  journal_entries: "D:/Claude/memories/journal/entries"
  lor_data: "D:/Claude/LoR/lor_data"   # shared forum
  legacy_db: "D:/Claude/memories/legacy.db"  # claude.ai export archive

channels:
  lor:
    author_name: "Valentine"
'''


def _use_root(tmp: Path) -> None:
    remote.ROOT = tmp
    remote.PERSONAS_DIR = tmp / "personas"


def test_rewrites_shared_paths_and_keeps_comments():
    text, changes = remote.box_config_text(SHARED_CONFIG, "valentine", PATH_MAP)
    assert '  memories: "/srv/claude-memories"\n' in text
    assert '  shared_database: "/srv/claude-memories/shared.db"\n' in text
    assert '  journal_entries: "/srv/claude-memories/journal/entries"\n' in text
    assert '  lor_data: "/srv/lor"   # shared forum\n' in text
    assert '  legacy_db: "/srv/claude-memories/legacy.db"  # claude.ai export archive\n' in text
    assert "# --- Shared data ---" in text and 'author_name: "Valentine"' in text
    assert {c["key"] for c in changes} == {"memories", "shared_database", "journal_entries", "lor_data", "legacy_db"}


def test_backslashes_and_own_folder_paths():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _use_root(tmp)
        own = str(tmp / "personas" / "nova" / "data" / "legacy.db").replace("\\", "/")
        config = (
            "model:\n"
            "  max_context: 32768\n"
            "paths:\n"
            f'  legacy_db: "{own}"  # ChatGPT export\n'
            '  lor_data: "D:\\\\Claude\\\\LoR\\\\lor_data"\n'
            "server:\n"
            '  models_dir: "C:\\\\Users\\\\x\\\\models"\n'
        )
        text, _ = remote.box_config_text(config, "nova", PATH_MAP)
        assert '  legacy_db: "personas/nova/data/legacy.db"  # ChatGPT export\n' in text
        assert '  lor_data: "/srv/lor"\n' in text
        # Outside paths: left alone (llama-server only starts for local providers)
        assert 'models_dir: "C:\\\\Users\\\\x\\\\models"' in text


def test_unmapped_windows_path_is_refused():
    try:
        remote.box_config_text('paths:\n  memories: "E:/elsewhere/memories"\n', "kai", PATH_MAP)
        raise AssertionError("unmapped path accepted")
    except remote.RemoteError as e:
        assert "E:/elsewhere/memories" in str(e) and "path_map" in str(e)


def test_bundle_skips_runtime_files_and_copies_wal_rows():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _use_root(tmp)
        persona = tmp / "personas" / "kai"
        (persona / "data" / "journal").mkdir(parents=True)
        (persona / ".env").write_text("TELEGRAM_BOT_TOKEN=x\n", encoding="utf-8")
        (persona / "config.yaml").write_text("paths:\n  lor_data: D:/Claude/LoR/lor_data\n", encoding="utf-8")
        (persona / "remote.json").write_text("{}", encoding="utf-8")
        (persona / "data" / "status.json").write_text("{}", encoding="utf-8")
        (persona / "data" / "journal" / "latest.md").write_text("hi", encoding="utf-8")
        (persona / "data" / "pulse.db").write_bytes(b"")  # empty placeholder, not SQLite

        # A row that lives only in the WAL must survive the copy
        db = sqlite3.connect(persona / "data" / "kai.db")
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA wal_autocheckpoint=0")
        db.execute("CREATE TABLE t (v TEXT)")
        db.execute("INSERT INTO t VALUES ('only-in-wal')")
        db.commit()
        assert (persona / "data" / "kai.db-wal").stat().st_size > 0

        new_text, _ = remote.box_config_text(
            (persona / "config.yaml").read_text(encoding="utf-8"), "kai", PATH_MAP)
        scratch = tmp / "scratch"
        scratch.mkdir()
        bundle = tmp / "kai.tar"
        counts = remote._build_bundle("kai", new_text, bundle, scratch)
        db.close()

        with tarfile.open(bundle) as tar:
            names = set(tar.getnames())
            assert "kai/.env" in names and "kai/data/journal/latest.md" in names
            assert "kai/remote.json" not in names and "kai/data/status.json" not in names
            assert not any(n.endswith(("-wal", "-shm")) for n in names)
            assert tar.getmember("kai/.env").mode == 0o640
            assert b"/srv/lor" in tar.extractfile("kai/config.yaml").read()
            out = tmp / "out"
            tar.extractall(out)
        copy = sqlite3.connect(out / "kai" / "data" / "kai.db")
        assert copy.execute("SELECT v FROM t").fetchall() == [("only-in-wal",)]
        copy.close()
        assert counts["databases"] == 1


def test_migrate_refuses_marked_or_envless_persona():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _use_root(tmp)
        persona = tmp / "personas" / "vale"
        persona.mkdir(parents=True)
        for setup, needle in (
            (lambda: None, ".env"),
            (lambda: (persona / "remote.json").write_text("{}"), "already marked"),
        ):
            setup()
            try:
                remote.migrate({"host": "box", "user": "lena", "box_root": "/opt/pulse"}, "vale")
                raise AssertionError("migrate went ahead")
            except remote.RemoteError as e:
                assert needle in str(e), e


def test_pulse_refuses_remote_persona():
    import pulse
    with tempfile.TemporaryDirectory() as tmp:
        persona = Path(tmp) / "kai"
        persona.mkdir()
        config = {"_persona_dir": str(persona), "_persona_name": "kai"}
        pulse.check_not_remote(config)  # no marker: fine
        (persona / "remote.json").write_text(json.dumps({"host": "192.168.0.203"}))
        try:
            pulse.check_not_remote(config)
            raise AssertionError("started a remote persona locally")
        except SystemExit as e:
            assert e.code == 2
        pulse.check_not_remote(config, local_anyway=True)  # explicit override


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
