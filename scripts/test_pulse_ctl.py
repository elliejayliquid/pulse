import io
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.append(os.getcwd())

import scripts.pulse_ctl as ctl


def _setup_box(tmp: Path) -> Path:
    """Fake /opt/pulse with app/personas/demo and point pulse_ctl at it."""
    app = tmp / "app"
    persona_dir = app / "personas" / "demo"
    data_dir = persona_dir / "data"
    data_dir.mkdir(parents=True)
    (persona_dir / "persona.yaml").write_bytes(b"name: Demo\n")
    (persona_dir / "config.yaml").write_text("provider:\n  type: openrouter\n", encoding="utf-8")
    (persona_dir / ".env").write_text("TELEGRAM_BOT_TOKEN=secret\n", encoding="utf-8")
    (data_dir / "status.json").write_text(json.dumps({"phase": "running"}), encoding="utf-8")

    conn = sqlite3.connect(str(data_dir / "demo.db"))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, text TEXT)")
    conn.execute("INSERT INTO memories (text) VALUES ('still in the WAL')")
    conn.commit()
    # Leave the connection open so the row only exists in the -wal file
    ctl._test_conn = conn

    ctl.APP_ROOT = app
    ctl.PERSONAS_DIR = app / "personas"
    ctl.LOGS_DIR = app / "logs"
    ctl.SNAPSHOTS_DIR = tmp / "snapshots"
    return persona_dir


def _run(argv, stdin: bytes = b""):
    out = io.StringIO()
    real_stdout, real_stdin = sys.stdout, sys.stdin
    sys.stdout = out
    sys.stdin = io.TextIOWrapper(io.BytesIO(stdin))
    try:
        code = ctl.main(argv)
    finally:
        sys.stdout, sys.stdin = real_stdout, real_stdin
    return code, json.loads(out.getvalue())


def test_snapshot_is_consistent_and_skips_secrets():
    with tempfile.TemporaryDirectory() as tmp:
        _setup_box(Path(tmp))
        try:
            code, result = _run(["snapshot", "demo"])
            assert code == 0, result
            snap = result["snapshots"][0]
            snap_dir = ctl.SNAPSHOTS_DIR / "demo" / snap["snapshot"]
            assert (ctl.SNAPSHOTS_DIR / "demo" / "LATEST").read_text() == snap["snapshot"]
            assert (snap_dir / "persona.yaml").exists()
            assert not (snap_dir / ".env").exists()
            assert not list(snap_dir.rglob("*.db-wal"))

            copy = sqlite3.connect(str(snap_dir / "data" / "demo.db"))
            rows = copy.execute("SELECT text FROM memories").fetchall()
            copy.close()
            assert rows == [("still in the WAL",)]
        finally:
            ctl._test_conn.close()


def test_snapshot_prunes_old_ones():
    with tempfile.TemporaryDirectory() as tmp:
        _setup_box(Path(tmp))
        try:
            root = ctl.SNAPSHOTS_DIR / "demo"
            for i in range(ctl.SNAPSHOT_KEEP + 3):
                (root / f"20200101T0000{i:02d}Z").mkdir(parents=True)
            _run(["snapshot", "demo"])
            kept = [d for d in root.iterdir() if d.is_dir()]
            assert len(kept) == ctl.SNAPSHOT_KEEP
            assert not (root / "20200101T000000Z").exists()
        finally:
            ctl._test_conn.close()


def test_get_put_round_trip_with_conflict_check():
    with tempfile.TemporaryDirectory() as tmp:
        persona_dir = _setup_box(Path(tmp))
        try:
            code, got = _run(["get", "demo", "persona.yaml"])
            assert code == 0 and got["content"] == "name: Demo\n"

            code, put = _run(["put", "demo", "persona.yaml", got["sha256"]], b"name: Demo II\n")
            assert code == 0, put
            assert (persona_dir / "persona.yaml").read_text() == "name: Demo II\n"
            assert put["backup"]

            # Stale hash → conflict, file untouched
            code, stale = _run(["put", "demo", "persona.yaml", got["sha256"]], b"name: Nope\n")
            assert code == 1 and stale["conflict"] is True
            assert (persona_dir / "persona.yaml").read_text() == "name: Demo II\n"
        finally:
            ctl._test_conn.close()


def test_put_rejects_invalid_yaml_and_other_files():
    with tempfile.TemporaryDirectory() as tmp:
        persona_dir = _setup_box(Path(tmp))
        try:
            _, got = _run(["get", "demo", "config.yaml"])
            code, bad = _run(["put", "demo", "config.yaml", got["sha256"]], b"a: 1\na: 2\n")
            assert code == 1 and "Invalid YAML" in bad["error"]
            assert "openrouter" in (persona_dir / "config.yaml").read_text()

            code, env = _run(["get", "demo", ".env"])
            assert code == 1 and "can be read or written" in env["error"]

            code, escape = _run(["get", "../demo", "config.yaml"])
            assert code == 1
        finally:
            ctl._test_conn.close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
