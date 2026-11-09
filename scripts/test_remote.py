import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.append(os.getcwd())

import scripts.remote as remote

SETTINGS = {"host": "box", "user": "lena", "box_root": "/opt/pulse"}


def _tar_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class _FakePopen:
    def __init__(self, payload: bytes, code: int = 0):
        self.stdout = io.BytesIO(payload)
        self._code = code

    def wait(self):
        return self._code


def _with_fake_box(tmp: Path, payload: bytes, code: int = 0):
    remote.ROOT = tmp
    remote.PERSONAS_DIR = tmp / "personas"
    remote.ssh = lambda settings, cmd, **kw: subprocess.CompletedProcess(cmd, 0, b"20261103T000000Z\n", b"")
    remote.subprocess.Popen = lambda *a, **kw: _FakePopen(payload, code)


def test_pull_replaces_backup_but_keeps_marker_and_env():
    real_popen, real_ssh = subprocess.Popen, remote.ssh
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        persona = tmp / "personas" / "demo"
        (persona / "data").mkdir(parents=True)
        (persona / "remote.json").write_text(json.dumps({"host": "box"}), encoding="utf-8")
        (persona / ".env").write_text("TOKEN=pc-copy\n", encoding="utf-8")
        (persona / "data" / "deleted_on_box.txt").write_text("old", encoding="utf-8")

        payload = _tar_bytes({
            "./persona.yaml": b"name: Demo\n",
            "./data/demo.db": b"db-bytes",
            "../escape.txt": b"nope",
        })
        try:
            _with_fake_box(tmp, payload)
            result = remote.pull(SETTINGS, "demo")
        finally:
            remote.subprocess.Popen, remote.ssh = real_popen, real_ssh

        assert result["snapshot"] == "20261103T000000Z"
        assert (persona / "persona.yaml").read_bytes() == b"name: Demo\n"
        assert (persona / "data" / "demo.db").read_bytes() == b"db-bytes"
        assert not (persona / "data" / "deleted_on_box.txt").exists()
        assert (persona / ".env").read_text(encoding="utf-8") == "TOKEN=pc-copy\n"
        assert json.loads((persona / "remote.json").read_text())["last_pull"]["snapshot"] == result["snapshot"]
        assert not (tmp / "personas" / "escape.txt").exists()
        assert not (tmp / "gui_data" / "escape.txt").exists()


def test_pull_refuses_local_persona_and_failed_copy():
    real_popen, real_ssh = subprocess.Popen, remote.ssh
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        persona = tmp / "personas" / "local"
        persona.mkdir(parents=True)
        (persona / "persona.yaml").write_text("name: Local\n", encoding="utf-8")
        try:
            _with_fake_box(tmp, _tar_bytes({"./persona.yaml": b"name: Overwritten\n"}))
            try:
                remote.pull(SETTINGS, "local")
                raise AssertionError("pulled into a local persona")
            except remote.RemoteError:
                pass
            assert "Local" in (persona / "persona.yaml").read_text()

            # Marked remote, but the copy fails → PC copy untouched
            (persona / "remote.json").write_text("{}", encoding="utf-8")
            _with_fake_box(tmp, _tar_bytes({"./persona.yaml": b"name: Overwritten\n"}), code=1)
            try:
                remote.pull(SETTINGS, "local")
                raise AssertionError("failed copy was applied")
            except remote.RemoteError:
                pass
            assert "Local" in (persona / "persona.yaml").read_text()
        finally:
            remote.subprocess.Popen, remote.ssh = real_popen, real_ssh


def _fake_sync_box(tmp: Path, box: dict):
    """A demo persona on the PC plus a fake box holding `box` (file -> text)."""
    import hashlib
    sha = lambda t: hashlib.sha256(t.encode()).hexdigest()
    remote.ROOT = tmp
    remote.PERSONAS_DIR = tmp / "personas"
    persona = remote.PERSONAS_DIR / "demo"
    persona.mkdir(parents=True)
    (persona / "remote.json").write_text("{}", encoding="utf-8")
    (persona / "persona.yaml").write_bytes(b"name: Demo\r\n")
    (persona / "config.yaml").write_bytes(
        b'provider:\r\n  model: "new-model"\r\npaths:\r\n  lor_data: "D:/Claude/LoR/lor_data"  # forum\r\n')
    puts = []
    remote.get_file = lambda s, p, f: {"ok": True, "content": box[f], "sha256": sha(box[f])}

    def fake_put(s, p, f, content, base):
        if base != sha(box[f]):
            return {"ok": False, "conflict": True, "error": "changed on the remote"}
        puts.append(f)
        box[f] = content
        return {"ok": True, "sha256": sha(content), "backup": "/opt/pulse/backups/x"}

    remote.put_file = fake_put
    return persona, puts


SYNC_SETTINGS = {**SETTINGS, "path_map": {"D:/Claude/LoR/lor_data": "/srv/lor"}}
OLD_BOX_CONFIG = 'provider:\n  model: "old-model"\npaths:\n  lor_data: "/srv/lor"  # forum\n'


def _states(persona: str = "demo") -> dict:
    return {r["file"]: r for r in remote.sync_state(SYNC_SETTINGS, persona)}


def test_sync_push_pull_conflict_directions():
    real = remote.get_file, remote.put_file, remote.ROOT, remote.PERSONAS_DIR
    with tempfile.TemporaryDirectory() as tmp:
        box = {"persona.yaml": "name: Demo\n", "config.yaml": OLD_BOX_CONFIG}
        persona, puts = _fake_sync_box(Path(tmp), box)
        try:
            rows = _states()
            assert rows["persona.yaml"]["state"] == "same"           # CRLF vs LF only
            assert rows["config.yaml"]["state"] == "unknown"         # differs, never synced
            assert '"/srv/lor"  # forum' in rows["config.yaml"]["pc_as_box"]
            assert "\r" not in rows["config.yaml"]["pc_as_box"]
            assert remote.local_unsynced(SYNC_SETTINGS, "demo") == []

            remote.push_files(SYNC_SETTINGS, "demo", list(rows.values()), ["config.yaml"])
            assert puts == ["config.yaml"] and 'model: "new-model"' in box["config.yaml"]
            assert _states()["config.yaml"]["state"] == "same"

            # Edited on the PC only -> push, and the cheap local check sees it too
            cfg = persona / "config.yaml"
            cfg.write_bytes(cfg.read_bytes().replace(b"new-model", b"pc-model"))
            assert _states()["config.yaml"]["state"] == "push"
            assert remote.local_unsynced(SYNC_SETTINGS, "demo") == ["config.yaml"]
            cfg.write_bytes(cfg.read_bytes().replace(b"pc-model", b"new-model"))

            # Edited on the box only -> pull; the diff reads PC -> box
            box["config.yaml"] = box["config.yaml"].replace("new-model", "box-model")
            row = _states()["config.yaml"]
            assert row["state"] == "pull" and "+  model: \"box-model\"" in row["diff"]
            remote.pull_files(SYNC_SETTINGS, "demo", [row], ["config.yaml"])
            pulled = cfg.read_bytes()
            assert b'model: "box-model"' in pulled
            assert b'"D:/Claude/LoR/lor_data"  # forum' in pulled             # path mapped home
            assert b"\r\n" in pulled and b"\n" not in pulled.replace(b"\r\n", b"")  # CRLF kept
            assert any((Path(tmp) / "gui_data" / "backups" / "demo").iterdir())  # PC backed up
            assert _states()["config.yaml"]["state"] == "same"

            # Both sides moved -> conflict; a stale put is refused, not overwritten
            cfg.write_bytes(cfg.read_bytes().replace(b"box-model", b"pc-2"))
            box["config.yaml"] = box["config.yaml"].replace("box-model", "box-2")
            rows = _states()
            assert rows["config.yaml"]["state"] == "conflict"
            box["config.yaml"] += "# one more box edit\n"
            try:
                remote.push_files(SYNC_SETTINGS, "demo", list(rows.values()), ["config.yaml"])
                raise AssertionError("pushed over a box edit it never saw")
            except remote.RemoteError:
                pass
            assert "one more box edit" in box["config.yaml"]
        finally:
            remote.get_file, remote.put_file, remote.ROOT, remote.PERSONAS_DIR = real


def test_pull_snapshot_keeps_pc_configs():
    real_popen, real_ssh = subprocess.Popen, remote.ssh
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        persona = tmp / "personas" / "demo"
        persona.mkdir(parents=True)
        (persona / "remote.json").write_text("{}", encoding="utf-8")
        (persona / "config.yaml").write_text("unpushed: edit\n", encoding="utf-8")
        payload = _tar_bytes({"./config.yaml": b"from: snapshot\n", "./data/demo.db": b"db"})
        try:
            _with_fake_box(tmp, payload)
            remote.pull(SETTINGS, "demo")
        finally:
            remote.subprocess.Popen, remote.ssh = real_popen, real_ssh
        assert (persona / "config.yaml").read_text() == "unpushed: edit\n"
        assert (persona / "data" / "demo.db").read_bytes() == b"db"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
