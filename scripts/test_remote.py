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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
