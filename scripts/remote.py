"""PC-side remote host tool: deploy Pulse to a home server and manage remote personas.

See designs/remote_host.md. Everything goes over the system OpenSSH client
(key-only, BatchMode) — the box opens no ports for Pulse. The GUI calls the
same functions.

    python scripts/remote.py init --host 192.168.0.203 --user lena [--key ~/.ssh/id_ed25519]
    python scripts/remote.py bootstrap            # first clone + venv on the box (after setup_box.sh)
    python scripts/remote.py deploy [--sha <commit>]
    python scripts/remote.py rollback
    python scripts/remote.py version
    python scripts/remote.py list | status <p> | start <p> | stop <p> | restart <p> | log <p> [n]
    python scripts/remote.py get <p> persona.yaml|config.yaml --out <file>
    python scripts/remote.py put <p> persona.yaml|config.yaml <file> --base <sha256 from get>
    python scripts/remote.py snapshot [<p>]
    python scripts/remote.py pull [<p>]           # newest snapshot -> personas/<p>/ (remote personas only)
    python scripts/remote.py mark-remote <p>      # PC copy becomes a backup of a box persona
    python scripts/remote.py migrate <p> [--no-health-check]   # move a persona to the box
    python scripts/remote.py who [--json]         # who's running on the box
    python scripts/remote.py diff <p>             # persona.yaml/config.yaml: which way each must go
    python scripts/remote.py push <p> [--yes] [--force] [--restart]   # PC edits (GUI or by hand) -> box
    python scripts/remote.py pull-config <p> [--yes] [--force]        # box edits -> PC

Settings live in gui_data/remote.json (git-ignored). Without that file nothing
here runs and nothing in Pulse changes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PERSONAS_DIR = ROOT / "personas"
SETTINGS_PATH = ROOT / "gui_data" / "remote.json"
GIT_REMOTE = "box"
DEPLOY_BRANCH = "deploy"
MARKER = "remote.json"
# Never replaced by a pull: the marker itself, and the PC's own copy of secrets
PULL_KEEP = {MARKER, ".env"}

DEFAULTS = {
    "box_root": "/opt/pulse",
}


class RemoteError(Exception):
    pass


# --- settings & transport --------------------------------------------------

def load_settings() -> dict:
    try:
        settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RemoteError("No remote host configured. Run: python scripts/remote.py init --host ... --user ...")
    return {**DEFAULTS, **settings}


def _ssh_base(settings: dict) -> list[str]:
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
    if settings.get("key"):
        cmd += ["-i", os.path.expanduser(settings["key"])]
    return cmd


def _target(settings: dict) -> str:
    return f"{settings['user']}@{settings['host']}"


# The GUI runs under pythonw (no console), and Windows gives every console program it
# starts (ssh, git) a console window of its own: a black flash per status poll.
# Captured calls never need one. Streamed CLI calls keep theirs, since that's where
# their output shows.
HIDDEN = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}


def ssh(settings: dict, remote_cmd: str, stdin: bytes | None = None,
        timeout: float | None = 60, stream: bool = False) -> subprocess.CompletedProcess:
    cmd = _ssh_base(settings) + [_target(settings), remote_cmd]
    if stream:  # long-running (pip, first clone): show output as it happens
        return subprocess.run(cmd, timeout=timeout)
    if stdin is None:  # never let ssh reach for a console or the parent's stdin
        return subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout, **HIDDEN)
    return subprocess.run(cmd, input=stdin, capture_output=True, timeout=timeout, **HIDDEN)


def ctl(settings: dict, *args: str, stdin: bytes | None = None, timeout: float | None = 60) -> dict:
    """Run scripts/pulse_ctl.py on the box and return its JSON result."""
    root = settings["box_root"]
    remote_cmd = " ".join(shlex.quote(a) for a in (
        f"{root}/venv/bin/python", f"{root}/app/scripts/pulse_ctl.py", *args
    ))
    result = ssh(settings, remote_cmd, stdin=stdin, timeout=timeout)
    out = result.stdout.decode("utf-8", errors="replace").strip()
    try:
        return json.loads(out.splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        err = result.stderr.decode("utf-8", errors="replace").strip()
        raise RemoteError(f"No answer from pulse_ctl ({result.returncode}): {err or out or 'empty output'}")


def _git(*args: str, env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, env=env,
                            stdin=subprocess.DEVNULL, **HIDDEN)
    if check and result.returncode != 0:
        raise RemoteError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def _git_ssh_env(settings: dict) -> dict:
    env = dict(os.environ)
    env["GIT_SSH_COMMAND"] = " ".join(shlex.quote(a) for a in _ssh_base(settings))
    return env


# --- deploy ----------------------------------------------------------------

def preflight(sha: str = "HEAD") -> dict:
    """What would be deployed, and what wouldn't (uncommitted work)."""
    resolved = _git("rev-parse", "--verify", f"{sha}^{{commit}}").stdout.strip()
    dirty = [
        line for line in _git("status", "--porcelain", "--untracked-files=no").stdout.splitlines()
        if line.strip()
    ]
    return {"sha": resolved, "subject": _git("log", "-1", "--format=%s", resolved).stdout.strip(),
            "uncommitted": len(dirty)}


def push(settings: dict, sha: str) -> None:
    _git("push", "-q", "--force", GIT_REMOTE, f"{sha}:refs/heads/{DEPLOY_BRANCH}",
         env=_git_ssh_env(settings))


def deploy(settings: dict, sha: str = "HEAD") -> dict:
    info = preflight(sha)
    push(settings, info["sha"])
    # Generous timeout: graceful stops can wait on replies in flight, pip may run
    result = ctl(settings, "deploy", info["sha"], timeout=1800)
    result["uncommitted"] = info["uncommitted"]
    return result


def version(settings: dict) -> dict:
    result = ctl(settings, "version")
    if result.get("ok"):
        remote_sha = result["sha"]
        ahead = _git("rev-list", "--count", f"{remote_sha}..HEAD", check=False)
        behind = _git("rev-list", "--count", f"HEAD..{remote_sha}", check=False)
        result["local_sha"] = _git("rev-parse", "HEAD").stdout.strip()
        result["ahead"] = int(ahead.stdout.strip()) if ahead.returncode == 0 else None
        result["behind"] = int(behind.stdout.strip()) if behind.returncode == 0 else None
    return result


def rollback(settings: dict) -> dict:
    history = ctl(settings, "version").get("history", [])
    deployed = [h for h in history if not h.get("rolled_back")]
    if not deployed:
        raise RemoteError("No deploy history to roll back to.")
    return deploy(settings, deployed[-1]["previous"])


# --- snapshots -------------------------------------------------------------

def _safe_members(tar: tarfile.TarFile):
    for member in tar:
        path = Path(member.name)
        if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
            continue
        yield member


def pull(settings: dict, persona: str) -> dict:
    """Replace personas/<persona>/ with the newest box snapshot (DBs, journal, avatar...).

    Keeps the marker, .env and the PC's persona.yaml/config.yaml: those sync per file.
    """
    persona_dir = PERSONAS_DIR / persona
    if not (persona_dir / MARKER).exists():
        raise RemoteError(f"{persona} isn't marked remote on this PC — refusing to overwrite a local persona.")

    snapshots = f"{settings['box_root']}/snapshots/{persona}"
    latest = ssh(settings, f"cat {shlex.quote(snapshots + '/LATEST')}")
    if latest.returncode != 0:
        raise RemoteError(f"No snapshot on the remote for {persona} yet. Run: remote.py snapshot {persona}")
    name = latest.stdout.decode().strip()

    staging_root = ROOT / "gui_data" / "pull_staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging_root) as tmp:
        proc = subprocess.Popen(
            _ssh_base(settings) + [_target(settings), f"tar -C {shlex.quote(snapshots + '/' + name)} -cf - ."],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            **HIDDEN,
        )
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            for member in _safe_members(tar):
                tar.extract(member, tmp)
        if proc.wait() != 0:
            raise RemoteError(f"Copying snapshot {name} failed; {persona} left untouched.")

        # Only now touch the PC copy: everything downloaded fine
        # persona.yaml/config.yaml stay: the PC copy is where Lena edits, and a snapshot
        # would silently wipe unpushed edits (sync_state/pull_files move those per file)
        keep = PULL_KEEP | {f for f in SYNC_FILES if (persona_dir / f).is_file()}
        for item in persona_dir.iterdir():
            if item.name in keep:
                continue
            shutil.rmtree(item) if item.is_dir() else item.unlink()
        for item in Path(tmp).iterdir():
            if item.name in keep:
                continue
            shutil.move(str(item), str(persona_dir / item.name))

    marker = json.loads((persona_dir / MARKER).read_text(encoding="utf-8"))
    marker["last_pull"] = {"snapshot": name, "at": _now()}
    (persona_dir / MARKER).write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return {"ok": True, "persona": persona, "snapshot": name}


def get_file(settings: dict, persona: str, filename: str) -> dict:
    """persona.yaml / config.yaml from the box, with the sha256 a later put must match."""
    return ctl(settings, "get", persona, filename)


def put_file(settings: dict, persona: str, filename: str, content: str, base_sha256: str) -> dict:
    """Write back an edited file. Refused (conflict) if it changed on the box since get."""
    return ctl(settings, "put", persona, filename, base_sha256, stdin=content.encode("utf-8"))


# --- syncing persona.yaml / config.yaml between the PC and the box ------------
#
# The PC copy is where Lena edits (GUI or by hand); the box copy is what runs.
# `synced` in remote.json holds, per file, the sha256 of the text both sides last
# agreed on (box form: box paths, LF endings). Comparing each side to it says which
# way a file has to go: only the PC moved -> push, only the box moved -> pull,
# both -> conflict (Lena picks). No record yet and they differ -> "unknown".

SYNC_FILES = ("config.yaml", "persona.yaml")
SYNC_STATES = ("same", "push", "pull", "conflict", "unknown")


def _lf(text: str) -> str:
    return text.replace("\r\n", "\n")


def _sha256_text(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_marker(persona: str) -> dict:
    path = PERSONAS_DIR / persona / MARKER
    if not path.is_file():
        raise RemoteError(f"{persona} isn't marked remote on this PC.")
    return json.loads(path.read_text(encoding="utf-8"))


def _remember_synced(persona: str, synced: dict[str, str]) -> None:
    """Record the agreed text's sha256 (box form, LF) per file."""
    if not synced:
        return
    marker = _read_marker(persona)
    marker.setdefault("synced", {}).update(synced)
    (PERSONAS_DIR / persona / MARKER).write_text(json.dumps(marker, indent=2), encoding="utf-8")


def _pc_as_box(settings: dict, persona: str, name: str) -> str | None:
    """The PC file in the form the box would hold (box paths, LF), or None if it's missing."""
    local = PERSONAS_DIR / persona / name
    if not local.is_file():
        return None
    text = _lf(local.read_text(encoding="utf-8"))
    if name == "config.yaml":
        text, _ = box_config_text(text, persona, settings.get("path_map") or {})
    return text


def local_unsynced(settings: dict, persona: str) -> list[str]:
    """Files edited on the PC since the last sync. No SSH, so it's cheap enough to poll.

    A file with no record yet isn't listed: only a real check against the box can tell.
    """
    synced = _read_marker(persona).get("synced", {})
    out = []
    for name in SYNC_FILES:
        known = synced.get(name)
        try:
            text = _pc_as_box(settings, persona, name)
        except RemoteError:
            text = None  # unmapped path: sync_state reports it properly
        if known and text is not None and _sha256_text(text) != known:
            out.append(name)
    return out


def _diff(a: str, b: str, a_label: str, b_label: str) -> str:
    import difflib
    return "".join(difflib.unified_diff(
        _lf(a).splitlines(keepends=True), _lf(b).splitlines(keepends=True), a_label, b_label,
    ))


def sync_state(settings: dict, persona: str) -> list[dict]:
    """Per file: which way it needs to go, with the diff for that direction.

    Each entry: file, state (SYNC_STATES), box, pc_as_box, box_sha256 (raw: what a put
    must match), diff. In the diff '-' is what the target loses and '+' what it gets:
    for a pull the target is the PC, otherwise it reads box -> PC.
    """
    synced = _read_marker(persona).get("synced", {})
    rows, agreed = [], {}
    for name in SYNC_FILES:
        pc = _pc_as_box(settings, persona, name) or ""
        got = get_file(settings, persona, name)
        if not got.get("ok"):
            raise RemoteError(f"Couldn't read {name} from the box: {got.get('error')}")
        box = got["content"]
        box_lf = _lf(box)
        known = synced.get(name)
        if box_lf == pc:
            state = "same"
            agreed[name] = _sha256_text(pc)
        elif not known:
            state = "unknown"
        else:
            pc_moved = _sha256_text(pc) != known
            box_moved = _sha256_text(box_lf) != known
            state = ("push" if pc_moved and not box_moved
                     else "pull" if box_moved and not pc_moved
                     else "conflict")
        box_label, pc_label = f"box/{persona}/{name}", f"pc/{persona}/{name}"
        if state == "same":
            diff = ""
        elif state == "pull":
            diff = _diff(pc, box_lf, pc_label, box_label)
        else:
            diff = _diff(box_lf, pc, box_label, pc_label)
        rows.append({
            "file": name, "state": state, "box": box, "pc_as_box": pc,
            "box_sha256": got["sha256"], "diff": diff,
        })
    _remember_synced(persona, agreed)
    return rows


def push_files(settings: dict, persona: str, rows: list[dict], files: list[str]) -> dict:
    """PC -> box for the named files (rows from sync_state). Hash-checked, backed up on the box."""
    pushed, agreed = [], {}
    for row in rows:
        if row["file"] not in files or row["state"] == "same":
            continue
        result = put_file(settings, persona, row["file"], row["pc_as_box"], row["box_sha256"])
        if not result.get("ok"):
            _remember_synced(persona, agreed)
            raise RemoteError(f"{row['file']}: {result.get('error')} Nothing else was written after it.")
        agreed[row["file"]] = _sha256_text(row["pc_as_box"])
        pushed.append({"file": row["file"], "backup": result.get("backup")})
    _remember_synced(persona, agreed)
    return {"ok": True, "persona": persona, "pushed": pushed}


def pull_files(settings: dict, persona: str, rows: list[dict], files: list[str]) -> dict:
    """Box -> PC for the named files: box paths mapped back to PC paths, PC line endings kept.

    The PC copy is backed up first (gui_data/backups, same as a GUI save).
    """
    todo = [r for r in rows if r["file"] in files and r["state"] != "same"]
    if not todo:
        return {"ok": True, "persona": persona, "pulled": []}
    from gui.backup import BackupManager
    backup = BackupManager(ROOT).create_backup(persona, reason="pre-pull")
    pulled, agreed = [], {}
    for row in todo:
        text = _lf(row["box"])
        if row["file"] == "config.yaml":
            text = pc_config_text(text, settings.get("path_map") or {})
        local = PERSONAS_DIR / persona / row["file"]
        crlf = local.is_file() and b"\r\n" in local.read_bytes()
        tmp = local.with_name(local.name + ".tmp")
        tmp.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
        os.replace(tmp, local)
        agreed[row["file"]] = _sha256_text(_lf(row["box"]))
        pulled.append({"file": row["file"]})
    _remember_synced(persona, agreed)
    return {"ok": True, "persona": persona, "pulled": pulled, "backup": backup.get("path")}


def backup_state(settings: dict, persona: str) -> dict:
    """Newest box snapshot vs the last one pulled to the PC (memories, journal, sessions)."""
    snapshots = f"{settings['box_root']}/snapshots/{persona}"
    latest = ssh(settings, f"cat {shlex.quote(snapshots + '/LATEST')}")
    newest = latest.stdout.decode().strip() if latest.returncode == 0 else ""
    last = _read_marker(persona).get("last_pull") or {}
    return {
        "ok": True,
        "persona": persona,
        "newest_snapshot": newest,
        "last_pull": last.get("snapshot", ""),
        "last_pull_at": last.get("at", ""),
        "behind": bool(newest) and newest != last.get("snapshot"),
    }


def remote_personas_here() -> list[str]:
    if not PERSONAS_DIR.is_dir():
        return []
    return sorted(d.name for d in PERSONAS_DIR.iterdir() if (d / MARKER).is_file())


def mark_remote(settings: dict, persona: str) -> dict:
    persona_dir = PERSONAS_DIR / persona
    if not persona_dir.is_dir():
        raise RemoteError(f"Persona not found: {persona}")
    marker = {"host": settings["host"], "remote_since": _now()}
    (persona_dir / MARKER).write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return {"ok": True, "persona": persona, **marker}


# --- migration (designs/remote_host.md §8) ------------------------------------

_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
# Never copied to the box: the PC-only marker, runtime state, SQLite side files
# (each DB goes over as one consistent copy made with the backup API)
_MIGRATE_SKIP_NAMES = {MARKER, "status.json", "shutdown_requested"}
_MIGRATE_SKIP_SUFFIXES = ("-wal", "-shm", "-journal")
# Keys whose box value must already exist there (shared resources the box owns)
_BOX_DIR_KEYS = {"lor_data", "memories"}
_BOX_FILE_KEYS = {"shared_database"}
# Shared files the box may not have yet: uploaded once, never overwritten
_UPLOADABLE_KEYS = {"legacy_db"}
_PATHS_LINE = re.compile(
    r"""^(?P<indent>\s+)(?P<key>[A-Za-z_][\w-]*):(?P<space>\s*)"""
    r"""(?P<value>"(?:[^"\\]|\\.)*"|'[^']*'|[^#\s][^#]*?)(?P<rest>\s*(?:#.*)?)$"""
)


def _norm(path: str) -> str:
    return path.replace("\\", "/").rstrip("/")


def _box_path(value: str, persona: str, path_map: dict) -> str | None:
    """Box equivalent of a Windows path, or None if nothing maps it."""
    norm = _norm(value)
    own = _norm(str(PERSONAS_DIR / persona))
    if norm.lower() == own.lower() or norm.lower().startswith(own.lower() + "/"):
        # Inside the persona's own folder: it travels with it, keep it relative
        return f"personas/{persona}{norm[len(own):]}"
    for pc_prefix, box_prefix in sorted(path_map.items(), key=lambda kv: -len(kv[0])):
        pc = _norm(pc_prefix)
        if norm.lower() == pc.lower() or norm.lower().startswith(pc.lower() + "/"):
            return box_prefix.rstrip("/") + norm[len(pc):]
    return None


def box_config_text(text: str, persona: str, path_map: dict) -> tuple[str, list[dict]]:
    """Rewrite the Windows paths under `paths:` for the box copy of a config.

    Works on the text, so Lena's comments survive. Raises RemoteError for a
    Windows path nothing maps (Pulse would refuse to start with it anyway).
    """
    import yaml

    out, changes, unmapped = [], [], []
    in_paths = False
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        if body and not body[0].isspace() and not body.lstrip().startswith("#"):
            in_paths = re.match(r"^paths:\s*(#.*)?$", body) is not None
        match = _PATHS_LINE.match(body) if in_paths else None
        if match:
            try:
                value = yaml.safe_load(match["value"])
            except yaml.YAMLError:
                value = None
            if isinstance(value, str) and _WINDOWS_PATH.match(value):
                new = _box_path(value, persona, path_map)
                if new is None:
                    unmapped.append(f"paths.{match['key']}: {value}")
                else:
                    changes.append({"key": match["key"], "pc": value, "box": new})
                    line = (f"{match['indent']}{match['key']}:{match['space']}"
                            f"{json.dumps(new)}{match['rest']}{line[len(body):]}")
        out.append(line)
    if unmapped:
        raise RemoteError(
            "No box path for: " + "; ".join(unmapped)
            + ". Add the PC folder to \"path_map\" in gui_data/remote.json."
        )
    new_text = "".join(out)
    leftover = {
        k: v for k, v in ((yaml.safe_load(new_text) or {}).get("paths") or {}).items()
        if isinstance(v, str) and _WINDOWS_PATH.match(v)
    }
    if leftover:  # a path the line rewrite couldn't see (flow style, odd quoting)
        raise RemoteError(f"Couldn't rewrite these paths, fix config.yaml by hand: {leftover}")
    return new_text, changes


def pc_config_text(text: str, path_map: dict) -> str:
    """Reverse of box_config_text for a pulled config: box paths under `paths:` -> PC paths.

    Paths inside the persona's own folder stay relative (they work on both machines).
    A box path nothing maps stays as it is: the PC copy is a backup, it never runs here.
    """
    import yaml

    reverse = sorted(((box.rstrip("/"), _norm(pc)) for pc, box in path_map.items()),
                     key=lambda kv: -len(kv[0]))
    out, in_paths = [], False
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        if body and not body[0].isspace() and not body.lstrip().startswith("#"):
            in_paths = re.match(r"^paths:\s*(#.*)?$", body) is not None
        match = _PATHS_LINE.match(body) if in_paths else None
        if match:
            try:
                value = yaml.safe_load(match["value"])
            except yaml.YAMLError:
                value = None
            if isinstance(value, str) and value.startswith("/"):
                for box_prefix, pc_prefix in reverse:
                    if value == box_prefix or value.startswith(box_prefix + "/"):
                        new = pc_prefix + value[len(box_prefix):]
                        line = (f"{match['indent']}{match['key']}:{match['space']}"
                                f"{json.dumps(new)}{match['rest']}{line[len(body):]}")
                        break
        out.append(line)
    return "".join(out)


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _running_locally(persona_dir: Path) -> bool:
    try:
        status = json.loads((persona_dir / "data" / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    pid = status.get("pid")
    return bool(status.get("running")) and isinstance(pid, int) and _pid_alive(pid)


def _is_sqlite(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(16) == b"SQLite format 3\x00"
    except OSError:
        return False


def _build_bundle(persona: str, config_text: str, out: Path, scratch: Path) -> dict:
    """Tar the persona folder for the box: rewritten config, consistent DB copies."""
    import sqlite3

    persona_dir = PERSONAS_DIR / persona
    counts = {"files": 0, "databases": 0, "bytes": 0}
    with tarfile.open(out, "w") as tar:
        for path in sorted(persona_dir.rglob("*")):
            rel = path.relative_to(persona_dir)
            if path.name in _MIGRATE_SKIP_NAMES or path.name.endswith(_MIGRATE_SKIP_SUFFIXES):
                continue
            if any(part in _MIGRATE_SKIP_NAMES for part in rel.parts):
                continue
            arcname = f"{persona}/{rel.as_posix()}"
            if path.is_dir():
                info = tar.gettarinfo(str(path), arcname)
                info.mode = 0o775
                tar.addfile(info)
                continue
            source = path
            if rel.as_posix() == "config.yaml":
                source = scratch / "config.yaml"
                source.write_text(config_text, encoding="utf-8", newline="\n")
            elif path.suffix == ".db" and _is_sqlite(path):
                source = scratch / f"db-{counts['databases']}.db"
                src = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
                dst = sqlite3.connect(source)
                try:
                    src.backup(dst)
                    if dst.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                        raise RemoteError(f"{rel} failed its integrity check; not migrating.")
                finally:
                    dst.close()
                    src.close()
                counts["databases"] += 1
            info = tar.gettarinfo(str(source), arcname)
            info.mode = 0o640 if path.name == ".env" else 0o664
            with open(source, "rb") as f:
                tar.addfile(info, f)
            counts["files"] += 1
            counts["bytes"] += info.size
    return counts


def _scp(settings: dict, local: Path, remote_path: str) -> None:
    cmd = ["scp", "-q", "-o", "BatchMode=yes"]
    if settings.get("key"):
        cmd += ["-i", os.path.expanduser(settings["key"])]
    subprocess.run(cmd + [str(local), f"{_target(settings)}:{remote_path}"], check=True, timeout=600)


def _health_check(settings: dict, persona: str, timeout: float = 150) -> dict:
    """Start the unit once, wait for status.json to say running, stop it again.

    Only right after a migration: the box copy has no status.json until this
    first start writes one, so whatever it says is fresh.
    """
    ctl(settings, "start", persona)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(5)
            info = ctl(settings, "status", persona).get("persona", {})
            status = info.get("status", {})
            if status.get("phase") == "running":
                return {"ok": True, "status": status}
            if info.get("systemd", {}).get("active") == "failed":
                break
        log = ctl(settings, "log", persona, "30").get("log", "")
        return {"ok": False, "error": f"{persona} didn't reach 'running'", "log": log}
    finally:
        ctl(settings, "stop", persona)


def migrate(settings: dict, persona: str, health_check: bool = True) -> dict:
    """Move a persona to the box: §8 step 2, minus `enable` (Lena starts boys herself).

    The PC folder itself is untouched apart from the remote.json marker, so it
    stays the pre-move backup. Rollback: delete the box folder and the marker.
    """
    persona_dir = PERSONAS_DIR / persona
    if not persona_dir.is_dir() or persona.startswith(("_", ".")):
        raise RemoteError(f"Persona not found: {persona}")
    if (persona_dir / MARKER).exists():
        raise RemoteError(f"{persona} is already marked remote.")
    if _running_locally(persona_dir):
        raise RemoteError(f"{persona} is running on this PC. Stop it first.")
    if not (persona_dir / ".env").is_file():
        raise RemoteError(f"{persona} has no .env (bot token, API key); nothing would run on the box.")

    config_path = persona_dir / "config.yaml"
    config_text = config_path.read_text(encoding="utf-8") if config_path.is_file() else ""
    new_text, changes = box_config_text(config_text, persona, settings.get("path_map") or {})

    root = settings["box_root"]
    app_personas = f"{root}/app/personas"
    if ssh(settings, "true").returncode != 0:
        raise RemoteError(f"Can't reach {_target(settings)} over SSH.")
    if ssh(settings, f"test ! -e {shlex.quote(app_personas + '/' + persona)}").returncode != 0:
        raise RemoteError(f"The box already has a {persona} folder; not overwriting it.")

    # Shared things the box must already have, and shared files it may still need
    uploads = []
    for change in changes:
        key, box = change["key"], change["box"]
        if not box.startswith("/"):
            continue  # inside the persona folder, travels with it
        if key in _BOX_DIR_KEYS and ssh(settings, f"test -d {shlex.quote(box)}").returncode != 0:
            raise RemoteError(f"paths.{key} → {box} doesn't exist on the box.")
        if key in _BOX_FILE_KEYS and ssh(settings, f"test -f {shlex.quote(box)}").returncode != 0:
            raise RemoteError(f"paths.{key} → {box} doesn't exist on the box.")
        if key in _UPLOADABLE_KEYS and ssh(settings, f"test -e {shlex.quote(box)}").returncode != 0:
            local = Path(change["pc"])
            if not local.is_file():
                raise RemoteError(f"paths.{key}: neither {local} here nor {box} on the box exists.")
            uploads.append((local, box))

    staging_root = ROOT / "gui_data" / "migrate_staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging_root) as tmp:
        tmp_path = Path(tmp)
        bundle = tmp_path / f"{persona}.tar"
        counts = _build_bundle(persona, new_text, bundle, tmp_path)

        for local, box in uploads:
            part = f"{box}.uploading"
            _scp(settings, local, part)
            done = ssh(settings, f"chmod 660 {shlex.quote(part)} && mv -n {shlex.quote(part)} {shlex.quote(box)}")
            if done.returncode != 0:
                raise RemoteError(f"Couldn't place {box}: {done.stderr.decode(errors='replace').strip()}")

        # Unpack into a hidden staging folder (pulse_ctl ignores dot-folders),
        # then move into place in one step
        script = f"""set -e
cd {shlex.quote(app_personas)}
[ ! -e {shlex.quote(persona)} ] || {{ echo "{persona} appeared meanwhile" >&2; exit 3; }}
stage=$(mktemp -d .migrate-XXXXXX)
trap 'rm -rf "$stage"' EXIT
tar -xf - -C "$stage"
chmod -R g+rwX "$stage"/{shlex.quote(persona)}
[ ! -f "$stage"/{shlex.quote(persona)}/.env ] || chmod 640 "$stage"/{shlex.quote(persona)}/.env
mv "$stage"/{shlex.quote(persona)} {shlex.quote(persona)}
echo migrate-ok
"""
        result = ssh(settings, f"bash -c {shlex.quote(script)}", stdin=bundle.read_bytes(), timeout=900)
        if result.returncode != 0 or b"migrate-ok" not in result.stdout:
            raise RemoteError(f"Upload failed: {result.stderr.decode(errors='replace').strip()}")

    # From here the box copy is the real one: mark the PC copy as its backup
    marker = mark_remote(settings, persona)
    report = {
        "ok": True, "persona": persona, "remote_since": marker["remote_since"],
        "paths": changes, "uploaded": [box for _, box in uploads], **counts,
        "unit": "installed, not enabled (start it with remote.bat start " + persona + ")",
    }
    if health_check:
        report["health"] = _health_check(settings, persona)
        report["ok"] = report["health"]["ok"]
    return report


# --- one-time setup ----------------------------------------------------------

def init(host: str, user: str, key: str | None, box_root: str) -> dict:
    settings = {"host": host, "user": user, "box_root": box_root}
    if key:
        settings["key"] = key
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    url = f"ssh://{user}@{host}{box_root}/repo.git"
    if _git("remote", "get-url", GIT_REMOTE, check=False).returncode == 0:
        _git("remote", "set-url", GIT_REMOTE, url)
    else:
        _git("remote", "add", GIT_REMOTE, url)

    if ssh(settings, "true").returncode != 0:
        raise RemoteError(f"Can't reach {user}@{host} over SSH with key auth.")
    scp = ["scp", "-q", "-o", "BatchMode=yes", "-r"]
    if key:
        scp += ["-i", os.path.expanduser(key)]
    subprocess.run(scp + [str(ROOT / "deploy" / "linux"), f"{user}@{host}:pulse-setup"], check=True)
    return {
        "ok": True,
        "settings": str(SETTINGS_PATH.relative_to(ROOT)),
        "next": f"ssh -t {user}@{host} sudo bash pulse-setup/setup_box.sh {user} Pacific/Auckland",
    }


def bootstrap(settings: dict) -> dict:
    """First clone + venv on the box. Needs setup_box.sh to have run."""
    info = preflight("HEAD")
    push(settings, info["sha"])
    root = settings["box_root"]
    script = f"""set -e
cd {shlex.quote(root)}
[ -d app/.git ] || git clone -q -b {DEPLOY_BRANCH} repo.git app
cd app
mkdir -p personas logs data
chmod 2775 personas logs data
[ -f host.yaml ] || cp host.example.yaml host.yaml
[ -d ../venv ] || python3 -m venv ../venv
../venv/bin/pip install -q --upgrade pip
../venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
../venv/bin/pip install -q -r requirements.txt
git checkout -q --detach {info['sha']}
echo bootstrap-ok
"""
    result = ssh(settings, f"bash -c {shlex.quote(script)}", timeout=3600, stream=True)
    if result.returncode != 0:
        raise RemoteError("Bootstrap failed (see output above).")
    return {"ok": True, "sha": info["sha"]}


def who_text(listing: dict) -> str:
    """Human summary of `pulse_ctl list`: who's up on the box right now."""
    lines = []
    for p in listing.get("personas", []):
        active = p.get("systemd", {}).get("active", "unknown")
        status = p.get("status", {})
        if active == "active":
            phase = status.get("phase") or "starting"
            hours = (status.get("uptime_seconds") or 0) / 3600
            state = f"RUNNING ({phase}, up {hours:.1f}h)" if phase == "running" else phase.upper()
        elif active in ("activating", "deactivating"):
            state = active.upper()
        elif active == "failed":
            state = "FAILED (see: log)"
        else:
            state = "stopped"
        boot = "  [starts at boot]" if p.get("systemd", {}).get("enabled") == "enabled" else ""
        lines.append(f"  {p.get('display_name', p['name']):<12} {p['name']:<12} {state}{boot}")
    return "\n".join(lines) if lines else "  (no personas on the box)"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# --- CLI -------------------------------------------------------------------

def _ask(question: str) -> bool:
    if not sys.stdin.isatty():
        return False
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


_STATE_TEXT = {
    "same": "same on both sides",
    "push": "edited on the PC -> push (- box now, + after push)",
    "pull": "edited ON THE BOX -> pull (- PC now, + after pull)",
    "conflict": "edited on BOTH sides (- box, + PC): pick one",
    "unknown": "differs, no record of the last sync (- box, + PC): read both sides",
}


def _sync_command(args) -> int:
    """`diff`, `push` and `pull-config`: human-readable, interactive unless --yes."""
    for stream in (sys.stdout, sys.stderr):
        try:  # configs hold →, — and emoji; the Windows console is cp1252
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    try:
        settings = load_settings()
        rows = sync_state(settings, args.persona)
    except (RemoteError, subprocess.SubprocessError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    for row in rows:
        print(f"\n  {row['file']}: {_STATE_TEXT[row['state']]}")
        if row["diff"]:
            print("\n" + row["diff"])
    if args.command == "diff":
        return 0

    pushing = args.command == "push"
    wanted = "push" if pushing else "pull"
    other = "pull" if pushing else "push"
    files = [r["file"] for r in rows if r["state"] == wanted]
    unsure = [r["file"] for r in rows if r["state"] in ("conflict", "unknown")]
    for r in rows:
        if r["state"] == other:
            print(f"  ({r['file']} goes the other way: remote.py {'pull-config' if pushing else 'push'} {args.persona})")
    if unsure and (args.force or _ask(
            f"\n{', '.join(unsure)}: {'overwrite the BOX with the PC' if pushing else 'overwrite the PC with the BOX'}"
            f" version anyway?")):
        files += unsure
    if not files:
        print(f"\n  Nothing to {wanted}.")
        return 0
    if not args.yes and not _ask(f"\n{'Push' if pushing else 'Pull'} {', '.join(files)}?"):
        print("Nothing changed.")
        return 1
    try:
        if not pushing:
            result = pull_files(settings, args.persona, rows, files)
            print(f"  pulled {', '.join(i['file'] for i in result['pulled'])}  (PC backup: {result.get('backup')})")
            return 0
        result = push_files(settings, args.persona, rows, files)
        for item in result["pushed"]:
            print(f"  pushed {item['file']}  (box backup: {item['backup']})")
        info = ctl(settings, "status", args.persona).get("persona") or {}
        if (info.get("systemd") or {}).get("active") == "active":
            if args.restart or _ask(f"{args.persona} is running with the old files. Restart him now?"):
                r = ctl(settings, "restart", args.persona)
                print(f"  restart: {'requested' if r.get('ok') else r.get('error')}")
            else:
                print(f"  He keeps the old settings until a restart: remote.bat restart {args.persona}")
        else:
            print(f"  {args.persona} is stopped: the new files apply on his next start.")
    except (RemoteError, subprocess.SubprocessError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Deploy Pulse to a remote host and manage remote personas.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init")
    p.add_argument("--host", required=True)
    p.add_argument("--user", required=True)
    p.add_argument("--key")
    p.add_argument("--box-root", default=DEFAULTS["box_root"])
    sub.add_parser("bootstrap")
    p = sub.add_parser("deploy")
    p.add_argument("--sha", default="HEAD")
    sub.add_parser("rollback")
    sub.add_parser("version")
    sub.add_parser("list")
    for verb in ("status", "start", "stop", "restart", "mark-remote"):
        sub.add_parser(verb).add_argument("persona")
    p = sub.add_parser("log")
    p.add_argument("persona")
    p.add_argument("lines", nargs="?", default="80")
    p = sub.add_parser("get", help="download persona.yaml/config.yaml; prints its sha256")
    p.add_argument("persona")
    p.add_argument("file", choices=["persona.yaml", "config.yaml"])
    p.add_argument("--out", required=True, help="local file to write")
    p = sub.add_parser("put", help="upload an edited file; --base is the sha256 from get")
    p.add_argument("persona")
    p.add_argument("file", choices=["persona.yaml", "config.yaml"])
    p.add_argument("source", help="local file to upload")
    p.add_argument("--base", required=True)
    sub.add_parser("snapshot").add_argument("persona", nargs="?")
    sub.add_parser("pull").add_argument("persona", nargs="?")
    p = sub.add_parser("migrate", help="move a persona to the box (unit installed, not enabled)")
    p.add_argument("persona")
    p.add_argument("--no-health-check", action="store_true",
                   help="skip the start-once-then-stop check")
    p = sub.add_parser("who", help="one line per remote persona: running or stopped")
    p.add_argument("--json", action="store_true")
    sub.add_parser("diff", help="persona.yaml/config.yaml: PC vs box, and which way each must go").add_argument("persona")
    for verb, text in (("push", "send PC edits of persona.yaml/config.yaml to the box"),
                       ("pull-config", "bring box edits of persona.yaml/config.yaml to the PC")):
        p = sub.add_parser(verb, help=text)
        p.add_argument("persona")
        p.add_argument("--yes", action="store_true", help="don't ask before writing")
        p.add_argument("--force", action="store_true", help="also overwrite files edited on both sides / never synced")
        if verb == "push":
            p.add_argument("--restart", action="store_true", help="restart him afterwards if he's running")

    args = parser.parse_args(argv)
    if args.command in ("diff", "push", "pull-config"):
        return _sync_command(args)
    try:
        if args.command == "init":
            result = init(args.host, args.user, args.key, args.box_root)
        else:
            settings = load_settings()
            if args.command == "bootstrap":
                result = bootstrap(settings)
            elif args.command == "deploy":
                info = preflight(args.sha)
                if info["uncommitted"]:
                    print(f"Note: {info['uncommitted']} uncommitted change(s) won't be deployed.", file=sys.stderr)
                result = deploy(settings, args.sha)
            elif args.command == "rollback":
                result = rollback(settings)
            elif args.command == "version":
                result = version(settings)
            elif args.command == "list":
                result = ctl(settings, "list")
            elif args.command in ("status", "start", "stop", "restart"):
                result = ctl(settings, args.command, args.persona)
            elif args.command == "log":
                result = ctl(settings, "log", args.persona, args.lines)
            elif args.command == "get":
                result = get_file(settings, args.persona, args.file)
                if result.get("ok"):
                    Path(args.out).write_bytes(result.pop("content").encode("utf-8"))
                    result["out"] = args.out
            elif args.command == "put":
                content = Path(args.source).read_bytes().decode("utf-8")
                result = put_file(settings, args.persona, args.file, content, args.base)
            elif args.command == "snapshot":
                result = ctl(settings, "snapshot", *([args.persona] if args.persona else []), timeout=600)
            elif args.command == "pull":
                personas = [args.persona] if args.persona else remote_personas_here()
                result = {"ok": True, "pulled": [pull(settings, p) for p in personas]}
            elif args.command == "mark-remote":
                result = mark_remote(settings, args.persona)
            elif args.command == "migrate":
                result = migrate(settings, args.persona, health_check=not args.no_health_check)
            elif args.command == "who":
                result = ctl(settings, "list")
    except (RemoteError, subprocess.SubprocessError, OSError) as e:
        result = {"ok": False, "error": str(e)}

    if args.command == "log" and result.get("ok"):
        print(result["log"])
    elif args.command == "who" and result.get("ok") and not args.json:
        print(who_text(result))
    else:
        print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
