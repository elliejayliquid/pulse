"""Box-side control script for remote personas (see designs/remote_host.md).

Runs on the remote host, invoked over SSH by the PC's scripts/remote.py and the
GUI. Every command prints exactly one JSON object on stdout; exit code 0 means
"ok": true. It is the only thing the PC runs remotely.

    python scripts/pulse_ctl.py list
    python scripts/pulse_ctl.py status <persona>
    python scripts/pulse_ctl.py start|stop|restart <persona>
    python scripts/pulse_ctl.py log <persona> [lines]
    python scripts/pulse_ctl.py get <persona> persona.yaml|config.yaml
    python scripts/pulse_ctl.py put <persona> persona.yaml|config.yaml <base_sha256>   (content on stdin)
    python scripts/pulse_ctl.py snapshot [<persona>]
    python scripts/pulse_ctl.py version
    python scripts/pulse_ctl.py deploy <sha>

Personas run as systemd units pulse@<persona>.service. start/stop/restart go
through `sudo -n /usr/local/sbin/pulse-unit`, a wrapper that the sudoers rule
from deploy/linux/setup_box.sh allows without a password. Deploy never runs `git clean`: persona data lives in
the checkout as git-ignored files.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
PERSONAS_DIR = APP_ROOT / "personas"
LOGS_DIR = APP_ROOT / "logs"
# Box layout: /opt/pulse/{app,snapshots,deploys.log}
BOX_ROOT = APP_ROOT.parent
SNAPSHOTS_DIR = Path(os.environ.get("PULSE_SNAPSHOTS_DIR", BOX_ROOT / "snapshots"))
DEPLOY_LOG = BOX_ROOT / "deploys.log"
DEPLOY_LOCK = BOX_ROOT / "deploy.lock"
VENV_PYTHON = Path(sys.executable)
UNIT_WRAPPER = "/usr/local/sbin/pulse-unit"

EDITABLE_FILES = ("persona.yaml", "config.yaml")
SNAPSHOT_KEEP = 14
# status.json is live runtime state; a copy would make the PC backup look "running"
SNAPSHOT_SKIP_NAMES = {".env", "status.json", "shutdown_requested", "__pycache__"}
SNAPSHOT_SKIP_SUFFIXES = (".db-wal", ".db-shm", ".db-journal", ".tmp")
HEALTH_TIMEOUT_SECONDS = 90

sys.path.insert(0, str(APP_ROOT))


class CtlError(Exception):
    """A failure reported to the caller as {"ok": false, "error": ...}."""


# --- helpers ---------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _persona_dir(persona: str) -> Path:
    if not persona or persona.startswith(("_", ".")) or "/" in persona or "\\" in persona:
        raise CtlError(f"Invalid persona name: {persona!r}")
    path = PERSONAS_DIR / persona
    if not path.is_dir():
        raise CtlError(f"Persona not found: {persona}")
    return path


def _persona_names() -> list[str]:
    if not PERSONAS_DIR.is_dir():
        return []
    return sorted(
        d.name for d in PERSONAS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(("_", "."))
    )


def _unit(persona: str) -> str:
    return f"pulse@{persona}.service"


def _run(cmd: list[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise CtlError(f"{' '.join(cmd)} failed: {detail}")
    return result


def _systemd_state(persona: str) -> dict:
    if not shutil.which("systemctl"):
        return {"active": "unknown", "enabled": "unknown"}
    result = _run(
        ["systemctl", "show", _unit(persona), "-p", "ActiveState", "-p", "UnitFileState"],
        check=False,
    )
    props = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    return {
        "active": props.get("ActiveState", "unknown"),
        "enabled": props.get("UnitFileState", "unknown"),
    }


def _read_status_file(persona: str) -> dict:
    path = PERSONAS_DIR / persona / "data" / "status.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _display_name(persona: str) -> str:
    persona_yaml = PERSONAS_DIR / persona / "persona.yaml"
    try:
        import yaml
        data = yaml.safe_load(persona_yaml.read_text(encoding="utf-8")) or {}
        return str(data.get("name") or persona)
    except Exception:
        return persona


def _persona_info(persona: str) -> dict:
    return {
        "name": persona,
        "display_name": _display_name(persona),
        "systemd": _systemd_state(persona),
        "status": _read_status_file(persona),
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return _run(["git", "-C", str(APP_ROOT), *args], check=check)


def _head_sha() -> str:
    return _git("rev-parse", "HEAD").stdout.strip()


# --- commands --------------------------------------------------------------

def cmd_list(_args: list[str]) -> dict:
    return {"personas": [_persona_info(p) for p in _persona_names()]}


def cmd_status(args: list[str]) -> dict:
    persona = _require(args, 1, "status <persona>")[0]
    _persona_dir(persona)
    return {"persona": _persona_info(persona)}


def _systemctl(verb: str, persona: str, block: bool = False) -> None:
    # pulse-unit is a root-owned wrapper that only accepts these verbs on
    # pulse@<name> units — sudoers allows exactly that, nothing else
    cmd = ["sudo", "-n", UNIT_WRAPPER, verb, persona]
    if not block:
        cmd.append("--no-block")
    _run(cmd)


def _make_control(verb: str):
    def command(args: list[str]) -> dict:
        persona = _require(args, 1, f"{verb} <persona>")[0]
        _persona_dir(persona)
        # --no-block: a graceful stop can take minutes; callers poll `status`
        _systemctl(verb, persona)
        return {"persona": persona, "requested": verb}
    return command


def cmd_log(args: list[str]) -> dict:
    persona = _require(args, 1, "log <persona> [lines]")[0]
    lines = int(args[1]) if len(args) > 1 else 80
    lines = max(1, min(lines, 500))
    path = LOGS_DIR / f"{persona}.log"
    if not path.exists():
        return {"persona": persona, "log": f"No log file yet: logs/{persona}.log"}
    data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"persona": persona, "log": "\n".join(data[-lines:])}


def _editable_path(persona: str, filename: str) -> Path:
    if filename not in EDITABLE_FILES:
        raise CtlError(f"Only {', '.join(EDITABLE_FILES)} can be read or written remotely.")
    return _persona_dir(persona) / filename


def cmd_get(args: list[str]) -> dict:
    persona, filename = _require(args, 2, "get <persona> <file>")
    path = _editable_path(persona, filename)
    data = path.read_bytes() if path.exists() else b""
    return {
        "persona": persona,
        "file": filename,
        "content": data.decode("utf-8"),
        "sha256": _sha256(data),
    }


def cmd_put(args: list[str]) -> dict:
    persona, filename, base_sha = _require(args, 3, "put <persona> <file> <base_sha256>")
    path = _editable_path(persona, filename)
    new_data = sys.stdin.buffer.read()

    current = path.read_bytes() if path.exists() else b""
    if _sha256(current) != base_sha:
        return {
            "ok": False,
            "conflict": True,
            "error": f"{filename} changed on the remote since it was opened.",
            "sha256": _sha256(current),
        }

    # Validate before touching anything — a broken YAML would stop the persona from starting
    from core import yaml_loader
    import yaml
    try:
        yaml_loader.safe_load(new_data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as e:
        raise CtlError(f"Invalid YAML, nothing written: {e}")

    from gui.backup import BackupManager
    backup = BackupManager(APP_ROOT).create_backup(persona, reason="remote-edit")

    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_bytes(new_data)
    os.chmod(tmp, 0o664)  # the pulse service user reads it via the shared group
    os.replace(tmp, path)
    return {
        "persona": persona,
        "file": filename,
        "sha256": _sha256(new_data),
        "backup": backup.get("path"),
    }


def _snapshot_one(persona: str) -> dict:
    """Copy a persona folder to snapshots/<persona>/<UTC stamp>/, consistently.

    SQLite files go through the online backup API, so this is safe while the
    persona is running. Secrets (.env) and SQLite side files are skipped.
    """
    source = _persona_dir(persona)
    stamp = _now().strftime("%Y%m%dT%H%M%SZ")
    persona_root = SNAPSHOTS_DIR / persona
    dest = persona_root / stamp
    suffix = 1
    while dest.exists():
        dest = persona_root / f"{stamp}_{suffix:02d}"
        suffix += 1
    partial = dest.with_name(dest.name + ".partial")
    partial.mkdir(parents=True)

    files = 0
    databases = 0
    for path in sorted(source.rglob("*")):
        rel = path.relative_to(source)
        if any(part in SNAPSHOT_SKIP_NAMES for part in rel.parts):
            continue
        if path.name.endswith(SNAPSHOT_SKIP_SUFFIXES) or path.is_symlink():
            continue
        target = partial / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in (".db", ".sqlite"):
            _backup_sqlite(path, target)
            databases += 1
        else:
            shutil.copy2(path, target)
        files += 1

    # Only a finished snapshot gets its final name, so a pull never sees half of one
    partial.rename(dest)
    (persona_root / "LATEST").write_text(dest.name, encoding="utf-8")
    _prune_snapshots(persona_root)
    return {"persona": persona, "snapshot": dest.name, "files": files, "databases": databases}


def _backup_sqlite(path: Path, target: Path) -> None:
    """Consistent copy of a live SQLite db (WAL included) via the online backup API."""
    src = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=30)
    dst = sqlite3.connect(str(target))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()


def _prune_snapshots(persona_root: Path) -> None:
    snapshots = sorted(
        (d for d in persona_root.iterdir() if d.is_dir() and not d.name.endswith(".partial")),
        key=lambda d: d.name,
        reverse=True,
    )
    for old in snapshots[SNAPSHOT_KEEP:]:
        shutil.rmtree(old)


def _shared_databases() -> dict[str, Path]:
    """Databases shared by several residents (LoR, Nebula), from PULSE_SHARED_DATABASES.

    Format: "name=/path/to.db,other=/path/other.db". Set on the nightly
    pulse-snapshot.service (a drop-in that also grants it access to those folders).
    """
    shared = {}
    for item in os.environ.get("PULSE_SHARED_DATABASES", "").split(","):
        name, sep, path = item.strip().partition("=")
        if sep and name.strip() and path.strip():
            shared[name.strip()] = Path(path.strip())
    return shared


def _snapshot_shared(name: str, path: Path) -> dict:
    """snapshots/_shared/<name>/<UTC stamp>/<file>, kept like persona snapshots."""
    if not path.is_file():
        raise CtlError(f"shared database {name} not found at {path}")
    root = SNAPSHOTS_DIR / "_shared" / name
    stamp = _now().strftime("%Y%m%dT%H%M%SZ")
    dest = root / stamp
    suffix = 1
    while dest.exists():
        dest = root / f"{stamp}_{suffix:02d}"
        suffix += 1
    partial = dest.with_name(dest.name + ".partial")
    partial.mkdir(parents=True)
    _backup_sqlite(path, partial / path.name)
    partial.rename(dest)
    (root / "LATEST").write_text(dest.name, encoding="utf-8")
    _prune_snapshots(root)
    return {"shared": name, "snapshot": dest.name, "source": str(path)}


def cmd_snapshot(args: list[str]) -> dict:
    if args:
        return {"snapshots": [_snapshot_one(args[0])]}
    result = {"snapshots": [_snapshot_one(p) for p in _persona_names()]}
    shared = []
    for name, path in _shared_databases().items():
        # One unreachable shared db must not cost the persona snapshots.
        try:
            shared.append(_snapshot_shared(name, path))
        except (CtlError, OSError, sqlite3.Error) as e:
            shared.append({"shared": name, "error": str(e)})
    if shared:
        result["shared"] = shared
    return result


def cmd_version(_args: list[str]) -> dict:
    history = []
    if DEPLOY_LOG.exists():
        for line in DEPLOY_LOG.read_text(encoding="utf-8").splitlines()[-10:]:
            try:
                history.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return {"sha": _head_sha(), "history": history}


def _wait_healthy(persona: str, since: datetime) -> str | None:
    """Return None once the persona reports running after `since`, else an error."""
    deadline = time.time() + HEALTH_TIMEOUT_SECONDS
    while time.time() < deadline:
        state = _systemd_state(persona)["active"]
        if state == "failed":
            return "systemd unit failed"
        status = _read_status_file(persona)
        updated = status.get("updated_at", "")
        try:
            fresh = datetime.fromisoformat(updated.replace("Z", "+00:00")) >= since
        except ValueError:
            fresh = False
        if fresh and status.get("phase") == "running":
            return None
        time.sleep(2)
    return f"not running after {HEALTH_TIMEOUT_SECONDS}s"


def _switch_to(sha: str, previous: str, running: list[str]) -> dict:
    """Stop running personas, check out `sha`, reinstall deps if needed, start them again."""
    for persona in running:
        _systemctl("stop", persona, block=True)  # graceful: SIGTERM → drain → exit

    _git("checkout", "--detach", sha)
    deps_changed = _git("diff", "--quiet", previous, sha, "--", "requirements.txt", check=False).returncode != 0
    if deps_changed:
        _run([str(VENV_PYTHON), "-m", "pip", "install", "-q", "-r", str(APP_ROOT / "requirements.txt")])

    started_at = _now()
    for persona in running:
        _systemctl("start", persona)
    failures = {}
    for persona in running:
        problem = _wait_healthy(persona, started_at)
        if problem:
            failures[persona] = problem
    return {"deps_changed": deps_changed, "failures": failures}


def _failure_log(persona: str, lines: int = 20) -> str:
    """Why a persona didn't come up: systemd journal first, then its Pulse log.

    A crash before Pulse switches to logs/<persona>.log only reaches the
    journal (stdout), so the Pulse log alone would show the previous run.
    """
    journal = _run(
        ["journalctl", "-u", _unit(persona), "-n", str(lines), "--no-pager", "-o", "cat"],
        check=False,
    ).stdout.strip() if shutil.which("journalctl") else ""
    pulse_log = cmd_log([persona, str(lines)])["log"]
    return f"--- journal ---\n{journal or '(empty)'}\n--- logs/{persona}.log ---\n{pulse_log}"


def cmd_deploy(args: list[str]) -> dict:
    sha = _require(args, 1, "deploy <sha>")[0]
    import fcntl  # box-side only

    with open(DEPLOY_LOCK, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise CtlError("Another deploy is already running.")

        # The box checkout is deploy-only. Local edits mean someone worked here —
        # surface it instead of overwriting their work.
        dirty = _git("status", "--porcelain", "--untracked-files=no").stdout.strip()
        if dirty:
            raise CtlError(f"Remote checkout has local changes, refusing to deploy:\n{dirty}")

        _git("fetch", "-q", "origin", "deploy")
        sha = _git("rev-parse", "--verify", f"{sha}^{{commit}}").stdout.strip()
        previous = _head_sha()
        if sha == previous:
            return {"sha": sha, "previous": previous, "changed": False}

        snapshots = cmd_snapshot([])["snapshots"]  # safety net before anything stops
        running = [p for p in _persona_names() if _systemd_state(p)["active"] == "active"]

        try:
            result = _switch_to(sha, previous, running)
        except CtlError:
            # Broke halfway (checkout or pip): put the old code back and restart the personas
            _git("checkout", "--detach", previous, check=False)
            for persona in running:
                _systemctl("start", persona)
            raise
        rolled_back = False
        logs = {}
        if result["failures"]:
            logs = {p: _failure_log(p) for p in result["failures"]}
            rollback = _switch_to(previous, sha, running)
            rolled_back = True
            result["rollback_failures"] = rollback["failures"]

        entry = {
            "at": _iso(_now()),
            "sha": previous if rolled_back else sha,
            "requested": sha,
            "previous": previous,
            "rolled_back": rolled_back,
            "restarted": running,
            "failures": result["failures"],
        }
        with open(DEPLOY_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        response = {
            **entry,
            "changed": not rolled_back,
            "deps_changed": result["deps_changed"],
            "snapshots": [s["snapshot"] for s in snapshots],
        }
        if rolled_back:
            response["ok"] = False
            response["error"] = (
                "Deploy failed health check for " + ", ".join(result["failures"])
                + f"; rolled back to {previous[:7]}."
            )
            response["logs"] = logs
        return response


COMMANDS = {
    "list": cmd_list,
    "status": cmd_status,
    "start": _make_control("start"),
    "stop": _make_control("stop"),
    "restart": _make_control("restart"),
    "log": cmd_log,
    "get": cmd_get,
    "put": cmd_put,
    "snapshot": cmd_snapshot,
    "version": cmd_version,
    "deploy": cmd_deploy,
}


def _require(args: list[str], count: int, usage: str) -> list[str]:
    if len(args) < count:
        raise CtlError(f"Usage: pulse_ctl.py {usage}")
    return args[:count]


def main(argv: list[str]) -> int:
    # lena (deploys, edits) and the pulse service user share group "pulse";
    # keep everything we create group-writable so either can prune/overwrite it
    os.umask(0o002)
    if not argv or argv[0] not in COMMANDS:
        result = {"ok": False, "error": f"Commands: {', '.join(COMMANDS)}"}
    else:
        try:
            result = {"ok": True, **COMMANDS[argv[0]](argv[1:])}
        except CtlError as e:
            result = {"ok": False, "error": str(e)}
        except Exception as e:  # report, don't traceback — the caller parses JSON
            result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(json.dumps(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
