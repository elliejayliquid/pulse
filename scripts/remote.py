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

Settings live in gui_data/remote.json (git-ignored). Without that file nothing
here runs and nothing in Pulse changes.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
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


def ssh(settings: dict, remote_cmd: str, stdin: bytes | None = None,
        timeout: float | None = 60, stream: bool = False) -> subprocess.CompletedProcess:
    cmd = _ssh_base(settings) + [_target(settings), remote_cmd]
    if stream:  # long-running (pip, first clone): show output as it happens
        return subprocess.run(cmd, timeout=timeout)
    return subprocess.run(cmd, input=stdin, capture_output=True, timeout=timeout)


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
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, env=env)
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
    """Replace personas/<persona>/ with the newest box snapshot, keeping the marker and .env."""
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
            stdout=subprocess.PIPE,
        )
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            for member in _safe_members(tar):
                tar.extract(member, tmp)
        if proc.wait() != 0:
            raise RemoteError(f"Copying snapshot {name} failed; {persona} left untouched.")

        # Only now touch the PC copy: everything downloaded fine
        for item in persona_dir.iterdir():
            if item.name in PULL_KEEP:
                continue
            shutil.rmtree(item) if item.is_dir() else item.unlink()
        for item in Path(tmp).iterdir():
            if item.name in PULL_KEEP:
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# --- CLI -------------------------------------------------------------------

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

    args = parser.parse_args(argv)
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
    except (RemoteError, subprocess.SubprocessError, OSError) as e:
        result = {"ok": False, "error": str(e)}

    if args.command == "log" and result.get("ok"):
        print(result["log"])
    else:
        print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
