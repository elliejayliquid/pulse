#!/bin/bash
# One-time root setup of a Linux home server for remote Pulse personas.
# See designs/remote_host.md. Safe to re-run.
#
# Usually driven from the PC:  python scripts/remote.py init ...  (copies this
# folder to the box and tells you the exact command), then on the box:
#
#   sudo bash setup_box.sh <deploy-user> [timezone]
#   e.g. sudo bash setup_box.sh lena Pacific/Auckland
#
# What it does: installs git/venv/ffmpeg, creates the "pulse" system user,
# lays out /opt/pulse, creates the bare repo the PC pushes to, installs the
# systemd units, and a sudoers rule that lets <deploy-user> run ONLY
# /usr/local/sbin/pulse-unit (start/stop/restart pulse@<persona>) without a password.
set -euo pipefail

DEPLOY_USER="${1:-${SUDO_USER:-}}"
TIMEZONE="${2:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/opt/pulse

if [[ $EUID -ne 0 ]]; then
  echo "Run with sudo." >&2
  exit 1
fi
if [[ -z "$DEPLOY_USER" ]] || ! id "$DEPLOY_USER" >/dev/null 2>&1; then
  echo "usage: sudo bash setup_box.sh <deploy-user> [timezone]" >&2
  exit 2
fi

echo "== Packages"
apt-get update -qq
# --no-install-recommends: ffmpeg otherwise pulls in GTK, mesa and friends on a headless box
apt-get install -y -qq --no-install-recommends git python3-venv ffmpeg sqlite3

echo "== Users"
if ! id pulse >/dev/null 2>&1; then
  useradd --system --user-group --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin pulse
fi
usermod -aG pulse "$DEPLOY_USER"

echo "== Layout ($ROOT)"
# setgid dirs: everything created inside stays in group "pulse"
install -d -o "$DEPLOY_USER" -g pulse -m 2775 "$ROOT" "$ROOT/cache" "$ROOT/snapshots"
if [[ ! -d "$ROOT/repo.git" ]]; then
  sudo -u "$DEPLOY_USER" git init -q --bare "$ROOT/repo.git"
fi

echo "== systemd units"
install -m 0755 -o root -g root "$HERE/pulse-unit" /usr/local/sbin/pulse-unit
install -m 0644 -o root -g root "$HERE/pulse@.service" /etc/systemd/system/pulse@.service
install -m 0644 -o root -g root "$HERE/pulse-snapshot.service" /etc/systemd/system/pulse-snapshot.service
install -m 0644 -o root -g root "$HERE/pulse-snapshot.timer" /etc/systemd/system/pulse-snapshot.timer
systemctl daemon-reload
systemctl enable --now pulse-snapshot.timer

echo "== sudoers"
SUDOERS=/etc/sudoers.d/pulse
TMP_SUDOERS="$(mktemp)"
echo "$DEPLOY_USER ALL=(root) NOPASSWD: /usr/local/sbin/pulse-unit" > "$TMP_SUDOERS"
visudo -cf "$TMP_SUDOERS" >/dev/null
install -m 0440 -o root -g root "$TMP_SUDOERS" "$SUDOERS"
rm -f "$TMP_SUDOERS"

if [[ -n "$TIMEZONE" ]]; then
  echo "== Timezone -> $TIMEZONE"
  # Pulse uses local system time for reminders and heartbeats
  timedatectl set-timezone "$TIMEZONE"
fi
echo "Timezone: $(timedatectl show -p Timezone --value)"

echo
echo "Done. $DEPLOY_USER must log in again to pick up the 'pulse' group."
echo "Next, on the PC:  python scripts/remote.py bootstrap"
