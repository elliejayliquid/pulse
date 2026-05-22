"""GUI-owned backup helpers for persona config files."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


IDENTITY_FILES = ("persona.yaml", "persona.yml", "persona.json")


class BackupManager:
    """Creates and lists GUI backups without touching persona runtime data."""

    def __init__(
        self,
        root: Path | str,
        keep_per_persona: int = 20,
        max_age_days: int = 30,
    ):
        self.root = Path(root).resolve()
        self.personas_dir = self.root / "personas"
        self.backup_root = self.root / "gui_data" / "backups"
        self.keep_per_persona = keep_per_persona
        self.max_age_days = max_age_days

    def create_backup(self, persona: str, reason: str = "manual") -> dict[str, Any]:
        """Back up user-authored persona config files and prune old backups."""
        persona_dir = self._persona_dir(persona)
        sources = self._source_files(persona_dir)
        if not sources:
            raise FileNotFoundError(f"No config files found for persona: {persona}")

        created_at = datetime.now(timezone.utc)
        backup_dir = self._new_backup_dir(persona, created_at)
        backup_dir.mkdir(parents=True, exist_ok=False)

        files: list[str] = []
        source_paths: dict[str, str] = {}
        for source in sources:
            dest_name = source.name
            shutil.copy2(source, backup_dir / dest_name)
            files.append(dest_name)
            source_paths[dest_name] = self._rel(source)

        metadata = {
            "persona": persona,
            "created_at": self._iso_z(created_at),
            "reason": reason,
            "files": files,
            "source_paths": source_paths,
        }
        (backup_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        self.prune(persona)
        metadata["path"] = self._rel(backup_dir)
        return metadata

    def list_backups(self, persona: str) -> list[dict[str, Any]]:
        """Return newest backups first."""
        persona_root = self.backup_root / persona
        if not persona_root.is_dir():
            return []

        backups = []
        for backup_dir in sorted(persona_root.iterdir(), reverse=True):
            if not backup_dir.is_dir():
                continue
            metadata = self._read_metadata(backup_dir)
            metadata["path"] = self._rel(backup_dir)
            backups.append(metadata)
        return backups

    def restore(self, persona: str, backup_path: str) -> dict[str, Any]:
        """Restore a backup, creating a pre-restore backup first."""
        persona_dir = self._persona_dir(persona)
        backup_dir = self._backup_dir(persona, backup_path)
        metadata = self._read_metadata(backup_dir)
        files = [
            name for name in metadata.get("files", [])
            if isinstance(name, str) and name != "metadata.json"
        ]
        if not files:
            return {
                "ok": True,
                "changed": False,
                "restored_from": self._rel(backup_dir),
                "safety_backup": None,
                "files": [],
            }

        payloads = []
        source_paths = metadata.get("source_paths", {}) or {}
        for filename in files:
            source = backup_dir / filename
            if not source.is_file():
                continue
            dest = self._restore_destination(persona_dir, filename, source_paths.get(filename))
            payloads.append((filename, dest, source.read_bytes()))

        if not payloads:
            return {
                "ok": True,
                "changed": False,
                "restored_from": self._rel(backup_dir),
                "safety_backup": None,
                "files": [],
            }

        safety_backup = self.create_backup(persona, reason="pre-restore")
        restored = []
        for filename, dest, data in payloads:
            self._atomic_write_bytes(dest, data)
            restored.append(filename)

        return {
            "ok": True,
            "changed": True,
            "restored_from": self._rel(backup_dir),
            "safety_backup": safety_backup,
            "files": restored,
        }

    def prune(self, persona: str) -> None:
        """Keep recent backups and remove expired backups for one persona."""
        persona_root = self.backup_root / persona
        if not persona_root.is_dir():
            return

        backups = []
        for backup_dir in persona_root.iterdir():
            if backup_dir.is_dir():
                backups.append((self._created_at(backup_dir), backup_dir))
        backups.sort(key=lambda item: (item[0], item[1].name), reverse=True)

        cutoff = datetime.now(timezone.utc) - timedelta(days=self.max_age_days)
        for index, (created_at, backup_dir) in enumerate(backups):
            if index >= self.keep_per_persona or created_at < cutoff:
                shutil.rmtree(backup_dir)

    def _persona_dir(self, persona: str) -> Path:
        if not persona or persona == "__base__":
            raise ValueError("Choose a persona before creating a backup.")
        persona_dir = (self.personas_dir / persona).resolve()
        try:
            persona_dir.relative_to(self.personas_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"Invalid persona name: {persona}") from exc
        if not persona_dir.is_dir():
            raise FileNotFoundError(f"Persona not found: {persona}")
        return persona_dir

    def _backup_dir(self, persona: str, backup_path: str) -> Path:
        if not backup_path:
            raise FileNotFoundError("Backup path is required.")
        raw = Path(backup_path)
        candidate = raw if raw.is_absolute() else self.root / raw
        backup_dir = candidate.resolve()
        persona_root = (self.backup_root / persona).resolve()
        try:
            backup_dir.relative_to(persona_root)
        except ValueError as exc:
            raise ValueError("Backup path is outside this persona's backup folder.") from exc
        if not backup_dir.is_dir():
            raise FileNotFoundError(f"Backup not found: {backup_path}")
        return backup_dir

    def _restore_destination(self, persona_dir: Path, filename: str, source_path: str | None) -> Path:
        if source_path:
            dest = (self.root / source_path).resolve()
        elif filename == "config.yaml":
            dest = persona_dir / "config.yaml"
        elif filename in IDENTITY_FILES:
            dest = persona_dir / filename
        else:
            raise ValueError(f"Cannot restore unknown backup file: {filename}")

        try:
            dest.relative_to(persona_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"Restore destination is outside persona folder: {filename}") from exc
        if dest.parent.resolve() != persona_dir.resolve():
            raise ValueError(f"Restore destination must be a top-level persona file: {filename}")
        if dest.name != filename or dest.name not in ("config.yaml", *IDENTITY_FILES):
            raise ValueError(f"Cannot restore unsupported file: {filename}")
        return dest

    def _atomic_write_bytes(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def _source_files(self, persona_dir: Path) -> list[Path]:
        files = []
        config_path = persona_dir / "config.yaml"
        if config_path.exists():
            files.append(config_path)
        for filename in IDENTITY_FILES:
            path = persona_dir / filename
            if path.exists():
                files.append(path)
                break
        return files

    def _new_backup_dir(self, persona: str, created_at: datetime) -> Path:
        stamp = created_at.strftime("%Y%m%d_%H%M%S")
        persona_root = self.backup_root / persona
        candidate = persona_root / stamp
        suffix = 1
        while candidate.exists():
            candidate = persona_root / f"{stamp}_{suffix:02d}"
            suffix += 1
        return candidate

    def _read_metadata(self, backup_dir: Path) -> dict[str, Any]:
        metadata_path = backup_dir / "metadata.json"
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {
                "persona": backup_dir.parent.name,
                "created_at": self._iso_z(self._created_at(backup_dir)),
                "reason": "unknown",
                "files": [
                    p.name for p in backup_dir.iterdir()
                    if p.is_file() and p.name != "metadata.json"
                ],
                "source_paths": {},
            }

    def _created_at(self, backup_dir: Path) -> datetime:
        metadata_path = backup_dir / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            created_at = str(metadata.get("created_at", "")).replace("Z", "+00:00")
            return datetime.fromisoformat(created_at)
        except (OSError, ValueError, json.JSONDecodeError):
            try:
                return datetime.strptime(backup_dir.name[:15], "%Y%m%d_%H%M%S").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                return datetime.fromtimestamp(backup_dir.stat().st_mtime, timezone.utc)

    def _rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return str(path)

    @staticmethod
    def _iso_z(value: datetime) -> str:
        return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        )
