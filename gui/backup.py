"""GUI-owned backup helpers for persona config files."""

from __future__ import annotations

import json
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
