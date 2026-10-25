"""Offline, verified retirement of a legacy data tree; no model dependencies."""

import hashlib
import json
import os
import shutil
import stat
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .state import AvatarState, atomic_write, load_state


class DataArchive(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    source: Path
    avatar_subdir: str
    created_at: str
    phase: Literal["planned", "prepared", "backed_up", "archived", "completed"] = "planned"
    backup_required: bool = True
    files: dict[str, str]
    directories: list[str]
    total_bytes: int = Field(ge=0)
    active_files: dict[str, str] = Field(default_factory=dict)
    active_directories: list[str] = Field(default_factory=list)
    reference_state: AvatarState | None = None

    @field_validator("avatar_subdir")
    @classmethod
    def relative_avatar(cls, value: str) -> str:
        _relative(value)
        return value

    @field_validator("files", "active_files")
    @classmethod
    def relative_files(cls, value: dict[str, str]) -> dict[str, str]:
        for name, digest in value.items():
            _relative(name)
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Invalid archive SHA-256")
        return value

    @field_validator("directories", "active_directories")
    @classmethod
    def relative_directories(cls, value: list[str]) -> list[str]:
        for name in value:
            _relative(name)
        return value


def _relative(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or ":" in value
        or str(path) != value
        or value == "."
    ):
        raise ValueError(f"Archive path must be relative: {value!r}")


def _inventory(root: Path) -> tuple[dict[str, str], list[str], int]:
    if root.is_symlink() or root.is_junction() or not root.is_dir():
        raise ValueError(f"Inventory root must be a real directory: {root}")
    files: dict[str, str] = {}
    directories: list[str] = []
    total = 0

    def fail(error: OSError) -> None:
        raise error

    for parent, dirs, names in os.walk(root, followlinks=False, onerror=fail):
        for name in sorted(dirs + names):
            path = Path(parent) / name
            info = path.lstat()
            if path.is_symlink() or path.is_junction():
                raise ValueError(f"Refusing linked data path: {path}")
            relative = path.relative_to(root).as_posix()
            _relative(relative)
            if stat.S_ISDIR(info.st_mode):
                directories.append(relative)
            elif stat.S_ISREG(info.st_mode):
                with path.open("rb") as stream:
                    files[relative] = hashlib.file_digest(stream, "sha256").hexdigest()
                after = path.stat()
                if (info.st_size, info.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError(f"Data changed while hashing: {path}")
                total += info.st_size
            else:
                raise ValueError(f"Not a regular data file: {path}")
    return files, sorted(directories), total


def _save(archive: Path, manifest: DataArchive) -> None:
    atomic_write(archive / "manifest.json", manifest.model_dump_json(indent=2).encode() + b"\n")


def _verify_zip(archive: Path, manifest: DataArchive) -> None:
    with zipfile.ZipFile(archive / "backup.zip") as backup:
        expected = set(manifest.files) | {name + "/" for name in manifest.directories}
        if set(backup.namelist()) != expected or len(backup.namelist()) != len(expected):
            raise ValueError("Backup entries do not match the manifest")
        if sum(i.file_size for i in backup.infolist()) != manifest.total_bytes:
            raise ValueError("Backup size does not match the manifest")
        for name, digest in manifest.files.items():
            with backup.open(name) as stream:
                actual = hashlib.sha256()
                while block := stream.read(1024**2):
                    actual.update(block)
                if actual.hexdigest() != digest:
                    raise ValueError(f"Backup SHA-256 mismatch: {name}")


def verify_archive(archive: Path, *, include_original: bool = True) -> DataArchive:
    """Validate archived bytes and, when requested at migration, the independent backup."""
    archive = archive.expanduser().resolve()
    manifest = DataArchive.model_validate_json((archive / "manifest.json").read_bytes())
    if manifest.backup_required:
        _verify_zip(archive, manifest)
    data = archive / "data"
    if include_original and manifest.phase in {"archived", "completed"} and not data.is_dir():
        raise FileNotFoundError("Original archived data directory is missing")
    if (
        include_original
        and data.exists()
        and _inventory(data)
        != (
            manifest.files,
            manifest.directories,
            manifest.total_bytes,
        )
    ):
        raise ValueError("Archived data does not match the manifest")
    return manifest


def migrate_data(
    source: Path,
    archive: Path,
    *,
    avatar_subdir: str = "avatar",
    apply: bool = False,
    backup: bool = True,
) -> DataArchive:
    """Archive a stopped data tree and retain only its explicitly selected trusted reference.

    Run without apply first. Any unreadable, linked, changing or corrupt data fails closed.
    A completed transaction is idempotent; an interrupted activation can be resumed.
    """
    _relative(avatar_subdir)
    if source.is_symlink() or source.is_junction() or archive.is_symlink() or archive.is_junction():
        raise ValueError("Migration roots must not be links")
    source = source.expanduser().resolve()
    archive = archive.expanduser().resolve()
    if source == source.parent or source.is_relative_to(archive) or archive.is_relative_to(source):
        raise ValueError("Source and archive must be separate, non-nested directories")
    if source.anchor != archive.anchor:
        raise ValueError("Archive must be on the source filesystem for atomic cutover")
    existing = archive
    while not existing.exists():
        existing = existing.parent
    source_volume = source if source.exists() else source.parent
    if source_volume.stat().st_dev != existing.stat().st_dev:
        raise ValueError("Archive must be on the source filesystem for atomic cutover")
    if (source / ".git").exists():
        raise ValueError("Refusing to migrate a repository root")
    if (archive / "manifest.json").is_file():
        manifest = DataArchive.model_validate_json((archive / "manifest.json").read_bytes())
        if manifest.source != source or manifest.avatar_subdir != avatar_subdir:
            raise ValueError("Archive belongs to a different source or avatar directory")
        if manifest.phase == "completed":
            return verify_archive(archive)
        if not apply:
            return manifest
        if manifest.phase not in {"prepared", "backed_up", "archived"}:
            raise ValueError("Incomplete backup; source is untouched, use a new archive directory")
        if manifest.backup_required:
            _verify_zip(archive, manifest)
        if (archive / "data").exists():
            verify_archive(archive)
    else:
        if not source.is_dir():
            raise FileNotFoundError(f"Data directory does not exist: {source}")
        if (source / "migration.json").exists():
            raise ValueError("This data tree was already migrated; verify its existing archive")
        files, directories, total = _inventory(source)
        reference_state = load_state(source / avatar_subdir)
        manifest = DataArchive(
            source=source,
            avatar_subdir=avatar_subdir,
            created_at=datetime.now(UTC).isoformat(),
            files=files,
            directories=directories,
            total_bytes=total,
            reference_state=reference_state,
            backup_required=backup,
        )
        if not apply:
            return manifest
        if archive.exists():
            raise FileExistsError(f"Archive destination already exists: {archive}")
        retained_bytes = (
            sum(
                (source / avatar_subdir / name).stat().st_size
                for name in (
                    "state.json",
                    f"references/{reference_state.reference_sha256}.png",
                )
            )
            if reference_state is not None
            else 0
        )
        if (
            shutil.disk_usage(source.parent).free
            < (total if backup else 0) + retained_bytes + 16 * 1024**2
        ):
            raise OSError("Insufficient space for migration staging and requested backup")
        archive.mkdir(parents=True)
        staged_avatar = archive / "active" / avatar_subdir
        staged_avatar.mkdir(parents=True)
        if reference_state is not None:
            old_avatar = source / avatar_subdir
            reference = Path("references") / f"{reference_state.reference_sha256}.png"
            (staged_avatar / "references").mkdir()
            shutil.copyfile(old_avatar / "state.json", staged_avatar / "state.json")
            shutil.copyfile(old_avatar / reference, staged_avatar / reference)
            if load_state(staged_avatar) != reference_state:
                raise ValueError("Staged reference does not match the original checkpoint")
        atomic_write(
            archive / "active" / "migration.json",
            json.dumps({"schema_version": 1, "archive": str(archive)}).encode() + b"\n",
        )
        manifest.active_files, manifest.active_directories, _ = _inventory(archive / "active")
        _save(archive, manifest)
        if backup:
            with zipfile.ZipFile(
                archive / "backup.zip", "w", compression=zipfile.ZIP_STORED
            ) as zipped:
                for name in manifest.directories:
                    zipped.write(source / name, name + "/")
                for name in manifest.files:
                    zipped.write(source / name, name)
            _verify_zip(archive, manifest)
        manifest.phase = "backed_up" if backup else "prepared"
        _save(archive, manifest)

    original = archive / "data"
    staged = archive / "active"
    if not original.exists():
        if _inventory(staged)[:2] != (manifest.active_files, manifest.active_directories):
            raise ValueError("Staged active data is corrupt; source was not moved")
        if _inventory(source) != (manifest.files, manifest.directories, manifest.total_bytes):
            raise RuntimeError("Source changed since inventory; cutover refused")
        source.rename(original)
    try:
        # Verify before activating even when resuming after the source rename.
        if _inventory(original) != (manifest.files, manifest.directories, manifest.total_bytes):
            raise ValueError("Archived data does not match the manifest")
        manifest.phase = "archived"
        _save(archive, manifest)
        if not source.exists():
            if _inventory(staged)[:2] != (manifest.active_files, manifest.active_directories):
                raise ValueError("Staged active data is corrupt; original remains in the archive")
            staged.rename(source)
        elif _inventory(source)[:2] != (manifest.active_files, manifest.active_directories):
            raise ValueError(
                "Active destination conflicts with this migration; nothing overwritten"
            )
    except (OSError, ValueError):
        if not source.exists():
            original.rename(source)
        raise
    verify_archive(archive)
    manifest.phase = "completed"
    _save(archive, manifest)
    return manifest


def restore_data(archive: Path, output: Path) -> DataArchive:
    """Restore verified ZIP or original archive bytes without overwriting existing data."""
    archive = archive.expanduser().resolve()
    if output.is_symlink() or output.is_junction():
        raise ValueError("Restore destination must not be a link")
    output = output.expanduser().resolve()
    if output.exists() or output.is_relative_to(archive) or archive.is_relative_to(output):
        raise FileExistsError("Restore destination must be new and outside the archive")
    manifest = verify_archive(archive, include_original=False)
    if not manifest.backup_required:
        verify_archive(archive)
        if not (archive / "data").is_dir():
            raise FileNotFoundError("Original archived data directory is missing")
    staged = output.with_name(f"{output.name}.restore-{uuid4().hex}")
    staged.mkdir(parents=True)
    if manifest.backup_required:
        with zipfile.ZipFile(archive / "backup.zip") as backup:
            for name in manifest.directories:
                (staged / name).mkdir(parents=True, exist_ok=True)
            for name in manifest.files:
                target = staged / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with backup.open(name) as incoming, target.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
    else:
        for name in manifest.directories:
            (staged / name).mkdir(parents=True, exist_ok=True)
        for name in manifest.files:
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(archive / "data" / name, target)
    if _inventory(staged) != (manifest.files, manifest.directories, manifest.total_bytes):
        raise ValueError("Restored data failed verification; destination was not activated")
    # rename refuses a nonempty destination; recheck also protects an empty concurrent directory.
    if output.exists():
        raise FileExistsError("Restore destination appeared during extraction")
    staged.rename(output)
    return manifest
