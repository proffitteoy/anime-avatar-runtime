import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from anime_avatar_runtime import cli, migration
from anime_avatar_runtime.state import initialize, load_state


def old_data(tmp_path: Path, *, initialized: bool = True) -> tuple[Path, Path]:
    source = tmp_path / "var"
    avatar = source / "avatar"
    avatar.mkdir(parents=True)
    (source / "old-experiment").mkdir()
    (source / "old-experiment" / "run.json").write_text('{"status":"failed"}')
    (source / "empty-experiment").mkdir()
    (avatar / "anchor-cache").mkdir()
    (avatar / "anchor-cache" / "private.bin").write_bytes(b"old cached features")
    if initialized:
        image = io.BytesIO()
        Image.new("RGB", (12, 16), (25, 50, 75)).save(image, format="PNG")
        initialize(avatar, image.getvalue())
    return source, tmp_path / "archive"


def test_plan_is_read_only_and_cutover_keeps_only_the_trusted_reference(tmp_path: Path) -> None:
    source, archive = old_data(tmp_path)
    original = migration._inventory(source)
    state = load_state(source / "avatar")
    plan = migration.migrate_data(source, archive)
    assert plan.phase == "planned"
    assert not archive.exists()
    assert migration._inventory(source) == original

    result = migration.migrate_data(source, archive, apply=True)
    assert result.phase == "completed"
    assert load_state(source / "avatar") == state
    assert not (source / "old-experiment").exists()
    assert not (source / "avatar" / "anchor-cache").exists()
    assert migration._inventory(archive / "data") == original
    assert migration.verify_archive(archive).phase == "completed"
    assert migration.migrate_data(source, archive, apply=True) == result
    with pytest.raises(ValueError, match="already migrated"):
        migration.migrate_data(source, tmp_path / "second-archive", apply=True)


def test_empty_active_avatar_does_not_import_a_test_checkpoint(tmp_path: Path) -> None:
    source, archive = old_data(tmp_path, initialized=False)
    other, _ = old_data(tmp_path / "fixtures")
    (source / "old-experiment" / "state.json").write_bytes(
        (other / "avatar" / "state.json").read_bytes()
    )
    result = migration.migrate_data(source, archive, apply=True)
    assert result.reference_state is None
    assert load_state(source / "avatar") is None
    assert list((source / "avatar").iterdir()) == []
    assert (archive / "data" / "old-experiment" / "state.json").is_file()


def test_corrupt_reference_fails_before_creating_an_archive(tmp_path: Path) -> None:
    source, archive = old_data(tmp_path)
    reference = next((source / "avatar" / "references").iterdir())
    reference.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        migration.migrate_data(source, archive, apply=True)
    assert reference.read_bytes() == b"corrupt"
    assert not archive.exists()


def test_unreadable_data_is_not_silently_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, archive = old_data(tmp_path)
    original_open = Path.open

    def deny(path, *args, **kwargs):
        if path.name == "private.bin":
            raise PermissionError("explicit unreadable data")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny)
    with pytest.raises(PermissionError, match="unreadable"):
        migration.migrate_data(source, archive, apply=True)
    assert source.is_dir()
    assert not archive.exists()


def test_linked_data_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, archive = old_data(tmp_path)
    monkeypatch.setattr(Path, "is_junction", lambda path: path.name == "anchor-cache")
    with pytest.raises(ValueError, match="linked"):
        migration.migrate_data(source, archive, apply=True)
    assert not archive.exists()


def test_disk_budget_is_checked_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, archive = old_data(tmp_path)
    usage = migration.shutil.disk_usage(tmp_path)._replace(free=0)
    monkeypatch.setattr(migration.shutil, "disk_usage", lambda _: usage)
    with pytest.raises(OSError, match="Insufficient space"):
        migration.migrate_data(source, archive, apply=True)
    assert not archive.exists()


def test_source_changed_during_backup_is_not_cut_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, archive = old_data(tmp_path)
    verify = migration._verify_zip

    def change_after_verification(*args):
        verify(*args)
        (source / "new-write.txt").write_text("concurrent writer")

    monkeypatch.setattr(migration, "_verify_zip", change_after_verification)
    with pytest.raises(RuntimeError, match="Source changed"):
        migration.migrate_data(source, archive, apply=True)
    assert (source / "new-write.txt").read_text() == "concurrent writer"
    assert not (archive / "data").exists()


@pytest.mark.parametrize("backup", [True, False])
def test_interrupted_activation_can_be_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backup: bool
) -> None:
    source, archive = old_data(tmp_path)
    original = migration._inventory(source)
    rename = Path.rename

    def fail_activation(path, target):
        if path == archive / "active":
            raise OSError("explicit activation failure")
        return rename(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rename", fail_activation)
        with pytest.raises(OSError, match="activation"):
            migration.migrate_data(source, archive, backup=backup, apply=True)
    assert migration._inventory(source) == original
    assert not (archive / "data").exists()
    assert migration.migrate_data(source, archive, apply=True).phase == "completed"
    assert source.is_dir()


def test_failed_terminal_write_is_not_reported_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, archive = old_data(tmp_path)
    save = migration._save

    def fail_completion(path, manifest):
        if manifest.phase == "completed":
            raise OSError("explicit manifest write failure")
        save(path, manifest)

    with monkeypatch.context() as patch:
        patch.setattr(migration, "_save", fail_completion)
        with pytest.raises(OSError, match="manifest write"):
            migration.migrate_data(source, archive, apply=True)
    assert json.loads((archive / "manifest.json").read_text())["phase"] == "archived"
    assert migration.migrate_data(source, archive, apply=True).phase == "completed"


def test_restore_uses_independent_backup_and_never_overwrites(tmp_path: Path) -> None:
    source, archive = old_data(tmp_path)
    original = migration._inventory(source)
    migration.migrate_data(source, archive, apply=True)
    (archive / "data" / "old-experiment" / "run.json").write_text("damaged original archive")
    with pytest.raises(ValueError, match="Archived data"):
        migration.verify_archive(archive)
    restored = tmp_path / "restored"
    migration.restore_data(archive, restored)
    assert migration._inventory(restored) == original
    with pytest.raises(FileExistsError, match="must be new"):
        migration.restore_data(archive, restored)
    assert migration._inventory(restored) == original


def test_corrupt_backup_cannot_be_restored(tmp_path: Path) -> None:
    source, archive = old_data(tmp_path)
    migration.migrate_data(source, archive, apply=True)
    (archive / "backup.zip").write_bytes(b"not a backup")
    with pytest.raises(migration.zipfile.BadZipFile):
        migration.restore_data(archive, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_backup_restores_even_if_original_archive_is_lost(tmp_path: Path) -> None:
    source, archive = old_data(tmp_path)
    original = migration._inventory(source)
    migration.migrate_data(source, archive, apply=True)
    (archive / "data").rename(tmp_path / "detached-original")
    with pytest.raises(FileNotFoundError, match="missing"):
        migration.verify_archive(archive)
    migration.restore_data(archive, tmp_path / "restored")
    assert migration._inventory(tmp_path / "restored") == original


@pytest.mark.parametrize("relative", ["../avatar", "/avatar", "C:/avatar", "a\\b", "."])
def test_unsafe_avatar_paths_fail_before_writing(tmp_path: Path, relative: str) -> None:
    source, archive = old_data(tmp_path)
    with pytest.raises(ValueError, match="relative"):
        migration.migrate_data(source, archive, avatar_subdir=relative, apply=True)
    assert not archive.exists()


def test_nested_archive_is_rejected(tmp_path: Path) -> None:
    source, _ = old_data(tmp_path)
    with pytest.raises(ValueError, match="non-nested"):
        migration.migrate_data(source, source / "archive", apply=True)
    assert not (source / "archive").exists()


def test_cross_filesystem_archive_is_rejected_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = old_data(tmp_path)
    other = tmp_path / "other-volume"
    other.mkdir()
    actual_stat = Path.stat

    def different_device(path, *args, **kwargs):
        result = actual_stat(path, *args, **kwargs)
        if path == other:
            fields = list(result)
            fields[2] += 1
            return os.stat_result(fields)
        return result

    monkeypatch.setattr(Path, "stat", different_device)
    with pytest.raises(ValueError, match="source filesystem"):
        migration.migrate_data(source, other / "archive", apply=True)
    assert not (other / "archive").exists()


def test_cli_plan_does_not_load_environment_or_mutate_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source, archive = old_data(tmp_path)
    monkeypatch.setenv("AVATAR_GENERATION_TIMEOUT_SECONDS", "invalid")
    monkeypatch.setattr(
        sys,
        "argv",
        ["avatar-runtime", "migrate-data", "--source", str(source), "--archive", str(archive)],
    )
    assert cli.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["phase"] == "planned"
    assert report["reference_initialized"] is True
    assert not archive.exists()


def test_data_cli_has_no_observer_or_model_imports() -> None:
    script = (
        "import sys; from anime_avatar_runtime.cli import main; "
        "assert 'anime_avatar_runtime.observers' not in sys.modules; "
        "assert 'anime_avatar_runtime.skyreels' not in sys.modules; "
        "assert 'anime_avatar_runtime.contracts' not in sys.modules; "
        "assert not {'torch', 'onnxruntime', 'numpy'} & set(sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cli_bad_backup_is_a_nonzero_error_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source, archive = old_data(tmp_path)
    migration.migrate_data(source, archive, apply=True)
    (archive / "backup.zip").write_bytes(b"corrupt")
    monkeypatch.setattr(sys, "argv", ["avatar-runtime", "verify-data", "--archive", str(archive)])
    assert cli.main() == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("error:")
    assert "Traceback" not in captured.err


def test_no_backup_cutover_and_restore_keep_verified_originals(tmp_path: Path) -> None:
    source, archive = old_data(tmp_path)
    original = migration._inventory(source)
    result = migration.migrate_data(source, archive, backup=False, apply=True)
    assert result.backup_required is False
    assert not (archive / "backup.zip").exists()
    assert migration.verify_archive(archive) == result
    # Retrying honors the recorded choice, without silently creating a backup.
    assert migration.migrate_data(source, archive, apply=True) == result
    restored = tmp_path / "restored"
    migration.restore_data(archive, restored)
    assert migration._inventory(restored) == original
    assert migration._inventory(archive / "data") == original
    assert not (archive / "backup.zip").exists()


@pytest.mark.parametrize("backup", [True, False])
def test_process_interruption_after_source_rename_can_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backup: bool
) -> None:
    source, archive = old_data(tmp_path)
    original = migration._inventory(source)
    save = migration._save

    def crash_after_rename(path, manifest):
        if manifest.phase == "archived":
            raise KeyboardInterrupt
        save(path, manifest)

    with monkeypatch.context() as patch:
        patch.setattr(migration, "_save", crash_after_rename)
        with pytest.raises(KeyboardInterrupt):
            migration.migrate_data(source, archive, backup=backup, apply=True)
    assert not source.exists()
    assert migration._inventory(archive / "data") == original
    assert migration.migrate_data(source, archive, backup=backup, apply=True).phase == "completed"


def test_legacy_checkpoint_is_preserved_byte_for_byte_through_api_restart(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from anime_avatar_runtime.api import create_app

    source, archive = old_data(tmp_path)
    expected = load_state(source / "avatar")
    before = (source / "avatar" / "state.json").read_bytes()
    migration.migrate_data(source, archive, backup=False, apply=True)
    with TestClient(create_app(source / "avatar")) as client:
        assert client.get("/avatar/state").json() == expected.model_dump(mode="json")
        assert client.get("/runtime/status").json()["mode"] == "paused"
        assert client.post("/runtime/resume").status_code == 200
        assert client.post("/avatar/generate", json={}).status_code == 503
        assert client.post("/observer/observe", content=b"frame").status_code == 503
        schema = client.get("/openapi.json").json()
        assert schema["paths"]["/avatar/generate"]["post"]["deprecated"] is True
    with TestClient(create_app(source / "avatar")) as restarted:
        assert restarted.get("/avatar/state").json() == expected.model_dump(mode="json")
        assert restarted.get("/runtime/status").json()["mode"] == "paused"
    assert (source / "avatar" / "state.json").read_bytes() == before
    assert (archive / "data" / "avatar" / "state.json").read_bytes() == before


def test_no_backup_corruption_cannot_be_restored(tmp_path: Path) -> None:
    source, archive = old_data(tmp_path)
    migration.migrate_data(source, archive, backup=False, apply=True)
    (archive / "data" / "old-experiment" / "run.json").write_text("corrupt")
    with pytest.raises(ValueError, match="Archived data"):
        migration.restore_data(archive, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


@pytest.mark.parametrize("backup", [False, True])
def test_prepared_data_tampering_cannot_remove_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backup: bool
) -> None:
    source, archive = old_data(tmp_path)
    original = migration._inventory(source)
    save = migration._save

    def corrupt_staging(path, manifest):
        save(path, manifest)
        if manifest.phase in {"prepared", "backed_up"}:
            (archive / "active" / "avatar" / "state.json").write_text("corrupt")

    monkeypatch.setattr(migration, "_save", corrupt_staging)
    with pytest.raises(ValueError, match="Staged active"):
        migration.migrate_data(source, archive, backup=backup, apply=True)
    assert migration._inventory(source) == original
    assert not (archive / "data").exists()


def test_no_backup_source_change_prevents_cutover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, archive = old_data(tmp_path)
    save = migration._save

    def change_source(path, manifest):
        save(path, manifest)
        if manifest.phase == "prepared":
            (source / "new-write.txt").write_text("concurrent writer")

    monkeypatch.setattr(migration, "_save", change_source)
    with pytest.raises(RuntimeError, match="Source changed"):
        migration.migrate_data(source, archive, backup=False, apply=True)
    assert (source / "new-write.txt").read_text() == "concurrent writer"
    assert not (archive / "data").exists()


def test_cli_no_backup_applies_without_loading_model_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source, archive = old_data(tmp_path)
    monkeypatch.setenv("AVATAR_GENERATION_TIMEOUT_SECONDS", "invalid")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "avatar-runtime",
            "migrate-data",
            "--source",
            str(source),
            "--archive",
            str(archive),
            "--no-backup",
            "--apply",
        ],
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["backup_required"] is False
    assert not (archive / "backup.zip").exists()
