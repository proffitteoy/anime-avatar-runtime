import json
import subprocess
import sys
from pathlib import Path

import pytest

from anime_avatar_runtime import cli, skyreels


def arguments(tmp_path: Path) -> list[str]:
    image = tmp_path / "input.png"
    image.touch()
    model = tmp_path / "model"
    model.mkdir()
    return [
        "avatar-runtime",
        "generate",
        "--image",
        str(image),
        "--python",
        sys.executable,
        "--repo",
        str(tmp_path),
        "--model",
        str(model),
        "--output",
        str(tmp_path / "run"),
    ]


def test_dry_run_does_not_launch_backend_or_write_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", arguments(tmp_path) + ["--dry-run"])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["executed"] is False
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("returncode", [0, 7])
def test_failure_or_missing_video_is_recorded_as_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
) -> None:
    monkeypatch.setattr(sys, "argv", arguments(tmp_path))
    monkeypatch.setattr(skyreels, "check_backend", lambda *a, **kw: None)
    monkeypatch.setattr(
        skyreels, "_run_process", lambda *a, **kw: subprocess.CompletedProcess(a, returncode)
    )
    assert cli.main() == 1
    manifest = json.loads((tmp_path / "run" / "run.json").read_text())
    assert manifest["status"] == "failed"
    assert "finished_at" in manifest


def test_existing_experiment_is_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", arguments(tmp_path))
    monkeypatch.setattr(skyreels, "check_backend", lambda *a, **kw: None)
    output = tmp_path / "run"
    output.mkdir()
    (output / "run.json").write_text("original", encoding="utf-8")
    assert cli.main() == 1
    assert (output / "run.json").read_text() == "original"


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf", "invalid"])
def test_invalid_configuration_fails_before_launch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], timeout: str
) -> None:
    monkeypatch.setenv("AVATAR_GENERATION_TIMEOUT_SECONDS", timeout)
    monkeypatch.setattr(sys, "argv", ["avatar-runtime", "doctor"])
    assert cli.main() == 1
    assert "generation_timeout_seconds" in capsys.readouterr().err


def test_environment_config_is_resolved_and_exported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AVATAR_DATA_DIR", "private-avatar")
    monkeypatch.setenv("SKYREELS_MODEL", "weights")
    monkeypatch.setenv("AVATAR_GENERATION_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setattr(sys, "argv", ["avatar-runtime", "doctor"])
    assert cli.main() == 0
    config = json.loads(capsys.readouterr().out)["config"]
    assert config["data_dir"] == str(tmp_path / "private-avatar")
    assert config["skyreels_model"] == str(tmp_path / "weights")
    assert config["generation_timeout_seconds"] == 12.5


def test_blank_backend_path_is_not_resolved_to_current_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKYREELS_PYTHON", "")
    monkeypatch.setattr(sys, "argv", ["avatar-runtime", "doctor"])
    assert cli.main() == 1
