import json
import subprocess
import sys
from pathlib import Path

import pytest

from anime_avatar_runtime import cli


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
    monkeypatch.setattr(cli, "check_backend", lambda *_: None)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, returncode)
    )
    assert cli.main() == 1
    manifest = json.loads((tmp_path / "run" / "run.json").read_text())
    assert manifest["status"] == "failed"
    assert "finished_at" in manifest


def test_existing_experiment_is_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", arguments(tmp_path))
    monkeypatch.setattr(cli, "check_backend", lambda *_: None)
    output = tmp_path / "run"
    output.mkdir()
    (output / "run.json").write_text("original", encoding="utf-8")
    assert cli.main() == 1
    assert (output / "run.json").read_text() == "original"
