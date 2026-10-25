import ast
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from uuid import uuid4

import pytest

from anime_avatar_runtime import skyreels
from anime_avatar_runtime.config import RuntimeConfig
from anime_avatar_runtime.contracts import GenerationRequest
from anime_avatar_runtime.skyreels import GenerationCancelled, build_command


def test_image_and_video_modes_use_distinct_official_flags(tmp_path: Path) -> None:
    source = tmp_path / "reference with spaces.png"
    source.touch()
    common = {"repo": tmp_path, "python": tmp_path / "python", "output": tmp_path / "output"}
    image_argv = build_command(**common, image=source, prompt="literal $(text); 'quoted'")
    assert image_argv[image_argv.index("--image") + 1] == str(source)
    assert image_argv[image_argv.index("--prompt") + 1] == "literal $(text); 'quoted'"
    assert "--video_path" not in image_argv
    video_argv = build_command(**common, video=source, frames=257)
    assert "--video_path" in video_argv and "--image" not in video_argv
    assert video_argv[video_argv.index("--overlap_history") + 1] == "17"


@pytest.mark.parametrize("frames", [0, 96, 98])
def test_invalid_frame_counts_are_rejected(tmp_path: Path, frames: int) -> None:
    with pytest.raises(ValueError, match="Frames"):
        build_command(
            repo=tmp_path, python=tmp_path, output=tmp_path, image=tmp_path, frames=frames
        )


def test_missing_or_ambiguous_sources_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        build_command(repo=tmp_path, python=tmp_path, output=tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        build_command(
            repo=tmp_path, python=tmp_path, output=tmp_path, image=tmp_path, video=tmp_path
        )


def test_generated_flags_exist_in_official_entrypoint(tmp_path: Path) -> None:
    upstream = Path("vendor/SkyReels-V2/generate_video_df.py")
    if not upstream.exists():
        upstream = Path(".cache/research/generate_video_df.py")
    if not upstream.exists():
        pytest.skip("Optional upstream source absent; see docs/backend-setup.md")
    content = upstream.read_bytes().replace(b"\r\n", b"\n")
    blob = b"blob " + str(len(content)).encode() + b"\0" + content
    assert hashlib.sha1(blob).hexdigest() == "990ef610af1798b09aaba53d96201bc716f9d731"
    # Parse upstream source without importing torch, downloading weights or running model code.
    tree = ast.parse(upstream.read_text(encoding="utf-8"))
    flags = {
        arg.value
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "add_argument"
        for arg in call.args
        if isinstance(arg, ast.Constant)
        and isinstance(arg.value, str)
        and arg.value.startswith("--")
    }
    source = tmp_path / "input.png"
    source.touch()
    argv = build_command(
        repo=upstream.parent, python=tmp_path, output=tmp_path, image=source, end_image=source
    )
    assert {arg for arg in argv if arg.startswith("--")} <= flags


def test_real_process_timeout_terminates_child(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    script = (
        "import os, pathlib, time; "
        "pathlib.Path('child.pid').write_text(str(os.getpid())); time.sleep(60)"
    )
    with pytest.raises(TimeoutError):
        skyreels._run_process([sys.executable, "-c", script], cwd=tmp_path, timeout=1)
    if pid_file.exists():
        assert not process_is_running(int(pid_file.read_text()))


def process_is_running(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x00100000, False, pid)
        if not handle:
            return False
        try:
            return kernel.WaitForSingleObject(handle, 0) == 258
        finally:
            kernel.CloseHandle(handle)
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists() and stat.read_text().split(") ", 1)[1].startswith("Z"):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_real_process_cancellation_stops_worker_tree(tmp_path: Path) -> None:
    # This launches only sleeping Python processes, never a generation model.
    child = (
        "import os, pathlib, time; "
        "pathlib.Path('grandchild.pid').write_text(str(os.getpid())); time.sleep(60)"
    )
    script = (
        "import os, pathlib, subprocess, sys, time; "
        "pathlib.Path('parent.pid').write_text(str(os.getpid())); "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(60)"
    )
    cancel = Event()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            skyreels._run_process,
            [sys.executable, "-c", script],
            cwd=tmp_path,
            timeout=10,
            cancel_event=cancel,
        )
        try:
            deadline = time.monotonic() + 5
            while not (tmp_path / "grandchild.pid").exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert (tmp_path / "grandchild.pid").exists()
        finally:
            cancel.set()
        with pytest.raises(GenerationCancelled):
            future.result(timeout=15)
    for filename in ("parent.pid", "grandchild.pid"):
        pid = int((tmp_path / filename).read_text())
        deadline = time.monotonic() + 2
        while process_is_running(pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not process_is_running(pid)


def test_adapter_records_real_subprocess_artifact_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Exercise the subprocess/file contract with an explicit test fixture, not inference.
    script = tmp_path / "generate_video_df.py"
    script.write_text(
        "import pathlib, sys\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--outdir') + 1])\n"
        "(out / 'fixture.mp4').write_bytes(b'test artifact, not real video')\n",
        encoding="utf-8",
    )
    image = tmp_path / "input.png"
    image.write_bytes(b"test input")
    config = RuntimeConfig(
        skyreels_repo=tmp_path, skyreels_python=Path(sys.executable), skyreels_model=tmp_path
    )
    monkeypatch.setattr(skyreels, "check_backend", lambda *a, **kw: None)
    run_id, avatar_id = uuid4(), uuid4()
    result = skyreels.run_generation(
        config,
        GenerationRequest(seed=7),
        tmp_path / "run",
        run_id=run_id,
        avatar_id=avatar_id,
        image=image,
    )
    manifest = json.loads((result.output / "run.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["run_id"] == str(run_id)
    assert manifest["avatar_id"] == str(avatar_id)
    assert manifest["request"]["seed"] == 7
    assert manifest["inputs"]["image"]["sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
    assert manifest["hardware"] is None
    assert manifest["model_revision_verified"] is False
    assert result.videos[0].is_file()
    assert manifest["elapsed_seconds"] > 0


@pytest.mark.parametrize("cancelled", [False, True])
def test_preflight_failure_and_cancellation_keep_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancelled: bool
) -> None:
    image = tmp_path / "input.png"
    image.touch()
    config = RuntimeConfig(
        skyreels_repo=tmp_path, skyreels_python=Path(sys.executable), skyreels_model=tmp_path
    )
    cancel = Event()
    if cancelled:
        cancel.set()
    with pytest.raises((GenerationCancelled, FileNotFoundError)):
        skyreels.run_generation(
            config,
            GenerationRequest(),
            tmp_path / "run",
            run_id=uuid4(),
            image=image,
            cancel_event=cancel,
        )
    manifest = json.loads((tmp_path / "run" / "run.json").read_text())
    assert manifest["status"] == ("cancelled" if cancelled else "failed")
    assert manifest["upstream_revision_verified"] is False
    assert manifest["error"]
    assert "finished_at" in manifest
