"""CLI bridge to a pinned official SkyReels checkout, without importing its CUDA stack."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, TextIO
from uuid import UUID

from .config import RuntimeConfig
from .state import atomic_write

if TYPE_CHECKING:
    from .contracts import GenerationRequest, GenerationResult

UPSTREAM_REVISION = "9351d13152207cc04de780e055346b08ade0b851"
MODEL_ID = "Skywork/SkyReels-V2-DF-1.3B-540P"
MODEL_REVISION = "1100111771ba2d921f10e76991f54db9f73edb3d"
DEFAULT_PROMPT = (
    "An anime character calmly idles, breathing softly and blinking naturally, "
    "subtle hair movement, consistent identity and outfit, static camera, stable background."
)


def build_command(
    *,
    repo: Path,
    python: Path,
    output: Path,
    image: Path | None = None,
    video: Path | None = None,
    end_image: Path | None = None,
    model: str = MODEL_ID,
    frames: int = 97,
    seed: int = 42,
    prompt: str = DEFAULT_PROMPT,
) -> list[str]:
    if (image is None) == (video is None):
        raise ValueError("Provide exactly one image or continuation video")
    if end_image is not None and image is None:
        raise ValueError("End-frame control requires an input image")
    if frames < 97 or (frames - 1) % 4:
        raise ValueError("Frames must be 4n+1 and at least 97 for this baseline")
    if not 0 <= seed < 2**32:
        raise ValueError("Seed must be an unsigned 32-bit integer")
    if not prompt.strip():
        raise ValueError("Prompt must not be empty")
    for path in (image, video, end_image):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    command = [
        str(python.resolve()),
        str((repo / "generate_video_df.py").resolve()),
        "--model_id",
        model,
        "--resolution",
        "540P",
        "--num_frames",
        str(frames),
        "--base_num_frames",
        "97",
        "--overlap_history",
        "17",
        "--ar_step",
        "0",
        "--addnoise_condition",
        "20",
        "--guidance_scale",
        "5.0",
        "--shift",
        "5.0",
        "--inference_steps",
        "30",
        "--seed",
        str(seed),
        "--fps",
        "24",
        "--offload",
        "--outdir",
        str(output.resolve()),
        "--prompt",
        prompt,
    ]
    source = image if image is not None else video
    assert source is not None
    command.extend(["--image" if image is not None else "--video_path", str(source.resolve())])
    if end_image is not None:
        command.extend(["--end_image", str(end_image.resolve())])
    return command


class GenerationCancelled(RuntimeError):
    """The caller stopped a generation, including its backend preflight."""


def backend_unavailable_reason(config: RuntimeConfig) -> str | None:
    """Check paths only; presence does not establish backend health or CUDA support."""
    if config.skyreels_python is None:
        return "Set SKYREELS_PYTHON to the separate CUDA environment"
    if not config.skyreels_python.is_file():
        return "SKYREELS_PYTHON does not point to an existing interpreter"
    if not (config.skyreels_repo / "generate_video_df.py").is_file():
        return "Install the pinned SkyReels checkout at SKYREELS_REPO"
    if not config.skyreels_model.is_dir():
        return "Download the pinned model snapshot to SKYREELS_MODEL"
    return None


def _stop_process_tree(process: subprocess.Popen[str]) -> None:
    """Stop the owned backend process and its children before returning control."""
    if sys.platform == "win32":
        # CUDA launchers can spawn workers; terminating only the parent leaks them.
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if result.returncode and process.poll() is None:
                detail = result.stderr.decode(errors="replace").strip()
                raise RuntimeError(f"Could not terminate the backend process tree: {detail}")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def _run_process(
    argv: list[str],
    *,
    cwd: Path,
    timeout: float,
    cancel_event: Event | None = None,
    log: TextIO | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run an isolated process with bounded waiting and cancellation, without a shell."""
    if cancel_event is not None and cancel_event.is_set():
        raise GenerationCancelled("Generation cancelled")
    if timeout <= 0:
        raise TimeoutError("Generation timed out")
    deadline = time.monotonic() + timeout
    process_flags = 0
    if sys.platform == "win32":
        process_flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    # A file avoids PIPE deadlocks and unbounded captured output in memory.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as capture:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            stdout=log if log is not None else capture,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=sys.platform != "win32",
            creationflags=process_flags,
        )
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise GenerationCancelled("Generation cancelled")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Generation process timed out after {timeout:g} seconds")
                try:
                    code = process.wait(timeout=min(0.1, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            _stop_process_tree(process)
            raise
        capture.seek(0)
        return subprocess.CompletedProcess(argv, code, stdout=capture.read())


def check_backend(
    repo: Path,
    python: Path,
    *,
    timeout: float = 60,
    cancel_event: Event | None = None,
    log: TextIO | None = None,
) -> None:
    if not (repo / "generate_video_df.py").is_file() or not python.is_file():
        raise FileNotFoundError(
            "Install the pinned SkyReels checkout and its separate Python first"
        )
    deadline = time.monotonic() + timeout
    result = _run_process(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        cwd=repo,
        timeout=timeout,
        cancel_event=cancel_event,
    )
    if result.returncode:
        raise RuntimeError(f"Cannot inspect SkyReels source revision: {result.stdout.strip()}")
    revision = result.stdout.strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"Expected SkyReels revision {UPSTREAM_REVISION}, found {revision}")
    # Runs only when explicitly generating, never on API startup or dry-run.
    result = _run_process(
        [
            str(python.resolve()),
            "-c",
            "import torch; from moviepy.editor import VideoFileClip; "
            "assert torch.cuda.is_available(), 'SkyReels baseline requires an NVIDIA CUDA GPU'",
        ],
        cwd=repo,
        timeout=deadline - time.monotonic(),
        cancel_event=cancel_event,
        log=log,
    )
    if result.returncode:
        raise RuntimeError("SkyReels CUDA environment preflight failed; inspect backend.log")


def run_generation(
    config: RuntimeConfig,
    request: GenerationRequest,
    output: Path,
    *,
    run_id: UUID,
    avatar_id: UUID | None = None,
    image: Path | None = None,
    video: Path | None = None,
    end_image: Path | None = None,
    cancel_event: Event | None = None,
) -> GenerationResult:
    """Execute the real backend once and retain evidence for every attempted run."""
    from .contracts import GenerationResult

    output = output.resolve()
    # Never overwrite a previous experiment, including a failed or cancelled run.
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + config.generation_timeout_seconds
    manifest: dict[str, object] = {
        "run_id": str(run_id),
        "avatar_id": str(avatar_id) if avatar_id is not None else None,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_revision_verified": False,
        "model_id": MODEL_ID,
        "model_revision_expected": MODEL_REVISION,
        "model_revision_verified": False,
        "hardware": None,
        "request": request.model_dump(mode="json"),
        "config": config.model_dump(mode="json"),
        "inputs": {},
        "argv": [],
        "cwd": str(config.skyreels_repo),
    }
    manifest_path = output / "run.json"

    def save_manifest() -> None:
        atomic_write(manifest_path, json.dumps(manifest, indent=2).encode() + b"\n")

    save_manifest()
    try:
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelled("Generation cancelled")
        if config.skyreels_python is None:
            raise ValueError("Set --python or SKYREELS_PYTHON to the separate CUDA environment")
        argv = build_command(
            repo=config.skyreels_repo,
            python=config.skyreels_python,
            output=output,
            image=image,
            video=video,
            end_image=end_image,
            model=str(config.skyreels_model),
            frames=request.frames,
            seed=request.seed,
            prompt=request.prompt,
        )
        inputs: dict[str, object] = {}
        for name, path in (("image", image), ("video", video), ("end_image", end_image)):
            if path is not None:
                with path.open("rb") as source:
                    digest = hashlib.file_digest(source, "sha256").hexdigest()
                inputs[name] = {"path": str(path.resolve()), "sha256": digest}
        manifest.update(argv=argv, inputs=inputs)
        save_manifest()
        if not config.skyreels_model.is_dir():
            raise FileNotFoundError(
                "Download the pinned model snapshot first; see docs/backend-setup.md"
            )
        with (output / "backend.log").open("w", encoding="utf-8") as log:
            check_backend(
                config.skyreels_repo,
                config.skyreels_python,
                timeout=min(60, deadline - time.monotonic()),
                cancel_event=cancel_event,
                log=log,
            )
            manifest["upstream_revision_verified"] = True
            save_manifest()
            result = _run_process(
                argv,
                cwd=config.skyreels_repo,
                timeout=deadline - time.monotonic(),
                cancel_event=cancel_event,
                log=log,
            )
        manifest["returncode"] = result.returncode
        videos = sorted(p for p in output.glob("*.mp4") if p.is_file() and p.stat().st_size > 0)
        if result.returncode != 0 or not videos:
            raise RuntimeError(
                f"Backend failed or produced no video; inspect {output / 'backend.log'}"
            )
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelled("Generation cancelled")
        manifest.update(status="completed", videos=[str(path) for path in videos])
        return GenerationResult(
            run_id=run_id,
            avatar_id=avatar_id,
            output=output,
            videos=videos,
            elapsed_seconds=time.monotonic() - started,
        )
    except BaseException as exc:
        status = (
            "cancelled" if isinstance(exc, (GenerationCancelled, KeyboardInterrupt)) else "failed"
        )
        manifest.update(status=status, error=str(exc) or type(exc).__name__)
        raise
    finally:
        manifest.update(
            finished_at=datetime.now(UTC).isoformat(),
            elapsed_seconds=time.monotonic() - started,
        )
        save_manifest()
