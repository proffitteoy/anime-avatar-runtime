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
from typing import TYPE_CHECKING, Final, TextIO
from uuid import UUID

from .config import RuntimeConfig
from .state import atomic_write

if TYPE_CHECKING:
    from .contracts import GenerationRequest, GenerationResult, GenerationWindow

UPSTREAM_REVISION = "9351d13152207cc04de780e055346b08ade0b851"
MODEL_ID = "Skywork/SkyReels-V2-DF-1.3B-540P"
MODEL_REVISION = "1100111771ba2d921f10e76991f54db9f73edb3d"
VENDOR_PATCH_NAME = "skyreels-v2-8gb-memory.patch"
VIDEO_FPS: Final = 24
WINDOW_OVERLAP: Final = 17


def vendor_patch_sha256() -> str | None:
    """Fingerprint the committed vendor patch, so run.json records what was applied."""
    patch_path = Path(__file__).resolve().parents[2] / "patches" / VENDOR_PATCH_NAME
    if not patch_path.is_file():
        return None
    with patch_path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


DEFAULT_PROMPT = (
    "An anime character standing still and idling naturally: slow gentle breathing, "
    "occasional soft blinks, natural gaze with subtle eye shifts, very slight head sway, "
    "light hair and clothing movement from breathing, consistent facial identity and outfit, "
    "static camera, stable background, no turning, no large gestures."
)
DEFAULT_NEGATIVE_PROMPT = (
    "turning around, spinning, large body movement, fast motion, running, jumping, "
    "complex hand gesture, camera pan, camera zoom, camera shake, scene change, "
    "background change, flickering, jitter, shaking, looping, repeated motion, "
    "warped face, deformed face, extra fingers, blurred, low quality, watermark, "
    "text, oversaturated"
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
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    inference_steps: int = 30,
    guidance_scale: float = 5.0,
    shift: float = 5.0,
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
        str(python.absolute()),
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
        str(WINDOW_OVERLAP),
        "--ar_step",
        "0",
        "--addnoise_condition",
        "20",
        "--guidance_scale",
        str(guidance_scale),
        "--shift",
        str(shift),
        "--inference_steps",
        str(inference_steps),
        "--seed",
        str(seed),
        "--fps",
        str(VIDEO_FPS),
        "--offload",
        "--outdir",
        str(output.resolve()),
        "--prompt",
        prompt,
        "--negative_prompt",
        negative_prompt,
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
            str(python.absolute()),
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
        "quantize_int8": os.environ.get("QUANTIZE_INT8") == "1",
        "vendor_patch_sha256": vendor_patch_sha256(),
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
        if request.windows is not None or request.previous_run_id is not None:
            raise ValueError("Bounded window requests must be driven by Runtime")
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
            negative_prompt=request.negative_prompt,
            inference_steps=request.inference_steps,
            guidance_scale=request.guidance_scale,
            shift=request.shift,
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


# Runs in the existing model environment (MoviePy 1.0.3 / imageio), never at import.
# Stream the cumulative video rather than retaining its frames in the controller.
_WINDOW_MEDIA_SCRIPT = r"""
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio
import imageio_ffmpeg
import numpy as np
from moviepy.editor import VideoFileClip

video, review, expected, previous = sys.argv[1:]
review = Path(review)
expected, previous = int(expected), int(previous)
with VideoFileClip(video, audio=False) as clip:
    if not math.isclose(clip.fps, 24, rel_tol=0, abs_tol=0.001):
        raise ValueError('Window video must have 24 FPS')
    width, height = clip.size
# Count actual FFmpeg output. MoviePy rounds duration and can repeat its last frame.
with tempfile.TemporaryFile() as errors:
    process = subprocess.Popen([
        imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-xerror', '-i', video,
        '-map', '0:v:0', '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1',
    ], stdout=subprocess.PIPE, stderr=errors,
       creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
    writer = None
    try:
        if previous:
            writer = imageio.get_writer(str(review / 'boundary.mp4'), fps=24,
                                        macro_block_size=1)
        count = 0
        while True:
            raw = process.stdout.read(width * height * 3)
            if not raw:
                break
            if len(raw) != width * height * 3:
                raise ValueError('Incomplete decoded frame')
            index = count
            count += 1
            if count > expected:
                raise ValueError('Window contains extra frames')
            if previous and previous - 17 <= index < previous + 17:
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                writer.append_data(frame)
            if previous and index in (previous - 1, previous):
                name = 'before.png' if index < previous else 'after.png'
                imageio.imwrite(str(review / name), frame)
        if process.wait():
            errors.seek(0)
            raise ValueError('FFmpeg decode failed: ' + errors.read().decode(errors='replace'))
        if count != expected:
            raise ValueError('Window is truncated or has missing frames')
    finally:
        if process.poll() is None:
            process.kill()
        process.stdout.close()
        process.wait()
        if writer is not None:
            writer.close()
    (review / 'media.json').write_text(json.dumps({
        'frame_count': count, 'fps': 24, 'size': [width, height],
        'video_seconds': count / 24,
        'added_frames': count - previous, 'added_seconds': (count - previous) / 24,
        'boundary_frame': previous if previous else None,
        'boundary_seconds': previous / 24 if previous else None,
        'boundary_clip_start_frame': previous - 17 if previous else None,
        'boundary_clip_frames': 34 if previous else None,
    }, indent=2), encoding='utf-8')
"""


def verify_window_source(config: RuntimeConfig, window: GenerationWindow) -> None:
    """Do not resume missing, replaced or redirected committed video context."""
    root = config.data_dir / "runs" / str(window.run_id) / "windows" / str(window.window_id)
    if not window.video.resolve().is_relative_to(root.resolve()):
        raise ValueError("Continuation video is outside its recorded window directory")
    with window.video.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != window.video_sha256:
        raise ValueError("Continuation video does not match its committed SHA-256")


def inspect_window(
    config: RuntimeConfig,
    video: Path,
    output: Path,
    *,
    expected_frames: int,
    previous_frames: int,
    cancel_event: Event,
) -> str:
    """Fully decode a window and retain seam evidence before Runtime commits it."""
    if config.skyreels_python is None:
        raise ValueError("Window inspection requires SKYREELS_PYTHON")
    review = output / "review"
    review.mkdir(exist_ok=False)
    started = time.monotonic()
    evidence: dict[str, object] = {
        "status": "running",
        "expected_frames": expected_frames,
        "previous_frames": previous_frames,
        "fps": VIDEO_FPS,
    }
    manifest = review / "validation.json"
    atomic_write(manifest, json.dumps(evidence, indent=2).encode() + b"\n")
    try:
        with (review / "decode.log").open("w", encoding="utf-8") as log:
            result = _run_process(
                [
                    str(config.skyreels_python.absolute()),
                    "-c",
                    _WINDOW_MEDIA_SCRIPT,
                    str(video.resolve()),
                    str(review.resolve()),
                    str(expected_frames),
                    str(previous_frames),
                ],
                cwd=output,
                timeout=config.generation_timeout_seconds,
                cancel_event=cancel_event,
                log=log,
            )
        if result.returncode or not (review / "media.json").is_file():
            raise RuntimeError(f"Window decode/timeline validation failed; inspect {review}")
        if previous_frames and not (review / "boundary.mp4").stat().st_size:
            raise RuntimeError("Window boundary review video is empty")
        with video.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if cancel_event.is_set():
            raise GenerationCancelled("Window inspection cancelled")
        evidence.update(status="validated", video_sha256=digest)
        return digest
    except BaseException as exc:
        evidence.update(
            status="cancelled" if isinstance(exc, GenerationCancelled) else "failed",
            error=str(exc) or type(exc).__name__,
        )
        raise
    finally:
        evidence["elapsed_seconds"] = time.monotonic() - started
        atomic_write(manifest, json.dumps(evidence, indent=2).encode() + b"\n")
