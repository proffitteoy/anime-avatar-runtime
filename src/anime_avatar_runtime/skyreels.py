"""CLI bridge to a pinned official SkyReels checkout, without importing its CUDA stack."""

import subprocess
from pathlib import Path

UPSTREAM_REVISION = "9351d13152207cc04de780e055346b08ade0b851"
MODEL_ID = "Skywork/SkyReels-V2-DF-1.3B-540P"
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


def check_backend(repo: Path, python: Path) -> None:
    if not (repo / "generate_video_df.py").is_file() or not python.is_file():
        raise FileNotFoundError(
            "Install the pinned SkyReels checkout and its separate Python first"
        )
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"Expected SkyReels revision {UPSTREAM_REVISION}, found {revision}")
    # Runs only when explicitly generating, never on API startup or dry-run.
    subprocess.run(
        [
            str(python.resolve()),
            "-c",
            "import torch; from moviepy.editor import VideoFileClip; "
            "assert torch.cuda.is_available(), 'SkyReels baseline requires an NVIDIA CUDA GPU'",
        ],
        cwd=repo,
        check=True,
    )
