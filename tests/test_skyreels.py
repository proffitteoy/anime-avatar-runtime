import ast
import hashlib
from pathlib import Path

import pytest

from anime_avatar_runtime.skyreels import build_command


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
