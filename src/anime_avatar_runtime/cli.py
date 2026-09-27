"""Local API entry point and reproducible invocation of upstream generation."""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from .skyreels import DEFAULT_PROMPT, UPSTREAM_REVISION, build_command, check_backend
from .state import atomic_write


def main() -> int:
    parser = argparse.ArgumentParser(prog="avatar-runtime")
    commands = parser.add_subparsers(dest="command", required=True)
    server = commands.add_parser("serve", help="Start the local control API")
    server.add_argument("--port", type=int, default=8000)
    commands.add_parser("doctor", help="Report control environment and inference configuration")
    generate = commands.add_parser("generate", help="Run the pinned official SkyReels backend")
    source = generate.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--video", type=Path, help="Continue an existing video")
    generate.add_argument("--end-image", type=Path)
    generate.add_argument(
        "--repo", type=Path, default=os.getenv("SKYREELS_REPO", "vendor/SkyReels-V2")
    )
    generate.add_argument("--python", type=Path, default=os.getenv("SKYREELS_PYTHON"))
    generate.add_argument("--model", type=Path, default=Path("models/skyreels-v2-df-1.3b-540p"))
    generate.add_argument("--output", type=Path, required=True, help="New experiment directory")
    generate.add_argument("--frames", type=int, default=97)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--prompt", default=DEFAULT_PROMPT)
    generate.add_argument(
        "--dry-run", action="store_true", help="Print argv without loading models"
    )
    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "anime_avatar_runtime.api:create_app",
            factory=True,
            host="127.0.0.1",
            port=args.port,
            workers=1,
        )
        return 0
    if args.command == "doctor":
        print(
            json.dumps(
                {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "packages": {
                        name: importlib.metadata.version(name)
                        for name in ("fastapi", "pydantic", "uvicorn", "pillow")
                    },
                    "data_dir": os.getenv("AVATAR_DATA_DIR", "var/avatar"),
                    "skyreels_python": os.getenv("SKYREELS_PYTHON"),
                    "skyreels_revision": UPSTREAM_REVISION,
                    "inference_verified": False,
                    "note": "generate checks CUDA and models in the separate backend environment",
                },
                indent=2,
            )
        )
        return 0

    try:
        if args.python is None:
            raise ValueError("Set --python or SKYREELS_PYTHON to the separate CUDA environment")
        repo = args.repo.resolve()
        output = args.output.resolve()
        argv = build_command(
            repo=repo,
            python=args.python,
            output=output,
            image=args.image,
            video=args.video,
            end_image=args.end_image,
            model=str(args.model.resolve()),
            frames=args.frames,
            seed=args.seed,
            prompt=args.prompt,
        )
        if args.dry_run:
            print(json.dumps({"cwd": str(repo), "argv": argv, "executed": False}, indent=2))
            return 0
        if not args.model.is_dir():
            raise FileNotFoundError(
                "Download the pinned model snapshot first; see docs/backend-setup.md"
            )
        check_backend(repo, args.python)
        # Never overwrite a previous experiment, including failed runs.
        output.mkdir(parents=True, exist_ok=False)
        manifest: dict[str, object] = {
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "upstream_revision": UPSTREAM_REVISION,
            "argv": argv,
            "cwd": str(repo),
        }
        manifest_path = output / "run.json"

        def save_manifest() -> None:
            atomic_write(manifest_path, json.dumps(manifest, indent=2).encode() + b"\n")

        save_manifest()
        try:
            with (output / "backend.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(argv, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
            videos = [str(p) for p in output.glob("*.mp4") if p.stat().st_size > 0]
            if result.returncode != 0 or not videos:
                raise RuntimeError(
                    f"Backend failed or produced no video; inspect {output / 'backend.log'}"
                )
            manifest.update(status="completed", videos=videos)
        except BaseException as exc:
            manifest.update(status="failed", error=str(exc))
            raise
        finally:
            manifest["finished_at"] = datetime.now(UTC).isoformat()
            save_manifest()
        print(manifest_path)
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
