"""Local API entry point and reproducible invocation of upstream generation."""

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from uuid import uuid4

from .config import RuntimeConfig
from .contracts import GenerationRequest
from .skyreels import DEFAULT_PROMPT, UPSTREAM_REVISION, build_command, run_generation


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
    generate.add_argument("--repo", type=Path)
    generate.add_argument("--python", type=Path)
    generate.add_argument("--model", type=Path)
    generate.add_argument("--timeout", type=float, help="Total preflight and generation seconds")
    generate.add_argument("--output", type=Path, required=True, help="New experiment directory")
    generate.add_argument("--frames", type=int, default=97)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--prompt", default=DEFAULT_PROMPT)
    generate.add_argument(
        "--dry-run", action="store_true", help="Print argv without loading models"
    )
    args = parser.parse_args()

    try:
        config = RuntimeConfig.from_env()
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
                        "data_dir": str(config.data_dir),
                        "skyreels_python": (
                            str(config.skyreels_python) if config.skyreels_python else None
                        ),
                        "config": config.model_dump(mode="json"),
                        "skyreels_revision": UPSTREAM_REVISION,
                        "inference_verified": False,
                        "note": (
                            "generate checks CUDA and models in the separate backend environment"
                        ),
                    },
                    indent=2,
                )
            )
            return 0

        values = config.model_dump()
        for argument, field in (
            ("repo", "skyreels_repo"),
            ("python", "skyreels_python"),
            ("model", "skyreels_model"),
            ("timeout", "generation_timeout_seconds"),
        ):
            if (value := getattr(args, argument)) is not None:
                values[field] = value
        config = RuntimeConfig.model_validate(values)
        request = GenerationRequest(frames=args.frames, seed=args.seed, prompt=args.prompt)
        if args.dry_run:
            if config.skyreels_python is None:
                raise ValueError("Set --python or SKYREELS_PYTHON to the separate CUDA environment")
            argv = build_command(
                repo=config.skyreels_repo,
                python=config.skyreels_python,
                output=args.output,
                image=args.image,
                video=args.video,
                end_image=args.end_image,
                model=str(config.skyreels_model),
                frames=request.frames,
                seed=request.seed,
                prompt=request.prompt,
            )
            print(
                json.dumps(
                    {"cwd": str(config.skyreels_repo), "argv": argv, "executed": False}, indent=2
                )
            )
            return 0
        result = run_generation(
            config,
            request,
            args.output,
            run_id=uuid4(),
            image=args.image,
            video=args.video,
            end_image=args.end_image,
        )
        print(result.output / "run.json")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("error: Generation cancelled", file=sys.stderr)
        return 130
