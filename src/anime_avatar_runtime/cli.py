"""Local API entry point and reproducible invocation of upstream generation."""

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path
from uuid import uuid4
from zipfile import BadZipFile

from .config import RuntimeConfig


def main() -> int:
    parser = argparse.ArgumentParser(prog="avatar-runtime")
    commands = parser.add_subparsers(dest="command", required=True)
    legacy = commands.add_parser(
        "legacy", help="Explicit historical experiment and checkpoint tools"
    )
    historical = legacy.add_subparsers(dest="legacy_command", required=True)
    server = historical.add_parser("serve", help="Start the legacy experiment API")
    server.add_argument("--port", type=int, default=8000)
    checkpoint = historical.add_parser(
        "state", help="Read and verify an old checkpoint without writes"
    )
    checkpoint.add_argument("--data-dir", type=Path, required=True)
    commands.add_parser("doctor", help="Report control environment and inference configuration")
    migrate = commands.add_parser(
        "migrate-data", help="Archive stopped legacy data and retain its reference"
    )
    migrate.add_argument("--source", type=Path, required=True)
    migrate.add_argument("--archive", type=Path, required=True)
    migrate.add_argument("--avatar-subdir", default="avatar")
    migrate.add_argument("--apply", action="store_true", help="Apply the verified offline cutover")
    migrate.add_argument(
        "--no-backup",
        action="store_true",
        help="Keep the original archive without creating a ZIP copy",
    )
    verify = commands.add_parser("verify-data", help="Verify a legacy data archive and backup")
    verify.add_argument("--archive", type=Path, required=True)
    restore = commands.add_parser("restore-data", help="Restore a backup to a new directory")
    restore.add_argument("--archive", type=Path, required=True)
    restore.add_argument("--output", type=Path, required=True)
    observe = historical.add_parser(
        "observe", help="Run pinned CPU identity and geometry observers"
    )
    observed_source = observe.add_mutually_exclusive_group(required=True)
    observed_source.add_argument("--image", type=Path)
    observed_source.add_argument("--video", type=Path)
    observe.add_argument("--anchor", type=Path, action="append", required=True)
    observe.add_argument("--python", type=Path)
    observe.add_argument("--models", type=Path)
    observe.add_argument("--output", type=Path, required=True)
    observe.add_argument("--sample-seconds", type=float, default=1)
    observe.add_argument("--max-samples", type=int, default=120)
    observe.add_argument("--identity-threshold", type=float)
    observe.add_argument("--geometry-threshold", type=float)
    generate = historical.add_parser("generate", help="Run the pinned official SkyReels backend")
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
    generate.add_argument("--prompt")
    generate.add_argument("--negative-prompt")
    generate.add_argument("--inference-steps", type=int, default=30)
    generate.add_argument("--guidance-scale", type=float, default=5.0)
    generate.add_argument("--shift", type=float, default=5.0)
    generate.add_argument(
        "--dry-run", action="store_true", help="Print argv without loading models"
    )
    for name in ("serve", "generate", "observe"):
        commands.add_parser(
            name,
            parents=[historical.choices[name]],
            add_help=False,
            help=f"Historical compatibility alias for legacy {name}",
        )
    args = parser.parse_args()
    if args.command == "legacy":
        args.command = args.legacy_command
    elif args.command in {"serve", "generate", "observe"}:
        print(f"note: historical interface; prefer legacy {args.command}", file=sys.stderr)

    try:
        if args.command == "state":
            from .state import load_state

            state = load_state(args.data_dir)
            if state is None:
                raise FileNotFoundError("No legacy checkpoint in the explicitly selected directory")
            print(state.model_dump_json(indent=2))
            return 0
        if args.command in {"migrate-data", "verify-data", "restore-data"}:
            from .migration import migrate_data, restore_data, verify_archive

            if args.command == "migrate-data":
                migrated = migrate_data(
                    args.source,
                    args.archive,
                    avatar_subdir=args.avatar_subdir,
                    apply=args.apply,
                    backup=not args.no_backup,
                )
            elif args.command == "verify-data":
                migrated = verify_archive(args.archive)
            else:
                migrated = restore_data(args.archive, args.output)
            print(
                json.dumps(
                    {
                        "phase": migrated.phase,
                        "backup_required": migrated.backup_required,
                        "files": len(migrated.files),
                        "bytes": migrated.total_bytes,
                        "reference_initialized": migrated.reference_state is not None,
                        "archive": str(args.archive.resolve()),
                        **(
                            {"restored_to": str(args.output.resolve())}
                            if args.command == "restore-data"
                            else {}
                        ),
                    },
                    indent=2,
                )
            )
            return 0
        config = RuntimeConfig.from_env()
        if args.command == "serve":
            if importlib.util.find_spec("numpy") is None:
                raise RuntimeError("Historical API requires uv sync --locked --extra legacy")
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
            from .skyreels import UPSTREAM_REVISION

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
                        "interface": "legacy_experiment_control",
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

        if args.command == "observe":
            if importlib.util.find_spec("numpy") is None:
                raise RuntimeError(
                    "Historical observation requires uv sync --locked --extra legacy"
                )
            from .contracts import ObservationRequest
            from .observers import run_observation

            values = config.model_dump()
            if args.python is not None:
                values["observer_python"] = args.python
            if args.models is not None:
                values["observer_models"] = args.models
            config = RuntimeConfig.model_validate(values)
            observation_request = ObservationRequest(
                sample_seconds=args.sample_seconds,
                max_samples=args.max_samples,
                identity_threshold=args.identity_threshold,
                geometry_threshold=args.geometry_threshold,
            )
            observed = run_observation(
                config,
                observation_request,
                args.output,
                anchors=args.anchor,
                image=args.image,
                video=args.video,
            )
            print(observed.output / "result.json")
            return 0
        from .contracts import GenerationRequest
        from .skyreels import (
            DEFAULT_NEGATIVE_PROMPT,
            DEFAULT_PROMPT,
            build_command,
            run_generation,
        )

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
        request = GenerationRequest(
            frames=args.frames,
            seed=args.seed,
            prompt=args.prompt if args.prompt is not None else DEFAULT_PROMPT,
            negative_prompt=(
                args.negative_prompt
                if args.negative_prompt is not None
                else DEFAULT_NEGATIVE_PROMPT
            ),
            inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            shift=args.shift,
        )
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
                negative_prompt=request.negative_prompt,
                inference_steps=request.inference_steps,
                guidance_scale=request.guidance_scale,
                shift=request.shift,
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
    except (OSError, ValueError, RuntimeError, BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("error: Generation cancelled", file=sys.stderr)
        return 130
