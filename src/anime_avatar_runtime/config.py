"""Validated configuration shared by the API, runtime and command line."""

import os
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    data_dir: Path = Path("var/avatar")
    skyreels_repo: Path = Path("vendor/SkyReels-V2")
    skyreels_python: Path | None = None
    skyreels_model: Path = Path("models/skyreels-v2-df-1.3b-540p")
    generation_timeout_seconds: float = Field(default=1800, gt=0, allow_inf_nan=False)
    observer_python: Path | None = None
    observer_models: Path = Path("models/observers")
    observation_timeout_seconds: float = Field(default=600, gt=0, allow_inf_nan=False)

    @field_validator(
        "data_dir",
        "skyreels_repo",
        "skyreels_python",
        "skyreels_model",
        "observer_python",
        "observer_models",
        mode="before",
    )
    @classmethod
    def reject_empty_path(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            raise ValueError("Path must not be blank; unset optional environment variables instead")
        return value

    @field_validator("data_dir", "skyreels_repo", "skyreels_model", "observer_models")
    @classmethod
    def resolve_path(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @field_validator("skyreels_python", "observer_python")
    @classmethod
    def resolve_interpreter(cls, value: Path | None) -> Path | None:
        # The venv interpreter is a symlink to the base python; resolving it
        # drops the venv's site-packages. Keep the symlink path.
        return value.expanduser().absolute() if value is not None else None

    @classmethod
    def from_env(cls) -> Self:
        """Read only supported settings; loading a .env file remains explicit."""
        variables = {
            "data_dir": "AVATAR_DATA_DIR",
            "skyreels_repo": "SKYREELS_REPO",
            "skyreels_python": "SKYREELS_PYTHON",
            "skyreels_model": "SKYREELS_MODEL",
            "generation_timeout_seconds": "AVATAR_GENERATION_TIMEOUT_SECONDS",
            "observer_python": "AVATAR_OBSERVER_PYTHON",
            "observer_models": "AVATAR_OBSERVER_MODELS",
            "observation_timeout_seconds": "AVATAR_OBSERVATION_TIMEOUT_SECONDS",
        }
        return cls.model_validate(
            {field: os.environ[name] for field, name in variables.items() if name in os.environ}
        )
