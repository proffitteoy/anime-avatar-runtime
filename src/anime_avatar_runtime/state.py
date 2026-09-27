"""Validated reference state and atomic local checkpoints; no synthetic model output."""

import hashlib
import io
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000


class AvatarState(BaseModel):
    """Only fields known at initialization; pose and embeddings are not fabricated."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    avatar_id: UUID
    mode: Literal["paused"] = "paused"
    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    created_at: AwareDatetime


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            name = stream.name
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


def load_state(data_dir: Path) -> AvatarState | None:
    checkpoint = data_dir / "state.json"
    if not checkpoint.exists():
        return None
    state = AvatarState.model_validate_json(checkpoint.read_bytes())
    reference = data_dir / "references" / f"{state.reference_sha256}.png"
    if hashlib.sha256(reference.read_bytes()).hexdigest() != state.reference_sha256:
        raise ValueError("Reference image does not match checkpoint SHA-256")
    return state


def initialize(data_dir: Path, payload: bytes) -> AvatarState:
    """Store one canonical reference; existing state must not be silently replaced."""
    if (data_dir / "state.json").exists():
        raise FileExistsError("An avatar already exists in this data directory")
    if not payload or len(payload) > MAX_IMAGE_BYTES:
        raise ValueError("Reference must contain 1 byte to 10 MiB")
    try:
        with Image.open(io.BytesIO(payload)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("Reference exceeds 16 million pixels")
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError("Reference must be a single still image")
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Reference is not a valid supported image") from exc
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    png = buffer.getvalue()
    digest = hashlib.sha256(png).hexdigest()
    state = AvatarState(
        avatar_id=uuid4(),
        reference_sha256=digest,
        width=image.width,
        height=image.height,
        created_at=datetime.now(UTC),
    )
    atomic_write(data_dir / "references" / f"{digest}.png", png)
    atomic_write(data_dir / "state.json", state.model_dump_json(indent=2).encode() + b"\n")
    return state
