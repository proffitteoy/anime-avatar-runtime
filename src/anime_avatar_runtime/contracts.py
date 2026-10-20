"""Shared control contracts; absent model measurements remain explicitly unknown."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self
from uuid import UUID, uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from .skyreels import DEFAULT_NEGATIVE_PROMPT, DEFAULT_PROMPT, VIDEO_FPS, WINDOW_OVERLAP


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    frames: int = Field(default=97, ge=97, strict=True)
    seed: int = Field(default=42, ge=0, lt=2**32, strict=True)
    prompt: str = DEFAULT_PROMPT
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    inference_steps: int = Field(default=30, ge=1, le=100, strict=True)
    guidance_scale: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    shift: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    windows: int | None = Field(default=None, ge=1, le=100, strict=True)
    previous_run_id: UUID | None = None

    @model_validator(mode="after")
    def validate_continuation(self) -> Self:
        if self.previous_run_id is not None and self.windows is None:
            raise ValueError("previous_run_id requires an explicit windows limit")
        return self

    @field_validator("frames")
    @classmethod
    def validate_frames(cls, value: int) -> int:
        if (value - 1) % 4:
            raise ValueError("Frames must be 4n+1 and at least 97 for this baseline")
        return value

    @field_validator("prompt", "negative_prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt must not be empty")
        return value


class GenerationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    run_id: UUID
    avatar_id: UUID | None = None
    output: Path
    videos: list[Path] = Field(min_length=1)
    elapsed_seconds: float = Field(ge=0)


class GenerationWindow(BaseModel):
    """One decoded cumulative video; committed only through its owning job record."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    window_id: UUID
    run_id: UUID
    avatar_id: UUID
    index: int = Field(ge=1, strict=True)
    previous_window_id: UUID | None = None
    request: GenerationRequest
    video: Path
    video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frame_count: int = Field(ge=97, strict=True)
    previous_frame_count: int = Field(default=0, ge=0, strict=True)
    fps: Literal[24] = VIDEO_FPS
    added_frames: int = Field(gt=0, strict=True)
    elapsed_seconds: float = Field(ge=0)
    boundary_video: Path | None = None

    @model_validator(mode="after")
    def validate_timeline(self) -> Self:
        if self.request.windows is not None or self.request.previous_run_id is not None:
            raise ValueError("Window requests must describe one backend invocation")
        continued = self.previous_window_id is not None
        if continued != (self.index > 1) or continued != (self.previous_frame_count > 0):
            raise ValueError("Window history does not match its index and prefix")
        expected = self.request.frames - (WINDOW_OVERLAP if continued else 0)
        if (
            self.added_frames != expected
            or self.frame_count != self.previous_frame_count + expected
        ):
            raise ValueError("Window timeline does not match the official overlap contract")
        if continued != (self.boundary_video is not None):
            raise ValueError("Continuation windows require a boundary review video")
        return self


class GenerationJob(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    run_id: UUID
    avatar_id: UUID
    state_schema_version: Literal[1] = 1
    request: GenerationRequest
    status: Literal["queued", "running", "completed", "failed", "cancelled"] = "queued"
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: AwareDatetime | None = None
    result: GenerationResult | None = None
    error: str | None = None
    accepted_windows: list[GenerationWindow] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        terminal = self.status in {"completed", "failed", "cancelled"}
        if terminal != (self.finished_at is not None):
            raise ValueError("Only terminal jobs must have a finished_at timestamp")
        if (self.status == "completed") != (self.result is not None):
            raise ValueError("Only completed jobs must have a generation result")
        if (self.status in {"failed", "cancelled"}) != (self.error is not None):
            raise ValueError("Only failed or cancelled jobs must have an error reason")
        if self.result and (
            self.result.run_id != self.run_id or self.result.avatar_id != self.avatar_id
        ):
            raise ValueError("Generation result does not belong to this job")
        if self.request.windows is None:
            if self.accepted_windows:
                raise ValueError("Single-segment experiments cannot commit window context")
        else:
            if len(self.accepted_windows) > self.request.windows or (
                self.status == "completed" and len(self.accepted_windows) != self.request.windows
            ):
                raise ValueError("Accepted window count does not match the requested limit")
            previous: GenerationWindow | None = None
            window_ids: set[UUID] = set()
            for window in self.accepted_windows:
                if window.run_id != self.run_id or window.avatar_id != self.avatar_id:
                    raise ValueError("Window does not belong to this job")
                if window.window_id in window_ids:
                    raise ValueError("Window IDs must be unique")
                window_ids.add(window.window_id)
                if previous is None and (
                    (self.request.previous_run_id is None) != (window.previous_window_id is None)
                ):
                    raise ValueError("First accepted window does not match the requested source")
                expected_request = GenerationRequest.model_validate(
                    self.request.model_dump()
                    | {
                        "windows": None,
                        "previous_run_id": None,
                        "seed": (self.request.seed + window.index - 1) % 2**32,
                    }
                )
                if window.request != expected_request:
                    raise ValueError("Window parameters do not match the requested seed strategy")
                if previous is not None and (
                    window.previous_window_id != previous.window_id
                    or window.index != previous.index + 1
                    or window.previous_frame_count != previous.frame_count
                ):
                    raise ValueError("Accepted windows are not contiguous")
                previous = window
            if self.result and self.result.videos != [self.accepted_windows[-1].video]:
                raise ValueError("Completed result must identify the last cumulative video")
        return self


EventKind = Literal[
    "runtime_started",
    "avatar_initialized",
    "runtime_resumed",
    "generation_submitted",
    "generation_started",
    "generation_completed",
    "generation_failed",
    "generation_cancelled",
    "generation_window_accepted",
    "runtime_paused",
    "runtime_stopped",
    "observation_recorded",
]


class RuntimeEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    event_id: UUID = Field(default_factory=uuid4)
    timestamp: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    kind: EventKind
    avatar_id: UUID | None = None
    run_id: UUID | None = None
    state_schema_version: Literal[1] = 1
    mode: str
    reason: str


class Observation(BaseModel):
    """Independent distances, tied to one run; unavailable values are never zero-filled."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal[1] = 1
    avatar_id: UUID | None = None
    run_id: UUID | None = None
    measured_at: AwareDatetime | None = None
    state_revision: int | None = Field(default=None, ge=0)
    identity_candidate_distance: float | None = Field(default=None, ge=0)
    identity_distance: float | None = Field(default=None, ge=0)
    geometry_distance: float | None = Field(default=None, ge=0)
    motion_distance: float | None = Field(default=None, ge=0)
    unavailable_reason: str | None = None
    subject_status: Literal["single_face_proxy", "unverified", "ambiguous"] = "unverified"
    subject_face_count: int | None = Field(default=None, ge=0)
    subject_reason: str = "No subject or tracking evidence supplied"

    @model_validator(mode="after")
    def validate_measurement(self) -> Self:
        measured = any(
            value is not None
            for value in (
                self.identity_candidate_distance,
                self.identity_distance,
                self.geometry_distance,
                self.motion_distance,
            )
        )
        if measured and (self.measured_at is None or self.run_id is None or self.avatar_id is None):
            raise ValueError("Measured distances require avatar_id, run_id and measured_at")
        if not measured and not self.unavailable_reason:
            raise ValueError("Absent measurements require an unavailable_reason")
        return self


class ObservationRequest(BaseModel):
    """Offline sampling and explicit research thresholds, not a recovery policy."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    sample_seconds: float = Field(default=1, gt=0)
    max_samples: int = Field(default=120, ge=1, le=10000, strict=True)
    identity_threshold: float | None = Field(default=None, gt=0)
    geometry_threshold: float | None = Field(default=None, gt=0)
    min_confidence: float = Field(default=0.3, gt=0, le=1)
    face_confidence: float = Field(default=0.5, gt=0, le=1)


class AnomalyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    identity_distance: float | None = Field(default=None, ge=0)
    geometry_distance: float | None = Field(default=None, ge=0)
    identity_threshold: float | None = Field(default=None, gt=0)
    geometry_threshold: float | None = Field(default=None, gt=0)


class AnomalyResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["anomaly", "within_thresholds", "unknown"]
    identity: Literal["exceeded", "within", "unknown"]
    geometry: Literal["exceeded", "within", "unknown"]
    reasons: list[str]


class ObservationSample(BaseModel):
    """Time-aligned, independent signals with raw detections for review."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation: Observation
    identity_status: str
    geometry_status: str
    identity_anchor: str | None = None
    geometry_anchor: str | None = None
    identity_elapsed_seconds: float = Field(ge=0)
    geometry_elapsed_seconds: float = Field(ge=0)
    expression_distance: float | None = Field(default=None, ge=0)
    identity_feature: str | None = None
    faces: list[dict[str, list[float] | list[list[float]]]] = Field(default_factory=list)
    overlay: str | None = None
    anomaly: AnomalyResult


class ObservationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal[1] = 1
    run_id: UUID
    avatar_id: UUID
    output: Path
    sample_count: int = Field(ge=1)
    truncated: bool
    elapsed_seconds: float = Field(ge=0)
    latest: ObservationSample

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        observation = self.latest.observation
        if observation.run_id != self.run_id or observation.avatar_id != self.avatar_id:
            raise ValueError("Observation result does not belong to this run and avatar")
        return self
