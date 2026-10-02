"""Single-process manual generation lifecycle, independent of HTTP and GPU imports."""

import asyncio
from collections import deque
from datetime import UTC, datetime
from threading import Event
from uuid import UUID, uuid4

from .config import RuntimeConfig
from .contracts import EventKind, GenerationJob, GenerationRequest, Observation, RuntimeEvent
from .skyreels import GenerationCancelled, backend_unavailable_reason, run_generation
from .state import AvatarState, atomic_write, initialize, load_state


class RuntimeConflict(RuntimeError):
    """A requested transition conflicts with the current runtime state."""


class CapabilityUnavailable(RuntimeError):
    """The requested operation has no configured implementation."""


class Runtime:
    """One avatar and one in-flight job on one event loop.

    Reference checkpoints stay at schema v1. Generation artifacts are experiments,
    not trusted anchors or committed visual state. Resume enables manual submission;
    it does not start an autonomous generation/observation loop.
    """

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.state = load_state(config.data_dir)
        self._control_lock = asyncio.Lock()
        self._paused = True
        self._closed = False
        self._task: asyncio.Task[None] | None = None
        self._cancel = Event()
        self._active_job: GenerationJob | None = None
        self._last_error: str | None = None
        self._events: deque[RuntimeEvent] = deque(maxlen=200)
        self._emit("runtime_started", "Reference checkpoint loaded; manual runtime starts paused")

    @property
    def mode(self) -> str:
        if self._closed:
            return "stopped"
        if self.state is None:
            return "uninitialized"
        if self._active_job is not None:
            return "pausing" if self._paused else "generating"
        if self._paused:
            return "paused"
        return "error" if self._last_error else "ready"

    def _emit(self, kind: EventKind, reason: str, run_id: UUID | None = None) -> None:
        self._events.append(
            RuntimeEvent(
                kind=kind,
                avatar_id=self.state.avatar_id if self.state else None,
                run_id=run_id,
                mode=self.mode,
                reason=reason,
            )
        )

    def status(self) -> dict[str, object]:
        reason = backend_unavailable_reason(self.config)
        return {
            "initialized": self.state is not None,
            "mode": self.mode,
            "generation": "managed_subprocess",
            "autonomous_loop": False,
            "observers_enabled": [],
            "recovery_available": False,
            "backend_configured": reason is None,
            "backend_unavailable_reason": reason,
            "inference_verified": False,
            "active_run_id": str(self._active_job.run_id) if self._active_job else None,
            "last_error": self._last_error,
        }

    def events(self, limit: int = 50) -> list[RuntimeEvent]:
        return list(self._events)[-limit:]

    def observation(self) -> Observation:
        return Observation(
            avatar_id=self.state.avatar_id if self.state else None,
            unavailable_reason="Identity, geometry and motion observers are not implemented",
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeConflict("Runtime is stopped")

    def avatar_state(self) -> AvatarState:
        # Revalidate references at reads and submission, including external tampering.
        state = load_state(self.config.data_dir)
        if state is None:
            raise LookupError("Initialize an avatar first")
        return state

    async def initialize(self, payload: bytes) -> AvatarState:
        async with self._control_lock:
            self._require_open()
            worker = asyncio.create_task(
                asyncio.to_thread(initialize, self.config.data_dir, payload)
            )
            try:
                self.state = await asyncio.shield(worker)
            except asyncio.CancelledError:
                # Do not release the initialization lock while its writer is still running.
                self.state = await worker
                raise
            self._emit("avatar_initialized", "Canonical reference checkpoint saved")
            return self.state

    async def resume(self) -> dict[str, object]:
        async with self._control_lock:
            self._require_open()
            self.avatar_state()
            if self._paused or self._last_error:
                self._paused = False
                self._last_error = None
                self._emit("runtime_resumed", "Manual generation submissions enabled")
            return self.status()

    async def pause(self) -> dict[str, object]:
        async with self._control_lock:
            self._require_open()
            self._paused = True
            self._cancel.set()
            if self._task is not None:
                await asyncio.shield(self._task)
            self._emit("runtime_paused", "Paused after releasing the active backend process")
            return self.status()

    async def close(self) -> None:
        async with self._control_lock:
            if self._closed:
                return
            self._paused = True
            self._cancel.set()
            if self._task is not None:
                await asyncio.shield(self._task)
            self._closed = True
            self._emit("runtime_stopped", "Shutdown completed; no generation is scheduled")

    def _save_job(self, job: GenerationJob) -> None:
        path = self.config.data_dir / "runs" / str(job.run_id) / "job.json"
        atomic_write(path, job.model_dump_json(indent=2).encode() + b"\n")

    def get_job(self, run_id: UUID) -> GenerationJob:
        if self._active_job is not None and self._active_job.run_id == run_id:
            return self._active_job
        path = self.config.data_dir / "runs" / str(run_id) / "job.json"
        if not path.is_file():
            raise LookupError(f"Unknown generation job: {run_id}")
        job = GenerationJob.model_validate_json(path.read_bytes())
        if job.run_id != run_id or self.state is None or job.avatar_id != self.state.avatar_id:
            raise ValueError("Job identity does not match its path or current avatar")
        if job.status in {"queued", "running"}:
            # No work survives process restart. Never auto-replay interrupted GPU work.
            job = GenerationJob.model_validate(
                job.model_dump()
                | {
                    "status": "failed",
                    "error": "Runtime interrupted before a terminal job record was saved",
                    "finished_at": datetime.now(UTC),
                }
            )
            self._save_job(job)
        return job

    async def submit(self, request: GenerationRequest) -> GenerationJob:
        async with self._control_lock:
            self._require_open()
            state = self.avatar_state()
            if self._task is not None:
                raise RuntimeConflict("A generation job is already active")
            if self._paused or self._last_error:
                raise RuntimeConflict("Resume the runtime before submitting generation")
            if reason := backend_unavailable_reason(self.config):
                raise CapabilityUnavailable(reason)
            job = GenerationJob(run_id=uuid4(), avatar_id=state.avatar_id, request=request)
            self._save_job(job)
            self._active_job = job
            self._cancel = Event()
            self._emit("generation_submitted", "One reference-image segment requested", job.run_id)
            self._task = asyncio.create_task(self._execute(job), name=f"generation-{job.run_id}")
            return job

    async def _execute(self, job: GenerationJob) -> None:
        terminal: GenerationJob | None = None
        kind: EventKind = "generation_failed"
        try:
            job = GenerationJob.model_validate(job.model_dump() | {"status": "running"})
            self._save_job(job)
            self._active_job = job
            self._emit(
                "generation_started", "Starting the separate backend environment", job.run_id
            )
            state = self.avatar_state()
            image = self.config.data_dir / "references" / f"{state.reference_sha256}.png"
            output = self.config.data_dir / "runs" / str(job.run_id) / "output"
            worker = asyncio.create_task(
                asyncio.to_thread(
                    run_generation,
                    self.config,
                    job.request,
                    output,
                    run_id=job.run_id,
                    avatar_id=job.avatar_id,
                    image=image,
                    cancel_event=self._cancel,
                )
            )
            try:
                result = await asyncio.shield(worker)
            except asyncio.CancelledError:
                self._cancel.set()
                # Await process cleanup even if the enclosing task was cancelled.
                try:
                    await worker
                except Exception:
                    pass
                raise GenerationCancelled("Runtime task cancelled") from None
            if self._cancel.is_set():
                raise GenerationCancelled("Generation cancelled before accepting the result")
            terminal = GenerationJob.model_validate(
                job.model_dump()
                | {"status": "completed", "finished_at": datetime.now(UTC), "result": result}
            )
            kind = "generation_completed"
        except Exception as exc:
            cancelled = isinstance(exc, GenerationCancelled)
            terminal = GenerationJob.model_validate(
                job.model_dump()
                | {
                    "status": "cancelled" if cancelled else "failed",
                    "finished_at": datetime.now(UTC),
                    "error": str(exc) or type(exc).__name__,
                }
            )
            kind = "generation_cancelled" if cancelled else "generation_failed"
            if not cancelled:
                self._last_error = terminal.error
        finally:
            try:
                if terminal is not None:
                    self._save_job(terminal)
            except OSError as exc:
                self._last_error = f"Could not save terminal job record: {exc}"
                kind = "generation_failed"
            self._active_job = None
            self._task = None
            reason = self._last_error or (
                terminal.error if terminal and terminal.error else "Generation artifacts saved"
            )
            self._emit(kind, reason, job.run_id)
