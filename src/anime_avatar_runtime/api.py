"""HTTP transport for the single-process runtime; importing it never loads a model."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from .config import RuntimeConfig
from .contracts import GenerationJob, GenerationRequest, Observation, RuntimeEvent
from .runtime import CapabilityUnavailable, Runtime, RuntimeConflict
from .state import MAX_IMAGE_BYTES, AvatarState


def create_app(data_dir: Path | None = None, *, config: RuntimeConfig | None = None) -> FastAPI:
    settings = config or RuntimeConfig.from_env()
    if data_dir is not None:
        settings = RuntimeConfig.model_validate(settings.model_dump() | {"data_dir": data_dir})
    runtime = Runtime(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await runtime.close()

    app = FastAPI(title="Anime Avatar Runtime", version="0.1.0", lifespan=lifespan)
    app.state.runtime = runtime

    @app.exception_handler(RuntimeConflict)
    async def conflict_handler(request: Request, exc: RuntimeConflict) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(CapabilityUnavailable)
    async def unavailable_handler(request: Request, exc: CapabilityUnavailable) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(LookupError)
    async def missing_handler(request: Request, exc: LookupError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/runtime/status")
    async def status() -> dict[str, object]:
        return runtime.status()

    @app.post("/avatar/init", status_code=201, response_model=AvatarState)
    async def init_avatar(request: Request) -> AvatarState:
        payload = bytearray()
        async for chunk in request.stream():
            if len(payload) + len(chunk) > MAX_IMAGE_BYTES:
                raise HTTPException(413, "Reference exceeds 10 MiB")
            payload.extend(chunk)
        try:
            return await runtime.initialize(bytes(payload))
        except FileExistsError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/avatar/state", response_model=AvatarState)
    async def avatar_state() -> AvatarState:
        return runtime.avatar_state()

    @app.post("/avatar/generate", status_code=202, response_model=GenerationJob)
    async def generate(request: GenerationRequest) -> GenerationJob:
        return await runtime.submit(request)

    @app.get("/runtime/jobs/{run_id}", response_model=GenerationJob)
    async def job(run_id: UUID) -> GenerationJob:
        return runtime.get_job(run_id)

    @app.post("/runtime/pause")
    async def pause() -> dict[str, object]:
        return await runtime.pause()

    @app.post("/runtime/resume")
    async def resume() -> dict[str, object]:
        return await runtime.resume()

    @app.get("/runtime/events", response_model=list[RuntimeEvent])
    async def events(limit: int = Query(default=50, ge=1, le=200)) -> list[RuntimeEvent]:
        return runtime.events(limit)

    @app.get("/observer/status", response_model=Observation)
    async def observer_status() -> Observation:
        return runtime.observation()

    @app.post("/recovery/trigger", status_code=503)
    async def trigger_recovery() -> None:
        raise CapabilityUnavailable("Recovery is not implemented; no recovery task was created")

    return app
