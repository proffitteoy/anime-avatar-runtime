"""Single-process local control API. Model inference runs in its own environment."""

import os
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException, Request

from .state import MAX_IMAGE_BYTES, AvatarState, initialize, load_state


def create_app(data_dir: Path | None = None) -> FastAPI:
    root = data_dir or Path(os.environ.get("AVATAR_DATA_DIR", "var/avatar"))
    # Fail visibly on damaged persisted state instead of silently losing a character.
    load_state(root)
    lock = Lock()
    app = FastAPI(title="Anime Avatar Runtime", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/runtime/status")
    def status() -> dict[str, object]:
        state = load_state(root)
        return {
            "initialized": state is not None,
            "mode": state.mode if state else "uninitialized",
            "generation": "external_cli",
            "autonomous_loop": False,
            "observers_enabled": [],
        }

    @app.post("/avatar/init", status_code=201, response_model=AvatarState)
    async def init_avatar(request: Request) -> AvatarState:
        payload = bytearray()
        async for chunk in request.stream():
            if len(payload) + len(chunk) > MAX_IMAGE_BYTES:
                raise HTTPException(413, "Reference exceeds 10 MiB")
            payload.extend(chunk)
        with lock:
            try:
                return initialize(root, bytes(payload))
            except FileExistsError as exc:
                raise HTTPException(409, str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.get("/avatar/state", response_model=AvatarState)
    def avatar_state() -> AvatarState:
        state = load_state(root)
        if state is None:
            raise HTTPException(404, "Initialize an avatar first")
        return state

    return app
