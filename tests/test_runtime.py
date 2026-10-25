import asyncio
import io
import json
import time
from pathlib import Path
from threading import Event
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from anime_avatar_runtime import runtime as runtime_module
from anime_avatar_runtime.api import create_app
from anime_avatar_runtime.config import RuntimeConfig
from anime_avatar_runtime.contracts import (
    GenerationJob,
    GenerationRequest,
    GenerationResult,
    Observation,
)
from anime_avatar_runtime.skyreels import GenerationCancelled
from anime_avatar_runtime.state import MAX_IMAGE_BYTES, initialize, load_state


def reference() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (32, 48), (80, 120, 160)).save(stream, format="PNG")
    return stream.getvalue()


def test_initialize_survives_restart_and_refuses_replacement(tmp_path: Path) -> None:
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/avatar/state").status_code == 404
        response = client.post("/avatar/init", content=reference())
        assert response.status_code == 201
        state = response.json()
        assert (state["width"], state["height"], state["mode"]) == (32, 48, "paused")
        assert client.post("/avatar/init", content=reference()).status_code == 409
        assert client.get("/runtime/status").json()["autonomous_loop"] is False
    with TestClient(create_app(tmp_path)) as restarted:
        assert restarted.get("/avatar/state").json() == state


@pytest.mark.parametrize("content", [b"", b"not an image"])
def test_invalid_reference_does_not_create_checkpoint(tmp_path: Path, content: bytes) -> None:
    with TestClient(create_app(tmp_path)) as client:
        assert client.post("/avatar/init", content=content).status_code == 422
    assert not (tmp_path / "state.json").exists()


def test_large_upload_is_rejected_before_decoding(tmp_path: Path) -> None:
    with TestClient(create_app(tmp_path)) as client:
        assert client.post("/avatar/init", content=b"x" * (MAX_IMAGE_BYTES + 1)).status_code == 413
    assert not (tmp_path / "state.json").exists()


def test_reference_tampering_is_not_silently_accepted(tmp_path: Path) -> None:
    state = initialize(tmp_path, reference())
    (tmp_path / "references" / f"{state.reference_sha256}.png").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="SHA-256"):
        create_app(tmp_path)


def test_invalid_checkpoint_is_not_overwritten(tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text('{"schema_version": 999}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_state(tmp_path)
    with pytest.raises(FileExistsError):
        initialize(tmp_path, reference())


def wait_for_job(client: TestClient, run_id: str) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = client.get(f"/runtime/jobs/{run_id}").json()
        if job["status"] not in {"queued", "running"}:
            return job
        time.sleep(0.01)
    pytest.fail("Job did not terminate")


def enable_test_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    # Explicit control-plane test doubles; these do not exercise GPU inference.
    monkeypatch.setattr(runtime_module, "backend_unavailable_reason", lambda _: None)


def test_manual_controls_report_missing_capabilities_without_work(tmp_path: Path) -> None:
    config = RuntimeConfig(data_dir=tmp_path, skyreels_python=None)
    with TestClient(create_app(config=config)) as client:
        assert client.post("/runtime/resume").status_code == 404
        assert client.post("/avatar/generate", json={}).status_code == 404
        assert client.post("/avatar/init", content=reference()).status_code == 201
        assert client.post("/avatar/generate", json={}).status_code == 409
        response = client.post("/runtime/resume")
        assert response.json()["mode"] == "ready"
        assert response.json()["backend_configured"] is False
        assert client.post("/avatar/generate", json={}).status_code == 503
        assert not (tmp_path / "runs").exists()
        observation = client.get("/observer/status").json()
        assert observation["identity_distance"] is None
        assert observation["geometry_distance"] is None
        assert observation["motion_distance"] is None
        assert observation["measured_at"] is None
        assert observation["unavailable_reason"]
        assert client.post("/recovery/trigger").status_code == 503
        assert client.get("/runtime/status").json()["autonomous_loop"] is False
        assert client.post("/runtime/pause").json()["mode"] == "paused"
        assert client.get(f"/runtime/jobs/{uuid4()}").status_code == 404
        assert client.get("/runtime/jobs/not-a-uuid").status_code == 422
        assert client.get("/runtime/events?limit=201").status_code == 422


@pytest.mark.parametrize(
    "payload",
    [{"frames": 98}, {"frames": True}, {"seed": -1}, {"prompt": " "}, {"unknown": 1}],
)
def test_generation_contract_rejects_invalid_input(tmp_path: Path, payload: dict) -> None:
    with pytest.raises(ValidationError):
        GenerationRequest.model_validate(payload)
    with TestClient(create_app(tmp_path)) as client:
        assert client.post("/avatar/generate", json=payload).status_code == 422
    assert not (tmp_path / "runs").exists()


def test_unknown_observation_cannot_masquerade_as_measured() -> None:
    with pytest.raises(ValidationError, match="unavailable_reason"):
        Observation()
    with pytest.raises(ValidationError, match="Measured distances require"):
        Observation(identity_distance=0)
    with pytest.raises(ValidationError):
        Observation(identity_distance=float("nan"), unavailable_reason="unavailable")


def test_successful_job_is_queryable_and_does_not_promote_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    enable_test_backend(monkeypatch)
    entered, release = Event(), Event()

    def generate(config, request, output, *, run_id, avatar_id, image, cancel_event):
        assert (
            image == config.data_dir / "references" / f"{load_state(tmp_path).reference_sha256}.png"
        )
        assert request.seed == 123
        entered.set()
        assert release.wait(5)
        output.mkdir(parents=True)
        video = output / "test-artifact.mp4"
        video.write_bytes(b"control-test artifact, not real video")
        return GenerationResult(
            run_id=run_id, avatar_id=avatar_id, output=output, videos=[video], elapsed_seconds=0.1
        )

    monkeypatch.setattr(runtime_module, "run_generation", generate)
    with TestClient(create_app(tmp_path)) as client:
        state = client.post("/avatar/init", content=reference()).json()
        checkpoint = (tmp_path / "state.json").read_bytes()
        client.post("/runtime/resume")
        response = client.post("/avatar/generate", json={"seed": 123})
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        try:
            assert entered.wait(5)
            assert client.get("/health").status_code == 200
            assert client.get("/runtime/status").json()["mode"] == "generating"
            assert client.post("/avatar/generate", json={}).status_code == 409
        finally:
            release.set()
        job = wait_for_job(client, run_id)
        assert job["status"] == "completed"
        assert job["result"]["run_id"] == run_id
        assert job["result"]["avatar_id"] == state["avatar_id"]
        assert client.get("/runtime/status").json()["mode"] == "ready"
        assert (tmp_path / "state.json").read_bytes() == checkpoint
        events = client.get("/runtime/events").json()
        kinds = [event["kind"] for event in events if event["run_id"] == run_id]
        assert kinds == ["generation_submitted", "generation_started", "generation_completed"]
    with TestClient(create_app(tmp_path)) as restarted:
        assert restarted.get(f"/runtime/jobs/{run_id}").json() == job
        assert restarted.get("/runtime/status").json()["mode"] == "paused"
        assert restarted.get("/avatar/state").json() == state


def test_failure_preserves_checkpoint_and_requires_explicit_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    enable_test_backend(monkeypatch)

    def fail(*args, **kwargs):
        raise RuntimeError("backend failed for control test")

    monkeypatch.setattr(runtime_module, "run_generation", fail)
    with TestClient(create_app(tmp_path)) as client:
        client.post("/avatar/init", content=reference())
        checkpoint = (tmp_path / "state.json").read_bytes()
        client.post("/runtime/resume")
        run_id = client.post("/avatar/generate", json={}).json()["run_id"]
        job = wait_for_job(client, run_id)
        assert job["status"] == "failed"
        assert job["result"] is None
        assert "backend failed" in job["error"]
        assert client.get("/runtime/status").json()["mode"] == "error"
        assert client.post("/avatar/generate", json={}).status_code == 409
        assert client.post("/runtime/resume").json()["mode"] == "ready"
        assert (tmp_path / "state.json").read_bytes() == checkpoint


@pytest.mark.parametrize("late_result", [False, True])
def test_pause_cleans_active_job_and_rejects_late_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, late_result: bool
) -> None:
    enable_test_backend(monkeypatch)
    entered, stopped = Event(), Event()

    def generate(config, request, output, *, run_id, avatar_id, image, cancel_event):
        entered.set()
        assert cancel_event.wait(5)
        stopped.set()
        if late_result:
            return GenerationResult(
                run_id=run_id,
                avatar_id=avatar_id,
                output=output,
                videos=[output / "late.mp4"],
                elapsed_seconds=0.1,
            )
        raise GenerationCancelled("cancelled by test")

    monkeypatch.setattr(runtime_module, "run_generation", generate)
    with TestClient(create_app(tmp_path)) as client:
        client.post("/avatar/init", content=reference())
        client.post("/runtime/resume")
        run_id = client.post("/avatar/generate", json={}).json()["run_id"]
        assert entered.wait(5)
        assert client.post("/runtime/pause").json()["mode"] == "paused"
        assert stopped.is_set()
        job = client.get(f"/runtime/jobs/{run_id}").json()
        assert job["status"] == "cancelled"
        assert job["result"] is None
        assert client.get("/runtime/status").json()["active_run_id"] is None
        assert client.post("/avatar/generate", json={}).status_code == 409


def test_shutdown_waits_for_backend_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    enable_test_backend(monkeypatch)
    entered, stopped = Event(), Event()

    def generate(*args, cancel_event, **kwargs):
        entered.set()
        assert cancel_event.wait(5)
        stopped.set()
        raise GenerationCancelled("shutdown")

    monkeypatch.setattr(runtime_module, "run_generation", generate)
    app = create_app(tmp_path)
    with TestClient(app) as client:
        client.post("/avatar/init", content=reference())
        client.post("/runtime/resume")
        run_id = client.post("/avatar/generate", json={}).json()["run_id"]
        assert entered.wait(5)
    assert stopped.is_set()
    assert app.state.runtime.status()["mode"] == "stopped"
    with TestClient(create_app(tmp_path)) as restarted:
        assert restarted.get(f"/runtime/jobs/{run_id}").json()["status"] == "cancelled"


def test_restart_marks_unfinished_job_failed_without_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = initialize(tmp_path, reference())
    job = GenerationJob(run_id=uuid4(), avatar_id=state.avatar_id, request=GenerationRequest())
    path = tmp_path / "runs" / str(job.run_id) / "job.json"
    path.parent.mkdir(parents=True)
    path.write_text(job.model_dump_json(), encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("Restart must not launch a backend")

    monkeypatch.setattr(runtime_module, "run_generation", forbidden)
    with TestClient(create_app(tmp_path)) as client:
        restored = client.get(f"/runtime/jobs/{job.run_id}").json()
        assert restored["status"] == "failed"
        assert "interrupted" in restored["error"]
        assert json.loads(path.read_text())["status"] == "failed"
        assert client.get("/runtime/status").json()["mode"] == "paused"


def test_mismatched_result_cannot_complete_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    enable_test_backend(monkeypatch)

    def generate(config, request, output, *, run_id, avatar_id, image, cancel_event):
        return GenerationResult(
            run_id=uuid4(),
            avatar_id=avatar_id,
            output=output,
            videos=[output / "wrong.mp4"],
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr(runtime_module, "run_generation", generate)
    with TestClient(create_app(tmp_path)) as client:
        client.post("/avatar/init", content=reference())
        client.post("/runtime/resume")
        run_id = client.post("/avatar/generate", json={}).json()["run_id"]
        assert UUID(run_id)
        job = wait_for_job(client, run_id)
        assert job["status"] == "failed"
        assert "does not belong" in job["error"]
        assert job["result"] is None


def test_cancelled_initialization_keeps_writer_serialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, release = Event(), Event()

    def slow_initialize(root, payload):
        entered.set()
        assert release.wait(5)
        return initialize(root, payload)

    monkeypatch.setattr(runtime_module, "initialize", slow_initialize)

    async def scenario():
        runtime = runtime_module.Runtime(RuntimeConfig(data_dir=tmp_path))
        first = asyncio.create_task(runtime.initialize(reference()))
        assert await asyncio.to_thread(entered.wait, 5)
        first.cancel()
        second = asyncio.create_task(runtime.initialize(reference()))
        await asyncio.sleep(0)
        assert not second.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        with pytest.raises(FileExistsError):
            await second
        assert runtime.avatar_state() == runtime.state
        await runtime.close()

    asyncio.run(scenario())
