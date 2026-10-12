"""Algorithm/control tests use explicit model doubles, never claim model inference."""

import asyncio
import io
import json
from threading import Event
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from anime_avatar_runtime import observers, runtime
from anime_avatar_runtime.api import create_app
from anime_avatar_runtime.config import RuntimeConfig
from anime_avatar_runtime.contracts import AnomalyRequest, ObservationRequest
from anime_avatar_runtime.observers import (
    EXPRESSION_POINTS,
    STABLE_POINTS,
    evaluate_anomaly,
    face_status,
    geometry_distance,
    observe_frame,
    run_observation,
)
from anime_avatar_runtime.skyreels import GenerationCancelled


def landmarks():
    points = np.random.default_rng(42).uniform(-1, 1, (28, 3))
    points[:, 2] = 1
    points[11, :2], points[13, :2] = (-1, 0), (-0.5, 0)
    points[17, :2], points[19, :2] = (0.5, 0), (1, 0)
    points[23, :2] = (0, 0.5)
    return points


def face(points=None):
    return {
        "bbox": [-2, -2, 2, 2, 0.9],
        "keypoints": (landmarks() if points is None else points).tolist(),
    }


def test_geometry_removes_similarity_but_retains_structure_and_reflection():
    reference = landmarks()
    theta = 0.7
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    transformed = reference.copy()
    transformed[:, :2] = reference[:, :2] @ rotation * 7 + [80, -35]
    assert geometry_distance(transformed, reference)[0] < 1e-12
    distorted = reference.copy()
    distorted[2, :2] += [0, 1.5]
    assert geometry_distance(distorted, reference)[0] > 0.2
    reflected = reference.copy()
    reflected[:, 0] *= -1
    assert geometry_distance(reflected, reference)[0] > 0.1


def test_expression_is_separate_and_confidence_excludes_occlusion():
    reference = landmarks()
    changed = reference.copy()
    changed[list(EXPRESSION_POINTS), 1] += 0.5
    structural, expression = geometry_distance(changed, reference)
    assert structural < 1e-12
    assert expression > 0.2
    changed = reference.copy()
    changed[0, :2] += 1000
    changed[0, 2] = 0.1
    assert geometry_distance(changed, reference)[0] < 1e-12
    changed[list(STABLE_POINTS[:3]), 2] = 0
    with pytest.raises(ValueError, match="Low confidence"):
        geometry_distance(changed, reference)


def test_degenerate_and_nonfinite_geometry_are_unavailable():
    collapsed = landmarks()
    collapsed[:, :2] = 0
    with pytest.raises(ValueError, match="Degenerate"):
        geometry_distance(collapsed, collapsed)
    points = landmarks()
    points[3, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        geometry_distance(points, landmarks())


def test_face_scope_is_explicit():
    request = ObservationRequest()
    assert face_status([], request) == "no_face"
    assert face_status([face(), face()], request) == "multiple_faces"
    assert face_status([face()], request) == "valid"
    points = landmarks()
    points[23, 0] = 2
    assert face_status([face(points)], request) == "unsupported_pose"
    points[list(STABLE_POINTS), 2] = 0.1
    assert face_status([face(points)], request) == "low_confidence_or_occlusion"


def test_anomaly_unknown_is_not_normal_and_threshold_is_not_implicit():
    assert evaluate_anomaly(AnomalyRequest()).status == "unknown"
    assert evaluate_anomaly(AnomalyRequest(identity_distance=0.9)).identity == "unknown"
    result = evaluate_anomaly(AnomalyRequest(identity_distance=0.3, identity_threshold=0.2))
    assert (result.status, result.identity, result.geometry) == ("anomaly", "exceeded", "unknown")
    assert (
        evaluate_anomaly(AnomalyRequest(identity_distance=0.1, identity_threshold=0.2)).status
        == "unknown"
    )
    result = evaluate_anomaly(
        AnomalyRequest(
            identity_distance=0.2,
            identity_threshold=0.2,
            geometry_distance=0.1,
            geometry_threshold=0.2,
        )
    )
    assert result.status == "within_thresholds"
    for kwargs in (
        {"identity_distance": np.nan},
        {"geometry_threshold": np.inf},
        {"identity_threshold": 0},
        {"geometry_distance": -1},
    ):
        with pytest.raises(ValidationError):
            AnomalyRequest(**kwargs)


class ModelDouble:
    ccip = SimpleNamespace(ccip_difference=lambda *args, **kwargs: 0.12)

    def __init__(self, faces=None, fail=False):
        self.faces = [face()] if faces is None else faces
        self.fail = fail

    def detect(self, image, request):
        return self.faces

    def extract(self, image):
        if self.fail:
            raise RuntimeError("explicit test failure")
        return np.ones(768, dtype=np.float32)


def test_independent_signals_do_not_zero_fill_failure(tmp_path):
    anchors = [("trusted.png", {"feature": [1] * 768, "faces": [face()]})]
    sample = observe_frame(
        ModelDouble(fail=True),
        Image.new("RGB", (20, 20)),
        anchors,
        ObservationRequest(),
        run_id=uuid4(),
        avatar_id=uuid4(),
        frame_index=17,
        timestamp=0.7,
        overlay=tmp_path / "overlay.png",
    )
    assert sample.observation.identity_distance is None
    assert sample.observation.geometry_distance < 1e-12
    assert "explicit test failure" in sample.identity_status
    assert sample.anomaly.status == "unknown"
    assert (sample.frame_index, sample.timestamp_seconds) == (17, 0.7)
    sample = observe_frame(
        ModelDouble(faces=[face(), face()]),
        Image.new("RGB", (20, 20)),
        anchors,
        ObservationRequest(),
        run_id=uuid4(),
        avatar_id=uuid4(),
        frame_index=0,
        timestamp=0,
        overlay=tmp_path / "multi.png",
    )
    assert sample.observation.geometry_distance is None
    assert sample.observation.identity_distance is None
    assert sample.geometry_status == "multiple_faces"
    sample = observe_frame(
        ModelDouble(faces=[]),
        Image.new("RGB", (20, 20)),
        anchors,
        ObservationRequest(),
        run_id=uuid4(),
        avatar_id=uuid4(),
        frame_index=1,
        timestamp=1,
        overlay=tmp_path / "no-face.png",
    )
    assert sample.observation.identity_distance is None
    assert sample.observation.geometry_distance is None
    assert sample.identity_feature is None


def test_anchor_cache_changes_with_input_and_model_and_repairs_corruption(tmp_path):
    # Only cache behavior is exercised here; features are explicitly synthetic.
    observer = observers.OfflineObserver.__new__(observers.OfflineObserver)
    observer.model_key = "model-a"
    observer.cache = tmp_path / "cache"
    calls = []
    observer.extract = lambda image: calls.append(1) or np.ones(768, dtype=np.float32)
    observer.detect = lambda image, request: [face()]
    image = tmp_path / "anchor.png"
    Image.new("RGB", (20, 20), "red").save(image)
    observer.anchor(image, ObservationRequest())
    observer.anchor(image, ObservationRequest())
    assert len(calls) == 1
    next(observer.cache.glob("*.json")).write_text("broken", encoding="utf-8")
    observer.anchor(image, ObservationRequest())
    assert len(calls) == 2
    Image.new("RGB", (20, 20), "blue").save(image)
    observer.anchor(image, ObservationRequest())
    observer.model_key = "model-b"
    observer.anchor(image, ObservationRequest())
    assert len(calls) == 4


def image_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (20, 20), "red").save(stream, format="PNG")
    return stream.getvalue()


def test_api_observe_and_anomaly_contract(tmp_path, monkeypatch):
    config = RuntimeConfig(data_dir=tmp_path)
    with TestClient(create_app(config=config)) as client:
        assert client.post("/observer/observe", content=image_bytes()).status_code == 404
        assert client.post("/avatar/init", content=image_bytes()).status_code == 201
        assert client.post("/observer/observe", content=image_bytes()).status_code == 503
        assert client.post("/observer/anomalies", json={}).json()["status"] == "unknown"
        assert (
            client.post("/observer/anomalies", json={"geometry_threshold": -1}).status_code == 422
        )
    python = tmp_path / "python"
    python.touch()
    models = tmp_path / "models"
    models.mkdir()
    (models / "manifest.json").write_text("{}")
    config = RuntimeConfig(data_dir=tmp_path, observer_python=python, observer_models=models)

    def run(config, request, output, *, run_id, avatar_id, **kwargs):
        assert kwargs["anchors"][0].parent.name == "references"
        sample = observe_frame(
            ModelDouble(),
            Image.new("RGB", (20, 20)),
            [("trusted", {"feature": [1] * 768, "faces": [face()]})],
            request,
            run_id=run_id,
            avatar_id=avatar_id,
            frame_index=0,
            timestamp=0,
            overlay=tmp_path / "test-overlay.png",
        )
        return observers.ObservationResult(
            run_id=run_id,
            avatar_id=avatar_id,
            output=output,
            sample_count=1,
            truncated=False,
            elapsed_seconds=0.1,
            latest=sample,
        )

    monkeypatch.setattr(runtime, "run_observation", run)
    with TestClient(create_app(config=config)) as client:
        checkpoint = (tmp_path / "state.json").read_bytes()
        assert client.post("/observer/observe", content=b"bad").status_code == 422
        assert (
            client.post(
                "/observer/observe?identity_threshold=nan", content=image_bytes()
            ).status_code
            == 422
        )
        observed = client.post("/observer/observe?identity_threshold=0.1", content=image_bytes())
        assert observed.status_code == 200
        assert observed.json()["latest"]["anomaly"]["identity"] == "exceeded"
        assert client.get("/observer/status").json()["identity_distance"] == 0.12
        assert client.get("/runtime/events").json()[-1]["kind"] == "observation_recorded"
        assert (tmp_path / "state.json").read_bytes() == checkpoint


def test_bridge_records_real_subprocess_failure_without_result(tmp_path, monkeypatch):
    python = tmp_path / "python"
    python.touch()
    image = tmp_path / "image.png"
    image.write_bytes(image_bytes())
    monkeypatch.setattr(observers, "_run_process", lambda *a, **k: SimpleNamespace(returncode=1))
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="backend failed"):
        run_observation(
            RuntimeConfig(observer_python=python),
            ObservationRequest(),
            output,
            image=image,
            anchors=[image],
        )
    assert json.loads((output / "run.json").read_bytes())["status"] == "failed"
    assert not (output / "result.json").exists()
    with pytest.raises(FileExistsError):
        run_observation(
            RuntimeConfig(observer_python=python),
            ObservationRequest(),
            output,
            image=image,
            anchors=[image],
        )


def test_tampered_weight_manifest_fails_before_model_import(tmp_path):
    manifest = {
        "schema_version": 1,
        "ccip_code": observers.CCIP_CODE,
        "face_code": observers.FACE_CODE,
        "files": {},
    }
    for name, (repo, revision, filename) in observers.MODEL_FILES.items():
        (tmp_path / name).write_bytes(b"tampered, not a real model")
        manifest["files"][name] = {
            "repo": repo,
            "revision": revision,
            "filename": filename,
            "sha256": observers.sha256(tmp_path / name),
        }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="weight hash mismatch"):
        observers.OfflineObserver(tmp_path, tmp_path / "cache")


def test_video_sampling_preserves_decode_failures_and_marks_limit(tmp_path, monkeypatch):
    class Capture:
        frame = -1

        def get(self, field):
            return {1: 10, 2: 60, 3: self.frame + 1}[field]

        def isOpened(self):
            return True

        def set(self, field, value):
            self.frame = value
            return True

        def read(self):
            return (
                (False, None) if self.frame == 10 else (True, np.zeros((20, 20, 3), dtype=np.uint8))
            )

        def release(self):
            pass

    fake_cv2 = SimpleNamespace(
        VideoCapture=lambda path: Capture(),
        CAP_PROP_FPS=1,
        CAP_PROP_FRAME_COUNT=2,
        CAP_PROP_POS_FRAMES=3,
        COLOR_BGR2RGB=4,
        cvtColor=lambda frame, code: frame,
    )
    model = ModelDouble()
    model.model_manifest = {"explicit": "model double"}
    model.anchor = lambda path, request: {"feature": [1] * 768, "faces": [face()]}
    monkeypatch.setattr(observers, "OfflineObserver", lambda *args: model)
    monkeypatch.setattr(observers.importlib, "import_module", lambda name: fake_cv2)
    media = tmp_path / "test.mp4"
    media.write_bytes(b"explicit fake capture test; not a real video")
    spec = {
        "request": {"sample_seconds": 1, "max_samples": 3},
        "run_id": str(uuid4()),
        "avatar_id": str(uuid4()),
        "models": str(tmp_path),
        "cache": str(tmp_path),
        "anchors": [str(tmp_path / "anchor.png")],
        "input": str(media),
        "kind": "video",
    }
    result = observers.analyze_media(spec, tmp_path)
    samples = [json.loads(line) for line in (tmp_path / "samples.jsonl").read_text().splitlines()]
    assert [s["frame_index"] for s in samples] == [0, 10, 20]
    assert [s["timestamp_seconds"] for s in samples] == [0, 1, 2]
    assert samples[1]["identity_status"] == samples[1]["geometry_status"] == "frame_decode_failed"
    assert samples[1]["observation"]["identity_distance"] is None
    assert samples[1]["observation"]["geometry_distance"] is None
    assert result.truncated is True
    assert result.sample_count == 3


def test_cancelled_observation_releases_process_before_close(tmp_path, monkeypatch):
    python = tmp_path / "python"
    python.touch()
    models = tmp_path / "models"
    models.mkdir()
    (models / "manifest.json").write_text("{}")
    config = RuntimeConfig(data_dir=tmp_path, observer_python=python, observer_models=models)
    entered, cancelled, release = Event(), Event(), Event()

    def run(*args, cancel_event, **kwargs):
        entered.set()
        assert cancel_event.wait(5)
        cancelled.set()
        assert release.wait(5)
        raise GenerationCancelled("explicit observation cancellation test")

    monkeypatch.setattr(runtime, "run_observation", run)

    async def scenario():
        instance = runtime.Runtime(config)
        await instance.initialize(image_bytes())
        task = asyncio.create_task(instance.observe(image_bytes(), ObservationRequest()))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            assert await asyncio.to_thread(cancelled.wait, 5)
            closing = asyncio.create_task(instance.close())
            await asyncio.sleep(0)
            assert not closing.done()
            assert instance.status()["observation_active"] is True
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await closing
        assert instance.status()["observation_active"] is False
        assert instance.observation().identity_distance is None
        assert "observation_recorded" not in [event.kind for event in instance.events()]

    asyncio.run(scenario())
