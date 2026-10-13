"""Algorithm/control tests use explicit model doubles, never claim model inference."""

import asyncio
import io
import json
import os
import runpy
from pathlib import Path
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
from anime_avatar_runtime.contracts import AnomalyRequest, GenerationRequest, ObservationRequest
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


def test_experimental_irls_preserves_anomaly_scoring_and_localizes_point_error():
    fit = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/observer_geometry_experiment.py")
    )["fit_residuals"]
    reference = landmarks()
    transformed = reference.copy()
    transformed[:, :2] = reference[:, :2] * 3 + [12, -7]
    assert fit(transformed, reference, delta=0.05)["structure"] < 1e-12
    changed = reference.copy()
    changed[0, :2] += [0, 0.8]
    wls, robust = fit(changed, reference), fit(changed, reference, delta=0.05)
    assert wls["structure"] == pytest.approx(geometry_distance(changed, reference)[0])
    assert robust["downweighted_points"] > 0
    # Robust fitting protects unaffected regions; the bad point still counts in the score.
    assert robust["eyes"] < wls["eyes"]
    assert robust["structure"] >= wls["structure"]
    changed[0, 2] = 0
    assert fit(changed, reference, delta=0.05)["structure"] < 1e-12


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
    assert sample.observation.identity_candidate_distance == 0.12
    assert sample.observation.subject_status == "ambiguous"
    assert sample.observation.subject_face_count == 2
    assert sample.identity_status == "candidate_only"
    assert sample.anomaly.identity == "unknown"
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
    assert sample.identity_feature is not None
    assert sample.observation.identity_candidate_distance == 0.12
    assert sample.observation.subject_status == "unverified"
    assert sample.observation.subject_face_count == 0
    assert sample.anomaly.identity == "unknown"


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


@pytest.mark.parametrize("failed", ["identity", "geometry"])
def test_model_components_load_independently(tmp_path, monkeypatch, failed):
    (tmp_path / "manifest.json").write_text('{"schema_version":1}')

    def identity(self, *args):
        if failed == "identity":
            raise RuntimeError("identity load failed")
        self.ccip = object()

    def geometry(self, *args):
        if failed == "geometry":
            raise RuntimeError("geometry load failed")
        self.detector = object()

    monkeypatch.setattr(observers.OfflineObserver, "_load_identity", identity)
    monkeypatch.setattr(observers.OfflineObserver, "_load_geometry", geometry)
    observer = observers.OfflineObserver(tmp_path, tmp_path / "cache")
    assert failed in observer.errors
    healthy = "geometry" if failed == "identity" else "identity"
    assert observer.model_manifest["components"][healthy] == "ready"


def test_partial_anchor_failure_does_not_block_identity_or_cache_failure(tmp_path):
    observer = observers.OfflineObserver.__new__(observers.OfflineObserver)
    observer.model_key, observer.cache = "model", tmp_path / "cache"
    observer.extract = lambda image: np.ones(768, dtype=np.float32)

    def fail(*args):
        raise RuntimeError("face detector failed")

    observer.detect = fail
    image = tmp_path / "anchor.png"
    Image.new("RGB", (20, 20)).save(image)
    anchor = observer.anchor(image, ObservationRequest())
    assert len(anchor["feature"]) == 768
    assert anchor["errors"]["geometry"] == "face detector failed"
    assert not list(observer.cache.glob("*.json"))
    observer.detect = lambda *args: [face()]
    assert "errors" not in observer.anchor(image, ObservationRequest())
    assert len(list(observer.cache.glob("*.json"))) == 1


def test_environment_report_does_not_require_both_optional_components(tmp_path, monkeypatch):
    request = tmp_path / "request.json"
    request.write_text("{}")
    calls = []

    def version(name):
        if name == "anime-face-detector":
            raise observers.importlib.metadata.PackageNotFoundError(name)
        return "explicit environment test"

    monkeypatch.setattr(observers.importlib.metadata, "version", version)
    monkeypatch.setattr(observers, "analyze_media", lambda *args: calls.append(args))
    monkeypatch.setattr(
        observers.sys,
        "argv",
        ["observers", "analyze", "--request", str(request), "--output", str(tmp_path)],
    )
    assert observers.main() == 0
    assert len(calls) == 1
    environment = json.loads((tmp_path / "environment.json").read_bytes())
    assert environment["packages"]["anime-face-detector"] is None
    assert environment["packages"]["dghs-imgutils"] == "explicit environment test"


@pytest.mark.parametrize("scope", ["failed_detection", "unsupported_pose", "multi_anchor"])
def test_identity_candidate_survives_geometry_restrictions(tmp_path, scope):
    model = ModelDouble()
    anchor_faces = [face()]
    if scope == "failed_detection":

        def fail(*args):
            raise RuntimeError("detector unavailable")

        model.detect = fail
    elif scope == "unsupported_pose":
        points = landmarks()
        points[23, 0] = 2
        model.faces = [face(points)]
    else:
        anchor_faces *= 2
    sample = observe_frame(
        model,
        Image.new("RGB", (20, 20)),
        [("anchor", {"feature": [1] * 768, "faces": anchor_faces})],
        ObservationRequest(identity_threshold=0.1),
        run_id=uuid4(),
        avatar_id=uuid4(),
        frame_index=0,
        timestamp=0,
        overlay=tmp_path / "overlays" / "sample.png",
    )
    assert sample.observation.identity_candidate_distance == 0.12
    assert sample.observation.geometry_distance is None
    if scope == "unsupported_pose":
        assert sample.observation.identity_distance == 0.12
        assert sample.anomaly.identity == "exceeded"
    else:
        assert sample.observation.identity_distance is None
        assert sample.anomaly.identity == "unknown"


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
        assert observed.json()["latest"]["observation"]["state_revision"] == 0
        assert observed.json()["latest"]["observation"]["identity_candidate_distance"] == 0.12
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
    with pytest.raises(RuntimeError, match="weight hash mismatch"):
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


@pytest.mark.parametrize("control", ["pause", "close", "resume"])
@pytest.mark.parametrize("late_result", [False, True])
def test_controls_cancel_observation_without_holding_lock_and_reject_stale_result(
    tmp_path, monkeypatch, control, late_result
):
    python = tmp_path / "python"
    python.touch()
    models = tmp_path / "models"
    models.mkdir()
    (models / "manifest.json").write_text("{}")
    instance = runtime.Runtime(
        RuntimeConfig(data_dir=tmp_path, observer_python=python, observer_models=models)
    )
    entered, cancelled, release = Event(), Event(), Event()

    def run(config, request, output, *, cancel_event, run_id, avatar_id, **kwargs):
        entered.set()
        assert cancel_event.wait(5)
        cancelled.set()
        assert release.wait(5)
        if not late_result:
            raise GenerationCancelled("explicit control test")
        if control == "resume":
            # Verify the revision guard independently of the cooperative cancel flag.
            cancel_event.clear()
        sample = observe_frame(
            ModelDouble(),
            Image.new("RGB", (20, 20)),
            [("anchor", {"feature": [1] * 768, "faces": [face()]})],
            request,
            run_id=run_id,
            avatar_id=avatar_id,
            frame_index=0,
            timestamp=0,
            overlay=tmp_path / "overlays" / "late.png",
        )
        return observers.ObservationResult(
            run_id=run_id,
            avatar_id=avatar_id,
            output=output,
            latest=sample,
            sample_count=1,
            truncated=False,
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr(runtime, "run_observation", run)

    async def scenario():
        await instance.initialize(image_bytes())
        checkpoint = (tmp_path / "state.json").read_bytes()
        revision = instance.status()["state_revision"]
        observing = asyncio.create_task(instance.observe(image_bytes(), ObservationRequest()))
        stopping = None
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            with pytest.raises(runtime.RuntimeConflict, match="already active"):
                await instance.observe(image_bytes(), ObservationRequest())
            with pytest.raises(runtime.RuntimeConflict, match="already active"):
                await instance.submit(GenerationRequest())
            stopping = asyncio.create_task(getattr(instance, control)())
            assert await asyncio.to_thread(cancelled.wait, 5)
            # The subprocess is still cleaning up, but a controller can acquire the lock.
            await asyncio.wait_for(instance._control_lock.acquire(), 1)
            instance._control_lock.release()
            assert instance.status()["state_revision"] > revision
            if control != "resume":
                assert not stopping.done()
                with pytest.raises(runtime.RuntimeConflict):
                    await instance.resume()
                # Client disconnect must not abandon owned process cleanup.
                stopping.cancel()
                await asyncio.sleep(0)
                assert not stopping.done()
        finally:
            release.set()
        with pytest.raises(runtime.RuntimeConflict, match="cancelled or stale|cancelled"):
            await observing
        if control != "resume":
            with pytest.raises(asyncio.CancelledError):
                await stopping
        else:
            await stopping
        assert not instance.status()["observation_active"]
        assert "observation_recorded" not in [event.kind for event in instance.events()]
        assert (tmp_path / "state.json").read_bytes() == checkpoint
        await instance.close()

    asyncio.run(scenario())


def test_real_observer_inference_opt_in(tmp_path):
    """Configured runs must infer; ordinary CI explicitly skips unavailable models."""
    names = ("OBSERVER_TEST_PYTHON", "OBSERVER_TEST_MODELS", "OBSERVER_TEST_IMAGE")
    if not any(os.environ.get(name) for name in names):
        pytest.skip("Set OBSERVER_TEST_PYTHON/MODELS/IMAGE for real CPU inference")
    assert all(os.environ.get(name) for name in names), "Supply all three real-model paths"
    python, models, image = (Path(os.environ[name]).resolve() for name in names)
    assert python.is_file() and image.is_file() and (models / "manifest.json").is_file()
    result = run_observation(
        RuntimeConfig(observer_python=python, observer_models=models, data_dir=tmp_path),
        ObservationRequest(),
        tmp_path / "real",
        anchors=[image],
        image=image,
    )
    sample = result.latest
    assert sample.identity_status == sample.geometry_status == "valid"
    assert sample.observation.subject_status == "single_face_proxy"
    assert sample.observation.identity_distance < 1e-4
    assert sample.observation.geometry_distance < 1e-8
    feature = np.load(sample.identity_feature, allow_pickle=False)
    assert feature.shape == (768,) and np.isfinite(feature).all() and np.any(feature)
    assert np.asarray(sample.faces[0]["keypoints"]).shape == (28, 3)
    actual_models = json.loads((result.output / "models.json").read_bytes())
    assert actual_models["ccip_code"] == observers.CCIP_CODE
    assert actual_models["face_code"] == observers.FACE_CODE
    assert actual_models["components"] == {"identity": "ready", "geometry": "ready"}
