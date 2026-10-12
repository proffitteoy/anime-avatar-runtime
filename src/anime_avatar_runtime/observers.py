"""Pinned offline observers and a shell-free bridge to their separate environment.

Only ``prepare_models`` downloads weights. Importing this module loads no model.
CCIP's pinned upstream functions retain their own preprocessing and metric;
only their model loaders are bound to verified local CPU sessions.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import io
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Any, Literal
from uuid import UUID, uuid4

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageOps

from .config import RuntimeConfig
from .contracts import (
    AnomalyRequest,
    AnomalyResult,
    Observation,
    ObservationRequest,
    ObservationResult,
    ObservationSample,
)
from .skyreels import _run_process
from .state import MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS, atomic_write

CCIP_CODE = "46df848dc4d20ac93f4919a40e2636d5f7c19766"
CCIP_MODEL = "ccip-caformer-24-randaug-pruned"
CCIP_REVISION = "eb2acdd29af1703388d3d0c04221add322bc9110"
FACE_CODE = "98a9fb480fa04bdd96acbe4d98da191a898267f3"
MODEL_FILES = {
    "ccip_feature.onnx": ("deepghs/ccip_onnx", CCIP_REVISION, f"{CCIP_MODEL}/model_feat.onnx"),
    "ccip_metric.onnx": ("deepghs/ccip_onnx", CCIP_REVISION, f"{CCIP_MODEL}/model_metrics.onnx"),
    "yolov3.safetensors": (
        "hysts/anime-face-detector-yolov3",
        "afdd4226a79ae8bb81f334dbcffd34f8cc000c38",
        "model.safetensors",
    ),
    "hrnetv2.safetensors": (
        "hysts/anime-face-detector-hrnetv2",
        "9b3435248b26aeb82e2a8578fe9d86d5d57158af",
        "model.safetensors",
    ),
}
MODEL_SHA256 = {
    "ccip_feature.onnx": "4ea118d16496274f4f6e08d3afc768cc592389e8f7f32f8732ce2215c228ac5f",
    "ccip_metric.onnx": "7e4646fdfbe369ad485b227ec47ac4092c98b8deef3af44e19fdbd8ba7bc25c1",
    "yolov3.safetensors": "23bbc708146bcbc1c910f00fe152adbc70d7658d875a0121eaf4ee61d978b2c4",
    "hrnetv2.safetensors": "e71271376406a743c01528a0460637fcc06e72aeeea583f85007cc72dc8b7a4a",
}
# Contour, eye corners and nose; eyelids, eyebrows and lips are expression signals.
STABLE_POINTS = (0, 1, 2, 3, 4, 11, 13, 17, 19, 23)
EXPRESSION_POINTS = tuple(i for i in range(28) if i not in STABLE_POINTS)
FloatArray = NDArray[np.float64]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_image(payload: bytes) -> Image.Image:
    if not payload or len(payload) > MAX_IMAGE_BYTES:
        raise ValueError("Image must contain 1 byte to 10 MiB")
    try:
        with Image.open(io.BytesIO(payload)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image exceeds 16 million pixels")
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError("Image must be a single still image")
            source.load()
            return ImageOps.exif_transpose(source).convert("RGB")
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Image is not a valid supported image") from exc


def evaluate_anomaly(request: AnomalyRequest) -> AnomalyResult:
    """Independent threshold flags; missing or uncalibrated signals stay unknown."""
    flags: list[Literal["exceeded", "within", "unknown"]] = []
    reasons = []
    for name in ("identity", "geometry"):
        distance = getattr(request, f"{name}_distance")
        threshold = getattr(request, f"{name}_threshold")
        if distance is None or threshold is None:
            flags.append("unknown")
            reasons.append(
                f"{name}: {'measurement unavailable' if distance is None else 'threshold unset'}"
            )
        elif distance > threshold:
            flags.append("exceeded")
            reasons.append(f"{name}: {distance:g} > {threshold:g}")
        else:
            flags.append("within")
    status: Literal["anomaly", "within_thresholds", "unknown"] = "unknown"
    if "exceeded" in flags:
        status = "anomaly"
    elif all(flag == "within" for flag in flags):
        status = "within_thresholds"
    return AnomalyResult(status=status, identity=flags[0], geometry=flags[1], reasons=reasons)


def geometry_distance(
    current: FloatArray, reference: FloatArray, min_confidence: float = 0.3
) -> tuple[float, float | None]:
    """Weighted proper similarity fit; residuals are relative to reference RMS size.

    Fit and structural score use stable points only. Expression points are scored
    separately using that same transform, so blinking/mouth opening cannot drive
    the structural alignment. Confidence values are heatmap scores, not probabilities.
    """
    if current.shape != (28, 3) or reference.shape != (28, 3):
        raise ValueError("Expected 28 landmarks with x, y, confidence")
    if not np.isfinite(current).all() or not np.isfinite(reference).all():
        raise ValueError("Non-finite landmarks")
    weights = np.clip(np.minimum(current[:, 2], reference[:, 2]), 0, 1)
    weights[weights < min_confidence] = 0
    stable = np.asarray(STABLE_POINTS)
    w = weights[stable]
    if np.count_nonzero(w) < 8:
        raise ValueError("Low confidence or occlusion: fewer than 8 stable points")
    x, y = current[stable, :2], reference[stable, :2]
    x_mean, y_mean = np.average(x, axis=0, weights=w), np.average(y, axis=0, weights=w)
    x_centered, y_centered = x - x_mean, y - y_mean
    x_size = float(np.sum(w[:, None] * x_centered**2))
    y_size = float(np.sum(w[:, None] * y_centered**2))
    if min(x_size, y_size) <= 1e-10:
        raise ValueError("Degenerate face scale")
    covariance = x_centered.T @ (w[:, None] * y_centered)
    u, singular, vt = np.linalg.svd(covariance)
    if singular[1] <= 1e-6 * singular[0]:
        raise ValueError("Degenerate or collinear landmarks")
    correction = np.diag([1.0, float(np.linalg.det(u @ vt))])
    rotation = u @ correction @ vt
    scale = float(np.sum(singular * np.diag(correction)) / x_size)
    if scale <= 1e-10:
        raise ValueError("Cannot align landmarks with positive scale")
    aligned = (current[:, :2] - x_mean) @ rotation * scale + y_mean
    errors = np.sum((aligned - reference[:, :2]) ** 2, axis=1)
    rms_size_squared = y_size / float(w.sum())
    distance = float(np.sqrt(np.average(errors[stable], weights=w) / rms_size_squared))
    expression = np.asarray(EXPRESSION_POINTS)
    ew = weights[expression]
    expression_distance = (
        float(np.sqrt(np.average(errors[expression], weights=ew) / rms_size_squared))
        if np.count_nonzero(ew) >= 8
        else None
    )
    return distance, expression_distance


def face_status(faces: list[dict[str, Any]], request: ObservationRequest) -> str:
    if not faces:
        return "no_face"
    if len(faces) != 1:
        return "multiple_faces"
    points = np.asarray(faces[0]["keypoints"], dtype=np.float64)
    box = np.asarray(faces[0]["bbox"], dtype=np.float64)
    if (
        points.shape != (28, 3)
        or box.shape != (5,)
        or not np.isfinite(points).all()
        or not np.isfinite(box).all()
        or box[2] <= box[0]
        or box[3] <= box[1]
    ):
        return "invalid_detection"
    if np.count_nonzero(points[list(STABLE_POINTS), 2] >= request.min_confidence) < 8:
        return "low_confidence_or_occlusion"
    # Conservative near-frontal proxy, invariant to roll and scale; not a 3D pose estimate.
    if np.any(points[[11, 13, 17, 19, 23], 2] < request.min_confidence):
        return "pose_unavailable"
    left = points[[11, 13], :2].mean(axis=0)
    right = points[[17, 19], :2].mean(axis=0)
    axis = right - left
    eye_distance = float(axis @ axis)
    if eye_distance <= 1e-10:
        return "degenerate_face"
    nose_offset = abs(float((points[23, :2] - (left + right) / 2) @ axis / eye_distance))
    widths = [float(np.linalg.norm(points[a, :2] - points[b, :2])) for a, b in ((11, 13), (17, 19))]
    if min(widths) <= 1e-8 or max(widths) / min(widths) > 2.5 or nose_offset > 0.35:
        return "unsupported_pose"
    return "valid"


def prepare_models(models: Path) -> None:
    from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]

    models.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, (repo, revision, filename) in MODEL_FILES.items():
        cached = Path(hf_hub_download(repo, filename, revision=revision))
        if sha256(cached) != MODEL_SHA256[name]:
            raise ValueError(f"Downloaded weight does not match pinned SHA-256: {name}")
        # Stream copies instead of holding hundreds of megabytes in memory.
        import shutil

        shutil.copyfile(cached, models / name)
        files[name] = {
            "repo": repo,
            "revision": revision,
            "filename": filename,
            "sha256": sha256(models / name),
        }
    for repo, revision in sorted({(repo, revision) for repo, revision, _ in MODEL_FILES.values()}):
        card = Path(hf_hub_download(repo, "README.md", revision=revision))
        atomic_write(models / f"{repo.split('/')[-1]}-README.md", card.read_bytes())
    atomic_write(
        models / "manifest.json",
        json.dumps(
            {
                "schema_version": 1,
                "ccip_code": CCIP_CODE,
                "face_code": FACE_CODE,
                "files": files,
                "licenses": {
                    "ccip_code": "MIT",
                    "ccip_weights": "OpenRAIL",
                    "face_code": "MIT and Apache-2.0 vendored code",
                    "face_weights": "MIT (model cards)",
                },
            },
            indent=2,
        ).encode()
        + b"\n",
    )


class OfflineObserver:
    """CPU model sessions, loaded only in the isolated observation process."""

    def __init__(self, models: Path, cache: Path) -> None:
        manifest_path = models / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        if (
            manifest.get("schema_version") != 1
            or manifest.get("ccip_code") != CCIP_CODE
            or manifest.get("face_code") != FACE_CODE
        ):
            raise ValueError("Observer model/source manifest mismatch; prepare pinned models")
        for name, (repo, revision, filename) in MODEL_FILES.items():
            expected = manifest["files"][name]
            if (expected["repo"], expected["revision"], expected["filename"]) != (
                repo,
                revision,
                filename,
            ):
                raise ValueError(f"Unpinned model: {name}")
            if (
                expected["sha256"] != MODEL_SHA256[name]
                or sha256(models / name) != MODEL_SHA256[name]
            ):
                raise ValueError(f"Observer weight hash mismatch: {name}")
        # uv installs exact source archives; check the installed direct URL as well.
        for package, commit in (("dghs-imgutils", CCIP_CODE), ("anime-face-detector", FACE_CODE)):
            direct = importlib.metadata.distribution(package).read_text("direct_url.json")
            if direct is None or commit not in json.loads(direct).get("url", ""):
                raise ValueError(f"Install {package} from the pinned source archive")
        self.model_key = sha256(manifest_path)
        self.model_manifest = manifest
        self.cache = cache
        ort = importlib.import_module("onnxruntime")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        feature = ort.InferenceSession(
            str(models / "ccip_feature.onnx"), options, providers=["CPUExecutionProvider"]
        )
        metric = ort.InferenceSession(
            str(models / "ccip_metric.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self.ccip: Any = importlib.import_module("imgutils.metrics.ccip")
        # This process owns these loaders; no unpinned Hub request is reachable.
        self.ccip.__dict__["_open_feat_model"] = lambda model: feature
        self.ccip.__dict__["_open_metric_model"] = lambda model: metric
        torch = importlib.import_module("torch")
        torch.set_num_threads(2)
        detector = importlib.import_module("anime_face_detector")
        self.detector = detector.LandmarkDetector(
            models / "hrnetv2.safetensors",
            face_detector_name="yolov3",
            face_detector_checkpoint_path=models / "yolov3.safetensors",
            device="cpu",
        )

    def extract(self, image: Image.Image) -> NDArray[np.float32]:
        feature = np.asarray(
            self.ccip.ccip_extract_feature(image, model=CCIP_MODEL), dtype=np.float32
        )
        if feature.shape != (768,) or not np.isfinite(feature).all() or not np.any(feature):
            raise ValueError("Unavailable CCIP embedding")
        return feature

    def detect(self, image: Image.Image, request: ObservationRequest) -> list[dict[str, Any]]:
        # Upstream expects BGR, whereas Pillow and CCIP use RGB.
        predictions = self.detector(np.asarray(image)[:, :, ::-1].copy())
        return [
            {"bbox": p["bbox"].tolist(), "keypoints": p["keypoints"].tolist()}
            for p in predictions
            if float(p["bbox"][4]) >= request.face_confidence
        ]

    def anchor(self, path: Path, request: ObservationRequest) -> dict[str, Any]:
        digest = sha256(path)
        key = hashlib.sha256(
            f"{digest}:{self.model_key}:rgb384:v1:{request.face_confidence}".encode()
        ).hexdigest()
        location = self.cache / f"{key}.json"
        if location.is_file():
            try:
                cached: dict[str, Any] = json.loads(location.read_bytes())
                feature = np.asarray(cached["feature"], dtype=np.float32)
                if (
                    cached["input_sha256"] == digest
                    and cached["model_key"] == self.model_key
                    and feature.shape == (768,)
                    and np.isfinite(feature).all()
                    and np.any(feature)
                    and isinstance(cached.get("faces"), list)
                    and all(
                        np.asarray(face["keypoints"]).shape == (28, 3)
                        and np.asarray(face["bbox"]).shape == (5,)
                        and np.isfinite(face["keypoints"]).all()
                        and np.isfinite(face["bbox"]).all()
                        for face in cached["faces"]
                    )
                ):
                    return cached
            except (KeyError, ValueError, TypeError):
                pass
        image = read_image(path.read_bytes())
        result = {
            "input_sha256": digest,
            "model_key": self.model_key,
            "feature": self.extract(image).tolist(),
            "faces": self.detect(image, request),
        }
        atomic_write(location, json.dumps(result, allow_nan=False).encode() + b"\n")
        return result


def observe_frame(
    observer: OfflineObserver,
    image: Image.Image,
    anchors: list[tuple[str, dict[str, Any]]],
    request: ObservationRequest,
    *,
    run_id: UUID,
    avatar_id: UUID,
    frame_index: int,
    timestamp: float,
    overlay: Path,
) -> ObservationSample:
    faces: list[dict[str, Any]] = []
    identity_distance = geometry_score = expression_score = None
    identity_anchor = geometry_anchor = None
    feature_path = None
    reasons = ["motion: observer not implemented"]
    started = time.monotonic()
    try:
        faces = observer.detect(image, request)
        geometry_status = face_status(faces, request)
        if geometry_status == "invalid_detection":
            faces = []
        if geometry_status == "valid":
            scores = []
            for name, anchor in anchors:
                if face_status(anchor["faces"], request) == "valid":
                    try:
                        score, expression = geometry_distance(
                            np.asarray(faces[0]["keypoints"], dtype=np.float64),
                            np.asarray(anchor["faces"][0]["keypoints"], dtype=np.float64),
                            request.min_confidence,
                        )
                        scores.append((score, name, expression))
                    except ValueError:
                        continue
            if scores:
                geometry_score, geometry_anchor, expression_score = min(scores, key=lambda x: x[0])
            else:
                geometry_status = "no_alignable_anchor"
    except Exception as exc:
        geometry_status = f"inference_failed: {type(exc).__name__}: {exc}"
        faces = []
    geometry_elapsed = time.monotonic() - started
    if geometry_score is None:
        reasons.append(f"geometry: {geometry_status}")
    started = time.monotonic()
    try:
        if geometry_status in {"multiple_faces", "no_face"}:
            raise ValueError(f"Single character cannot be confirmed: {geometry_status}")
        feature = observer.extract(image)
        feature_path = overlay.parent.parent / "features" / f"{overlay.stem}.npy"
        encoded_feature = io.BytesIO()
        np.save(encoded_feature, feature, allow_pickle=False)
        atomic_write(feature_path, encoded_feature.getvalue())
        identity_scores = [
            (
                float(
                    observer.ccip.ccip_difference(
                        feature,
                        np.asarray(anchor["feature"], dtype=np.float32),
                        model=CCIP_MODEL,
                    )
                ),
                name,
            )
            for name, anchor in anchors
        ]
        if not identity_scores or any(
            not np.isfinite(score) or score < 0 for score, _ in identity_scores
        ):
            raise ValueError("Invalid CCIP metric output")
        identity_distance, identity_anchor = min(identity_scores)
        identity_status = "valid"
    except Exception as exc:
        identity_status = f"unavailable: {type(exc).__name__}: {exc}"
        reasons.append(f"identity: {identity_status}")
    identity_elapsed = time.monotonic() - started
    overlay_image = image.copy()
    draw = ImageDraw.Draw(overlay_image)
    for face in faces:
        box = face["bbox"]
        if len(box) == 5 and np.isfinite(box).all() and box[2] > box[0] and box[3] > box[1]:
            draw.rectangle(box[:4], outline="yellow", width=2)
        for i, (x, y, confidence) in enumerate(face["keypoints"]):
            if np.isfinite([x, y, confidence]).all():
                color = "cyan" if i in STABLE_POINTS else "orange"
                if confidence < request.min_confidence:
                    color = "red"
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
                draw.text((x + 2, y + 2), str(i), fill=color)
    draw.text((5, 5), f"frame={frame_index} t={timestamp:.3f}s {geometry_status}", fill="yellow")
    overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay_image.save(overlay)
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    measurement = Observation(
        avatar_id=avatar_id,
        run_id=run_id,
        measured_at=datetime.now(UTC),
        identity_distance=identity_distance,
        geometry_distance=geometry_score,
        unavailable_reason="; ".join(reasons),
    )
    return ObservationSample(
        frame_index=frame_index,
        timestamp_seconds=timestamp,
        input_sha256=hashlib.sha256(encoded.getvalue()).hexdigest(),
        observation=measurement,
        identity_status=identity_status,
        geometry_status=geometry_status,
        identity_anchor=identity_anchor,
        geometry_anchor=geometry_anchor,
        identity_elapsed_seconds=identity_elapsed,
        geometry_elapsed_seconds=geometry_elapsed,
        expression_distance=expression_score,
        identity_feature=str(feature_path) if feature_path is not None else None,
        faces=faces,
        overlay=str(overlay),
        anomaly=evaluate_anomaly(
            AnomalyRequest(
                identity_distance=identity_distance,
                geometry_distance=geometry_score,
                identity_threshold=request.identity_threshold,
                geometry_threshold=request.geometry_threshold,
            )
        ),
    )


def write_curves(samples: list[ObservationSample], output: Path) -> None:
    """Standalone SVG curves with gaps for unavailable samples; no plotting dependency."""
    for name in ("identity", "geometry"):
        values = [getattr(s.observation, f"{name}_distance") for s in samples]
        ymax = max([v for v in values if v is not None] + [0.01]) * 1.1
        xmax = max(samples[-1].timestamp_seconds, 1)
        elements = [
            '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="300" '
            'viewBox="0 0 900 300">',
            '<rect width="900" height="300" fill="white"/>',
            f'<text x="50" y="22">{name} distance; y=0..{ymax:.4f}; time=0..{xmax:.3f}s</text>',
            '<path d="M50 40 V260 H880" fill="none" stroke="black"/>',
        ]
        previous = None
        for sample, value in zip(samples, values, strict=True):
            x = 50 + 830 * sample.timestamp_seconds / xmax
            if value is None:
                elements.append(f'<text x="{x:.2f}" y="280" fill="red">x</text>')
                previous = None
                continue
            y = 260 - 220 * value / ymax
            if previous is not None:
                elements.append(
                    f'<path d="M{previous[0]:.2f} {previous[1]:.2f} L{x:.2f} {y:.2f}" '
                    'stroke="blue"/>'
                )
            elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="blue"/>')
            previous = (x, y)
        elements.append("</svg>")
        atomic_write(output / f"{name}.svg", "\n".join(elements).encode())


def analyze_media(spec: dict[str, Any], output: Path) -> ObservationResult:
    request = ObservationRequest.model_validate(spec["request"])
    run_id, avatar_id = UUID(spec["run_id"]), UUID(spec["avatar_id"])
    observer = OfflineObserver(Path(spec["models"]), Path(spec["cache"]))
    atomic_write(output / "models.json", json.dumps(observer.model_manifest, indent=2).encode())
    anchors = [(str(Path(p).resolve()), observer.anchor(Path(p), request)) for p in spec["anchors"]]
    if any(len(a["faces"]) > 1 for _, a in anchors):
        raise ValueError("Trusted anchors must contain a single character; multiple faces detected")
    samples: list[ObservationSample] = []
    started = time.monotonic()
    truncated = False
    capture = None
    media = Path(spec["input"])
    try:
        if spec["kind"] == "video":
            cv2 = importlib.import_module("cv2")
            capture = cv2.VideoCapture(str(media))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not capture.isOpened() or not np.isfinite(fps) or fps <= 0:
                raise ValueError("Video cannot be decoded or has invalid FPS")
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                raise ValueError("Video has no reliable frame count")
        else:
            fps, total = 1.0, 1
        last_frame = -1
        with (output / "samples.jsonl").open("w", encoding="utf-8") as stream:
            for index in range(request.max_samples):
                frame_index = round(index * request.sample_seconds * fps) if capture else 0
                if frame_index >= total:
                    break
                if frame_index == last_frame:
                    continue
                last_frame = frame_index
                if capture is not None:
                    positioned = capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame = capture.read()
                    actual_position = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
                    seek_ok = positioned and abs(actual_position - (frame_index + 1)) < 0.5
                    if not ok or not seek_ok:
                        # Preserve decode failure at its requested timestamp; never zero-fill.
                        reason = "frame_decode_failed" if not ok else "frame_seek_failed"
                        sample = ObservationSample(
                            frame_index=frame_index,
                            timestamp_seconds=frame_index / fps,
                            input_sha256=sha256(media),
                            identity_status=reason,
                            geometry_status=reason,
                            identity_elapsed_seconds=0,
                            geometry_elapsed_seconds=0,
                            observation=Observation(
                                avatar_id=avatar_id, run_id=run_id, unavailable_reason=reason
                            ),
                            anomaly=evaluate_anomaly(AnomalyRequest()),
                        )
                    else:
                        if frame.shape[0] * frame.shape[1] > MAX_IMAGE_PIXELS:
                            raise ValueError("Video frame exceeds 16 million pixels")
                        sample = observe_frame(
                            observer,
                            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                            anchors,
                            request,
                            run_id=run_id,
                            avatar_id=avatar_id,
                            frame_index=frame_index,
                            timestamp=frame_index / fps,
                            overlay=output / "overlays" / f"{frame_index:08d}.png",
                        )
                else:
                    sample = observe_frame(
                        observer,
                        read_image(media.read_bytes()),
                        anchors,
                        request,
                        run_id=run_id,
                        avatar_id=avatar_id,
                        frame_index=0,
                        timestamp=0,
                        overlay=output / "overlays" / "00000000.png",
                    )
                samples.append(sample)
                stream.write(sample.model_dump_json() + "\n")
                stream.flush()
                if capture is None:
                    break
            truncated = (
                capture is not None
                and round(request.max_samples * request.sample_seconds * fps) < total
            )
        if not samples:
            raise ValueError("No samples were decoded")
        write_curves(samples, output)
        result = ObservationResult(
            run_id=run_id,
            avatar_id=avatar_id,
            output=output,
            sample_count=len(samples),
            truncated=truncated,
            elapsed_seconds=time.monotonic() - started,
            latest=samples[-1],
        )
        atomic_write(output / "result.json", result.model_dump_json(indent=2).encode() + b"\n")
        return result
    finally:
        if capture is not None:
            capture.release()


def run_observation(
    config: RuntimeConfig,
    request: ObservationRequest,
    output: Path,
    *,
    anchors: list[Path],
    image: Path | None = None,
    video: Path | None = None,
    run_id: UUID | None = None,
    avatar_id: UUID | None = None,
    cancel_event: Event | None = None,
) -> ObservationResult:
    if config.observer_python is None or not config.observer_python.is_file():
        raise RuntimeError("Configure AVATAR_OBSERVER_PYTHON; see docs/observers.md")
    if (image is None) == (video is None) or not anchors:
        raise ValueError("Provide one image/video and at least one trusted anchor")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    media = image if image is not None else video
    assert media is not None
    spec = {
        "run_id": str(run_id or uuid4()),
        "avatar_id": str(avatar_id or uuid4()),
        "models": str(config.observer_models),
        "cache": str(config.data_dir / "anchor-cache"),
        "anchors": [str(p.resolve()) for p in anchors],
        "input": str(media.resolve()),
        "kind": "image" if image else "video",
        "request": request.model_dump(mode="json"),
    }
    atomic_write(output / "request.json", json.dumps(spec, indent=2).encode() + b"\n")
    started = time.monotonic()
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "spec": spec,
        "config": config.model_dump(mode="json"),
        "inputs": {},
        "hardware": None,
    }

    def save() -> None:
        atomic_write(output / "run.json", json.dumps(manifest, indent=2).encode() + b"\n")

    save()
    try:
        manifest["inputs"] = {str(p.resolve()): sha256(p) for p in [media, *anchors]}
        argv = [
            str(config.observer_python),
            "-m",
            "anime_avatar_runtime.observers",
            "analyze",
            "--request",
            str(output / "request.json"),
            "--output",
            str(output),
        ]
        manifest["argv"] = argv
        save()
        with (output / "backend.log").open("w", encoding="utf-8") as log:
            process = _run_process(
                argv,
                cwd=Path(__file__).resolve().parents[2],
                timeout=config.observation_timeout_seconds,
                cancel_event=cancel_event,
                log=log,
            )
        if process.returncode != 0:
            raise RuntimeError(f"Observation backend failed; inspect {output / 'backend.log'}")
        result = ObservationResult.model_validate_json((output / "result.json").read_bytes())
        if (str(result.run_id), str(result.avatar_id), result.output) != (
            spec["run_id"],
            spec["avatar_id"],
            output,
        ):
            raise ValueError("Observation result identity/path mismatch")
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Observation cancelled before accepting result")
        manifest.update(
            status="completed", sample_count=result.sample_count, truncated=result.truncated
        )
        environment = output / "environment.json"
        if environment.is_file():
            manifest["hardware"] = json.loads(environment.read_bytes())
        manifest["models"] = json.loads((output / "models.json").read_bytes())
        return result
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc) or type(exc).__name__)
        raise
    finally:
        manifest.update(
            elapsed_seconds=time.monotonic() - started, finished_at=datetime.now(UTC).isoformat()
        )
        save()


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--models", type=Path, default=Path("models/observers"))
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--request", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare_models(args.models)
        else:
            environment = {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "device": "cpu",
                "threads": 2,
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "dghs-imgutils",
                        "anime-face-detector",
                        "numpy",
                        "torch",
                        "torchvision",
                        "onnxruntime",
                        "pillow",
                        "opencv-python-headless",
                    )
                },
            }
            atomic_write(
                args.output / "environment.json", json.dumps(environment, indent=2).encode()
            )
            analyze_media(json.loads(args.request.read_bytes()), args.output)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
