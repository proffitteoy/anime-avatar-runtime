"""Opt-in real CPU experiment; publishes numbers, keeps images in the output directory.

Run in the pinned observer environment. IRLS is an experimental comparator and does
not change Runtime scoring. Local warps are controlled inputs, not generated failures.
"""

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from anime_avatar_runtime.contracts import ObservationRequest
from anime_avatar_runtime.observers import (
    STABLE_POINTS,
    OfflineObserver,
    face_status,
    geometry_distance,
    observe_frame,
    read_image,
    sha256,
)


def fit_residuals(
    current: NDArray[np.float64],
    reference: NDArray[np.float64],
    *,
    delta: float | None = None,
) -> dict[str, float | int | None]:
    """Fit Huber IRLS, but score with original weights so outliers remain visible."""
    if current.shape != (28, 3) or reference.shape != (28, 3):
        raise ValueError("Expected 28 landmarks")
    if not np.isfinite(current).all() or not np.isfinite(reference).all():
        raise ValueError("Non-finite landmarks")
    stable = list(STABLE_POINTS)
    confidence = np.clip(np.minimum(current[:, 2], reference[:, 2]), 0, 1)
    confidence[confidence < 0.3] = 0
    base = confidence[stable]
    if np.count_nonzero(base) < 8:
        raise ValueError("Fewer than 8 reliable structural points")
    x, y = current[stable, :2], reference[stable, :2]
    center = np.average(y, axis=0, weights=base)
    radius = np.sqrt(np.average(np.sum((y - center) ** 2, axis=1), weights=base))
    if radius < 1e-5:
        raise ValueError("Degenerate reference")
    weights = base.copy()
    iterations = 0
    for _iteration in range(1, 21 if delta else 2):
        iterations = _iteration
        xm, ym = np.average(x, axis=0, weights=weights), np.average(y, axis=0, weights=weights)
        xx, yy = x - xm, y - ym
        u, singular, vt = np.linalg.svd(xx.T @ (weights[:, None] * yy))
        correction = np.diag([1.0, np.linalg.det(u @ vt)])
        scale = np.sum(singular * np.diag(correction)) / np.sum(weights[:, None] * xx**2)
        if singular[1] <= 1e-6 * singular[0] or scale <= 1e-10:
            raise ValueError("Degenerate fit")
        aligned = (current[:, :2] - xm) @ (u @ correction @ vt) * scale + ym
        residual = np.linalg.norm(aligned - reference[:, :2], axis=1) / radius
        if delta is None:
            break
        updated = base * np.minimum(1, delta / np.maximum(residual[stable], 1e-12))
        if np.max(np.abs(updated - weights)) < 1e-7:
            break
        weights = updated
    regions = {
        "structure": stable,
        "eyes": list(range(11, 23)),
        "nose_mouth": list(range(23, 28)),
        "contour": list(range(5)),
    }
    scores: dict[str, float | int | None] = {}
    for name, indices in regions.items():
        w = confidence[indices]
        scores[name] = (
            float(np.sqrt(np.average(residual[indices] ** 2, weights=w)))
            if np.count_nonzero(w) >= (8 if name == "structure" else 3)
            else None
        )
    scores["iterations"] = iterations
    scores["downweighted_points"] = int(np.count_nonzero(weights < base * 0.99))
    return scores


def warp(
    image: Image.Image, points: NDArray[np.float64], kind: str, strength: float
) -> Image.Image:
    """Smooth inverse Gaussian displacements, scaled by detected interocular span."""
    import cv2

    left, right = points[[11, 13], :2].mean(0), points[[17, 19], :2].mean(0)
    span = np.linalg.norm(right - left)
    yy, xx = np.mgrid[: image.height, : image.width].astype(np.float32)
    map_x, map_y = xx.copy(), yy.copy()
    if kind == "eye_spacing":
        controls = [(left, (-strength * span, 0)), (right, (strength * span, 0))]
        sigma = span * 0.24
    elif kind == "nose_mouth":
        controls = [(points[23:28, :2].mean(0), (strength * span, 0))]
        sigma = span * 0.3
    else:
        controls = [(points[2, :2], (0, strength * span))]
        sigma = span * 0.35
    for (cx, cy), (dx, dy) in controls:
        influence = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
        map_x -= dx * influence
        map_y -= dy * influence
    return Image.fromarray(
        cv2.remap(np.asarray(image), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--normal-gif", type=Path, required=True)
    parser.add_argument("--crop", nargs=4, type=int, required=True)
    parser.add_argument("--normal-reference", type=int, default=5)
    parser.add_argument("--normal-frames", nargs="+", type=int, default=[0, 10, 15, 25, 36, 51, 56])
    parser.add_argument("--blank", type=Path, required=True)
    parser.add_argument("--multiple", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.normal_reference in args.normal_frames:
        raise ValueError("Do not count the reference itself as an independent normal sample")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    observer = OfflineObserver(args.models, args.output / "cache")
    load_seconds = time.monotonic() - started
    request = ObservationRequest()
    anchor = observer.anchor(args.anchor, request)
    if face_status(anchor["faces"], request) != "valid":
        raise ValueError("Experiment requires an alignable anchor")
    points = np.asarray(anchor["faces"][0]["keypoints"])
    source = read_image(args.anchor.read_bytes())
    cases = [("self", "self_control", source, anchor, 0)]
    cases.append(("rotate12", "similarity_control", source.rotate(12), anchor, 0))
    for kind in ("eye_spacing", "nose_mouth", "chin"):
        for strength in (0.05, 0.1, 0.2):
            cases.append(
                (
                    f"{kind}_{strength}",
                    "synthetic_local_warp",
                    warp(source, points, kind, strength),
                    anchor,
                    strength,
                )
            )
    for name, path in (("blank", args.blank), ("multiple", args.multiple)):
        cases.append((name, "invalid_subject", read_image(path.read_bytes()), anchor, None))
    with Image.open(args.normal_gif) as gif:
        gif.seek(args.normal_reference)
        normal = gif.convert("RGB").crop(tuple(args.crop))
        normal.save(args.output / "normal-reference.png")
        normal_anchor = observer.anchor(args.output / "normal-reference.png", request)
        for index in args.normal_frames:
            gif.seek(index)
            cases.append(
                (
                    f"normal_frame_{index}",
                    "visually_reviewed_normal",
                    gif.convert("RGB").crop(tuple(args.crop)),
                    normal_anchor,
                    None,
                )
            )
    samples = []
    for name, label, image, reference, strength in cases:
        image.save(args.output / f"{name}.png")
        sample = observe_frame(
            observer,
            image,
            [("reference", reference)],
            request,
            run_id=uuid4(),
            avatar_id=uuid4(),
            frame_index=len(samples),
            timestamp=0,
            overlay=args.output / "overlays" / f"{name}.png",
        )
        row: dict[str, Any] = {
            "name": name,
            "label": label,
            "strength": strength,
            "input_sha256": sha256(args.output / f"{name}.png"),
            "reference_sha256": reference["input_sha256"],
            "observation": sample.observation.model_dump(mode="json"),
            "geometry_status": sample.geometry_status,
            "identity_status": sample.identity_status,
            "identity_seconds": sample.identity_elapsed_seconds,
            "geometry_seconds": sample.geometry_elapsed_seconds,
            "faces": sample.faces,
            "wls": None,
            "huber": None,
        }
        # Private identities and time stamps are not needed for public numerical evidence.
        for key in ("avatar_id", "run_id", "measured_at"):
            row["observation"].pop(key)
        if sample.geometry_status == "valid":
            current = np.asarray(sample.faces[0]["keypoints"])
            reference_points = np.asarray(reference["faces"][0]["keypoints"])
            row["wls"] = fit_residuals(current, reference_points)
            row["huber"] = fit_residuals(current, reference_points, delta=0.05)
            assert (
                abs(row["wls"]["structure"] - geometry_distance(current, reference_points)[0])
                < 1e-9
            )
        samples.append(row)
        print(name, sample.geometry_status, row["wls"], flush=True)
    result = {
        "schema_version": 1,
        "method": "WLS vs Huber IRLS; delta=0.05 reference RMS radius; max_iterations=20",
        "score": (
            "Original confidence weighted RMS after fit; robust fitting never drops scoring weights"
        ),
        "warp": "Inverse Gaussian remap; eye/nose/chin displacement 0.05/0.1/0.2 interocular span",
        "normal_annotation": (
            "One visual reviewer: eye opening/gaze, mouth opening/closure, head tilt; "
            "no obvious local structural artifact. Weak labels, not independent ground truth."
        ),
        "normal_crop": args.crop,
        "normal_reference_frame": args.normal_reference,
        "source_hashes": {
            "anchor": sha256(args.anchor),
            "normal_gif": sha256(args.normal_gif),
            "blank": sha256(args.blank),
            "multiple": sha256(args.multiple),
        },
        "implementation_sha256": sha256(Path(__file__)),
        "observer_source_sha256": sha256(
            Path(__import__("anime_avatar_runtime.observers", fromlist=[""]).__file__)
        ),
        "models": observer.model_manifest,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "device": "cpu",
            "threads": 2,
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "numpy",
                    "torch",
                    "torchvision",
                    "onnxruntime",
                    "opencv-contrib-python-headless",
                )
            },
        },
        "load_seconds": load_seconds,
        "total_seconds": time.monotonic() - started,
        "samples": samples,
    }
    (args.output / "evidence.json").write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
