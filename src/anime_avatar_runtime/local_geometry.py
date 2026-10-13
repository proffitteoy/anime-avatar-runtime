"""Offline local geometry and calibration. Never imported by the Runtime/API.

Human labels, independent source groups and immutable stage bindings are required
for statistical conclusions. Model scores are observations, never ground truth.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import ObservationRequest
from .observers import STABLE_POINTS, FloatArray, face_status, geometry_distance

POSES = ("front", "left_turn", "right_turn", "down", "up")
REGIONS = {
    "left_eye": (11, 13),
    "right_eye": (17, 19),
    "eye_spacing": (11, 13, 17, 19),
    "nose": (23,),
    "mouth": (24, 26),
    "contour": (0, 1, 2, 3, 4),
}
VARIANTS = ("wls", "single_local", "pose_local", "oracle", "intrinsic_local")
POLICY = {
    "min_confidence": 0.3,
    "mad_floor": 1e-6,
    "min_development": 20,
    "min_calibration": 39,
    "min_anomaly_sources": 20,
    "alpha": 0.05,
    "coverage_target": 0.8,
    "bootstrap_repeats": 2000,
    "seed": 42,
}
Json = dict[str, Any]


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reviewer: str = Field(min_length=1)
    kind: Literal["human", "assistant"]
    label: Literal["normal", "geometry_anomaly", "uncertain"]
    pose: Literal["front", "left_turn", "right_turn", "down", "up", "unsupported", "unknown"]
    mode: Literal["idle", "interaction", "unknown"]
    occlusion: Literal["none", "mild", "severe", "unknown"]
    parts: list[Literal["left_eye", "right_eye", "eye_spacing", "nose", "mouth", "contour"]] = (
        Field(default_factory=list)
    )
    severity: Literal["none", "mild", "moderate", "severe", "unknown"] = "unknown"
    approve_anchor: bool = False


class StudySample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    source_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image: Path
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_id: str | None = None
    primary: bool = False
    frame_index: int | None = Field(default=None, ge=0)
    timestamp_seconds: float | None = Field(default=None, ge=0)
    transform: Json | None = None
    reviews: list[Review] = Field(default_factory=list)
    adjudication: Review | None = None


class StudyManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal[1] = 1
    character_id: str = Field(min_length=1)
    samples: list[StudySample]
    provenance: Json = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_groups(self) -> Self:
        by_id = {s.sample_id: s for s in self.samples}
        if len(by_id) != len(self.samples):
            raise ValueError("Duplicate sample IDs")
        source_hashes: dict[str, str] = {}
        image_groups: dict[str, str] = {}
        primaries: set[str] = set()
        for s in self.samples:
            previous = source_hashes.setdefault(s.source_sha256, s.source_id)
            if previous != s.source_id:
                raise ValueError(
                    "One source hash cannot be counted as multiple independent sources"
                )
            previous = image_groups.setdefault(s.input_sha256, s.source_id)
            if previous != s.source_id:
                raise ValueError("Duplicate image across independent source groups")
            if s.primary:
                if s.parent_id is not None or s.transform is not None or s.source_id in primaries:
                    raise ValueError("Only one preselected original primary per source")
                primaries.add(s.source_id)
            visited = {s.sample_id}
            current = s
            while current.parent_id is not None:
                if current.parent_id not in by_id or current.parent_id in visited:
                    raise ValueError("Missing or cyclic parent")
                visited.add(current.parent_id)
                parent = by_id[current.parent_id]
                if (parent.source_id, parent.source_sha256) != (s.source_id, s.source_sha256):
                    raise ValueError("Derived samples must inherit the complete source group")
                current = parent
        return self


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def implementation_hash() -> str:
    return digest(
        {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("local_geometry.py", "observers.py")
        }
    )


def data_hash(manifest: StudyManifest) -> str:
    value = manifest.model_dump(mode="json")
    for sample in value["samples"]:
        sample.pop("reviews")
        sample.pop("adjudication")
    return digest(value)


def resolved_review(sample: StudySample) -> Json:
    human = [r for r in sample.reviews if r.kind == "human"]
    if sample.adjudication is not None:
        if sample.adjudication.kind != "human" or not human:
            raise ValueError("Adjudication needs human reviews and a human adjudicator")
        return sample.adjudication.model_dump() | {"review_status": "adjudicated"}
    if not human:
        return {"label": "uncertain", "review_status": "pending_human"}
    comparable = [
        {k: v for k, v in r.model_dump().items() if k not in {"reviewer", "approve_anchor"}}
        for r in human
    ]
    if any(r != comparable[0] for r in comparable[1:]):
        return {"label": "uncertain", "review_status": "disagreement"}
    return human[0].model_dump() | {
        "review_status": "confirmed",
        "approve_anchor": all(r.approve_anchor for r in human),
    }


def in_scope(review: Json) -> bool:
    return (
        review.get("label") in {"normal", "geometry_anomaly"}
        and review.get("pose") in POSES
        and review.get("mode") == "idle"
        and review.get("occlusion") in {"none", "mild"}
    )


def split_sources(manifest: StudyManifest) -> dict[str, str]:
    sources = sorted({s.source_id for s in manifest.samples})
    np.random.default_rng(42).shuffle(sources)
    n_dev = min(len(sources), max(1, round(len(sources) * 0.6))) if sources else 0
    n_cal = min(len(sources) - n_dev, round(len(sources) * 0.2))
    return {
        source: "development" if i < n_dev else "calibration" if i < n_dev + n_cal else "validation"
        for i, source in enumerate(sources)
    }


def points(record: Json) -> FloatArray | None:
    faces = record.get("faces", [])
    if len(faces) != 1:
        return None
    p = np.asarray(faces[0]["keypoints"], dtype=np.float64)
    return p if p.shape == (28, 3) and np.isfinite(p).all() else None


def aligned_residuals(
    current: FloatArray, reference: FloatArray, fit_points: tuple[int, ...]
) -> tuple[FloatArray, FloatArray, float, float]:
    """Fit only complementary points; radius comes from the reference, not the ROI."""
    if current.shape != (28, 3) or reference.shape != (28, 3):
        raise ValueError("Expected 28 landmarks")
    if not np.isfinite(current).all() or not np.isfinite(reference).all():
        raise ValueError("Non-finite landmarks")
    weights = np.clip(np.minimum(current[:, 2], reference[:, 2]), 0, 1)
    weights[weights < POLICY["min_confidence"]] = 0
    fit = list(fit_points)
    w = weights[fit]
    if np.count_nonzero(w) < max(4, math.ceil(len(fit) * 0.75)):
        raise ValueError("Insufficient complementary fit points")
    x, y = current[fit, :2], reference[fit, :2]
    xm, ym = np.average(x, axis=0, weights=w), np.average(y, axis=0, weights=w)
    xx, yy = x - xm, y - ym
    x_size = float(np.sum(w[:, None] * xx**2))
    if x_size <= 1e-10:
        raise ValueError("Degenerate current scale")
    u, singular, vt = np.linalg.svd(xx.T @ (w[:, None] * yy))
    if singular[0] <= 1e-10 or singular[1] <= 1e-6 * singular[0]:
        raise ValueError("Degenerate complementary geometry")
    correction = np.diag([1.0, float(np.linalg.det(u @ vt))])
    scale = float(np.sum(singular * np.diag(correction)) / x_size)
    if scale <= 1e-10:
        raise ValueError("Cannot align with positive scale")
    aligned = (current[:, :2] - xm) @ (u @ correction @ vt) * scale + ym
    stable = list(STABLE_POINTS)
    rw = np.clip(reference[stable, 2], 0, 1)
    rw[rw < POLICY["min_confidence"]] = 0
    if np.count_nonzero(rw) < 4:
        raise ValueError("Insufficient reference scale points")
    center = np.average(reference[stable, :2], axis=0, weights=rw)
    radius = float(
        np.sqrt(np.average(np.sum((reference[stable, :2] - center) ** 2, axis=1), weights=rw))
    )
    if radius <= 1e-10:
        raise ValueError("Degenerate reference scale")
    residual = np.linalg.norm(aligned - reference[:, :2], axis=1) / radius
    cost = float(np.sqrt(np.average(residual[fit] ** 2, weights=w)))
    return aligned, residual, radius, cost


def local_metrics(current: FloatArray, reference: FloatArray) -> Json:
    result: Json = {}
    for name, indices in REGIONS.items():
        try:
            fit = tuple(i for i in STABLE_POINTS if i not in indices)
            aligned, residual, radius, cost = aligned_residuals(current, reference, fit)
            w = np.clip(np.minimum(current[list(indices), 2], reference[list(indices), 2]), 0, 1)
            w[w < POLICY["min_confidence"]] = 0
            required = 3 if name == "contour" else len(indices)
            if np.count_nonzero(w) < required or (name == "contour" and w[2] == 0):
                raise ValueError("Insufficient target points")
            value = float(np.sqrt(np.average(residual[list(indices)] ** 2, weights=w)))
            shape: Json = {}
            if name in {"left_eye", "right_eye", "mouth"}:
                a, b = indices
                shape["width_error"] = (
                    abs(
                        float(np.linalg.norm(aligned[a] - aligned[b]))
                        - float(np.linalg.norm(reference[a, :2] - reference[b, :2]))
                    )
                    / radius
                )
            if name == "eye_spacing":
                left, right = aligned[[11, 13]].mean(0), aligned[[17, 19]].mean(0)
                rl, rr = reference[[11, 13], :2].mean(0), reference[[17, 19], :2].mean(0)
                shape["center_relation_error"] = (
                    float(np.linalg.norm((right - left) - (rr - rl))) / radius
                )
            expression_pair = {"left_eye": (12, 15), "right_eye": (18, 21), "mouth": (25, 27)}
            if name in expression_pair:
                a, b = expression_pair[name]
                shape["opening"] = (
                    float(np.linalg.norm(aligned[a] - aligned[b])) / radius
                    if min(current[a, 2], current[b, 2]) >= POLICY["min_confidence"]
                    else None
                )
            result[name] = {"value": value, "fit_cost": cost, "status": "valid", **shape}
        except ValueError as exc:
            result[name] = {"value": None, "fit_cost": None, "status": str(exc)}
    return result


def contour_shape(points: FloatArray) -> FloatArray:
    """Signed contour ratios in its own proper-similarity-invariant local frame.

    This frontal research observable uses all five contour points. It removes
    cross-region fitting, not detector coupling or three-dimensional pose effects.
    """
    if points.shape != (28, 3) or not np.isfinite(points).all():
        raise ValueError("Expected finite 28 landmarks")
    if np.any(points[:5, 2] < POLICY["min_confidence"]):
        raise ValueError("All five contour points must be reliable")
    a, b = points[0, :2], points[4, :2]
    axis = b - a
    width = float(np.linalg.norm(axis))
    if width <= 1e-8:
        raise ValueError("Degenerate contour width")
    along = axis / width
    normal = np.array([-along[1], along[0]])
    relative = (points[[1, 2, 3], :2] - (a + b) / 2) / width
    return np.array(
        [
            relative[1] @ normal,
            relative[1] @ along,
            relative[0] @ normal,
            relative[2] @ normal,
            relative[0] @ along,
            relative[2] @ along,
        ]
    )


def feature_variants(current: FloatArray | None, frozen: Json, oracle_pose: str | None) -> Json:
    empty = {r: {"value": None, "status": "No usable reference or detection"} for r in REGIONS}
    result: Json = {
        "wls": {"global": {"value": None, "status": "No usable reference or detection"}}
    }
    result.update({v: {r: dict(row) for r, row in empty.items()} for v in VARIANTS[1:]})
    if current is None:
        return result
    refs = {
        pose: np.asarray(ref["points"], dtype=np.float64)
        for pose, ref in frozen["references"].items()
    }
    if "front" in refs:
        result["single_local"] = local_metrics(current, refs["front"])
        result["intrinsic_local"] = {k: dict(v) for k, v in result["single_local"].items()}
        try:
            shape_delta = contour_shape(current) - contour_shape(refs["front"])
            result["intrinsic_local"]["contour"] = {
                "value": float(np.sqrt(np.mean(shape_delta**2))),
                "status": "valid",
                "chin_height_error": abs(float(shape_delta[0])),
                "chin_lateral_error": abs(float(shape_delta[1])),
            }
        except ValueError as exc:
            result["intrinsic_local"]["contour"] = {"value": None, "status": str(exc)}
        try:
            if (
                face_status(
                    [{"bbox": [0, 0, 1, 1, 1], "keypoints": current.tolist()}], ObservationRequest()
                )
                != "valid"
            ):
                raise ValueError("Legacy face/pose gate rejected current sample")
            if (
                face_status(
                    [{"bbox": [0, 0, 1, 1, 1], "keypoints": refs["front"].tolist()}],
                    ObservationRequest(),
                )
                != "valid"
            ):
                raise ValueError("Legacy face/pose gate rejected reference")
            value, expression = geometry_distance(current, refs["front"])
            result["wls"] = {
                "global": {"value": value, "expression": expression, "status": "valid"}
            }
        except ValueError as exc:
            result["wls"]["global"]["status"] = str(exc)
    if oracle_pose in refs:
        result["oracle"] = local_metrics(current, refs[oracle_pose])
    candidates = {pose: local_metrics(current, ref) for pose, ref in refs.items()}
    for region in REGIONS:
        ranked = sorted(
            (float(metrics[region]["fit_cost"]), pose)
            for pose, metrics in candidates.items()
            if metrics[region]["fit_cost"] is not None
        )
        if not ranked:
            continue
        cost, best = ranked[0]
        raw = candidates[best][region] | {
            "matched_pose": best,
            "reference_id": frozen["references"][best]["sample_id"],
        }
        gate = frozen.get("pose_gates", {}).get(best, {}).get(region)
        gap = ranked[1][0] - cost if len(ranked) > 1 else None
        reason = None
        if gate is None or gap is None:
            reason = "Uncalibrated or incomplete pose bank"
        elif cost > gate["max_cost"]:
            reason = "Pose match outside development tolerance"
        elif gap < gate["min_gap"]:
            reason = "Ambiguous pose match"
        result["pose_local"][region] = raw | {"raw_value": raw["value"], "gap": gap}
        if reason:
            result["pose_local"][region].update(value=None, status=reason)
    return result


def verify_extraction(manifest: StudyManifest, extracted: Json) -> None:
    if extracted["data_hash"] != data_hash(manifest):
        raise ValueError("Extraction does not match the frozen input manifest")
    expected = {s.sample_id: s.input_sha256 for s in manifest.samples}
    actual = {r["sample_id"]: r["input_sha256"] for r in extracted["records"]}
    if expected != actual or len(actual) != len(extracted["records"]):
        raise ValueError("Extraction IDs/hashes mismatch or duplicate records")


def profile_pose(metrics: Json, variant: str, human_pose: str) -> str | None:
    """Human pose routes the diagnostic oracle only, never the automatic candidate."""
    if variant == "oracle":
        return human_pose
    if variant in {"wls", "single_local", "intrinsic_local"}:
        return "front"  # These baselines use one fixed frontal reference/profile.
    matched = {row["matched_pose"] for row in metrics.values() if row["value"] is not None}
    return next(iter(matched)) if len(matched) == 1 else None


def freeze_study(
    manifest: StudyManifest, extracted: Json, candidate_variant: str = "pose_local"
) -> Json:
    if candidate_variant not in {"single_local", "pose_local", "intrinsic_local"}:
        raise ValueError("Select a local candidate before calibration/validation")
    verify_extraction(manifest, extracted)
    records = {r["sample_id"]: r for r in extracted["records"]}
    splits = split_sources(manifest)
    references: Json = {}
    dev = [s for s in manifest.samples if splits[s.source_id] == "development"]
    for pose in POSES:
        candidates = [
            s
            for s in dev
            if s.parent_id is None
            and s.transform is None
            and in_scope(resolved_review(s))
            and resolved_review(s)["label"] == "normal"
            and resolved_review(s).get("approve_anchor")
            and resolved_review(s)["pose"] == pose
            and points(records[s.sample_id]) is not None
        ]
        costs = []
        for candidate in candidates:
            p = points(records[candidate.sample_id])
            assert p is not None
            distances = []
            for other in candidates:
                q = points(records[other.sample_id])
                assert q is not None
                try:
                    _, _, _, cost = aligned_residuals(p, q, STABLE_POINTS)
                    distances.append(cost)
                except ValueError:
                    continue
            if distances:
                costs.append((float(np.median(distances)), candidate.sample_id))
        if costs:
            _, chosen = min(costs)
            sample = next(s for s in candidates if s.sample_id == chosen)
            references[pose] = {
                "sample_id": chosen,
                "source_id": sample.source_id,
                "input_sha256": sample.input_sha256,
                "points": records[chosen]["faces"][0]["keypoints"],
            }
    frozen: Json = {
        "schema_version": 1,
        "stage": "frozen_development",
        "character_id": manifest.character_id,
        "data_hash": data_hash(manifest),
        "labels_hash": digest(manifest.model_dump(mode="json")),
        "extraction_hash": digest(extracted),
        "model_hash": digest(extracted["models"]),
        "implementation_hash": implementation_hash(),
        "policy": dict(POLICY),
        "candidate_variant": candidate_variant,
        "splits": splits,
        "references": references,
        "pose_gates": {},
        "normalizers": {},
    }
    # Independent normal primaries only; a reference's own source never trains its pose gate.
    for pose in references:
        ref_points = np.asarray(references[pose]["points"], dtype=np.float64)
        gates: Json = {}
        normal = [
            s
            for s in dev
            if s.primary
            and in_scope(resolved_review(s))
            and resolved_review(s)["label"] == "normal"
            and resolved_review(s)["pose"] == pose
            and s.source_id != references[pose]["source_id"]
        ]
        for region in REGIONS:
            fit_costs, gaps = [], []
            for s in normal:
                current = points(records[s.sample_id])
                if current is None:
                    continue
                metrics = local_metrics(current, ref_points)[region]
                if metrics["fit_cost"] is None:
                    continue
                competitors = [
                    local_metrics(current, np.asarray(ref["points"], dtype=np.float64))[region][
                        "fit_cost"
                    ]
                    for other, ref in references.items()
                    if other != pose
                ]
                competitors = [c for c in competitors if c is not None]
                if competitors and min(competitors) > metrics["fit_cost"]:
                    fit_costs.append(metrics["fit_cost"])
                    gaps.append(min(competitors) - metrics["fit_cost"])
            if len(fit_costs) >= POLICY["min_development"]:
                gates[region] = {
                    "max_cost": float(np.quantile(fit_costs, 0.95)),
                    "min_gap": float(np.quantile(gaps, 0.05)),
                    "n_sources": len(fit_costs),
                }
        frozen["pose_gates"][pose] = gates
    dev_features = {
        s.sample_id: feature_variants(
            points(records[s.sample_id]), frozen, resolved_review(s)["pose"]
        )
        for s in dev
        if s.primary
        and in_scope(resolved_review(s))
        and resolved_review(s)["label"] == "normal"
        and s.source_id not in {r["source_id"] for r in references.values()}
    }
    for variant in VARIANTS:
        frozen["normalizers"][variant] = {}
        for pose in POSES:
            values: Json = {}
            for s in dev:
                review = resolved_review(s)
                if not (s.primary and in_scope(review) and review["label"] == "normal"):
                    continue
                if s.source_id in {r["source_id"] for r in references.values()}:
                    continue
                metrics = dev_features[s.sample_id][variant]
                if profile_pose(metrics, variant, review["pose"]) != pose:
                    continue
                for region, row in metrics.items():
                    if row["value"] is not None:
                        values.setdefault(region, []).append(row["value"])
            normalizers: Json = {}
            for region, array in values.items():
                if len(array) >= POLICY["min_development"]:
                    median = float(np.median(array))
                    mad = 1.4826 * float(np.median(np.abs(np.asarray(array) - median)))
                    normalizers[region] = {
                        "median": median,
                        "scale": max(mad, POLICY["mad_floor"]),
                        "mad_floored": mad < POLICY["mad_floor"],
                        "n_sources": len(array),
                    }
            frozen["normalizers"][variant][pose] = normalizers
    frozen["artifact_hash"] = digest(frozen)
    return frozen


def verify_frozen(manifest: StudyManifest, extracted: Json, frozen: Json) -> None:
    verify_extraction(manifest, extracted)
    if digest({k: v for k, v in frozen.items() if k != "artifact_hash"}) != frozen["artifact_hash"]:
        raise ValueError("Frozen artifact tampered")
    if (
        frozen["data_hash"],
        frozen["labels_hash"],
        frozen["extraction_hash"],
        frozen["implementation_hash"],
        frozen["policy"],
    ) != (
        data_hash(manifest),
        digest(manifest.model_dump(mode="json")),
        digest(extracted),
        implementation_hash(),
        POLICY,
    ):
        raise ValueError("Frozen inputs/labels/models/implementation/policy changed")


def standardized_score(metrics: Json, normalizers: Json) -> Json:
    observed = {k: row["value"] for k, row in metrics.items() if row["value"] is not None}
    if not observed or any(k not in normalizers for k in observed):
        return {"score": None, "mask": sorted(observed), "reason": "Uncalibrated or absent region"}
    z = {
        k: max(0.0, (v - normalizers[k]["median"]) / normalizers[k]["scale"])
        for k, v in observed.items()
    }
    winner = max(z, key=lambda k: z[k])
    return {"score": z[winner], "mask": sorted(z), "winner": winner, "standardized": z}


def rank_threshold(values: list[float]) -> tuple[float | None, int]:
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Invalid calibration score")
    rank = math.ceil((len(values) + 1) * 0.95)
    return (sorted(values)[rank - 1] if len(values) >= 39 and rank <= len(values) else None, rank)


def calibrate_study(manifest: StudyManifest, extracted: Json, frozen: Json) -> Json:
    verify_frozen(manifest, extracted, frozen)
    records = {r["sample_id"]: r for r in extracted["records"]}
    profiles: Json = {v: {} for v in VARIANTS}
    cal_features = {
        s.sample_id: feature_variants(
            points(records[s.sample_id]), frozen, resolved_review(s)["pose"]
        )
        for s in manifest.samples
        if s.primary
        and frozen["splits"][s.source_id] == "calibration"
        and in_scope(resolved_review(s))
        and resolved_review(s)["label"] == "normal"
    }
    for variant in VARIANTS:
        for pose in POSES:
            values, masks, sources = [], [], []
            for s in manifest.samples:
                review = resolved_review(s)
                if not (
                    s.primary
                    and frozen["splits"][s.source_id] == "calibration"
                    and in_scope(review)
                    and review["label"] == "normal"
                ):
                    continue
                metrics = cal_features[s.sample_id][variant]
                if profile_pose(metrics, variant, review["pose"]) != pose:
                    continue
                score = standardized_score(metrics, frozen["normalizers"][variant][pose])
                if score["score"] is not None:
                    values.append(score["score"])
                    masks.append(score["mask"])
                    sources.append(s.source_id)
            threshold, rank = rank_threshold(values)
            profiles[variant][pose] = {
                "threshold": threshold,
                "rank": rank,
                "n_sources": len(sources),
                "missing_to_minimum": max(0, 39 - len(sources)),
                "masks": sorted({tuple(m) for m in masks}),
                "status": "calibrated_rank_only" if threshold is not None else "insufficient_data",
            }
    calibrated: Json = {
        "schema_version": 1,
        "stage": "calibration",
        "frozen_hash": frozen["artifact_hash"],
        "profiles": profiles,
    }
    calibrated["artifact_hash"] = digest(calibrated)
    return calibrated


def decide(metrics: Json, normalizers: Json, profile: Json) -> Json:
    score = standardized_score(metrics, normalizers)
    result = score | {"decision": "unknown", "statistical_flag": None}
    if score["score"] is None or profile["threshold"] is None:
        result["reason"] = "Measurements or calibration unavailable"
        return result
    if score["mask"] not in [list(mask) for mask in profile["masks"]]:
        result["reason"] = "Unseen availability combination"
        return result
    exceeded = score["score"] > profile["threshold"]
    result["statistical_flag"] = exceeded
    full = set(score["mask"]) == set(metrics)
    result["decision"] = "anomaly" if exceeded else "within_thresholds" if full else "unknown"
    result["reason"] = (
        "Strict threshold exceeded"
        if exceeded
        else "Observed regions within; critical regions missing"
        if not full
        else "All measured regions within thresholds"
    )
    return result


def clopper_pearson_upper(errors: int, n: int, confidence: float = 0.95) -> float | None:
    """One-sided exact binomial interval, solved numerically without a SciPy dependency."""
    if n < 0 or not 0 <= errors <= n or not 0 < confidence < 1:
        raise ValueError("Invalid binomial counts or confidence")
    if n == 0:
        return None
    if errors == n:
        return 1.0
    if errors == 0:
        return float(1 - (1 - confidence) ** (1 / n))
    lo, hi = errors / n, 1.0
    for _ in range(80):
        p = (lo + hi) / 2
        cdf = sum(
            math.exp(
                math.lgamma(n + 1)
                - math.lgamma(j + 1)
                - math.lgamma(n - j + 1)
                + j * math.log(p)
                + (n - j) * math.log1p(-p)
            )
            for j in range(errors + 1)
        )
        if cdf > 1 - confidence:
            lo = p
        else:
            hi = p
    return (lo + hi) / 2


def summarize_decisions(rows: list[Json], variant: str) -> Json:
    normal = [r for r in rows if r["label"] == "normal" and r["primary"]]
    abnormal = [r for r in rows if r["label"] == "geometry_anomaly"]
    valid = [r for r in normal if r["decisions"][variant]["statistical_flag"] is not None]
    errors = sum(r["decisions"][variant]["statistical_flag"] is True for r in valid)
    detected = sum(r["decisions"][variant]["decision"] == "anomaly" for r in abnormal)
    covered = sum(r["decisions"][variant]["decision"] != "unknown" for r in normal)
    return {
        "normal_primary_sources": len(normal),
        "valid_normal_sources": len(valid),
        "false_positives": errors,
        "fpr": errors / len(valid) if valid else None,
        "fpr_upper95": clopper_pearson_upper(errors, len(valid)),
        "normal_coverage": covered / len(normal) if normal else None,
        "normal_unknown": len(normal) - covered,
        "anomaly_samples": len(abnormal),
        "anomaly_sources": len({r["source_id"] for r in abnormal}),
        "detected": detected,
        "recall": detected / len(abnormal) if abnormal else None,
        "anomaly_unknown_counted_missed": sum(
            r["decisions"][variant]["decision"] == "unknown" for r in abnormal
        ),
        "region_coverage": {
            region: sum(
                r["metrics"][variant].get(region, {}).get("value") is not None for r in normal
            )
            / len(normal)
            if normal
            else None
            for region in ("global",)
            if variant == "wls"
        }
        if variant == "wls"
        else {
            region: sum(r["metrics"][variant][region]["value"] is not None for r in normal)
            / len(normal)
            if normal
            else None
            for region in REGIONS
        },
    }


def paired_bootstrap(rows: list[Json], variant: str) -> Json:
    """Source-cluster bootstrap; sign test concerns source direction, not mean recall."""
    abnormal = [r for r in rows if r["label"] == "geometry_anomaly"]
    sources = sorted({r["source_id"] for r in abnormal})
    counts = np.asarray(
        [
            [
                sum(
                    r["decisions"][variant]["decision"] == "anomaly"
                    for r in abnormal
                    if r["source_id"] == source
                ),
                sum(
                    r["decisions"]["wls"]["decision"] == "anomaly"
                    for r in abnormal
                    if r["source_id"] == source
                ),
                sum(r["source_id"] == source for r in abnormal),
            ]
            for source in sources
        ],
        dtype=np.float64,
    ).reshape(-1, 3)
    differences = counts[:, 0] - counts[:, 1]
    positive = int(np.count_nonzero(differences > 0))
    negative = int(np.count_nonzero(differences < 0))
    nonzero = positive + negative
    sign_test = {
        "positive_sources": positive,
        "negative_sources": negative,
        "tied_sources": len(sources) - nonzero,
        "p_one_sided": sum(math.comb(nonzero, j) for j in range(positive, nonzero + 1)) / 2**nonzero
        if nonzero
        else None,
        "target": "P(source recall difference > 0 | nonzero difference) > 0.5",
    }
    common = {
        "n_sources": len(sources),
        "minimum_sources": POLICY["min_anomaly_sources"],
        "sign_test": sign_test,
        "interval_target": (
            "Sample-weighted recall difference; source-cluster percentile approximation"
        ),
    }
    if len(sources) < POLICY["min_anomaly_sources"]:
        return common | {
            "status": "insufficient_independent_anomaly_sources",
            "interval95": None,
        }
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(2000):
        totals = counts[rng.integers(0, len(sources), len(sources))].sum(0)
        deltas.append(float((totals[0] - totals[1]) / totals[2]))
    return common | {
        "status": "estimated",
        "repeats": 2000,
        "seed": 42,
        "interval95": np.quantile(deltas, [0.025, 0.975]).tolist(),
    }


def evaluate_study(
    manifest: StudyManifest, extracted: Json, frozen: Json, calibrated: Json
) -> Json:
    verify_frozen(manifest, extracted, frozen)
    if (
        calibrated["frozen_hash"] != frozen["artifact_hash"]
        or digest({k: v for k, v in calibrated.items() if k != "artifact_hash"})
        != calibrated["artifact_hash"]
    ):
        raise ValueError("Calibration tampered or belongs to a different frozen study")
    records = {r["sample_id"]: r for r in extracted["records"]}
    rows, exclusions = [], []
    for s in manifest.samples:
        review = resolved_review(s)
        if frozen["splits"][s.source_id] != "validation" or not in_scope(review):
            exclusions.append(
                {
                    "sample_id": s.sample_id,
                    "split": frozen["splits"][s.source_id],
                    "review_status": review["review_status"],
                    "label": review["label"],
                    "reason": "Not validation or not human-reviewed in-scope idle sample",
                }
            )
            continue
        pose = review["pose"]
        metrics = feature_variants(points(records[s.sample_id]), frozen, pose)
        decisions = {}
        routed = {v: profile_pose(metrics[v], v, pose) for v in VARIANTS}
        for variant, assigned in routed.items():
            decisions[variant] = (
                decide(
                    metrics[variant],
                    frozen["normalizers"][variant][assigned],
                    calibrated["profiles"][variant][assigned],
                )
                if assigned is not None
                else {
                    "decision": "unknown",
                    "statistical_flag": None,
                    "reason": "Inconsistent or unavailable automatic pose matches",
                }
            )
        rows.append(
            {
                "sample_id": s.sample_id,
                "source_id": s.source_id,
                "label": review["label"],
                "pose": pose,
                "parts": review["parts"],
                "severity": review["severity"],
                "primary": s.primary,
                "synthetic": s.transform is not None,
                "transform": s.transform,
                "strength": s.transform.get("strength") if s.transform is not None else None,
                "metrics": metrics,
                "assigned_profile_pose": routed,
                "decisions": decisions,
            }
        )
    summaries = {v: summarize_decisions(rows, v) for v in VARIANTS}
    adoption: Json = {}
    for variant in ("single_local", "pose_local", "intrinsic_local"):
        summary = summaries[variant]
        baseline = summaries["wls"]
        bootstrap = paired_bootstrap(rows, variant)
        real = [r for r in rows if r["label"] == "geometry_anomaly" and not r["synthetic"]]
        real_bootstrap = paired_bootstrap(real, variant)
        passed = (
            summary["fpr_upper95"] is not None
            and summary["fpr_upper95"] <= 0.05
            and summary["normal_coverage"] is not None
            and summary["normal_coverage"] >= 0.8
            and baseline["fpr_upper95"] is not None
            and baseline["fpr_upper95"] <= 0.05
            and bootstrap["interval95"] is not None
            and bootstrap["interval95"][0] > 0
            and bootstrap["sign_test"]["p_one_sided"] is not None
            and bootstrap["sign_test"]["p_one_sided"] <= 0.05
        )
        selected = variant == frozen["candidate_variant"]
        adoption[variant] = {
            "adopt": passed and selected,
            "selected_before_calibration": selected,
            "status": "supported_offline_only" if passed and selected else "not_accepted",
            "bootstrap_all": bootstrap,
            "bootstrap_real_anomalies": real_bootstrap,
            "requirements": (
                "FPR upper95 <=5%; coverage>=80%; calibrated WLS comparator; "
                "positive paired source interval on human-reviewed anomalies; "
                "at least 20 independent anomaly sources; source-direction sign p<=0.05; "
                "synthetic and real evidence reported separately"
            ),
        }
    strata: Json = {}
    for field in ("pose", "severity", "synthetic"):
        strata[field] = {
            str(value): {
                v: summarize_decisions([r for r in rows if r[field] == value], v) for v in VARIANTS
            }
            for value in sorted({r[field] for r in rows})
        }
    strata["parts"] = {
        part: {
            v: summarize_decisions(
                [r for r in rows if r["label"] == "normal" or part in r["parts"]], v
            )
            for v in VARIANTS
        }
        for part in REGIONS
    }
    strata["strength"] = {
        str(strength): {
            v: summarize_decisions([r for r in rows if r["strength"] == strength], v)
            for v in VARIANTS
        }
        for strength in sorted({r["strength"] for r in rows if r["strength"] is not None})
    }
    return {
        "schema_version": 1,
        "stage": "independent_evaluation",
        "character_id": manifest.character_id,
        "frozen_hash": frozen["artifact_hash"],
        "calibration_hash": calibrated["artifact_hash"],
        "source_counts": {
            split: sum(value == split for value in frozen["splits"].values())
            for split in ("development", "calibration", "validation")
        },
        "summary": summaries,
        "strata": strata,
        "adoption": adoption,
        "rows": rows,
        "exclusions": exclusions,
        "human_review_counts": {
            status: sum(resolved_review(s)["review_status"] == status for s in manifest.samples)
            for status in ("pending_human", "disagreement", "confirmed", "adjudicated")
        },
        "scope": (
            "Single character idle; statistical FPR on supported observations, "
            "complete geometry coverage reported separately; no Runtime/recovery integration"
        ),
    }
