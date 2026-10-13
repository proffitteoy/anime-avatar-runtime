"""Synthetic coordinates/labels test research behavior, never model accuracy."""

import copy
import hashlib
import json
import re
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from anime_avatar_runtime import local_geometry
from anime_avatar_runtime.local_geometry import (
    REGIONS,
    Review,
    StudyManifest,
    StudySample,
    calibrate_study,
    clopper_pearson_upper,
    data_hash,
    decide,
    digest,
    evaluate_study,
    feature_variants,
    freeze_study,
    local_metrics,
    paired_bootstrap,
    profile_pose,
    rank_threshold,
    resolved_review,
    split_sources,
    summarize_decisions,
)


def landmarks():
    p = np.random.default_rng(12).uniform(-1, 1, (28, 3))
    p[:, 2] = 1
    p[[11, 13, 17, 19], :2] = [[-1, 0], [-0.5, 0.1], [0.5, 0.1], [1, 0]]
    p[23, :2] = [0, 0.4]
    return p


def hash_id(value):
    return hashlib.sha256(value.encode()).hexdigest()


def test_endpoint_jacobian_and_procrustes_have_the_same_similarity_kernel():
    script = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/observer_geometry_experiment.py")
    )
    p = landmarks()
    jacobian = script["contour_jacobian"](p)
    assert np.linalg.matrix_rank(jacobian) == 6
    for column in range(10):
        plus, minus = p.copy(), p.copy()
        plus[column // 2, column % 2] += 1e-6
        minus[column // 2, column % 2] -= 1e-6
        actual = (local_geometry.contour_shape(plus) - local_geometry.contour_shape(minus)) / 2e-6
        np.testing.assert_allclose(actual, jacobian[:, column], atol=1e-8)
    theta = 0.4
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    current = p.copy()
    current[:, :2] = p[:, :2] @ rotation * 1.4 + [5, -2]
    vectors, errors = script["shape_vectors"](current, p)
    assert not errors
    assert max(np.linalg.norm(v) for v in vectors.values()) < 1e-12
    reflected = p.copy()
    reflected[:, 0] *= -1
    reflected_vectors = script["shape_vectors"](reflected, p)[0]
    assert np.linalg.norm(reflected_vectors["procrustes"]) > 0.01


def test_stability_four_methods_and_whitened_chart_control():
    script = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/observer_geometry_experiment.py")
    )
    p = landmarks()
    manifest = StudyManifest(character_id="fixture", samples=[sample(0)])
    extracted = {
        "models": {},
        "records": [
            {
                "sample_id": "sample0",
                "faces": [{"bbox": [0, 0, 100, 100, 1], "keypoints": p.tolist()}],
            }
        ],
    }
    result = script["stability_experiment"](manifest, extracted, "sample0")
    assert set(result["operators"]) == {"wls", "complement", "endpoint", "procrustes"}
    assert result["checks"]["whitened_chart_gram_error"] < 1e-12
    assert result["operators"]["procrustes"]["noise_to_min_shape_response"] == pytest.approx(1)
    assert result["operators"]["endpoint"]["noise_to_min_shape_response"] > 1
    for name in result["operators"]["endpoint"]["noise_models"]:
        endpoint = result["operators"]["endpoint"]
        procrustes = result["operators"]["procrustes"]
        endpoint_noise = endpoint["noise_models"][name]["expected_squared_score"]
        procrustes_noise = procrustes["noise_models"][name]["expected_squared_score"]
        assert endpoint_noise / endpoint["target_shape_min_gain"] ** 2 >= (
            procrustes_noise / procrustes["target_shape_min_gain"] ** 2 - 1e-12
        )
    assert result["coordinate_controls"]["contour_isotropic_size"]["endpoint"] < 1e-12
    assert result["coordinate_controls"]["contour_isotropic_size"]["procrustes"] < 1e-12
    assert result["rows"][0]["trusted_anomaly_judgment"].startswith("unknown")


def test_stability_missing_endpoints_do_not_lower_confidence_gate():
    script = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/observer_geometry_experiment.py")
    )
    reference, current = landmarks(), landmarks()
    current[0, 2] = 0.29
    vectors, errors = script["shape_vectors"](current, reference)
    assert "endpoint" in errors and "procrustes" in errors
    assert "complement" in vectors
    assert current[0, 2] == 0.29
    assert np.linalg.norm(vectors["wls"]) == pytest.approx(
        local_geometry.geometry_distance(current, reference)[0], abs=1e-12
    )


def test_similarity_probe_maps_pixel_centers_and_rejects_unknown_probe():
    script = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/observer_geometry_experiment.py")
    )
    image = Image.new("RGB", (20, 20), "black")
    image.putpixel((6, 8), (255, 255, 255))
    translated, matrix = script["similarity_probe"](image, "translate")
    assert translated.getpixel((9, 6)) == (255, 255, 255)
    np.testing.assert_allclose(matrix @ [6, 8, 1], [9, 6, 1])
    _, matrix = script["similarity_probe"](image, "rotate")
    np.testing.assert_allclose(matrix @ [9.5, 9.5, 1], [9.5, 9.5, 1])
    _, matrix = script["similarity_probe"](image, "scale")
    np.testing.assert_allclose(matrix @ [6, 8, 1], [(6.5) * 0.95 - 0.5, (8.5) * 0.95 - 0.5, 1])
    with pytest.raises(ValueError, match="Unknown"):
        script["similarity_probe"](image, "other")


def sample(index, **kwargs):
    return StudySample(
        sample_id=f"sample{index}",
        source_id=f"source{index}",
        source_sha256=hash_id(f"source{index}"),
        image=f"{index}.png",
        input_sha256=hash_id(f"image{index}"),
        **kwargs,
    )


def review(**kwargs):
    return Review(
        reviewer="explicit synthetic fixture",
        kind="human",
        label="normal",
        pose="front",
        mode="idle",
        occlusion="none",
        **kwargs,
    )


def test_local_similarity_invariance_and_reflection_cannot_be_removed():
    p = landmarks()
    theta = 0.8
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    q = p.copy()
    q[:, :2] = p[:, :2] @ rotation * 4 + [21, -7]
    assert all(r["value"] < 1e-12 for r in local_metrics(q, p).values())
    q = p.copy()
    q[:, 0] *= -1
    assert max(r["value"] for r in local_metrics(q, p).values()) > 0.1


@pytest.mark.parametrize("region", ["left_eye", "right_eye", "nose", "mouth", "contour"])
def test_target_movement_never_drives_its_own_fit(region):
    p = landmarks()
    q = p.copy()
    q[list(REGIONS[region]), :2] += [0.3, 0.2]
    result = local_metrics(q, p)[region]
    assert result["fit_cost"] < 1e-12
    assert result["value"] > 0.1


def test_coupling_operator_matches_refitting_and_removes_similarity():
    script = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/observer_geometry_experiment.py")
    )
    p = landmarks()
    fit = tuple(i for i in local_geometry.STABLE_POINTS if i not in REGIONS["contour"])
    operator, radius, _, jacobian = script["coupling_operator"](p, fit)
    assert np.max(np.abs(operator @ jacobian)) < 1e-12
    # Moving an eye leaks into a stationary chin through the complementary fit.
    delta = np.zeros((28, 2))
    delta[11, 0] = 1e-5
    q = p.copy()
    q[:, :2] += delta * radius
    aligned, _, _, _ = local_geometry.aligned_residuals(q, p, fit)
    predicted = (operator @ delta.reshape(-1)).reshape(28, 2)
    assert np.linalg.norm(predicted[2]) > 1e-7
    assert np.max(np.abs((aligned - p[:, :2]) / radius - predicted)) < 1e-9
    # A held-out chin movement retains its full amplitude.
    delta[:] = 0
    delta[2, 1] = 0.05
    assert (operator @ delta.reshape(-1)).reshape(28, 2)[2, 1] == pytest.approx(0.05)


def test_coupling_operator_freezes_confidence_and_rejects_unobservable_fit():
    script = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/observer_geometry_experiment.py")
    )
    p = landmarks()
    fit = tuple(i for i in local_geometry.STABLE_POINTS if i not in REGIONS["contour"])
    confidence = p[:, 2].copy()
    confidence[11] = 0
    operator, _, _, _ = script["coupling_operator"](p, fit, confidence)
    # A masked fit point cannot propagate coordinate changes to another region.
    assert np.max(np.abs(operator[4:6, 22:24])) < 1e-12
    confidence[list(fit)] = 0
    with pytest.raises(ValueError, match="Insufficient complementary"):
        script["coupling_operator"](p, fit, confidence)


def test_coupling_experiment_rejects_unapproved_reference():
    script = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/observer_geometry_experiment.py")
    )
    manifest = StudyManifest(character_id="synthetic", samples=[sample(0, reviews=[review()])])
    with pytest.raises(ValueError, match="approved human-normal original"):
        script["coupling_experiment"](manifest, {"records": []}, "sample0")


def test_intrinsic_contour_similarity_invariance_and_unrelated_point_isolation():
    script = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/observer_geometry_experiment.py")
    )
    shape = script["contour_shape"]
    p = landmarks()
    q = p.copy()
    theta = 0.7
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    q[:, :2] = p[:, :2] @ rotation * 3 + [40, -12]
    assert np.allclose(shape(p), shape(q), atol=1e-12)
    q = p.copy()
    q[5:, :2] += 1000
    assert np.array_equal(shape(p), shape(q))
    frozen = {
        "references": {"front": {"sample_id": "approved", "points": p.tolist()}},
        "pose_gates": {},
    }
    result = feature_variants(q, frozen, "front")
    assert result["intrinsic_local"]["contour"]["value"] == 0
    assert result["single_local"]["contour"]["value"] > 0
    assert profile_pose(result["intrinsic_local"], "intrinsic_local", "down") == "front"


def test_intrinsic_contour_retains_chin_movement_and_reflection():
    script = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/observer_geometry_experiment.py")
    )
    shape = script["contour_shape"]
    p = landmarks()
    axis = p[4, :2] - p[0, :2]
    q = p.copy()
    q[2, :2] += np.array([-axis[1], axis[0]]) * 0.2
    delta = shape(q) - shape(p)
    assert delta.shape == (6,)
    assert delta[0] == pytest.approx(0.2)
    assert np.linalg.norm(delta[1:]) < 1e-12
    assert np.sqrt(np.mean(delta**2)) == pytest.approx(0.2 / np.sqrt(6))
    q = p.copy()
    q[:, 0] *= -1
    assert np.linalg.norm(shape(p) - shape(q)) > 0.1
    q = p.copy()
    q[4, :2] = q[0, :2]
    with pytest.raises(ValueError, match="Degenerate contour width"):
        shape(q)
    p[2, 2] = 0
    with pytest.raises(ValueError, match="All five contour"):
        shape(p)


@pytest.mark.parametrize("index,component", [(1, 4), (3, 5)])
def test_intrinsic_contour_retains_side_movement_along_local_axis(index, component):
    p = landmarks()
    q = p.copy()
    q[index, :2] += 0.2 * (p[4, :2] - p[0, :2])
    delta = local_geometry.contour_shape(q) - local_geometry.contour_shape(p)
    assert delta[component] == pytest.approx(0.2)
    assert np.linalg.norm(np.delete(delta, component)) < 1e-12
    frozen = {
        "references": {"front": {"sample_id": "approved", "points": p.tolist()}},
        "pose_gates": {},
    }
    result = feature_variants(q, frozen, "front")
    assert result["intrinsic_local"]["contour"]["value"] == pytest.approx(0.2 / np.sqrt(6))
    assert result["single_local"]["contour"]["value"] > 0.1


def test_causal_replay_dwell_unknown_reset_hysteresis_and_prefix_invariance():
    script = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/observer_geometry_experiment.py")
    )
    replay = script["causal_replay"]
    timeline = [{"time": i / 10, "value": 0.1, "mode": "idle"} for i in range(10)]
    rows = replay(timeline, 0.03, 0.015, 0.5, 0.15)
    assert sum(r["new_recommendation"] for r in rows) == 1
    assert rows[5]["new_recommendation"]
    assert replay(timeline[:4], 0.03, 0.015, 0.5, 0.15) == rows[:4]
    timeline[4]["value"] = None
    rows = replay(timeline, 0.03, 0.015, 0.5, 0.15)
    assert rows[4]["state"] == "UNKNOWN"
    assert not any(r["new_recommendation"] for r in rows)
    timeline = [
        {"time": 0.0, "value": 0.1, "mode": "idle"},
        {"time": 0.1, "value": 0.02, "mode": "idle"},
        {"time": 0.2, "value": 0.01, "mode": "idle"},
        {"time": 1.0, "value": 0.1, "mode": "idle"},
        {"time": 1.1, "value": 0.1, "mode": "interaction"},
    ]
    rows = replay(timeline, 0.03, 0.015, 0.5, 0.15)
    assert [r["state"] for r in rows] == ["WATCH", "WATCH", "NORMAL", "WATCH", "UNKNOWN"]
    assert rows[3]["gap_reset"] and rows[3]["watch_seconds"] == 0
    with pytest.raises(ValueError, match="strictly increasing"):
        replay(timeline + [timeline[-1]], 0.03, 0.015, 0.5, 0.15)


def test_statistical_budget_uses_independent_valid_sources_and_conditional_control():
    script = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/observer_geometry_experiment.py")
    )
    budget = script["statistical_budget"]()
    validation = budget["zero_false_positive_validation"]
    assert validation[0]["minimum_valid_normals"] == 59
    assert all(row["upper95"] <= 0.05 for row in validation)
    assert budget["validity_sampling"]["minimum_sampled_normals"] > 59
    control = budget["uniform_rank_calibration_control"]
    assert control["marginal_false_positive_probability"] == 0.05
    assert control["probability_frozen_true_fpr_exceeds_5_percent"] > 0.4


def test_regional_flow_replay_preserves_known_translation_and_input_binding(tmp_path):
    script = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/observer_geometry_experiment.py")
    )
    s = sample(0, reviews=[review(approve_anchor=True)])
    manifest = StudyManifest(character_id="synthetic", samples=[s])
    Image.new("RGB", (256, 256)).save(tmp_path / s.image)
    p = landmarks()
    p[:, :2] = p[:, :2] * 50 + 128
    extracted = {"records": [{"sample_id": s.sample_id, "faces": [{"keypoints": p.tolist()}]}]}
    flow = np.zeros((2, 2, 256, 256), dtype=np.float32)
    flow[0, 0] = 1
    flow[1, 0] = -1
    np.savez_compressed(tmp_path / f"{s.sample_id}-flow.npz", flow=flow)
    report = {
        "data_hash": data_hash(manifest),
        "reference_id": s.sample_id,
        "rows": [
            {
                "sample_id": s.sample_id,
                "input_sha256": s.input_sha256,
                "consistent_fraction_in_bounds": 1.0,
            }
        ],
    }
    location = tmp_path / "flow.json"
    location.write_text(json.dumps(report), encoding="utf-8")
    result = script["regional_flow_experiment"](
        manifest, extracted, tmp_path, s.sample_id, location
    )
    assert result["global_consistency_max_difference"] == 0
    for region in result["rows"][0]["regions"].values():
        assert region["median_pixels"] == 1
        assert region["p95_pixels"] == 1
        assert region["consistent_fraction_in_bounds"] == 1
    report["rows"][0]["input_sha256"] = "0" * 64
    location.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="input hash mismatch"):
        script["regional_flow_experiment"](manifest, extracted, tmp_path, s.sample_id, location)


def test_expression_aperture_is_not_eye_structure_and_partial_failure_is_local():
    p = landmarks()
    q = p.copy()
    q[15, 1] += 1
    result = local_metrics(q, p)
    assert result["left_eye"]["value"] < 1e-12
    assert result["left_eye"]["opening"] != local_metrics(p, p)["left_eye"]["opening"]
    q[23, 2] = 0
    result = local_metrics(q, p)
    assert result["nose"]["value"] is None
    assert result["mouth"]["value"] < 1e-12
    q[:, :2] = 0
    assert all(r["value"] is None for r in local_metrics(q, p).values())


def test_pose_reference_selection_excludes_nose_and_preserves_rejected_raw_value():
    p = landmarks()
    other = p.copy()
    other[:5, 0] *= 0.7
    frozen = {
        "references": {
            "front": {"sample_id": "approved", "points": p.tolist()},
            "left_turn": {"sample_id": "other", "points": other.tolist()},
        },
        "pose_gates": {"front": {r: {"max_cost": 0.1, "min_gap": 0.001} for r in REGIONS}},
    }
    q = p.copy()
    q[23, 0] += 1.0  # The production nose-based pose gate rejects this.
    features = feature_variants(q, frozen, "front")
    assert features["wls"]["global"]["value"] is None
    assert features["pose_local"]["nose"]["matched_pose"] == "front"
    assert features["pose_local"]["nose"]["value"] > 0.2
    frozen["pose_gates"]["front"]["nose"]["min_gap"] = 10
    result = feature_variants(q, frozen, "front")["pose_local"]["nose"]
    assert result["value"] is None and result["raw_value"] > 0.2
    assert result["status"] == "Ambiguous pose match"
    assert profile_pose(features["pose_local"], "pose_local", "down") == "front"
    assert profile_pose(features["oracle"], "oracle", "down") == "down"
    assert profile_pose(features["single_local"], "single_local", "down") == "front"


def test_human_disagreement_and_assistant_labels_cannot_train():
    s = sample(0, reviews=[review().model_copy(update={"kind": "assistant"})])
    assert resolved_review(s)["review_status"] == "pending_human"
    reviews = [review(), review().model_copy(update={"label": "geometry_anomaly"})]
    s = sample(0, reviews=reviews)
    assert resolved_review(s)["label"] == "uncertain"
    assert resolved_review(s)["review_status"] == "disagreement"
    s = s.model_copy(update={"adjudication": review()})
    assert resolved_review(s)["review_status"] == "adjudicated"
    s = sample(1, reviews=[review(approve_anchor=True), review(approve_anchor=False)])
    assert resolved_review(s)["review_status"] == "confirmed"
    assert resolved_review(s)["approve_anchor"] is False


def test_source_splits_are_deterministic_and_derived_samples_inherit():
    samples = [sample(i, primary=True) for i in range(100)]
    child = samples[0].model_copy(
        update={
            "sample_id": "child",
            "parent_id": samples[0].sample_id,
            "primary": False,
            "transform": {"kind": "chin"},
        }
    )
    manifest = StudyManifest(character_id="fixture", samples=[*samples, child])
    splits = split_sources(manifest)
    assert splits == split_sources(manifest)
    assert [
        list(splits.values()).count(s) for s in ["development", "calibration", "validation"]
    ] == [60, 20, 20]
    assert splits[child.source_id] == splits[samples[0].source_id]
    assert split_sources(StudyManifest(character_id="fixture", samples=[sample(0)])) == {
        "source0": "development"
    }


@pytest.mark.parametrize(
    "kind", ["duplicate_source", "duplicate_image", "wrong_parent", "two_primary", "cycle"]
)
def test_false_independence_and_invalid_lineage_are_rejected(kind):
    a, b = sample(0, primary=True), sample(1, primary=True)
    if kind == "duplicate_source":
        b = b.model_copy(update={"source_sha256": a.source_sha256})
    elif kind == "duplicate_image":
        b = b.model_copy(update={"input_sha256": a.input_sha256})
    elif kind == "wrong_parent":
        b = b.model_copy(update={"primary": False, "parent_id": a.sample_id})
    elif kind == "two_primary":
        b = b.model_copy(update={"source_id": a.source_id, "source_sha256": a.source_sha256})
    else:
        b = b.model_copy(update={"parent_id": b.sample_id, "primary": False})
    with pytest.raises(ValidationError):
        StudyManifest(character_id="fixture", samples=[a, b])


def test_rank_threshold_strict_equality_missing_masks_and_partial_unknown():
    assert rank_threshold([0.0] * 38)[0] is None
    threshold, rank = rank_threshold(list(range(39)))
    assert (threshold, rank) == (37, 38)
    profile = {"threshold": 1.0, "masks": [["nose"], ["nose", "mouth"]]}
    normalizers = {r: {"median": 0, "scale": 1} for r in ["nose", "mouth"]}
    metrics = {"nose": {"value": 1.0}, "mouth": {"value": None}}
    result = decide(metrics, normalizers, profile)
    assert result["decision"] == "unknown" and result["statistical_flag"] is False
    metrics["nose"]["value"] = 1.001
    assert decide(metrics, normalizers, profile)["decision"] == "anomaly"
    assert decide({"left_eye": {"value": 0.1}}, normalizers, profile)["decision"] == "unknown"
    assert decide(metrics, normalizers, profile | {"threshold": None})["statistical_flag"] is None


def test_exact_binomial_upper_bound_and_empty_denominators():
    assert clopper_pearson_upper(0, 59) < 0.05
    assert clopper_pearson_upper(0, 58) > 0.05
    assert clopper_pearson_upper(1, 10) == pytest.approx(0.3941633024, abs=1e-9)
    assert clopper_pearson_upper(10, 10) == 1
    assert clopper_pearson_upper(0, 0) is None
    with pytest.raises(ValueError):
        clopper_pearson_upper(2, 1)


def corpus(n=300):
    p = landmarks()
    samples = [sample(i, primary=True, reviews=[review(approve_anchor=i == 0)]) for i in range(n)]
    manifest = StudyManifest(character_id="synthetic_algorithm_fixture", samples=samples)
    # Ensure a development-only reference independently of the deterministic shuffle.
    dev_id = next(
        s.sample_id for s in samples if split_sources(manifest)[s.source_id] == "development"
    )
    samples = [
        s.model_copy(update={"reviews": [review(approve_anchor=s.sample_id == dev_id)]})
        for s in samples
    ]
    manifest = StudyManifest(character_id=manifest.character_id, samples=samples)
    extracted = {
        "data_hash": data_hash(manifest),
        "models": {"fixture": "no real model"},
        "records": [
            {
                "sample_id": s.sample_id,
                "input_sha256": s.input_sha256,
                "faces": [{"keypoints": p.tolist()}],
            }
            for s in samples
        ],
    }
    return manifest, extracted


def test_full_frozen_calibration_evaluation_and_inputs_invalidate_artifacts(monkeypatch):
    manifest, extracted = corpus()
    frozen = freeze_study(manifest, extracted)
    calibrated = calibrate_study(manifest, extracted, frozen)
    assert calibrated["profiles"]["single_local"]["front"]["n_sources"] == 60
    assert calibrated["profiles"]["single_local"]["front"]["threshold"] is not None
    report = evaluate_study(manifest, extracted, frozen, calibrated)
    assert report["summary"]["single_local"]["valid_normal_sources"] == 60
    assert report["summary"]["single_local"]["fpr_upper95"] < 0.05
    assert report["adoption"]["single_local"]["adopt"] is False  # No anomaly evidence.
    damaged = copy.deepcopy(calibrated)
    damaged["profiles"]["single_local"]["front"]["threshold"] = 100
    with pytest.raises(ValueError, match="tampered"):
        evaluate_study(manifest, extracted, frozen, damaged)
    changed = copy.deepcopy(extracted)
    changed["models"]["fixture"] = "changed"
    with pytest.raises(ValueError, match="changed"):
        calibrate_study(manifest, changed, frozen)
    changed = manifest.model_copy(
        update={"samples": [s.model_copy(update={"reviews": []}) for s in manifest.samples]}
    )
    with pytest.raises(ValueError, match="changed"):
        calibrate_study(changed, extracted, frozen)
    monkeypatch.setattr(local_geometry, "implementation_hash", lambda: "changed scoring code")
    with pytest.raises(ValueError, match="changed"):
        calibrate_study(manifest, extracted, frozen)


def test_no_labels_produce_honest_insufficient_calibration():
    manifest, extracted = corpus(1)
    manifest = manifest.model_copy(
        update={"samples": [s.model_copy(update={"reviews": []}) for s in manifest.samples]}
    )
    frozen = freeze_study(manifest, extracted)
    assert frozen["references"] == {}
    calibrated = calibrate_study(manifest, extracted, frozen)
    assert calibrated["profiles"]["single_local"]["front"]["missing_to_minimum"] == 39
    report = evaluate_study(manifest, extracted, frozen, calibrated)
    assert report["summary"]["single_local"]["fpr"] is None
    assert report["human_review_counts"]["pending_human"] == 1


def test_paired_bootstrap_resamples_sources_and_unknown_anomalies_are_missed():
    rows = [
        {
            "source_id": f"source{i}",
            "label": "geometry_anomaly",
            "decisions": {
                "single_local": {"decision": "anomaly" if i < 16 else "unknown"},
                "wls": {"decision": "within_thresholds"},
            },
        }
        for i in range(20)
    ]
    result = paired_bootstrap(rows, "single_local")
    assert result["interval95"][0] > 0
    assert result["sign_test"]["positive_sources"] == 16
    assert result["sign_test"]["tied_sources"] == 4
    assert result["sign_test"]["p_one_sided"] == 2**-16
    assert result == paired_bootstrap(rows, "single_local")
    assert paired_bootstrap([rows[0]] * 20, "single_local")["interval95"] is None
    for row in rows:
        row["primary"] = False
    summary = summarize_decisions(rows, "single_local")
    assert summary["recall"] == 0.8
    assert summary["anomaly_unknown_counted_missed"] == 4


def test_two_positive_anomaly_sources_cannot_support_adoption():
    rows = [
        {
            "source_id": f"source{i}",
            "label": "geometry_anomaly",
            "decisions": {
                "single_local": {"decision": "anomaly"},
                "wls": {"decision": "within_thresholds"},
            },
        }
        for i in range(2)
    ]
    result = paired_bootstrap(rows, "single_local")
    assert result["status"] == "insufficient_independent_anomaly_sources"
    assert result["interval95"] is None
    assert result["sign_test"]["p_one_sided"] == 0.25


def test_source_sign_test_uses_direction_and_excludes_ties():
    rows = [
        {
            "source_id": f"source{i}",
            "label": "geometry_anomaly",
            "decisions": {
                "single_local": {"decision": "anomaly" if i < 8 else "within_thresholds"},
                "wls": {"decision": "anomaly" if 8 <= i < 10 else "within_thresholds"},
            },
        }
        for i in range(20)
    ]
    result = paired_bootstrap(rows, "single_local")
    assert result["sign_test"]["positive_sources"] == 8
    assert result["sign_test"]["negative_sources"] == 2
    assert result["sign_test"]["tied_sources"] == 10
    assert result["sign_test"]["p_one_sided"] == pytest.approx(56 / 1024)


def test_review_import_preserves_data_binding_and_rejects_wrong_dataset(tmp_path, monkeypatch):
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "blue").save(image)
    s = sample(0).model_copy(
        update={
            "image": Path("image.png"),
            "input_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        }
    )
    manifest = StudyManifest(character_id="explicit_test_fixture", samples=[s])
    path = tmp_path / "manifest.json"
    path.write_text(manifest.model_dump_json())
    annotations = {
        "data_hash": data_hash(manifest),
        "reviews": [{"sample_id": s.sample_id, "review": review().model_dump()}],
    }
    labels = tmp_path / "annotations.json"
    labels.write_text(json.dumps(annotations))
    monkeypatch.setattr(sys, "argv", ["script", "study"])
    cli = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/observer_geometry_experiment.py")
    )["study_main"]
    output = tmp_path / "reviewed.json"
    argv = [
        "review",
        "--manifest",
        str(path),
        "--annotations",
        str(labels),
        "--output",
        str(output),
    ]
    cli(argv)
    reviewed = StudyManifest.model_validate_json(output.read_bytes())
    assert data_hash(reviewed) == data_hash(manifest)
    assert resolved_review(reviewed.samples[0])["review_status"] == "confirmed"
    with pytest.raises(FileExistsError):
        cli(argv)
    annotations["data_hash"] = "wrong dataset"
    labels.write_text(json.dumps(annotations))
    with pytest.raises(ValueError, match="another dataset"):
        cli(argv)
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Input changed"):
        cli(argv)


def test_blind_review_page_preserves_human_fields_and_hides_provenance(tmp_path, monkeypatch):
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "blue").save(image)
    prior = review(approve_anchor=True).model_copy(
        update={"reviewer": 'fixture "<&', "parts": ["nose"], "pose": "left_turn"}
    )
    reviewed = sample(0).model_copy(
        update={
            "image": Path("image.png"),
            "input_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "transform": {"kind": "hidden_warp", "strength": 0.1},
            "reviews": [prior],
        }
    )
    manifest = StudyManifest(character_id="fixture", samples=[reviewed])
    path = tmp_path / "manifest.json"
    path.write_text(manifest.model_dump_json())
    monkeypatch.setattr(sys, "argv", ["script", "study"])
    script = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/observer_geometry_experiment.py")
    )
    output = tmp_path / "review.html"
    argv = ["review-page", "--manifest", str(path), "--output", str(output)]
    script["study_main"](argv)
    page = output.read_text(encoding="utf-8")
    assert 'value="normal" selected>正常' in page
    assert 'value="left_turn" selected>向画面左侧转头' in page
    assert 'data-part="nose" checked' in page
    assert "data-anchor checked" in page
    assert 'value="fixture &quot;&lt;&amp;"' in page
    assert data_hash(manifest) in page
    assert "hidden_warp" not in page and reviewed.source_sha256 not in page
    with pytest.raises(FileExistsError):
        script["study_main"](argv)
    pending = manifest.model_copy(update={"samples": [reviewed.model_copy(update={"reviews": []})]})
    unreviewed_page = script["blind_review_page"](pending, tmp_path)
    assert 'value="" selected>请选择分类' in unreviewed_page
    assert not re.search(r'<option value="uncertain" selected', unreviewed_page)
    assert "reviews.some(r=>!r.review.label)" in unreviewed_page
    assert "data:image/png;base64," in unreviewed_page
    assert "geometry-review:" in unreviewed_page
    assert 'data-bulk="pose"' in unreviewed_page
    assert "showModal()" in unreviewed_page


def test_prepare_still_image_keeps_source_and_primary_without_auto_labels(tmp_path, monkeypatch):
    image = tmp_path / "image.png"
    Image.new("RGB", (80, 80), "blue").save(image)
    monkeypatch.setattr(sys, "argv", ["script", "study"])
    script = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/observer_geometry_experiment.py")
    )
    cli = script["study_main"]

    class SyntheticObserver:
        model_manifest = {"synthetic_test_fixture": True}

        def __init__(self, *_args):
            pass

        def detect(self, *_args):
            return [{"keypoints": landmarks().tolist()}]

    monkeypatch.setitem(cli.__globals__, "OfflineObserver", SyntheticObserver)
    monkeypatch.setitem(cli.__globals__, "warp", lambda image, *_args, **_kwargs: image.copy())
    output = tmp_path / "prepared"
    args = [
        "prepare",
        "--image",
        str(image),
        "--character",
        "synthetic_test_fixture",
        "--models",
        str(tmp_path),
        "--output",
        str(output),
        "--similarity-controls",
        "--compact-warps",
    ]
    cli(args)
    manifest = StudyManifest.model_validate_json((output / "manifest.json").read_bytes())
    assert len(manifest.samples) == 13
    assert sum(s.primary for s in manifest.samples) == 1
    assert manifest.samples[0].input_sha256 == hashlib.sha256(image.read_bytes()).hexdigest()
    assert all(s.parent_id == "s0000" and not s.primary for s in manifest.samples[1:])
    assert len({s.source_id for s in manifest.samples}) == 1
    assert all(not s.reviews for s in manifest.samples)
    assert manifest.provenance["crop"] == [0, 0, 80, 80]
    assert manifest.provenance["source_kind"] == "still_image"
    assert all(
        s.transform["method"] == "inverse_compact_elliptic_displacement"
        for s in manifest.samples[4:]
    )
    assert set(split_sources(manifest).values()) == {"development"}
    with pytest.raises(FileExistsError):
        cli(args)
    invalid_output = tmp_path / "invalid"
    bad_args = [*args]
    bad_args[bad_args.index(str(output))] = str(invalid_output)
    with pytest.raises(ValueError, match="inside the input"):
        cli([*bad_args, "--crop", "-1", "0", "80", "80"])
    assert not invalid_output.exists()
    with pytest.raises(ValueError, match="only frame 0"):
        cli([*bad_args, "--frames", "0", "5"])
    assert not invalid_output.exists()


@pytest.mark.parametrize("kind", ["eye_spacing", "nose_mouth", "chin"])
def test_compact_warp_preserves_distant_pixels_and_has_no_foldovers(kind, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["script", "study"])
    script = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/observer_geometry_experiment.py")
    )
    maps = []

    def capture_maps(image, map_x, map_y, *_args, **_kwargs):
        maps.append((map_x, map_y))
        return image  # Tests the mapping, not interpolation or model accuracy.

    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(remap=capture_maps, INTER_LINEAR=1, BORDER_REFLECT=2),
    )
    p = landmarks()
    p[:, :2] = p[:, :2] * 20 + 40
    image = Image.new("RGB", (80, 80), "blue")
    yy, xx = np.mgrid[:80, :80]
    warp = script["warp"]
    warp(image, p, kind, 0, compact=True)
    assert np.array_equal(maps[-1][0], xx) and np.array_equal(maps[-1][1], yy)
    for strength in (0.05, 0.1, 0.2):
        warp(image, p, kind, strength, compact=True)
        mx, my = maps[-1]
        assert np.array_equal(mx[:10], xx[:10]) and np.array_equal(my[:10], yy[:10])
        assert np.any(mx != xx) or np.any(my != yy)
        dx_dy, dx_dx = np.gradient(mx)
        dy_dy, dy_dx = np.gradient(my)
        assert np.min(dx_dx * dy_dy - dx_dy * dy_dx) > 0
    warp(image, p, kind, 0.2)
    assert np.any(maps[-1][0][:10] != xx[:10]) or np.any(maps[-1][1][:10] != yy[:10])


def test_freeze_never_reads_calibration_or_validation_labels_for_references():
    manifest, extracted = corpus(100)
    splits = split_sources(manifest)
    changed = manifest.model_copy(
        update={
            "samples": [
                s.model_copy(update={"reviews": [review(approve_anchor=True)]})
                if splits[s.source_id] != "development"
                else s
                for s in manifest.samples
            ]
        }
    )
    before, after = freeze_study(manifest, extracted), freeze_study(changed, extracted)
    assert before["references"] == after["references"]
    assert before["normalizers"] == after["normalizers"]
    assert before["labels_hash"] != after["labels_hash"]
    assert digest(before) != digest(after)


@pytest.mark.parametrize("anomaly_sources,accepted", [(2, False), (20, True)])
def test_adoption_uses_reviewed_anomalies_and_separately_reports_real_evidence(
    anomaly_sources, accepted
):
    # Controlled fixture, not an actual human dataset or accuracy result.
    manifest, extracted = corpus()
    splits = split_sources(manifest)
    originals = [s for s in manifest.samples if splits[s.source_id] == "validation"][
        :anomaly_sources
    ]
    children = []
    for i, parent in enumerate(originals):
        child = parent.model_copy(
            update={
                "sample_id": f"synthetic_child{i}",
                "parent_id": parent.sample_id,
                "primary": False,
                "input_sha256": hash_id(f"synthetic_child{i}"),
                "transform": {"kind": "nose_mouth", "strength": 0.2},
                "reviews": [
                    review().model_copy(
                        update={
                            "label": "geometry_anomaly",
                            "parts": ["nose"],
                            "severity": "severe",
                        }
                    )
                ],
            }
        )
        children.append(child)
        p = landmarks()
        p[23, 0] += 0.8
        extracted["records"].append(
            {
                "sample_id": child.sample_id,
                "input_sha256": child.input_sha256,
                "faces": [{"keypoints": p.tolist()}],
            }
        )
    manifest = StudyManifest(
        character_id=manifest.character_id, samples=[*manifest.samples, *children]
    )
    extracted["data_hash"] = data_hash(manifest)
    frozen = freeze_study(manifest, extracted, candidate_variant="single_local")
    calibrated = calibrate_study(manifest, extracted, frozen)
    report = evaluate_study(manifest, extracted, frozen, calibrated)
    assert report["adoption"]["single_local"]["adopt"] is accepted
    assert report["adoption"]["single_local"]["bootstrap_real_anomalies"]["interval95"] is None
    assert report["strata"]["synthetic"]["True"]["single_local"]["recall"] == 1
