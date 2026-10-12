"""Synthetic coordinates/labels test research behavior, never model accuracy."""

import copy
import hashlib
import json
import runpy
import sys
from pathlib import Path

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
                "single_local": {"decision": "anomaly" if i < 8 else "unknown"},
                "wls": {"decision": "within_thresholds"},
            },
        }
        for i in range(10)
    ]
    result = paired_bootstrap(rows, "single_local")
    assert result["interval95"][0] > 0
    assert result == paired_bootstrap(rows, "single_local")
    assert paired_bootstrap([rows[0]] * 20, "single_local")["interval95"] is None
    for row in rows:
        row["primary"] = False
    summary = summarize_decisions(rows, "single_local")
    assert summary["recall"] == 0.8
    assert summary["anomaly_unknown_counted_missed"] == 2


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


def test_adoption_uses_reviewed_anomalies_and_separately_reports_real_evidence():
    # Controlled fixture, not an actual human dataset or accuracy result.
    manifest, extracted = corpus()
    splits = split_sources(manifest)
    originals = [s for s in manifest.samples if splits[s.source_id] == "validation"][:10]
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
    assert report["adoption"]["single_local"]["adopt"] is True
    assert report["adoption"]["single_local"]["bootstrap_real_anomalies"]["interval95"] is None
    assert report["strata"]["synthetic"]["True"]["single_local"]["recall"] == 1
