"""Opt-in real CPU experiment; publishes numbers, keeps images in the output directory.

Run in the pinned observer environment. IRLS is an experimental comparator and does
not change Runtime scoring. Local warps are controlled inputs, not generated failures.
"""

import argparse
import base64
import html
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from anime_avatar_runtime.contracts import ObservationRequest
from anime_avatar_runtime.local_geometry import StudyManifest, data_hash, resolved_review
from anime_avatar_runtime.observers import (
    STABLE_POINTS,
    OfflineObserver,
    face_status,
    geometry_distance,
    observe_frame,
    read_image,
    sha256,
)


def blind_review_page(manifest: StudyManifest, directory: Path) -> str:
    """Render anonymous inputs with Chinese controls, preserving prior human entries."""
    import io

    fields = {
        "label": (
            "图片分类",
            [
                ("", "请选择分类"),
                ("normal", "正常"),
                ("geometry_anomaly", "几何畸变"),
                ("uncertain", "不确定／无法判断"),
            ],
        ),
        "pose": (
            "头部姿态",
            [
                ("unknown", "无法判断"),
                ("front", "正脸"),
                ("left_turn", "向画面左侧转头"),
                ("right_turn", "向画面右侧转头"),
                ("down", "低头"),
                ("up", "抬头"),
                ("unsupported", "超出范围（如大角度侧脸）"),
            ],
        ),
        "mode": (
            "场景模式",
            [("unknown", "无法判断"), ("idle", "待机／轻微动作"), ("interaction", "互动场景")],
        ),
        "occlusion": (
            "遮挡情况",
            [
                ("unknown", "无法判断"),
                ("none", "无遮挡"),
                ("mild", "少量遮挡"),
                ("severe", "严重遮挡"),
            ],
        ),
        "severity": (
            "畸变程度",
            [
                ("unknown", "无法判断"),
                ("none", "无畸变"),
                ("mild", "轻微"),
                ("moderate", "中等"),
                ("severe", "严重"),
            ],
        ),
    }
    regions = {
        "left_eye": "画面左眼",
        "right_eye": "画面右眼",
        "eye_spacing": "双眼关系／眼距",
        "nose": "鼻",
        "mouth": "嘴",
        "contour": "脸部轮廓／下巴",
    }
    cards = []
    ordered = list(manifest.samples)
    np.random.default_rng(42).shuffle(ordered)
    for sample in ordered:
        prior = resolved_review(sample)
        image_path = sample.image if sample.image.is_absolute() else directory / sample.image
        image = read_image(image_path.read_bytes())
        image.thumbnail((360, 420))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        encoded = base64.b64encode(buffer.getvalue()).decode()
        controls = ""
        for key, (label, options) in fields.items():
            selected = prior.get(key, "" if key == "label" else "unknown")
            if key == "label" and prior["review_status"] == "pending_human":
                selected = ""
            controls += f'<label>{label}<select data-field="{key}">'
            controls += "".join(
                f'<option value="{value}"'
                + (" selected" if value == selected else "")
                + f">{label}</option>"
                for value, label in options
            )
            controls += "</select></label>"
        controls += "<fieldset><legend>异常部位（仅畸变时勾选）</legend>"
        for region, label in regions.items():
            checked = " checked" if region in prior.get("parts", []) else ""
            controls += (
                f'<label><input type="checkbox" data-part="{region}"{checked}>{label}</label>'
            )
        controls += "</fieldset>"
        checked = " checked" if prior.get("approve_anchor") else ""
        controls += f'<label><input type="checkbox" data-anchor{checked}>确认作为正常参考</label>'
        cards.append(
            f'<section data-id="{sample.sample_id}"><h2>{sample.sample_id}</h2>'
            f'<img src="data:image/jpeg;base64,{encoded}">{controls}</section>'
        )
    reviewers = {
        r.reviewer for sample in manifest.samples for r in sample.reviews if r.kind == "human"
    }
    reviewer = html.escape(next(iter(reviewers)) if len(reviewers) == 1 else "", quote=True)
    return (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>局部几何盲审</title>"
        "<style>body{font:16px/1.6 sans-serif;background:#edf1f3;margin:25px}"
        "main{display:flex;flex-wrap:wrap;gap:20px}"
        "section{background:white;padding:18px;width:360px}"
        "img{max-width:100%}label{display:block;margin:8px}"
        "select{display:block;width:100%;padding:6px}button{padding:12px}"
        "#summary{position:sticky;top:0;background:#fff3cd;padding:12px;z-index:1}</style>"
        "<h1>单角色局部几何盲审</h1>"
        "<p>每张图先选择分类：正常表情、眨眼、张嘴、轻微转头本身不算畸变；"
        "五官错位、比例或脸形不自然才选几何畸变。看不清时可明确选不确定。</p>"
        "<p>左右按画面方向。请独立填写姿态、场景、遮挡和畸变程度；"
        "正常图选择无畸变，并取消异常部位勾选。未知是有效记录，但不能用于正式拟合。</p>"
        "<p>参考勾选不能代替分类。只有人工确认正常的原图才可能进入可信参考；"
        "派生图由程序排除。此页面不显示分数、变形参数或关键点。</p>"
        f'<label>人工复核者标识<input id="reviewer" value="{reviewer}"></label>'
        '<label><input type="checkbox" id="human">我已人工查看并确认这些标注</label>'
        '<button onclick="download()">导出 annotations.json</button>'
        '<p id="summary" role="status" aria-live="polite"></p><main>'
        + "".join(cards)
        + "</main><script>"
        'function entries(){return Array.from(document.querySelectorAll("section")).map(s=>{'
        'const review={reviewer:document.getElementById("reviewer").value.trim(),kind:"human",'
        'parts:Array.from(s.querySelectorAll("[data-part]:checked")).map(p=>p.dataset.part),'
        'approve_anchor:s.querySelector("[data-anchor]").checked};'
        's.querySelectorAll("select").forEach(e=>review[e.dataset.field]=e.value);'
        "return {sample_id:s.dataset.id,review};});}"
        "function summary(){const rows=entries(),"
        "count=v=>rows.filter(r=>r.review.label===v).length;"
        'const unknown=rows.filter(r=>["pose","mode","occlusion"]'
        '.some(k=>r.review[k]==="unknown")).length;'
        'const anchors=rows.filter(r=>r.review.approve_anchor&&r.review.label!=="normal").length;'
        'document.getElementById("summary").textContent='
        '`共 ${rows.length} 张：未选 ${count("")}，正常 ${count("normal")}，'
        '畸变 ${count("geometry_anomaly")}，不确定 ${count("uncertain")}；'
        "姿态／场景／遮挡含未知 ${unknown} 张；勾参考但分类非正常 ${anchors} 张。`;}"
        'document.addEventListener("change",summary);summary();'
        "function download(){const reviews=entries();"
        'if(!document.getElementById("reviewer").value.trim()||'
        '!document.getElementById("human").checked){alert("请填写标识并确认人工复核");return;}'
        'if(reviews.some(r=>!r.review.label)){alert("请逐张选择分类；无法判断时可选不确定。");return;}'
        "const payload={schema_version:1,data_hash:"
        + json.dumps(data_hash(manifest))
        + ',reviews};const a=document.createElement("a");'
        "a.href=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],"
        '{type:"application/json"}));a.download="annotations.json";a.click();'
        "URL.revokeObjectURL(a.href);}</script></html>"
    )


def study_main(argv: list[str]) -> None:
    """Manifest-driven stages; legacy image comparison remains available below."""
    from anime_avatar_runtime.local_geometry import (
        Review,
        StudySample,
        calibrate_study,
        evaluate_study,
        feature_variants,
        freeze_study,
        implementation_hash,
        points,
        split_sources,
        verify_extraction,
    )

    parser = argparse.ArgumentParser(description="Offline, human-reviewed geometry study")
    commands = parser.add_subparsers(dest="stage", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--gif", type=Path, required=True)
    prepare.add_argument("--crop", nargs=4, type=int, required=True)
    prepare.add_argument("--frames", nargs="+", type=int, default=[0, 5, 10, 15, 25, 36, 51, 56])
    prepare.add_argument("--primary-frame", type=int, default=0)
    prepare.add_argument("--character", required=True)
    prepare.add_argument("--models", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    review = commands.add_parser("review")
    review.add_argument("--manifest", type=Path, required=True)
    review.add_argument("--annotations", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    review_page = commands.add_parser("review-page")
    review_page.add_argument("--manifest", type=Path, required=True)
    review_page.add_argument("--output", type=Path, required=True)
    for stage in ("extract", "explore", "freeze", "calibrate", "evaluate"):
        cmd = commands.add_parser(stage)
        cmd.add_argument("--manifest", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
        if stage == "extract":
            cmd.add_argument("--models", type=Path, required=True)
        else:
            cmd.add_argument("--extraction", type=Path, required=True)
        if stage == "explore":
            cmd.add_argument("--reference", required=True)
        if stage in {"calibrate", "evaluate"}:
            cmd.add_argument("--frozen", type=Path, required=True)
        if stage == "evaluate":
            cmd.add_argument("--calibration", type=Path, required=True)
        if stage == "freeze":
            cmd.add_argument(
                "--candidate", choices=["single_local", "pose_local"], default="pose_local"
            )
    args = parser.parse_args(argv)

    def save(path: Path, value: Any) -> None:
        if path.exists():
            raise FileExistsError(f"Do not overwrite a study artifact: {path}")
        path.write_text(
            json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if args.stage == "prepare":
        if args.primary_frame not in args.frames or len(set(args.frames)) != len(args.frames):
            raise ValueError("Select unique frames including the preselected primary")
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "images").mkdir()
        started = time.monotonic()
        observer = OfflineObserver(args.models, args.output / "cache")
        request = ObservationRequest()
        samples = []
        skipped = []
        source_hash = sha256(args.gif)
        with Image.open(args.gif) as gif:
            duration = 0.0
            for index in range(gif.n_frames):
                gif.seek(index)
                if index in args.frames:
                    image = gif.convert("RGB").crop(tuple(args.crop))
                    sample_id = f"s{len(samples):04d}"
                    location = args.output / "images" / f"{sample_id}.png"
                    image.save(location)
                    parent = StudySample(
                        sample_id=sample_id,
                        source_id="anisora_portrait_source",
                        source_sha256=source_hash,
                        image=Path("images") / location.name,
                        input_sha256=sha256(location),
                        primary=index == args.primary_frame,
                        frame_index=index,
                        timestamp_seconds=duration,
                    )
                    samples.append(parent)
                    try:
                        detected = observer.detect(image, request)
                        p = points({"faces": detected})
                        if p is None or np.any(p[[11, 13, 17, 19], 2] < 0.3):
                            raise ValueError("Unavailable eyes for deterministic local warp")
                    except Exception as exc:
                        skipped.append({"sample_id": sample_id, "reason": str(exc)})
                        duration += gif.info.get("duration", 0) / 1000
                        continue
                    for kind in ("eye_spacing", "nose_mouth", "chin"):
                        for strength in (0.05, 0.1, 0.2):
                            warped = warp(image, p, kind, strength)
                            child_id = f"s{len(samples):04d}"
                            location = args.output / "images" / f"{child_id}.png"
                            warped.save(location)
                            samples.append(
                                StudySample(
                                    sample_id=child_id,
                                    source_id=parent.source_id,
                                    source_sha256=source_hash,
                                    image=Path("images") / location.name,
                                    input_sha256=sha256(location),
                                    parent_id=sample_id,
                                    frame_index=index,
                                    timestamp_seconds=duration,
                                    transform={
                                        "kind": kind,
                                        "strength": strength,
                                        "method": "inverse_gaussian_interocular_displacement",
                                    },
                                )
                            )
                duration += gif.info.get("duration", 0) / 1000
        if not samples or not any(s.primary for s in samples):
            raise ValueError("Selected frame/primary does not exist in input")
        manifest = StudyManifest(
            character_id=args.character,
            samples=samples,
            provenance={
                "source_kind": "public_gif",
                "crop": args.crop,
                "source_sha256": source_hash,
                "warp_skips": skipped,
                "prepare_seconds": time.monotonic() - started,
                "models": observer.model_manifest,
                "seed": 42,
            },
        )
        save(args.output / "manifest.json", manifest.model_dump(mode="json"))
        save(args.output / "splits.json", split_sources(manifest))
        page = blind_review_page(manifest, args.output)
        (args.output / "review.html").write_text(page, encoding="utf-8")
        print(f"Prepared {len(samples)} samples, {len(skipped)} warp skips; labels pending human")
        return

    manifest = StudyManifest.model_validate_json(args.manifest.read_bytes())
    for s in manifest.samples:
        location = s.image if s.image.is_absolute() else args.manifest.parent / s.image
        if sha256(location) != s.input_sha256:
            raise ValueError(f"Input changed: {s.sample_id}")
    if args.stage == "review-page":
        if args.output.exists():
            raise FileExistsError(f"Do not overwrite a blind review page: {args.output}")
        page = blind_review_page(manifest, args.manifest.parent)
        args.output.write_text(page, encoding="utf-8")
        return
    if args.stage == "review":
        if args.output.parent.resolve() != args.manifest.parent.resolve():
            raise ValueError(
                "Save reviewed manifests beside the original to preserve relative paths"
            )
        annotations = json.loads(args.annotations.read_bytes())
        if annotations["data_hash"] != data_hash(manifest):
            raise ValueError("Annotations belong to another dataset")
        updates = {item["sample_id"]: item for item in annotations["reviews"]}
        if len(updates) != len(annotations["reviews"]) or not set(updates) <= {
            s.sample_id for s in manifest.samples
        }:
            raise ValueError("Duplicate or unknown annotation IDs")
        value = manifest.model_dump(mode="json")
        for sample in value["samples"]:
            if sample["sample_id"] in updates:
                item = updates[sample["sample_id"]]
                verified = Review.model_validate(item["review"])
                if verified.kind != "human":
                    raise ValueError("Blind review import requires an explicit human declaration")
                if item.get("adjudication"):
                    sample["adjudication"] = verified.model_dump()
                else:
                    sample["reviews"].append(verified.model_dump())
        reviewed = StudyManifest.model_validate(value)
        save(args.output, reviewed.model_dump(mode="json"))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    if args.stage == "extract":
        started = time.monotonic()
        observer = OfflineObserver(args.models, args.output / "cache")
        load_seconds = time.monotonic() - started
        records = []
        (args.output / "features").mkdir()
        for s in manifest.samples:
            image = read_image(
                (s.image if s.image.is_absolute() else args.manifest.parent / s.image).read_bytes()
            )
            record: dict[str, Any] = {
                "sample_id": s.sample_id,
                "input_sha256": s.input_sha256,
                "faces": [],
                "errors": {},
            }
            begin = time.monotonic()
            try:
                record["faces"] = observer.detect(image, ObservationRequest())
            except Exception as exc:
                record["errors"]["geometry"] = str(exc)
            record["geometry_seconds"] = time.monotonic() - begin
            begin = time.monotonic()
            try:
                feature = observer.extract(image)
                location = args.output / "features" / f"{s.sample_id}.npy"
                np.save(location, feature, allow_pickle=False)
                record["identity_feature_sha256"] = sha256(location)
            except Exception as exc:
                record["errors"]["identity"] = str(exc)
            record["identity_seconds"] = time.monotonic() - begin
            records.append(record)
            print(s.sample_id, "faces", len(record["faces"]), flush=True)
        save(
            args.output / "extraction.json",
            {
                "schema_version": 1,
                "data_hash": data_hash(manifest),
                "models": observer.model_manifest,
                "implementation_hash": implementation_hash(),
                "records": records,
                "load_seconds": load_seconds,
                "total_seconds": time.monotonic() - started,
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "processor": platform.processor(),
                    "device": "cpu",
                    "threads": 2,
                },
            },
        )
        return
    extracted = json.loads(args.extraction.read_bytes())
    verify_extraction(manifest, extracted)
    if args.stage == "explore":
        reference = next(s for s in manifest.samples if s.sample_id == args.reference)
        records = {r["sample_id"]: r for r in extracted["records"]}
        p = points(records[reference.sample_id])
        if p is None or reference.transform is not None:
            raise ValueError("Exploratory reference must be a detectable original")
        diagnostic = {
            "references": {"front": {"sample_id": reference.sample_id, "points": p.tolist()}},
            "pose_gates": {},
        }
        rows = [
            {
                "sample_id": s.sample_id,
                "parent_id": s.parent_id,
                "frame_index": s.frame_index,
                "transform": s.transform,
                "input_sha256": s.input_sha256,
                "metrics": feature_variants(points(records[s.sample_id]), diagnostic, "front"),
            }
            for s in manifest.samples
        ]
        save(
            args.output / "diagnostics.json",
            {
                "diagnostic_only": True,
                "not_human_ground_truth": True,
                "reference_id": reference.sample_id,
                "data_hash": data_hash(manifest),
                "models": extracted["models"],
                "rows": rows,
            },
        )
        return
    if args.stage == "freeze":
        save(args.output / "frozen.json", freeze_study(manifest, extracted, args.candidate))
        return
    frozen = json.loads(args.frozen.read_bytes())
    if args.stage == "calibrate":
        save(args.output / "calibration.json", calibrate_study(manifest, extracted, frozen))
        return
    calibrated = json.loads(args.calibration.read_bytes())
    report = evaluate_study(manifest, extracted, frozen, calibrated)
    save(args.output / "evaluation.json", report)
    table = "\n".join(
        f"| {v} | {row['valid_normal_sources']} | {row['fpr_upper95']} | "
        f"{row['normal_coverage']} | {row['recall']} |"
        for v, row in report["summary"].items()
    )
    text = (
        "# 单角色局部几何独立评估\n\n"
        f"角色：{manifest.character_id}；来源分组：{report['source_counts']}。\n\n"
        f"人工标注状态：{report['human_review_counts']}。\n\n"
        "| 方法 | 有效正常独立来源 | 误报上界95% | 完整判定覆盖率 | 畸变检出率 |\n"
        "| --- | --- | --- | --- | --- |\n" + table + "\n\n"
        "null 表示没有可估计分母。异常 unknown 计为未检出；不确定标签不充当正常标签。\n\n"
        "采用条件与分层数值见 evaluation.json。数据不足时不授予 5% 验收，不接 Runtime/恢复。\n"
    )
    (args.output / "report.md").write_text(text, encoding="utf-8")
    (args.output / "report.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>独立评估</title><pre>'
        + html.escape(text)
        + "</pre></html>",
        encoding="utf-8",
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
    if len(sys.argv) > 1 and sys.argv[1] == "study":
        study_main(sys.argv[2:])
        return
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
