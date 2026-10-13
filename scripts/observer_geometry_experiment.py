"""Opt-in real CPU experiment; publishes numbers, keeps images in the output directory.

Run in the pinned observer environment. IRLS is an experimental comparator and does
not change Runtime scoring. Local warps are controlled inputs, not generated failures.
"""

import argparse
import base64
import html
import importlib.metadata
import json
import math
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
from anime_avatar_runtime.local_geometry import (
    POLICY,
    REGIONS,
    StudyManifest,
    aligned_residuals,
    contour_shape,
    data_hash,
    digest,
    points,
    resolved_review,
)
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
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
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
            f'<img alt="匿名样本 {sample.sample_id}" src="data:image/png;base64,{encoded}" '
            "onclick=\"document.getElementById('fullimage').src=this.src;"
            "document.getElementById('zoom').showModal()\">"
            f"{controls}</section>"
        )
    reviewers = {
        r.reviewer for sample in manifest.samples for r in sample.reviews if r.kind == "human"
    }
    reviewer = html.escape(next(iter(reviewers)) if len(reviewers) == 1 else "", quote=True)
    bulk_controls = ""
    for key in ("pose", "mode", "occlusion"):
        label, options = fields[key]
        bulk_controls += (
            f'<label>{label}<select data-bulk="{key}" onchange="bulk(\'{key}\',this.value)">'
            '<option value="">不批量设置</option>'
            + "".join(f'<option value="{value}">{text}</option>' for value, text in options)
            + "</select></label>"
        )
    return (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>局部几何盲审</title>"
        "<style>body{font:16px/1.6 sans-serif;background:#edf1f3;margin:25px}"
        "main{display:flex;flex-wrap:wrap;gap:20px}"
        "section{background:white;padding:18px;width:360px}"
        "img{max-width:100%;cursor:zoom-in}label{display:block;margin:8px}"
        "select{display:block;width:100%;padding:6px}button{padding:12px}"
        "dialog{max-width:95vw;max-height:95vh}dialog img{display:block;cursor:default}"
        "#summary{position:sticky;top:0;background:#fff3cd;padding:12px;z-index:1}</style>"
        "<h1>单角色局部几何盲审</h1>"
        "<p>每张图先选择分类：正常表情、眨眼、张嘴、轻微转头本身不算畸变；"
        "五官错位、比例或脸形不自然才选几何畸变。看不清时可明确选不确定。</p>"
        "<p>左右按画面方向。请独立填写姿态、场景、遮挡和畸变程度；"
        "正常图选择无畸变，并取消异常部位勾选。未知是有效记录，但不能用于正式拟合。</p>"
        "<p>参考勾选不能代替分类。只有人工确认正常的原图才可能进入可信参考；"
        "派生图由程序排除。此页面不显示分数、变形参数或关键点。</p>"
        "<p>点击图片查看完整尺寸。若所有图的姿态、场景或遮挡一致，可明确选择下方批量设置；"
        "有例外的图片再单独修改。分类和畸变程度仍须逐张填写。</p>"
        "<fieldset><legend>人工批量设置（选择后应用到所有样本）</legend>"
        + bulk_controls
        + "</fieldset>"
        f'<label>人工复核者标识<input id="reviewer" value="{reviewer}"></label>'
        '<label><input type="checkbox" id="human">我已人工查看并确认这些标注</label>'
        '<button onclick="download()">导出 annotations.json</button>'
        '<p id="saved" role="status">填写后请导出文件；浏览器暂存不能代替导出。</p>'
        '<dialog id="zoom">'
        "<button onclick=\"document.getElementById('zoom').close()\">关闭</button>"
        '<img id="fullimage" alt="完整尺寸匿名样本"></dialog>'
        '<p id="summary" role="status" aria-live="polite"></p><main>'
        + "".join(cards)
        + "</main><script>"
        + "const dataHash="
        + json.dumps(data_hash(manifest))
        + ";"
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
        "function bulk(key,value){if(!value)return;"
        'document.querySelectorAll("section select").forEach(e=>{'
        "if(e.dataset.field===key)e.value=value;});summary();saveReview();}"
        'function saveReview(){try{localStorage.setItem("geometry-review:"+dataHash,'
        'JSON.stringify({reviews:entries()}));document.getElementById("saved").textContent='
        '"已在当前浏览器暂存；完成后仍须导出 annotations.json。";'
        '}catch{document.getElementById("saved").textContent='
        '"浏览器无法暂存，请导出文件保存；刷新可能丢失填写。";}}'
        'try{const saved=JSON.parse(localStorage.getItem("geometry-review:"+dataHash)||"null");'
        "if(saved){const byId=new Map(saved.reviews.map(r=>[r.sample_id,r.review]));"
        'document.querySelectorAll("section").forEach(s=>{const r=byId.get(s.dataset.id);'
        'if(!r)return;document.getElementById("reviewer").value=r.reviewer||"";'
        's.querySelectorAll("select").forEach(e=>{'
        "if(Array.from(e.options).some(o=>o.value===r[e.dataset.field]))e.value=r[e.dataset.field];});"
        's.querySelectorAll("[data-part]").forEach(e=>e.checked=(r.parts||[]).includes(e.dataset.part));'
        's.querySelector("[data-anchor]").checked=r.approve_anchor===true;});}}catch{}'
        'document.addEventListener("change",()=>{summary();saveReview();});'
        'document.getElementById("reviewer").addEventListener("input",saveReview);summary();'
        "function download(){const reviews=entries();"
        'if(!document.getElementById("reviewer").value.trim()||'
        '!document.getElementById("human").checked){alert("请填写标识并确认人工复核");return;}'
        'if(reviews.some(r=>!r.review.label)){alert("请逐张选择分类；无法判断时可选不确定。");return;}'
        "const payload={schema_version:1,data_hash:dataHash,reviews};"
        'const a=document.createElement("a");'
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
    source = prepare.add_mutually_exclusive_group(required=True)
    source.add_argument("--gif", type=Path)
    source.add_argument("--image", type=Path)
    prepare.add_argument("--crop", nargs=4, type=int)
    prepare.add_argument("--frames", nargs="+", type=int)
    prepare.add_argument("--primary-frame", type=int, default=0)
    prepare.add_argument("--similarity-controls", action="store_true")
    prepare.add_argument("--compact-warps", action="store_true")
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
    budget = commands.add_parser("budget")
    budget.add_argument("--output", type=Path, required=True)
    for stage in (
        "extract",
        "explore",
        "coupling",
        "intrinsic",
        "stability",
        "repeatability",
        "flow",
        "flow-regions",
        "replay",
        "freeze",
        "calibrate",
        "evaluate",
    ):
        cmd = commands.add_parser(stage)
        cmd.add_argument("--manifest", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
        if stage == "extract":
            cmd.add_argument("--models", type=Path, required=True)
        else:
            cmd.add_argument("--extraction", type=Path, required=True)
        if stage == "repeatability":
            cmd.add_argument("--models", type=Path, required=True)
        if stage in {
            "explore",
            "coupling",
            "intrinsic",
            "stability",
            "flow",
            "flow-regions",
            "replay",
        }:
            cmd.add_argument("--reference", required=True)
        if stage == "flow":
            cmd.add_argument("--gmflow-repo", type=Path, required=True)
            cmd.add_argument("--gmflow-model", type=Path, required=True)
            cmd.add_argument("--gmflow-provenance", type=Path, required=True)
        if stage == "flow-regions":
            cmd.add_argument("--flow-report", type=Path, required=True)
        if stage in {"calibrate", "evaluate"}:
            cmd.add_argument("--frozen", type=Path, required=True)
        if stage == "evaluate":
            cmd.add_argument("--calibration", type=Path, required=True)
        if stage == "freeze":
            cmd.add_argument(
                "--candidate",
                choices=["single_local", "pose_local", "intrinsic_local"],
                default="pose_local",
            )
    args = parser.parse_args(argv)

    def save(path: Path, value: Any) -> None:
        if path.exists():
            raise FileExistsError(f"Do not overwrite a study artifact: {path}")
        path.write_text(
            json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if args.stage == "budget":
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output / "budget.json", statistical_budget())
        return
    if args.stage == "prepare":
        if args.frames is None:
            args.frames = [0] if args.image else [0, 5, 10, 15, 25, 36, 51, 56]
        if args.image and (args.frames != [0] or args.primary_frame != 0):
            raise ValueError("A still image has only frame 0")
        if args.gif and args.crop is None:
            raise ValueError("GIF preparation requires an explicit crop")
        if args.primary_frame not in args.frames or len(set(args.frames)) != len(args.frames):
            raise ValueError("Select unique frames including the preselected primary")
        source_path = args.image or args.gif
        with Image.open(source_path) as source_image:
            if args.image and getattr(source_image, "n_frames", 1) != 1:
                raise ValueError("Use --gif with explicit frames for an animated input")
            crop = args.crop or [0, 0, source_image.width, source_image.height]
            if not (
                0 <= crop[0] < crop[2] <= source_image.width
                and 0 <= crop[1] < crop[3] <= source_image.height
            ):
                raise ValueError("Crop must be nonempty and inside the input")
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "images").mkdir()
        started = time.monotonic()
        observer = OfflineObserver(args.models, args.output / "cache")
        request = ObservationRequest()
        samples = []
        skipped = []
        source_hash = sha256(source_path)
        with Image.open(source_path) as gif:
            duration = 0.0
            for index in range(getattr(gif, "n_frames", 1)):
                gif.seek(index)
                if index in args.frames:
                    image = gif.convert("RGB").crop(tuple(crop))
                    sample_id = f"s{len(samples):04d}"
                    location = args.output / "images" / f"{sample_id}.png"
                    image.save(location)
                    parent = StudySample(
                        sample_id=sample_id,
                        source_id="still_image_source" if args.image else "anisora_portrait_source",
                        source_sha256=source_hash,
                        image=Path("images") / location.name,
                        input_sha256=sha256(location),
                        primary=index == args.primary_frame,
                        frame_index=index,
                        timestamp_seconds=duration,
                    )
                    samples.append(parent)
                    if args.similarity_controls:
                        fill = image.getpixel((0, 0))
                        controls = [
                            (
                                image.rotate(
                                    angle, resample=Image.Resampling.BICUBIC, fillcolor=fill
                                ),
                                {"kind": "similarity_rotation", "degrees": angle},
                            )
                            for angle in (-8, 8)
                        ]
                        size = (round(image.width * 0.9), round(image.height * 0.9))
                        scaled = Image.new("RGB", image.size, fill)
                        scaled.paste(
                            image.resize(size, Image.Resampling.BICUBIC),
                            ((image.width - size[0]) // 2, (image.height - size[1]) // 2),
                        )
                        controls.append((scaled, {"kind": "similarity_scale", "scale": 0.9}))
                        for transformed, transform in controls:
                            child_id = f"s{len(samples):04d}"
                            location = args.output / "images" / f"{child_id}.png"
                            transformed.save(location)
                            samples.append(
                                parent.model_copy(
                                    update={
                                        "sample_id": child_id,
                                        "image": Path("images") / location.name,
                                        "input_sha256": sha256(location),
                                        "parent_id": sample_id,
                                        "primary": False,
                                        "transform": transform,
                                    }
                                )
                            )
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
                            warped = warp(image, p, kind, strength, compact=args.compact_warps)
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
                                        "method": (
                                            "inverse_compact_elliptic_displacement"
                                            if args.compact_warps
                                            else "inverse_gaussian_interocular_displacement"
                                        ),
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
                "source_kind": "still_image" if args.image else "public_gif",
                "crop": crop,
                "source_sha256": source_hash,
                "prepare_code_sha256": sha256(Path(__file__)),
                "compact_warps": args.compact_warps,
                "similarity_controls": args.similarity_controls,
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
    if args.stage == "coupling":
        save(
            args.output / "coupling.json", coupling_experiment(manifest, extracted, args.reference)
        )
        return
    if args.stage == "intrinsic":
        save(
            args.output / "intrinsic.json",
            intrinsic_experiment(manifest, extracted, args.reference),
        )
        return
    if args.stage == "stability":
        save(
            args.output / "stability.json",
            stability_experiment(manifest, extracted, args.reference),
        )
        return
    if args.stage == "repeatability":
        save(
            args.output / "repeatability.json",
            repeatability_experiment(
                manifest, extracted, args.manifest.parent, args.models, args.output
            ),
        )
        return
    if args.stage == "flow":
        save(
            args.output / "flow.json",
            frontal_flow_experiment(
                manifest,
                args.manifest.parent,
                args.reference,
                args.gmflow_repo,
                args.gmflow_model,
                args.gmflow_provenance,
                args.output,
            ),
        )
        return
    if args.stage == "flow-regions":
        save(
            args.output / "flow-regions.json",
            regional_flow_experiment(
                manifest, extracted, args.manifest.parent, args.reference, args.flow_report
            ),
        )
        return
    if args.stage == "replay":
        shape = intrinsic_experiment(manifest, extracted, args.reference)
        scores = {row["sample_id"]: row["intrinsic"]["shape_rms"] for row in shape["rows"]}
        scenarios = {
            "normal": ["s0000"] * 12,
            "normal_eye_transient": ["s0000"] * 3 + ["s0005"] + ["s0000"] * 8,
            "anomaly_transient": ["s0000"] * 3 + ["s0012"] + ["s0000"] * 8,
            "anomaly_persistent": ["s0000"] * 3 + ["s0012"] * 12,
            "gap_interrupts_anomaly": ["s0012"] * 4 + [None] + ["s0012"] * 4,
        }
        if not set(s for ids in scenarios.values() for s in ids if s).issubset(scores):
            raise ValueError("Replay fixture requires the documented #39 frontal sample IDs")
        timelines = {}
        for name, ids in scenarios.items():
            timeline = [
                {
                    "time": i * 0.1,
                    "sample_id": sample_id,
                    "value": scores[sample_id] if sample_id else None,
                    "mode": "idle",
                }
                for i, sample_id in enumerate(ids)
            ]
            timelines[name] = causal_replay(timeline, 0.03, 0.015, 0.5, 0.15)
        save(
            args.output / "replay.json",
            {
                "schema_version": 1,
                "diagnostic_only": True,
                "synthetic_timeline": True,
                "data_hash": data_hash(manifest),
                "implementation_sha256": sha256(Path(__file__)),
                "policy": {
                    "enter": 0.03,
                    "exit": 0.015,
                    "dwell_seconds": 0.5,
                    "max_gap_seconds": 0.15,
                },
                "scenarios": timelines,
                "scope": (
                    "Measured frontal still scores in software fixtures; no real video, "
                    "calibrated threshold, prediction or recovery experiment"
                ),
            },
        )
        return
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


def coupling_operator(
    reference: NDArray[np.float64],
    fit: tuple[int, ...],
    confidence: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], float, float, NDArray[np.float64]]:
    """Fixed-weight derivative of signed, reference-radius-normalized residuals.

    Parameters are normalized x/y translation, scale and rotation at identity.
    Confidence is frozen, including its threshold mask; it is not a covariance.
    The SVD fit supplies the same feasibility checks and radius as the study.
    """
    base = reference.copy()
    if confidence is not None:
        base[:, 2] = confidence
    _, _, radius, _ = aligned_residuals(base, reference, fit)
    xy = (reference[:, :2] - reference[:, :2].mean(axis=0)) / radius
    jacobian = np.zeros((56, 4))
    jacobian[0::2] = np.column_stack((np.ones(28), np.zeros(28), xy[:, 0], -xy[:, 1]))
    jacobian[1::2] = np.column_stack((np.zeros(28), np.ones(28), xy[:, 1], xy[:, 0]))
    weights = np.clip(np.minimum(base[:, 2], reference[:, 2]), 0, 1)
    weights[weights < POLICY["min_confidence"]] = 0
    rows = np.array([2 * i + d for i in fit for d in (0, 1)])
    root_weights = np.sqrt(np.repeat(weights[list(fit)], 2))
    design = jacobian[rows] * root_weights[:, None]
    selector = np.eye(56)[rows] * root_weights[:, None]
    solution, _, rank, singular = np.linalg.lstsq(design, selector, rcond=None)
    if rank != 4:
        raise ValueError("Rank-deficient similarity Jacobian")
    return np.eye(56) - jacobian @ solution, radius, float(singular[0] / singular[-1]), jacobian


def coupling_experiment(
    manifest: StudyManifest, extracted: dict[str, Any], reference_id: str
) -> dict[str, Any]:
    """Mechanism diagnostics and predeclared fit ablations, without threshold fitting."""
    reference_sample = next(s for s in manifest.samples if s.sample_id == reference_id)
    review = resolved_review(reference_sample)
    if (
        reference_sample.parent_id is not None
        or reference_sample.transform is not None
        or review.get("label") != "normal"
        or not review.get("approve_anchor")
        or review["review_status"] not in {"confirmed", "adjudicated"}
    ):
        raise ValueError("Coupling reference must be an approved human-normal original")
    records = {r["sample_id"]: r for r in extracted["records"]}
    reference = points(records[reference_id])
    if reference is None:
        raise ValueError("Reference has no valid single-face landmarks")
    # Declared before examining new scores. Every target remains outside its own fit.
    fits = {
        variant: {
            region: tuple(i for i in pool if i not in target) for region, target in REGIONS.items()
        }
        for variant, pool in {
            "complement": STABLE_POINTS,
            "without_nose": tuple(i for i in STABLE_POINTS if i != 23),
            "with_mouth": (*STABLE_POINTS, 24, 26),
        }.items()
    }
    diagnostics = {}
    for variant, regions in fits.items():
        diagnostics[variant] = {}
        for region, fit in regions.items():
            operator, radius, condition, jacobian = coupling_operator(reference, fit)
            target = list(REGIONS[region])
            target_rows = [2 * i + d for i in target for d in (0, 1)]
            w = np.clip(reference[target, 2], 0, 1)
            w[w < POLICY["min_confidence"]] = 0
            if not np.any(w):
                raise ValueError("Reference target has no reliable points")
            weighted = operator[target_rows] * np.sqrt(np.repeat(w / w.sum(), 2))[:, None]
            gains = {}
            for source, indices in REGIONS.items():
                columns = [2 * i + d for i in indices for d in (0, 1)]
                gains[source] = float(
                    np.linalg.norm(weighted[:, columns], 2) * np.sqrt(len(indices))
                )
            finite_difference = {}
            for step in (1e-6, 1e-4):
                maximum = 0.0
                for column in range(56):
                    plus, minus = reference.copy(), reference.copy()
                    i, d = divmod(column, 2)
                    plus[i, d] += step * radius
                    minus[i, d] -= step * radius
                    a, _, _, _ = aligned_residuals(plus, reference, fit)
                    b, _, _, _ = aligned_residuals(minus, reference, fit)
                    derivative = ((a - b) / (2 * step * radius)).reshape(-1)
                    maximum = max(maximum, float(np.max(np.abs(derivative - operator[:, column]))))
                finite_difference[str(step)] = maximum
            theta = 0.3
            rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            similarity = reference.copy()
            similarity[:, :2] = reference[:, :2] @ rotation * 1.15 + [5 * radius, -3 * radius]
            _, errors, _, _ = aligned_residuals(similarity, reference, fit)
            chin_control = reference.copy()
            chin_control[2, 1] += 0.05 * radius
            _, chin_errors, _, _ = aligned_residuals(chin_control, reference, fit)
            diagnostics[variant][region] = {
                "fit_points": list(fit),
                "design_condition": condition,
                "spectral_coupling_rms_gain": gains,
                "isotropic_unit_coordinate_noise_rms_gain": float(np.linalg.norm(weighted)),
                "finite_difference_max_abs_error": finite_difference,
                "similarity_tangent_max_abs_error": float(np.max(np.abs(operator @ jacobian))),
                "exact_similarity_max_residual": float(errors.max()),
                "chin_shift_control_residual": float(chin_errors[2]),
            }
    rows = []
    for sample in manifest.samples:
        current = points(records[sample.sample_id])
        human = resolved_review(sample)
        scores = {}
        for variant, regions in fits.items():
            scores[variant] = {}
            for region, fit in regions.items():
                try:
                    if current is None:
                        raise ValueError("No valid single-face landmarks")
                    target = list(REGIONS[region])
                    w = np.clip(np.minimum(current[target, 2], reference[target, 2]), 0, 1)
                    w[w < POLICY["min_confidence"]] = 0
                    required = 3 if region == "contour" else len(target)
                    if np.count_nonzero(w) < required or (region == "contour" and w[2] == 0):
                        raise ValueError("Insufficient target points")
                    aligned, residual, radius, cost = aligned_residuals(current, reference, fit)
                    operator, _, _, _ = coupling_operator(reference, fit, current[:, 2])
                    delta = (current[:, :2] - reference[:, :2]) / radius
                    predicted = (operator @ delta.reshape(-1)).reshape(28, 2)
                    actual = (aligned - reference[:, :2]) / radius
                    value = float(np.sqrt(np.average(residual[target] ** 2, weights=w)))
                    entry = {
                        "status": "valid",
                        "value": value,
                        "fit_cost": cost,
                        "linear_prediction_error_rms": float(
                            np.sqrt(
                                np.average(
                                    np.sum((predicted[target] - actual[target]) ** 2, axis=1),
                                    weights=w,
                                )
                            )
                        ),
                    }
                    if region == "contour":
                        entry.update(
                            {
                                "chin": float(residual[2]),
                                "sides": float(
                                    np.sqrt(
                                        np.average(
                                            residual[[0, 1, 3, 4]] ** 2, weights=w[[0, 1, 3, 4]]
                                        )
                                    )
                                ),
                                "chin_before_alignment": float(np.linalg.norm(delta[2])),
                                "chin_linear_prediction": float(np.linalg.norm(predicted[2])),
                                "chin_linear_fit_contribution": float(
                                    np.linalg.norm(predicted[2] - delta[2])
                                ),
                            }
                        )
                    scores[variant][region] = entry
                except ValueError as exc:
                    scores[variant][region] = {"status": str(exc), "value": None}
        rows.append(
            {
                "sample_id": sample.sample_id,
                "input_sha256": sample.input_sha256,
                "transform": sample.transform,
                "human": {k: human.get(k) for k in ("label", "parts", "severity", "review_status")},
                "scores": scores,
            }
        )
    checks = {
        "derivative_max_abs_error": max(
            d["finite_difference_max_abs_error"]["1e-06"]
            for v in diagnostics.values()
            for d in v.values()
        ),
        "similarity_max_residual": max(
            d["exact_similarity_max_residual"] for v in diagnostics.values() for d in v.values()
        ),
        "contour_chin_control_max_error": max(
            abs(v["contour"]["chin_shift_control_residual"] - 0.05) for v in diagnostics.values()
        ),
    }
    if (
        checks["derivative_max_abs_error"] > 1e-8
        or checks["similarity_max_residual"] > 1e-10
        or checks["contour_chin_control_max_error"] > 1e-10
    ):
        raise ArithmeticError(f"Mechanism checks failed: {checks}")
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "evidence_level": "numerical-evidence",
        "reference_id": reference_id,
        "data_hash": data_hash(manifest),
        "manifest_sha256": digest(manifest.model_dump(mode="json")),
        "extraction_sha256": digest(extracted),
        "model_hash": digest(extracted["models"]),
        "implementation_sha256": sha256(Path(__file__)),
        "local_geometry_sha256": sha256(Path(sys.modules[aligned_residuals.__module__].__file__)),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "independent_sources": len({s.source_id for s in manifest.samples}),
        "protocol": {
            "precision": "float64; fixed confidence mask; reference RMS radius",
            "variants": fits,
            "derivative_steps": [1e-6, 1e-4],
            "acceptance_tolerances": {"derivative": 1e-8, "similarity_and_chin": 1e-10},
            "gain_definition": (
                "Worst weighted target RMS / unweighted source point RMS, over coordinate "
                "directions; overlapping regions are not a partition"
            ),
            "noise_assumption": (
                "Independent isotropic coordinate noise with unit variance, a sensitivity "
                "diagnostic only; confidence is not a calibrated probability"
            ),
            "selection": (
                "Predeclared diagnostic ablations; known single-source development labels; "
                "no candidate selection, threshold calibration or statistical acceptance"
            ),
        },
        "checks": checks,
        "diagnostics": diagnostics,
        "rows": rows,
        "scope": (
            "Reuses real extracted coordinates; perturbations and fit ablations do not rerun "
            "the detector or validate GPU generation, pose robustness or recovery"
        ),
    }


def intrinsic_experiment(
    manifest: StudyManifest, extracted: dict[str, Any], reference_id: str
) -> dict[str, Any]:
    """Frontal-only contour alternative and an identifiability negative control."""
    sample = next(s for s in manifest.samples if s.sample_id == reference_id)
    review = resolved_review(sample)
    if (
        sample.parent_id is not None
        or sample.transform is not None
        or review.get("label") != "normal"
        or review.get("pose") != "front"
        or not review.get("approve_anchor")
    ):
        raise ValueError("Intrinsic reference must be an approved normal frontal original")
    records = {r["sample_id"]: r for r in extracted["records"]}
    reference = points(records[reference_id])
    if reference is None:
        raise ValueError("Reference has no single-face landmarks")
    shape = contour_shape(reference)
    _, _, radius, _ = aligned_residuals(reference, reference, STABLE_POINTS)
    theta = 0.4
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    similarity = reference.copy()
    similarity[:, :2] = reference[:, :2] @ rotation * 1.7 + [200, -50]
    changed_eyes_mouth = reference.copy()
    changed_eyes_mouth[[11, 13, 17, 19, 23, 24, 26], :2] += [radius * 0.1, radius * 0.05]
    chin = reference.copy()
    axis = reference[4, :2] - reference[0, :2]
    width = float(np.linalg.norm(axis))
    chin[2, :2] += np.array([-axis[1], axis[0]]) / width * 0.05 * radius
    side = reference.copy()
    side[1, :2] += axis * 0.2
    side_delta = contour_shape(side) - shape
    checks = {
        "similarity_max_abs_error": float(np.max(np.abs(contour_shape(similarity) - shape))),
        "unrelated_points_max_abs_error": float(
            np.max(np.abs(contour_shape(changed_eyes_mouth) - shape))
        ),
        "chin_height_change": float((contour_shape(chin) - shape)[0]),
        "chin_height_expected": 0.05 * radius / width,
        "side_lateral_old_four_rms": float(np.sqrt(np.mean(side_delta[:4] ** 2))),
        "side_lateral_six_rms": float(np.sqrt(np.mean(side_delta**2))),
        "side_lateral_six_expected": 0.2 / np.sqrt(6),
    }
    if (
        checks["similarity_max_abs_error"] > 1e-12
        or checks["unrelated_points_max_abs_error"] != 0
        or abs(checks["chin_height_change"] - checks["chin_height_expected"]) > 1e-12
        or checks["side_lateral_old_four_rms"] > 1e-12
        or abs(checks["side_lateral_six_rms"] - checks["side_lateral_six_expected"]) > 1e-12
    ):
        raise ArithmeticError(f"Intrinsic controls failed: {checks}")
    rows = []
    for sample in manifest.samples:
        current = points(records[sample.sample_id])
        human = resolved_review(sample)
        result: dict[str, Any] = {"status": "unavailable", "shape_rms": None}
        if current is not None:
            try:
                delta = contour_shape(current) - shape
                result = {
                    "status": "valid",
                    "shape_rms": float(np.sqrt(np.mean(delta**2))),
                    "chin_height_error": abs(float(delta[0])),
                    "chin_lateral_error": abs(float(delta[1])),
                    "left_side_height_error": abs(float(delta[2])),
                    "right_side_height_error": abs(float(delta[3])),
                    "left_side_lateral_error": abs(float(delta[4])),
                    "right_side_lateral_error": abs(float(delta[5])),
                }
            except ValueError as exc:
                result["status"] = str(exc)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "transform": sample.transform,
                "label": human["label"],
                "pose": human.get("pose"),
                "parts": human.get("parts", []),
                "intrinsic": result,
            }
        )
    # An eye-spacing nuisance direction absorbs all amplitudes, including the
    # known severe 20% deformation. This is a negative control, never a detector.
    nuisance = np.zeros((28, 2))
    eye_axis = reference[[17, 19], :2].mean(0) - reference[[11, 13], :2].mean(0)
    eye_axis /= np.linalg.norm(eye_axis)
    nuisance[[11, 13]] = -eye_axis
    nuisance[[17, 19]] = eye_axis
    operator, _, _, _ = coupling_operator(
        reference, tuple(i for i in STABLE_POINTS if i not in REGIONS["eye_spacing"])
    )
    direction = (
        (operator @ nuisance.reshape(-1)).reshape(28, 2)[list(REGIONS["eye_spacing"])].reshape(-1)
    )
    direction /= np.linalg.norm(direction)
    projection = np.eye(8) - np.outer(direction, direction)
    nuisance_rows = []
    for sample in manifest.samples:
        if sample.transform is None or sample.transform["kind"] != "eye_spacing":
            continue
        current = points(records[sample.sample_id])
        if current is None:
            continue
        fit = tuple(i for i in STABLE_POINTS if i not in REGIONS["eye_spacing"])
        aligned, _, _, _ = aligned_residuals(current, reference, fit)
        residual = ((aligned - reference[:, :2]) / radius)[list(REGIONS["eye_spacing"])].reshape(-1)
        after = projection @ residual
        nuisance_rows.append(
            {
                "sample_id": sample.sample_id,
                "label": resolved_review(sample)["label"],
                "strength": sample.transform["strength"],
                "before_unweighted_rms": float(np.linalg.norm(residual) / 2),
                "after_unweighted_rms": float(np.linalg.norm(after) / 2),
                "fraction_removed_squared_norm": float(1 - (after @ after) / (residual @ residual)),
            }
        )
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "evidence_level": "numerical-evidence",
        "reference_id": reference_id,
        "data_hash": data_hash(manifest),
        "labels_hash": digest(manifest.model_dump(mode="json")),
        "extraction_hash": digest(extracted),
        "model_hash": digest(extracted["models"]),
        "implementation_sha256": sha256(Path(__file__)),
        "independent_sources": len({s.source_id for s in manifest.samples}),
        "protocol": {
            "shape_components": [
                "chin_height",
                "chin_lateral",
                "left_side_height",
                "right_side_height",
                "left_side_lateral",
                "right_side_lateral",
            ],
            "target": "Frontal contour internal ratios only; all five contour points required",
            "selection": "Development diagnostic; no threshold fitting or method adoption",
            "negative_control": "Unbounded eye-spacing projection; no normal range learned",
        },
        "checks": checks,
        "rows": rows,
        "eye_nuisance_negative_control": nuisance_rows,
        "limitations": [
            "Internal ratios are different observables from aligned RMS, not comparable thresholds",
            "Does not detect isotropic contour enlargement or guarantee 3D pose invariance",
            "Contour detection can still shift when other image regions change",
            "One labelled frontal source; no independent calibration or validation",
        ],
    }


def contour_jacobian(reference: NDArray[np.float64]) -> NDArray[np.float64]:
    """Derivative of six ratios w.r.t. ten contour coordinates, in pixel units."""
    contour_shape(reference)  # Same reliability/width contract as the observable.
    a, b = reference[[0, 4], :2]
    d = b - a
    squared_width = float(d @ d)
    rotated = np.array([-d[1], d[0]])
    jacobian = np.zeros((6, 10))
    for row, (index, vertical) in enumerate(
        [(2, True), (2, False), (1, True), (3, True), (1, False), (3, False)]
    ):
        v = reference[index, :2] - (a + b) / 2
        axis = rotated if vertical else d
        value = float(v @ axis / squared_width)
        gradient_v = axis / squared_width
        gradient_d = (
            np.array([v[1], -v[0]]) if vertical else v
        ) / squared_width - 2 * value * d / squared_width
        jacobian[row, 2 * index : 2 * index + 2] = gradient_v
        jacobian[row, :2] = -gradient_v / 2 - gradient_d
        jacobian[row, 8:10] = -gradient_v / 2 + gradient_d
    return jacobian


def shape_vectors(
    current: NDArray[np.float64], reference: NDArray[np.float64]
) -> tuple[dict[str, NDArray[np.float64]], dict[str, str]]:
    """Four research observables; raw WLS is separate from the production pose gate."""
    vectors: dict[str, NDArray[np.float64]] = {}
    errors = {}
    for method in ("wls", "complement", "endpoint", "procrustes"):
        try:
            if method == "endpoint":
                vector = (contour_shape(current) - contour_shape(reference)) / np.sqrt(6)
            elif method == "procrustes":
                contour_shape(current)
                contour_shape(reference)
                x, y = current[:5, :2], reference[:5, :2]
                xx, yy = x - x.mean(0), y - y.mean(0)
                u, singular, vt = np.linalg.svd(xx.T @ yy)
                correction = np.diag([1.0, float(np.linalg.det(u @ vt))])
                scale = float(np.sum(singular * np.diag(correction)) / np.sum(xx**2))
                if singular[1] <= 1e-6 * singular[0] or scale <= 1e-10:
                    raise ValueError("Degenerate proper contour Procrustes fit")
                width = float(np.linalg.norm(reference[4, :2] - reference[0, :2]))
                vector = ((xx @ (u @ correction @ vt) * scale - yy) / width / np.sqrt(5)).reshape(
                    -1
                )
            else:
                fit = (
                    STABLE_POINTS if method == "wls" else tuple(i for i in STABLE_POINTS if i >= 5)
                )
                aligned, _, radius, _ = aligned_residuals(current, reference, fit)
                weights = np.clip(np.minimum(current[:, 2], reference[:, 2]), 0, 1)
                weights[weights < POLICY["min_confidence"]] = 0
                target = list(STABLE_POINTS) if method == "wls" else list(range(5))
                w = weights[target]
                if method == "wls":
                    if np.count_nonzero(w) < 8:
                        raise ValueError("Fewer than eight reliable WLS structural points")
                    center = np.average(reference[target, :2], axis=0, weights=w)
                    radius = float(
                        np.sqrt(
                            np.average(
                                np.sum((reference[target, :2] - center) ** 2, axis=1), weights=w
                            )
                        )
                    )
                elif np.count_nonzero(w) < 3 or w[2] == 0:
                    raise ValueError("Insufficient contour targets or chin")
                vector = (
                    ((aligned - reference[:, :2])[target] / radius) * np.sqrt(w / w.sum())[:, None]
                ).reshape(-1)
            vectors[method] = vector
        except ValueError as exc:
            errors[method] = str(exc)
    return vectors, errors


def stability_experiment(
    manifest: StudyManifest, extracted: dict[str, Any], reference_id: str
) -> dict[str, Any]:
    """Frozen-confidence differential comparison, not detector noise calibration."""
    records = {r["sample_id"]: r for r in extracted["records"]}
    reference = points(records[reference_id])
    if reference is None:
        raise ValueError("Exploratory reference requires single-face measurements")
    width = float(np.linalg.norm(reference[4, :2] - reference[0, :2]))
    chart = contour_jacobian(reference) * width / np.sqrt(6)
    xy = (reference[:5, :2] - reference[:5, :2].mean(0)) / width
    tangent = np.empty((10, 4))
    tangent[0::2] = np.column_stack((np.ones(5), np.zeros(5), xy[:, 0], -xy[:, 1]))
    tangent[1::2] = np.column_stack((np.zeros(5), np.ones(5), xy[:, 1], xy[:, 0]))
    u, singular, _ = np.linalg.svd(tangent, full_matrices=True)
    if singular[-1] <= 1e-10:
        raise ValueError("Degenerate contour similarity tangent")
    shape_basis = u[:, 4:]
    projection = shape_basis @ shape_basis.T
    embedded_shape = np.zeros((56, 6))
    embedded_shape[:10] = shape_basis
    endpoint = np.zeros((6, 56))
    endpoint[:, :10] = chart
    procrustes = np.zeros((10, 56))
    procrustes[:, :10] = projection / np.sqrt(5)
    operators = {"endpoint": endpoint, "procrustes": procrustes}
    frozen_vectors, operator_errors = shape_vectors(reference, reference)
    for method in ("wls", "complement"):
        if method not in frozen_vectors:
            continue
        fit = STABLE_POINTS if method == "wls" else tuple(i for i in STABLE_POINTS if i >= 5)
        operator, radius, _, _ = coupling_operator(reference, fit)
        target = list(STABLE_POINTS) if method == "wls" else list(range(5))
        weights = np.clip(reference[target, 2], 0, 1)
        weights[weights < POLICY["min_confidence"]] = 0
        rows = [2 * i + d for i in target for d in (0, 1)]
        operators[method] = (
            operator[rows]
            * np.sqrt(np.repeat(weights / weights.sum(), 2))[:, None]
            * width
            / radius
        )
    eigenvalues, eigenvectors = np.linalg.eigh(chart @ chart.T)
    whitened_chart = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T @ chart
    checks = {
        "endpoint_rank": int(np.linalg.matrix_rank(chart)),
        "endpoint_similarity_null_error": float(np.max(np.abs(chart @ tangent))),
        "whitened_chart_gram_error": float(
            np.max(np.abs(whitened_chart.T @ whitened_chart - projection))
        ),
        "finite_difference_max_errors": {},
    }
    diagnostics = {}
    for method, operator in operators.items():
        max_error = 0.0
        for step in (1e-5, 1e-6):
            for column in range(56):
                plus, minus = reference.copy(), reference.copy()
                plus[column // 2, column % 2] += step * width
                minus[column // 2, column % 2] -= step * width
                actual = (
                    shape_vectors(plus, reference)[0][method]
                    - shape_vectors(minus, reference)[0][method]
                ) / (2 * step)
                max_error = max(max_error, float(np.max(np.abs(actual - operator[:, column]))))
        checks["finite_difference_max_errors"][method] = max_error
        target_singular = np.linalg.svd(operator @ embedded_shape, compute_uv=False)
        target_gain = float(target_singular[-1])
        global_gain = float(np.linalg.norm(operator, 2))
        endpoint_columns = [0, 1, 8, 9]
        internal_columns = [2, 3, 4, 5, 6, 7]
        endpoint_common = np.zeros((56, 2))
        endpoint_common[[0, 8], 0] = 1
        endpoint_common[[1, 9], 1] = 1
        noise_factors = {
            "iid_all_coordinates": np.eye(56),
            "iid_endpoints_only": np.eye(56)[:, endpoint_columns],
            "iid_internal_only": np.eye(56)[:, internal_columns],
            "endpoint_common_translation": endpoint_common,
        }
        diagnostics[method] = {
            "target_shape_min_gain": target_gain,
            "target_shape_max_gain": float(target_singular[0]),
            "all_coordinate_worst_noise_gain": global_gain,
            "one_pixel_worst_noise_gain": global_gain / width,
            "noise_to_min_shape_response": global_gain / target_gain
            if target_gain > 1e-12
            else None,
            "cross_region_worst_gain": float(np.linalg.norm(operator[:, 10:], 2)),
            "noise_models": {
                name: {
                    "expected_squared_score": float(np.sum((operator @ factor) ** 2)),
                    "worst_gain": float(np.linalg.norm(operator @ factor, 2)),
                }
                for name, factor in noise_factors.items()
            },
        }
    if (
        checks["endpoint_rank"] != 6
        or checks["endpoint_similarity_null_error"] > 1e-10
        or checks["whitened_chart_gram_error"] > 1e-10
        or max(checks["finite_difference_max_errors"].values()) > 1e-7
    ):
        raise ArithmeticError(f"Stability controls failed: {checks}")
    direction = (reference[4, :2] - reference[0, :2]) / width
    normal = np.array([-direction[1], direction[0]])
    controls = {}
    for name in ("chin_normal", "side_along", "contour_isotropic_size"):
        delta = np.zeros((28, 2))
        if name == "chin_normal":
            delta[2] = normal
        elif name == "side_along":
            delta[1] = direction
        else:
            delta[:5] = xy
        delta /= np.linalg.norm(delta)
        controls[name] = {
            method: float(np.linalg.norm(operator @ delta.reshape(-1)))
            for method, operator in operators.items()
        }
    rows = []
    for sample in manifest.samples:
        current = points(records[sample.sample_id])
        vectors, reasons = (
            shape_vectors(current, reference)
            if current is not None
            else ({}, {m: "No single face" for m in operators})
        )
        gate = face_status(records[sample.sample_id].get("faces", []), ObservationRequest())
        review = resolved_review(sample)
        scores = {method: float(np.linalg.norm(vector)) for method, vector in vectors.items()}
        if "wls" in scores and current is not None:
            if abs(scores["wls"] - geometry_distance(current, reference)[0]) > 1e-10:
                raise ArithmeticError("Raw WLS comparator differs from existing WLS")
        rows.append(
            {
                "sample_id": sample.sample_id,
                "label": review["label"],
                "human_review_status": review["review_status"],
                "transform": sample.transform,
                "reliable_landmark_count": int(np.count_nonzero(current[:, 2] >= 0.3))
                if current is not None
                else 0,
                "reliable_contour_count": int(np.count_nonzero(current[:5, 2] >= 0.3))
                if current is not None
                else 0,
                "production_pose_gate": gate,
                "raw_scores": scores,
                "unavailable_reasons": reasons,
                "trusted_anomaly_judgment": "unknown: no calibrated conditions or thresholds",
            }
        )
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "reference_id": reference_id,
        "data_hash": data_hash(manifest),
        "extraction_hash": digest(extracted),
        "implementation_sha256": sha256(Path(__file__)),
        "model_hash": digest(extracted["models"]),
        "reference_contour_width_pixels": width,
        "protocol": {
            "input_norm": "Euclidean coordinate perturbation divided by reference contour width",
            "methods": {
                "wls": "Existing structural WLS; raw numeric value, production gate separate",
                "complement": "Existing complementary contour RMS / reference stable radius",
                "endpoint": "Six endpoint ratios, Euclidean norm / sqrt(6)",
                "procrustes": "Equal-weight proper five-point Procrustes RMS / contour width",
            },
            "noise": "Ideal frozen-confidence covariance factors, not measured detector covariance",
            "target_class": "Unit directions orthogonal to the five-point similarity tangent",
            "not_comparable_raw_scales": True,
            "no_calibration_or_adoption": True,
        },
        "checks": checks,
        "operators": diagnostics,
        "coordinate_controls": controls,
        "rows": rows,
        "operator_unavailable_reasons": operator_errors,
        "scope": (
            "First-order frozen-confidence stability and unlabelled expression scores; "
            "not semantic anomaly separation or statistical coverage"
        ),
    }


def similarity_probe(image: Image.Image, kind: str) -> tuple[Image.Image, NDArray[np.float64]]:
    """Three fixed image-space probes and forward matrices for pixel-center indices."""
    fill = image.getpixel((0, 0))
    matrix = np.eye(3)
    if kind == "translate":
        matrix[:2, 2] = [3, -2]
        result = image.transform(
            image.size, Image.Transform.AFFINE, (1, 0, -3, 0, 1, 2), fillcolor=fill
        )
    elif kind == "rotate":
        angle = np.deg2rad(4)
        matrix[:2, :2] = [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        center = (np.array(image.size) - 1) / 2
        matrix[:2, 2] = center - matrix[:2, :2] @ center
        result = image.rotate(4, resample=Image.Resampling.BICUBIC, fillcolor=fill)
    elif kind == "scale":
        size = tuple(round(v * 0.95) for v in image.size)
        offset = (np.array(image.size) - size) // 2
        scale = np.array(size) / image.size
        matrix[:2, :2] = np.diag(scale)
        matrix[:2, 2] = offset + (scale - 1) / 2
        result = Image.new("RGB", image.size, fill)
        result.paste(image.resize(size, Image.Resampling.BICUBIC), tuple(offset))
    else:
        raise ValueError("Unknown similarity probe")
    return result, matrix


def repeatability_experiment(
    manifest: StudyManifest,
    extracted: dict[str, Any],
    directory: Path,
    models: Path,
    output: Path,
) -> dict[str, Any]:
    """Actual detector equivariance; no accuracy truth or empirical noise covariance."""
    observer = OfflineObserver(models, output / "cache")
    if digest(observer.model_manifest) != digest(extracted["models"]):
        raise ValueError("Repeatability must use the exact extraction models")
    records = {r["sample_id"]: r for r in extracted["records"]}
    rows = []
    started = time.monotonic()
    for sample in manifest.samples:
        reference = points(records[sample.sample_id])
        if reference is None:
            rows.append({"sample_id": sample.sample_id, "status": "No single-face baseline"})
            continue
        image_path = sample.image if sample.image.is_absolute() else directory / sample.image
        image = read_image(image_path.read_bytes())
        for kind in ("translate", "rotate", "scale"):
            probe, matrix = similarity_probe(image, kind)
            begin = time.monotonic()
            faces = observer.detect(probe, ObservationRequest())
            current = points({"faces": faces})
            row: dict[str, Any] = {
                "sample_id": sample.sample_id,
                "probe": kind,
                "forward_matrix": matrix.tolist(),
                "detector_seconds": time.monotonic() - begin,
                "faces": faces,
                "production_pose_gate": face_status(faces, ObservationRequest()),
                "baseline_production_gate": face_status(
                    records[sample.sample_id]["faces"], ObservationRequest()
                ),
                "status": "No single-face probe",
            }
            if current is not None:
                corrected = current.copy()
                corrected[:, :2] = (current[:, :2] - matrix[:2, 2]) @ np.linalg.inv(
                    matrix[:2, :2]
                ).T
                delta = corrected[:, :2] - reference[:, :2]
                vectors, errors = shape_vectors(corrected, reference)
                row.update(
                    status="measured",
                    error_pixels=np.linalg.norm(delta, axis=1).tolist(),
                    baseline_confidence=reference[:, 2].tolist(),
                    probe_confidence=current[:, 2].tolist(),
                    paired_reliable=((reference[:, 2] >= 0.3) & (current[:, 2] >= 0.3)).tolist(),
                    raw_scores={m: float(np.linalg.norm(v)) for m, v in vectors.items()},
                    unavailable_reasons=errors,
                )
            rows.append(row)
            print(sample.sample_id, kind, len(faces), flush=True)
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "data_hash": data_hash(manifest),
        "extraction_hash": digest(extracted),
        "model_hash": digest(observer.model_manifest),
        "implementation_sha256": sha256(Path(__file__)),
        "probe_count": len(rows),
        "rows": rows,
        "seconds": time.monotonic() - started,
        "environment": {"python": platform.python_version(), "device": "cpu", "threads": 2},
        "protocol": (
            "Same sample under integer translation (+3,-2), +4 degree rotation, "
            "0.95 rounded resize; fixed pixel-center inverse map; confidence unchanged"
        ),
        "scope": (
            "Equivariance error mixes detector, baseline, resampling and coordinate convention; "
            "not landmark truth, stochastic covariance, human labels or independent validation"
        ),
    }


def statistical_budget() -> dict[str, Any]:
    """Exact-count planning and a rank-calibration conditional-risk diagnostic."""
    from anime_avatar_runtime.local_geometry import clopper_pearson_upper

    validation = []
    for errors in range(6):
        lo, hi = errors + 1, 5000
        while lo < hi:
            n = (lo + hi) // 2
            upper = clopper_pearson_upper(errors, n)
            assert upper is not None
            if upper <= 0.05:
                hi = n
            else:
                lo = n + 1
        validation.append(
            {
                "false_positives": errors,
                "minimum_valid_normals": lo,
                "upper95": clopper_pearson_upper(errors, lo),
            }
        )
    target = validation[0]["minimum_valid_normals"]
    coverage_n = next(
        n
        for n in range(target, 500)
        if 1 - sum(math.comb(n, j) * 0.8**j * 0.2 ** (n - j) for j in range(target)) >= 0.95
    )
    splits = []
    for minimum_valid in (target, coverage_n):
        for n in range(1, 5000):
            dev = min(n, max(1, round(n * 0.6)))
            cal = min(n - dev, round(n * 0.2))
            valid = n - dev - cal
            if dev >= 21 and cal >= 39 and valid >= minimum_valid:
                splits.append(
                    {
                        "required_validation_sources": minimum_valid,
                        "total_sources": n,
                        "development": dev,
                        "calibration": cal,
                        "validation": valid,
                    }
                )
                break
    n = 39
    rank = math.ceil((n + 1) * 0.95)
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "zero_false_positive_validation": validation,
        "validity_sampling": {
            "assumed_probability": 0.8,
            "independent_trials": True,
            "required_valid": target,
            "probability_target": 0.95,
            "minimum_sampled_normals": coverage_n,
        },
        "current_60_20_20_source_split": splits,
        "uniform_rank_calibration_control": {
            "n": n,
            "rank": rank,
            "marginal_false_positive_probability": (n + 1 - rank) / (n + 1),
            "probability_frozen_true_fpr_exceeds_5_percent": sum(
                math.comb(n, j) * 0.95**j * 0.05 ** (n - j) for j in range(rank, n + 1)
            ),
        },
        "scope": (
            "Binomial/uniform planning assumptions, not new normal data or measured FPR. "
            "Rank calibration controls marginal risk; independent validation is required"
        ),
    }


def causal_replay(
    timeline: list[dict[str, Any]], enter: float, exit: float, dwell: float, max_gap: float
) -> list[dict[str, Any]]:
    """Causal shadow recommendations; missing data breaks a persistence interval."""
    if not all(math.isfinite(v) for v in (enter, exit, dwell, max_gap)) or not (
        enter > exit >= 0 and dwell > 0 and max_gap > 0
    ):
        raise ValueError("Invalid replay policy")
    state = "NORMAL"
    start = previous = None
    result = []
    for row in timeline:
        t, value = float(row["time"]), row["value"]
        if not math.isfinite(t) or t < 0 or (previous is not None and t <= previous):
            raise ValueError("Replay times must be finite and strictly increasing")
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("Invalid replay observation")
        old_state = state
        gap = previous is not None and t - previous > max_gap
        if gap:
            state, start = "UNKNOWN", None
        if value is None or row.get("mode") != "idle":
            state, start = "UNKNOWN", None
        elif value <= exit:
            state, start = "NORMAL", None
        elif value > enter and start is None:
            state, start = "WATCH", t
        elif start is None:
            state = "UNKNOWN" if state == "UNKNOWN" else "NORMAL"
        if start is not None and t - start >= dwell - 1e-12:
            state = "RECOVER_RECOMMENDED"
        result.append(
            row
            | {
                "state": state,
                "gap_reset": gap,
                "watch_seconds": t - start if start is not None else 0.0,
                "new_recommendation": state == "RECOVER_RECOMMENDED" and old_state != state,
            }
        )
        previous = t
    return result


def frontal_flow_experiment(
    manifest: StudyManifest,
    directory: Path,
    reference_id: str,
    repo: Path,
    weights: Path,
    provenance: Path,
    output: Path,
) -> dict[str, Any]:
    """Real CPU GMFlow on frontal still-image pairs; no temporal or anomaly claims."""
    import torch

    reference = next(s for s in manifest.samples if s.sample_id == reference_id)
    review = resolved_review(reference)
    if (
        reference.parent_id is not None
        or review.get("label") != "normal"
        or not review.get("approve_anchor")
        or review.get("pose") != "front"
    ):
        raise ValueError("Flow reference must be an approved normal frontal original")
    declared = json.loads(provenance.read_bytes())
    source_files = {
        r["name"]: r["sha256"] for r in declared if r["target"].startswith("vendor/gmflow/")
    }
    for name, expected in source_files.items():
        if sha256(repo / name) != expected:
            raise ValueError("GMFlow source changed: " + name)
    expected_weights = "e9887eda6165e0813c03f35998d6af86e265aafce2e3349af93d691a59696088"
    if not source_files or sha256(weights) != expected_weights:
        raise ValueError("GMFlow source provenance or fixed weights unavailable")
    sys.path.insert(0, str(repo.resolve()))
    from gmflow.geometry import flow_warp, forward_backward_consistency_check
    from gmflow.gmflow import GMFlow

    started = time.monotonic()
    torch.set_num_threads(2)
    model = GMFlow().cpu().eval()
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    load_seconds = time.monotonic() - started
    reference_path = (
        reference.image if reference.image.is_absolute() else directory / reference.image
    )
    anchor = read_image(reference_path.read_bytes()).resize((256, 256), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.asarray(anchor).copy()).permute(2, 0, 1).unsqueeze(0).float()
    rows = []
    for sample in manifest.samples:
        human = resolved_review(sample)
        if human.get("pose") != "front":
            raise ValueError("Only explicitly reviewed frontal inputs are allowed")
        path = sample.image if sample.image.is_absolute() else directory / sample.image
        image = read_image(path.read_bytes()).resize((256, 256), Image.Resampling.BILINEAR)
        other = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float()
        begin = time.monotonic()
        with torch.inference_mode():
            flow = model(
                tensor,
                other,
                attn_splits_list=[2],
                corr_radius_list=[-1],
                prop_radius_list=[-1],
                pred_bidir_flow=True,
            )["flow_preds"][-1]
            if flow.shape != (2, 2, 256, 256) or not torch.isfinite(flow).all():
                raise ValueError("Invalid bidirectional GMFlow output")
            fwd, bwd = flow[:1], flow[1:]
            warped, inside = flow_warp(bwd, fwd, mask=True)
            occluded, _ = forward_backward_consistency_check(fwd, bwd)
            consistent = inside & (occluded == 0)
            magnitude = torch.linalg.vector_norm(fwd, dim=1)
            discrepancy = torch.linalg.vector_norm(fwd + warped, dim=1)
            median = float(magnitude[consistent].median()) if torch.any(consistent) else None
            fbc = float(discrepancy[inside].mean()) if torch.any(inside) else None
        np.savez_compressed(output / f"{sample.sample_id}-flow.npz", flow=flow.numpy())
        rows.append(
            {
                "sample_id": sample.sample_id,
                "input_sha256": sample.input_sha256,
                "label": human["label"],
                "parts": human.get("parts", []),
                "transform": sample.transform,
                "flow_seconds": time.monotonic() - begin,
                "in_bounds_fraction": float(inside.float().mean()),
                "consistent_fraction_in_bounds": float(consistent.sum() / inside.sum())
                if torch.any(inside)
                else None,
                "median_pixels_on_consistent": median,
                "median_diagonal_on_consistent": median / math.hypot(256, 256)
                if median is not None
                else None,
                "mean_bidirectional_error_in_bounds": fbc,
            }
        )
        print("frontal flow", sample.sample_id, flush=True)
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "real_model_inference": True,
        "reference_id": reference_id,
        "data_hash": data_hash(manifest),
        "labels_hash": digest(manifest.model_dump(mode="json")),
        "implementation_sha256": sha256(Path(__file__)),
        "model": {
            "code_commit": "b5123431164d01ec14526a1c3d22218aecb62024",
            "source_files_sha256": source_files,
            "weights_sha256": expected_weights,
            "weights_url": "https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale1-things-e9887eda.pth",
            "source_license": "Apache-2.0",
            "weight_license": "Independent weight license not verified; not redistributed",
        },
        "environment": {
            "device": "cpu",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "threads": 2,
            "inference_size": [256, 256],
        },
        "load_seconds": load_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "rows": rows,
        "scope": (
            "Frontal still pairs, not video. Consistency is a reliability proxy, "
            "not geometry validity; no flow truth, calibrated anomaly, GPU or recovery test"
        ),
    }


def regional_flow_experiment(
    manifest: StudyManifest,
    extracted: dict[str, Any],
    directory: Path,
    reference_id: str,
    flow_report: Path,
) -> dict[str, Any]:
    """Read frozen real flows and score fixed reference regions, without new inference."""
    report = json.loads(flow_report.read_bytes())
    if report["data_hash"] != data_hash(manifest) or report["reference_id"] != reference_id:
        raise ValueError("Flow report belongs to different data/reference")
    reference = next(s for s in manifest.samples if s.sample_id == reference_id)
    p = points(next(r for r in extracted["records"] if r["sample_id"] == reference_id))
    if p is None:
        raise ValueError("No reference landmarks")
    path = reference.image if reference.image.is_absolute() else directory / reference.image
    with Image.open(path) as im:
        reference_size = np.array(im.size)
    xy = p[:, :2] / reference_size * 256
    radius = float(
        np.sqrt(
            np.mean(
                np.sum((xy[list(STABLE_POINTS)] - xy[list(STABLE_POINTS)].mean(0)) ** 2, axis=1)
            )
        )
    )
    padding = radius * 0.15
    yy, xx = np.mgrid[:256, :256].astype(np.float32)
    masks = {}
    for region, indices in REGIONS.items():
        lower, upper = xy[list(indices)].min(0) - padding, xy[list(indices)].max(0) + padding
        masks[region] = (xx >= lower[0]) & (xx <= upper[0]) & (yy >= lower[1]) & (yy <= upper[1])
    rows = []
    by_id = {r["sample_id"]: r for r in report["rows"]}
    if set(by_id) != {s.sample_id for s in manifest.samples}:
        raise ValueError("Flow sample IDs do not match manifest")
    agreement_error = 0.0
    for sample in manifest.samples:
        if by_id[sample.sample_id]["input_sha256"] != sample.input_sha256:
            raise ValueError("Flow sample input hash mismatch")
        location = flow_report.parent / f"{sample.sample_id}-flow.npz"
        with np.load(location, allow_pickle=False) as data:
            flow = data["flow"]
        if flow.shape != (2, 2, 256, 256) or not np.isfinite(flow).all():
            raise ValueError("Invalid frozen flow array")
        fwd, bwd = flow
        x, y = xx + fwd[0], yy + fwd[1]
        inside = (x >= 0) & (x <= 255) & (y >= 0) & (y <= 255)
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = x - x0, y - y0
        warped = np.zeros_like(bwd)
        for dx, dy, weight in (
            (0, 0, (1 - fx) * (1 - fy)),
            (1, 0, fx * (1 - fy)),
            (0, 1, (1 - fx) * fy),
            (1, 1, fx * fy),
        ):
            warped += bwd[:, np.clip(y0 + dy, 0, 255), np.clip(x0 + dx, 0, 255)] * weight
        magnitude = np.linalg.norm(fwd, axis=0)
        error = np.linalg.norm(fwd + warped, axis=0)
        threshold = 0.01 * (magnitude + np.linalg.norm(bwd, axis=0)) + 0.5
        consistent = inside & (error <= threshold)
        fraction = float(consistent.sum() / inside.sum()) if inside.any() else None
        old = by_id[sample.sample_id]["consistent_fraction_in_bounds"]
        if fraction is not None and old is not None:
            agreement_error = max(agreement_error, abs(fraction - old))
        regions = {}
        for region, mask in masks.items():
            valid = mask & consistent
            eligible = mask & inside
            regions[region] = {
                "mask_pixels": int(mask.sum()),
                "consistent_pixels": int(valid.sum()),
                "consistent_fraction_in_bounds": float(valid.sum() / eligible.sum())
                if eligible.any()
                else None,
                "median_pixels": float(np.median(magnitude[valid])) if valid.any() else None,
                "p95_pixels": float(np.quantile(magnitude[valid], 0.95)) if valid.any() else None,
            }
        rows.append(
            {
                "sample_id": sample.sample_id,
                "label": resolved_review(sample)["label"],
                "flow_sha256": sha256(location),
                "regions": regions,
            }
        )
    if agreement_error > 1e-4:
        raise ArithmeticError("Saved-flow consistency disagrees with original Torch computation")
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "data_hash": data_hash(manifest),
        "flow_report_sha256": sha256(flow_report),
        "reference_id": reference_id,
        "implementation_sha256": sha256(Path(__file__)),
        "global_consistency_max_difference": agreement_error,
        "protocol": {
            "region_masks": "Reference boxes padded by 0.15 RMS; no score-based tuning",
            "sampling": "256x256; fixed regions; pinned upstream consistency rule 0.01/0.5",
        },
        "rows": rows,
        "scope": (
            "Real frontal still pairs; regional flow is an observation, not a decision; "
            "no video, flow truth or calibrated recovery"
        ),
    }


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
    image: Image.Image,
    points: NDArray[np.float64],
    kind: str,
    strength: float,
    *,
    compact: bool = False,
) -> Image.Image:
    """Inverse displacements; compact supports preserve all pixels outside each ellipse."""
    import cv2

    left, right = points[[11, 13], :2].mean(0), points[[17, 19], :2].mean(0)
    span = np.linalg.norm(right - left)
    if not np.isfinite(span) or span <= 1e-5:
        raise ValueError("Invalid interocular span")
    yy, xx = np.mgrid[: image.height, : image.width].astype(np.float32)
    map_x, map_y = xx.copy(), yy.copy()
    if kind == "eye_spacing":
        controls = [(left, (-strength * span, 0)), (right, (strength * span, 0))]
        sigma = span * 0.24
        radii = (span * 0.55, span * 0.16)
    elif kind == "nose_mouth":
        controls = [(points[23:28, :2].mean(0), (strength * span, 0))]
        sigma = span * 0.3
        radii = (span * 0.5, span * 0.42)
    elif kind == "chin":
        controls = [(points[2, :2], (0, strength * span))]
        sigma = span * 0.35
        radii = (span * 0.65, span * 0.35)
    else:
        raise ValueError(f"Unknown local warp: {kind}")
    for (cx, cy), (dx, dy) in controls:
        if compact:
            radius_squared = ((xx - cx) / radii[0]) ** 2 + ((yy - cy) / radii[1]) ** 2
            influence = np.clip(1 - radius_squared, 0, 1) ** 2
        else:
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
