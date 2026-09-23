"""Read-only source audit for the 12 images lacking canonical labeling-set TXT."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from create_label_review import measures, parse_xml, parse_yolo, render


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
ERRORS = ROOT / "outputs" / "clean_512_candidate" / "validation_errors.csv"
METADATA = ROOT / "outputs" / "unique_512_review" / "metadata.csv"
MANIFEST = ROOT / "outputs" / "selected_512_original_bmp" / "manifest.csv"
LINKS = ROOT / "outputs" / "unique_512_audit" / "label_links.csv"
PRIOR = ROOT / "outputs" / "eda_image_metrics.csv"
OUT = ROOT / "outputs" / "missing_12_label_audit"
FOLDERS = (
    DATASET / "라벨링 6종 세트" / "labels",
    DATASET / "OpenLabeling-master" / "main" / "output" / "YOLO_darknet",
    DATASET / "OpenLabeling-master" / "main" / "output" / "PASCAL_VOC",
    DATASET / "test1" / "yolov3" / "labels",
)


def hash_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def save(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def box_equivalent(a: list[dict], b: list[dict], tolerance: float = 1.25) -> bool:
    if len(a) != len(b):
        return False
    unmatched = list(b)
    for box in a:
        found = next((i for i, other in enumerate(unmatched)
                      if box["cls"] == other["cls"] and
                      max(abs(x - y) for x, y in zip(box["coords"], other["coords"])) <= tolerance), None)
        if found is None:
            return False
        unmatched.pop(found)
    return True


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT}")
    errors = [row for row in load(ERRORS) if row["code"] == "MISSING_CANONICAL_LABEL"]
    if len(errors) != 12 or len({row["unique_id"] for row in errors}) != 12:
        raise ValueError(f"Expected 12 distinct missing canonical labels, found {len(errors)}")
    metadata = {row["unique_id"]: row for row in load(METADATA)}
    manifest = {row["unique_id"]: row for row in load(MANIFEST)}
    link_rows = load(LINKS)
    prior = {row["asset_id"]: row for row in load(PRIOR)}
    protected = [ERRORS, METADATA, MANIFEST, LINKS, PRIOR]
    baseline = {str(path): hash_file(path) for path in protected}
    targets = []
    for entry in errors:
        uid = entry["unique_id"]
        meta = metadata[uid]
        image = manifest[uid]
        if image["representative_path"] != meta["representative_path"]:
            raise ValueError(f"Representative path mismatch: {uid}")
        representative = Path(meta["representative_path"])
        selected = Path(image["copied_path"])
        if not representative.is_file() or not selected.is_file():
            raise FileNotFoundError(uid)
        if hash_file(representative) != image["sha256"] or hash_file(selected) != image["sha256"]:
            raise ValueError(f"Image SHA mismatch: {uid}")
        stem = representative.stem
        discovered = {path.resolve() for folder in FOLDERS
                      for suffix in (".txt", ".xml")
                      if (path := folder / f"{stem}{suffix}").is_file()}
        recorded = {Path(path).resolve() for path in json.loads(meta["yolo_label_path"]) + json.loads(meta["xml_label_path"])}
        recorded.update(Path(row["label_path"]).resolve() for row in link_rows
                        if row["unique_id"] == uid and row["label_path"] and row["label_format"] != "IMAGE_COPY")
        if not recorded <= discovered:
            raise ValueError(f"Audit has label path outside discovered labels: {uid}")
        paths = sorted(discovered | recorded)
        for path in paths:
            baseline[str(path)] = hash_file(path)
        baseline[str(representative)] = hash_file(representative)
        baseline[str(selected)] = hash_file(selected)
        targets.append((uid, meta, image, representative, paths))

    (OUT / "visualization").mkdir(parents=True, exist_ok=False)
    summary_rows = []
    detail_rows = []
    type_counts = Counter()
    same_yolo_bytes = 0
    equivalent_xml_yolo = 0
    for uid, meta, manifest_row, representative, paths in targets:
        with Image.open(representative) as opened:
            base = opened.convert("RGB")
        canonical = FOLDERS[0] / f"{representative.stem}.txt"
        yolo_paths = [path for path in paths if path.suffix.lower() == ".txt"]
        xml_paths = [path for path in paths if path.suffix.lower() == ".xml"]
        label_boxes = {}
        for path in paths:
            boxes = parse_xml(path) if path.suffix.lower() == ".xml" else parse_yolo(path, base.width, base.height)
            label_boxes[path] = boxes
            role = "canonical" if path.parent == FOLDERS[0] else (
                "openlabeling_yolo" if path.parent == FOLDERS[1] else
                "openlabeling_xml" if path.parent == FOLDERS[2] else "test1_yolo")
            detail_rows.append(dict(unique_id=uid, bmp_filename=representative.name,
                                    label_path=str(path), source_folder=str(path.parent), role=role,
                                    format=path.suffix.lower(), sha256=baseline[str(path)],
                                    bbox_count=len(boxes), classes=json.dumps(sorted({box["cls"] for box in boxes})),
                                    boxes_px=json.dumps(boxes, ensure_ascii=False)))
            render(base, boxes if path.suffix.lower() == ".xml" else [],
                   boxes if path.suffix.lower() == ".txt" else [],
                   OUT / "visualization" / f"{uid}_{role}.png",
                   [f"{uid} | {representative.name}", f"{role} | bbox={len(boxes)} | classes={','.join(sorted({box['cls'] for box in boxes})) or 'none'}"])
        base.save(OUT / "visualization" / f"{uid}_original.png")
        if xml_paths and yolo_paths:
            category = "C"
        elif yolo_paths:
            category = "A"
        elif xml_paths:
            category = "B"
        else:
            category = "D"
        type_counts[category] += 1
        yolo_hashes = {baseline[str(path)] for path in yolo_paths}
        yolo_identical = len(yolo_hashes) <= 1 if yolo_paths else None
        same_yolo_bytes += yolo_identical is True and len(yolo_paths) > 1
        first_yolo = yolo_paths[0] if yolo_paths else None
        xml_vs_yolo = [box_equivalent(label_boxes[path], label_boxes[first_yolo])
                       for path in xml_paths] if first_yolo else []
        all_equivalent = bool(xml_vs_yolo) and all(xml_vs_yolo) and yolo_identical
        equivalent_xml_yolo += all_equivalent
        pair_metrics = measures(label_boxes[xml_paths[0]], label_boxes[first_yolo])[0] if xml_paths and first_yolo else {}
        linked = prior.get(meta["historical_asset_id"])
        if linked is None:
            raise ValueError(f"Historical EDA reference missing: {uid}")
        audit_reason = ("기존 EDA는 VOC XML을 대표 라벨로 기록했고, review audit도 유효한 XML을 우선 선택함"
                        if linked["label_format"] == "VOC" and str(meta["label_source"]).lower().endswith(".xml")
                        else "기존 EDA와 review audit의 대표 라벨 경로를 확인할 것")
        summary_rows.append(dict(unique_id=uid, bmp_filename=representative.name,
                                 representative_path=str(representative), canonical_txt_exists=canonical.is_file(),
                                 canonical_txt_path=str(canonical),
                                 metadata_yolo_label_path=meta["yolo_label_path"],
                                 metadata_xml_label_path=meta["xml_label_path"],
                                 discovered_yolo_paths=json.dumps([str(path) for path in yolo_paths], ensure_ascii=False),
                                 discovered_xml_paths=json.dumps([str(path) for path in xml_paths], ensure_ascii=False),
                                 label_bbox_counts=json.dumps({str(path): len(label_boxes[path]) for path in paths}, ensure_ascii=False),
                                 label_classes=json.dumps({str(path): sorted({box["cls"] for box in label_boxes[path]})
                                                           for path in paths}, ensure_ascii=False),
                                 label_sha256=json.dumps({str(path): baseline[str(path)] for path in paths}, ensure_ascii=False),
                                 yolo_txt_byte_identical=yolo_identical,
                                 xml_yolo_bbox_equivalent_1_25px=all_equivalent,
                                 xml_yolo_mean_iou=pair_metrics.get("mean_iou", ""),
                                 prior_eda_label_path=str(DATASET / linked["label_path"]),
                                 prior_eda_label_format=linked["label_format"],
                                 metadata_label_source=meta["label_source"],
                                 audit_inclusion_reason=audit_reason,
                                 category=category, selected_bmp_path=manifest_row["copied_path"]))
        if first_yolo and xml_paths:
            render(base, label_boxes[xml_paths[0]], label_boxes[first_yolo],
                   OUT / "visualization" / f"{uid}_compare.png",
                   [f"{uid} | XML vs OpenLabeling YOLO", "RED solid = XML | BLUE dashed = YOLO"])

    save(OUT / "missing_12_label_sources.csv", summary_rows, list(summary_rows[0]))
    save(OUT / "missing_12_label_files.csv", detail_rows, list(detail_rows[0]))
    expected_png = sum(1 + len(paths) + (bool([p for p in paths if p.suffix.lower() == ".xml"]) and
                                           bool([p for p in paths if p.suffix.lower() == ".txt"]))
                       for _, _, _, _, paths in targets)
    actual_png = len(list((OUT / "visualization").glob("*.png")))
    if actual_png != expected_png:
        raise AssertionError(f"Visualization count {actual_png} != {expected_png}")
    if any(hash_file(Path(path)) != value for path, value in baseline.items()):
        raise AssertionError("An input file changed during the audit")
    report = dict(images=len(targets), categories={key: type_counts[key] for key in "ABCD"},
                  canonical_txt_missing=sum(not row["canonical_txt_exists"] for row in summary_rows),
                  yolo_txt_files=sum(row["format"] == ".txt" for row in detail_rows),
                  xml_files=sum(row["format"] == ".xml" for row in detail_rows),
                  yolo_pair_byte_identical_images=same_yolo_bytes,
                  xml_and_yolo_bbox_equivalent_1_25px_images=equivalent_xml_yolo,
                  all_prior_eda_representative_xml=all(row["prior_eda_label_format"] == "VOC" for row in summary_rows),
                  visualization_files=actual_png, source_files_unchanged=len(baseline), final_gt_selected=False)
    (OUT / "missing_12_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# canonical TXT 누락 12장 출처 감사\n\n"
        "`missing_12_label_sources.csv`는 이미지별 결과, `missing_12_label_files.csv`는 라벨 파일별 "
        "경로·SHA-256·bbox·class를 기록합니다. `visualization`에는 원본과 XML, OpenLabeling YOLO, "
        "test1 YOLO를 각각 표시하고 XML/YOLO 겹침 이미지를 추가했습니다. "
        "빨간 실선은 XML, 파란 점선은 YOLO입니다.\n\n"
        "분류는 서로 배타적입니다: A=YOLO만, B=XML만, C=YOLO와 XML 모두, D=둘 다 없음. "
        "기존 512장 집계는 canonical TXT 512개를 뜻하지 않습니다. 이전 EDA의 대표 라벨과 "
        "`audit_unique_512.py`의 XML 우선 로직을 사용했기 때문에, canonical TXT가 없는 이미지도 "
        "유효한 VOC XML로 라벨 연결된 그룹에 포함됐습니다. 다른 TXT를 최종 GT로 선택하지 않았습니다.\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
