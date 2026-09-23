"""Recheck the historical labeled 512-image cohort without touching source data."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
AUDIT = ROOT / "outputs" / "unique_512_audit"
REVIEW = ROOT / "outputs" / "unique_512_review"
REFERENCE = ROOT / "outputs" / "eda_image_metrics.csv"
EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png"}
ORIGINAL = "test1/yolov3/X선이물검출기(06.23_09.22)/"
SOURCES = (
    "test1/yolov3/X선이물검출기(06.23_09.22)",
    "test1/yolov3/images",
    "라벨링 6종 세트",
    "OpenLabeling-master/main/input",
)
LABEL_SOURCES = (
    "라벨링 6종 세트/labels",
    "OpenLabeling-master/main/output/YOLO_darknet",
    "OpenLabeling-master/main/output/PASCAL_VOC",
    "test1/yolov3/labels",
)


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phash(gray: np.ndarray) -> str:
    reduced = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coeff = cv2.dct(reduced)[:8, :8]
    median = np.median(coeff[1:, :])
    bits = (coeff > median).ravel()
    return f"{int(''.join('1' if bit else '0' for bit in bits), 2):016x}"


def source_priority(rel: str) -> tuple[int, str]:
    if rel.startswith(ORIGINAL) and rel.lower().endswith(".bmp"):
        rank = 0
    elif rel.startswith("OpenLabeling-master/main/input/") and rel.lower().endswith(".bmp"):
        rank = 1
    elif rel.startswith("라벨링 6종 세트/") and rel.lower().endswith(".jpg"):
        rank = 2
    elif rel.startswith("test1/yolov3/images/") and rel.lower().endswith(".jpg"):
        rank = 3
    else:
        rank = 4
    return rank, rel


def context(rel: str) -> tuple[str, str, str]:
    machine = re.search(r"([123])호기", rel)
    serial = re.search(r"(SN\d+)", rel, re.IGNORECASE)
    date = re.search(r"(?:^|[_/])(20\d{6})(?:[_/]|$)", Path(rel).stem)
    if not date:
        date = re.search(r"(?:^|[_/])(20\d{6})", Path(rel).stem)
    if not date:
        date = re.search(r"(?:^|[_/])(20\d{6})", rel)
    day = date.group(1) if date else ""
    return (machine.group(1) if machine else "", serial.group(1) if serial else "",
            f"{day[:4]}-{day[4:6]}-{day[6:]}" if day else "")


def inventory() -> list[dict]:
    paths = []
    for folder in SOURCES:
        base = DATASET / folder
        paths += [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS]
    rows = []
    for index, path in enumerate(sorted(set(paths))):
        rel = path.relative_to(DATASET).as_posix()
        with Image.open(path) as image:
            actual_format = image.format
            rgb = np.asarray(image.convert("RGB"))
            width, height = image.size
        pixel_digest = hashlib.sha256(f"{width}x{height}:".encode() + rgb.tobytes()).hexdigest()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        machine, serial, capture_date = context(rel)
        rows.append(dict(full_path=str(path), relative_path=rel, filename=path.name,
                         stem=path.stem, extension=path.suffix.lower(), actual_format=actual_format,
                         width=width,
                         height=height, file_size=path.stat().st_size, sha256=sha256(path),
                         pixel_hash=pixel_digest, perceptual_hash=phash(gray),
                         machine=machine, serial=serial, capture_date=capture_date,
                         source_folder=path.parent.relative_to(DATASET).as_posix()))
        if (index + 1) % 500 == 0:
            print(f"inventory: {index + 1}/{len(paths)}", flush=True)
    return rows


def small_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return cv2.resize(np.asarray(image.convert("L")), (144, 111), interpolation=cv2.INTER_AREA)


def compare(a: dict, b: dict) -> tuple[float, float]:
    aa = small_gray(Path(a["full_path"])).astype(np.float32)
    bb = small_gray(Path(b["full_path"])).astype(np.float32)
    mae = float(np.mean(np.abs(aa - bb)))
    corr = float(np.corrcoef(aa.ravel(), bb.ravel())[0, 1])
    return mae, corr


def parse_label(path: Path, width: int, height: int) -> tuple[list[tuple], str]:
    boxes = []
    try:
        if path.suffix.lower() == ".xml":
            root = ET.parse(path).getroot()
            declared = root.find("size")
            if declared is not None:
                dw = int(declared.findtext("width", "0"))
                dh = int(declared.findtext("height", "0"))
                if (dw, dh) != (width, height):
                    return [], f"XML_SIZE_MISMATCH:{dw}x{dh}"
            for obj in root.findall("object"):
                name = obj.findtext("name", "").strip().lower()
                if name != "defect":
                    return [], f"UNKNOWN_CLASS:{name}"
                box = obj.find("bndbox")
                values = [float(box.findtext(k, "nan")) for k in ("xmin", "ymin", "xmax", "ymax")]
                boxes.append((0, *values))
        else:
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    return [], "YOLO_COLUMN_COUNT"
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:])
                if cls != 0 or not all(0 <= v <= 1 for v in (x, y, w, h)) or w <= 0 or h <= 0:
                    return [], "YOLO_INVALID_VALUE_OR_CLASS"
                boxes.append((cls, (x - w / 2) * width, (y - h / 2) * height,
                              (x + w / 2) * width, (y + h / 2) * height))
        if any(not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height)
               for _, x1, y1, x2, y2 in boxes):
            return [], "BBOX_OUT_OF_RANGE"
    except (OSError, ValueError, TypeError, ET.ParseError, AttributeError) as exc:
        return [], f"PARSE_ERROR:{type(exc).__name__}"
    return sorted(boxes), ""


def equivalent(a: list[tuple], b: list[tuple], tolerance: float = 1.25) -> bool:
    if len(a) != len(b):
        return False
    remaining = list(b)
    for box in a:
        found = next((i for i, other in enumerate(remaining)
                      if box[0] == other[0] and
                      max(abs(x - y) for x, y in zip(box[1:], other[1:])) <= tolerance), None)
        if found is None:
            return False
        remaining.pop(found)
    return True


def to_yolo(boxes: list[tuple], width: int, height: int) -> str:
    return "".join(f"{cls} {(x1+x2)/(2*width):.12f} {(y1+y2)/(2*height):.12f} "
                   f"{(x2-x1)/width:.12f} {(y2-y1)/height:.12f}\n"
                   for cls, x1, y1, x2, y2 in boxes)


def visualize(image_path: Path, target: Path, uid: str, boxes: list[tuple], note: str) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for _, x1, y1, x2, y2 in boxes:
        draw.rectangle((x1, y1, x2, y2), outline=(255, 40, 20), width=2)
        draw.text((x1, max(0, y1 - 12)), "defect", fill=(255, 40, 20), font=font)
    draw.rectangle((0, 0, min(image.width, 260), 23), fill=(0, 0, 0))
    draw.text((5, 5), f"{uid}  {note}", fill=(255, 255, 255), font=font)
    image.save(target, quality=92)


def contact_sheet(items: list[dict], folder: Path, kind: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(items), 30):
        sheet = Image.new("RGB", (1000, 6 * 150), "white")
        draw = ImageDraw.Draw(sheet)
        for local, row in enumerate(items[start:start + 30]):
            path = Path(row["review_image_path"] if kind == "original" else row["visualization_path"])
            with Image.open(path) as source:
                thumb = source.convert("RGB")
                thumb.thumbnail((190, 125))
            x, y = (local % 5) * 200, (local // 5) * 150
            sheet.paste(thumb, (x + (200 - thumb.width) // 2, y))
            draw.text((x + 5, y + 127), row["unique_id"], fill="black")
        sheet.save(folder / f"{kind}_sheet_{start // 30 + 1:03d}.jpg", quality=88)


def main() -> None:
    if AUDIT.exists() or REVIEW.exists():
        raise FileExistsError("Audit/review output already exists; refusing to overwrite it")
    for path in (AUDIT, REVIEW / "images", REVIEW / "labels", REVIEW / "visualization"):
        path.mkdir(parents=True, exist_ok=False)
    with REFERENCE.open(newline="", encoding="utf-8-sig") as stream:
        reference = list(csv.DictReader(stream))
    assert len(reference) == 512 and len({r["asset_id"] for r in reference}) == 512

    source_hashes_before = {p.relative_to(DATASET).as_posix(): sha256(p)
                            for p in DATASET.rglob("*") if p.is_file()}

    rows = inventory()
    write_csv(AUDIT / "all_images_inventory.csv", rows, list(rows[0]))
    by_rel = {r["relative_path"]: r for r in rows}
    by_sha = defaultdict(list)
    by_pixel = defaultdict(list)
    by_stem = defaultdict(list)
    for row in rows:
        by_sha[row["sha256"]].append(row)
        by_pixel[row["pixel_hash"]].append(row)
        by_stem[row["stem"].lower()].append(row)

    # Different filenames can still be visually similar. Report perceptual matches,
    # but never merge them without shared provenance and a pixel-level check.
    distinct = [min(group, key=lambda r: source_priority(r["relative_path"]))
                for group in by_sha.values()]
    near_rows = []
    for i, left in enumerate(distinct):
        for right in distinct[i + 1:]:
            if (left["width"], left["height"]) != (right["width"], right["height"]):
                continue
            distance = (int(left["perceptual_hash"], 16) ^ int(right["perceptual_hash"], 16)).bit_count()
            if distance <= 4:
                mae, corr = compare(left, right) if distance <= 1 else (None, None)
                near_rows.append(dict(left_path=left["full_path"], right_path=right["full_path"],
                                      phash_distance=distance, same_stem=left["stem"].lower() == right["stem"].lower(),
                                      grayscale_mae=f"{mae:.5f}" if mae is not None else "",
                                      grayscale_correlation=f"{corr:.7f}" if corr is not None else "",
                                      decision="REVIEW_ONLY"))
    write_csv(AUDIT / "perceptual_near_candidates.csv", near_rows,
              ["left_path", "right_path", "phash_distance", "same_stem",
               "grayscale_mae", "grayscale_correlation", "decision"])

    labels_by_stem = defaultdict(list)
    for folder in LABEL_SOURCES:
        for path in (DATASET / folder).glob("*"):
            if path.suffix.lower() in {".txt", ".xml"}:
                labels_by_stem[path.stem.lower()].append(path)

    candidates = []
    metadata = []
    conflicts = []
    link_rows = []
    mismatch = []
    for ref in sorted(reference, key=lambda r: r["asset_id"]):
        anchor = by_rel.get(ref["image_path"])
        if anchor is None or anchor["sha256"] != ref["asset_id"]:
            mismatch.append(dict(asset_id=ref["asset_id"], image_path=ref["image_path"],
                                 reason="MISSING_OR_SHA_CHANGED"))
            continue
        members = {r["relative_path"]: r for r in by_sha[anchor["sha256"]] + by_pixel[anchor["pixel_hash"]]}
        for other in by_stem[anchor["stem"].lower()]:
            if other["relative_path"] in members:
                continue
            if (other["width"], other["height"]) != (anchor["width"], anchor["height"]):
                candidates.append(dict(asset_id=ref["asset_id"], anchor_path=anchor["full_path"],
                                       candidate_path=other["full_path"], reason="SAME_STEM_DIMENSION_MISMATCH",
                                       phash_distance=(int(anchor["perceptual_hash"], 16) ^ int(other["perceptual_hash"], 16)).bit_count(),
                                       grayscale_mae="", grayscale_correlation="", decision="REVIEW"))
                continue
            mae, corr = compare(anchor, other)
            dist = (int(anchor["perceptual_hash"], 16) ^ int(other["perceptual_hash"], 16)).bit_count()
            # Same basename, same dimensions, and strict normalized similarity jointly
            # establish a file-format conversion. A perceptual hash alone never merges.
            confirmed = mae <= 4.0 and corr >= 0.995
            candidates.append(dict(asset_id=ref["asset_id"], anchor_path=anchor["full_path"],
                                   candidate_path=other["full_path"], reason="SAME_STEM_COMPARISON",
                                   phash_distance=dist, grayscale_mae=f"{mae:.5f}",
                                   grayscale_correlation=f"{corr:.7f}",
                                   decision="VERIFIED_CONVERSION" if confirmed else "REVIEW"))
            if confirmed:
                members[other["relative_path"]] = other
        representative = min(members.values(), key=lambda r: source_priority(r["relative_path"]))
        uid = f"U{len(metadata) + 1:04d}"
        review_image = REVIEW / "images" / f"{uid}{representative['extension']}"
        shutil.copyfile(representative["full_path"], review_image)
        assert sha256(review_image) == representative["sha256"]
        stem_set = {member["stem"].lower() for member in members.values()}
        label_paths = sorted({p for stem in stem_set for p in labels_by_stem[stem]})
        parsed = []
        for path in label_paths:
            boxes, error = parse_label(path, representative["width"], representative["height"])
            parsed.append((path, boxes, error))
            link_rows.append(dict(unique_id=uid, label_path=str(path), label_format=path.suffix.lower(),
                                  bbox_count=len(boxes), error=error,
                                  boxes=json.dumps(boxes, ensure_ascii=False)))
        valid = [(p, b) for p, b, error in parsed if not error]
        errors = [(p, error) for p, _, error in parsed if error]
        conflict = bool(errors) or any(not equivalent(valid[0][1], b) for _, b in valid[1:]) if valid else bool(errors)
        if conflict:
            conflicts.append(dict(unique_id=uid, representative_path=representative["full_path"],
                                  label_paths=json.dumps([str(p) for p in label_paths], ensure_ascii=False),
                                  details=json.dumps([{"path": str(p), "boxes": b, "error": e}
                                                      for p, b, e in parsed], ensure_ascii=False)))
        preferred = next(((p, b) for p, b in valid if p.suffix.lower() == ".xml"), None)
        if preferred is None:
            preferred = next(((p, b) for p, b in valid if "라벨링 6종 세트" in str(p)), None)
        if preferred is None and valid:
            preferred = valid[0]
        selected_boxes = preferred[1] if preferred else []
        # Historical representative remains visible on conflict, but is not exported
        # as a resolved training label until a reviewer adjudicates it.
        if preferred and not conflict:
            (REVIEW / "labels" / f"{uid}.txt").write_text(
                to_yolo(selected_boxes, representative["width"], representative["height"]), encoding="utf-8")
        viz = REVIEW / "visualization" / f"{uid}.jpg"
        visualize(review_image, viz, uid, selected_boxes,
                  "LABEL CONFLICT" if conflict else ("NO LABEL" if not preferred else "reference boxes"))
        machine, serial, capture_date = context(representative["relative_path"])
        if not machine:
            machine = ref["machine"]
        if not capture_date:
            capture_date = ref["date"]
        for member in members.values():
            link_rows.append(dict(unique_id=uid, label_path="", label_format="IMAGE_COPY",
                                  bbox_count="", error="", boxes=member["full_path"]))
        metadata.append(dict(unique_id=uid, representative_path=representative["full_path"],
                             review_image_path=str(review_image), visualization_path=str(viz),
                             original_filename=representative["filename"], extension=representative["extension"],
                             width=representative["width"], height=representative["height"],
                             machine=machine, serial=serial, capture_date=capture_date,
                             duplicate_count=len(members) - 1,
                             duplicate_paths=json.dumps([r["full_path"] for r in sorted(members.values(), key=lambda x: x["relative_path"])
                                                         if r is not representative], ensure_ascii=False),
                             sha256=representative["sha256"], pixel_hash=representative["pixel_hash"],
                             perceptual_hash=representative["perceptual_hash"],
                             yolo_label_path=json.dumps([str(p) for p in label_paths if p.suffix.lower() == ".txt"], ensure_ascii=False),
                             xml_label_path=json.dumps([str(p) for p in label_paths if p.suffix.lower() == ".xml"], ensure_ascii=False),
                             label_source=str(preferred[0]) if preferred else "",
                             bbox_count=len(selected_boxes), class_ids="0" if selected_boxes else "",
                             class_names="defect" if selected_boxes else "", label_conflict=conflict,
                             duplicate_confidence="SHA_OR_PIXEL_AND_VERIFIED_SAME_STEM_CONVERSION",
                             representative_reason=f"priority_{source_priority(representative['relative_path'])[0]}",
                             historical_asset_id=ref["asset_id"], historical_label_path=str(DATASET / ref["label_path"]),
                             review_status="UNREVIEWED"))
        if len(metadata) % 100 == 0:
            print(f"review: {len(metadata)}/{len(reference)}", flush=True)

    candidates.extend(dict(asset_id="", anchor_path=row["left_path"],
                           candidate_path=row["right_path"], reason="PERCEPTUAL_NEAR_ONLY",
                           phash_distance=row["phash_distance"],
                           grayscale_mae=row["grayscale_mae"],
                           grayscale_correlation=row["grayscale_correlation"],
                           decision="REVIEW") for row in near_rows)
    write_csv(AUDIT / "duplicate_candidates.csv", candidates,
              ["asset_id", "anchor_path", "candidate_path", "reason", "phash_distance",
               "grayscale_mae", "grayscale_correlation", "decision"])
    write_csv(AUDIT / "label_conflicts.csv", conflicts,
              ["unique_id", "representative_path", "label_paths", "details"])
    write_csv(AUDIT / "label_links.csv", link_rows,
              ["unique_id", "label_path", "label_format", "bbox_count", "error", "boxes"])
    write_csv(AUDIT / "reference_mismatches.csv", mismatch, ["asset_id", "image_path", "reason"])
    write_csv(REVIEW / "metadata.csv", metadata, list(metadata[0]))
    contact_sheet(metadata, REVIEW / "contact_sheets", "original")
    contact_sheet(metadata, REVIEW / "contact_sheets", "bbox")

    exact_extra = sum(len(group) - 1 for group in by_sha.values())
    pixel_only_extra = len(by_sha) - len(by_pixel)
    mislabeled_bmp = sum(r["extension"] == ".jpg" and r["actual_format"] == "BMP" for r in rows)
    true_jpeg = sum(r["actual_format"] == "JPEG" for r in rows)
    stats = dict(candidate_image_files=len(rows), all_dataset_unique_sha=len(by_sha),
                 exact_duplicate_files=exact_extra, pixel_identical_additional_files=pixel_only_extra,
                 jpg_extension_with_bmp_bytes=mislabeled_bmp, actual_jpeg_files=true_jpeg,
                 historical_reference_rows=len(reference), verified_reference_rows=len(metadata),
                 reference_mismatches=len(mismatch), verified_conversion_links=sum(c["decision"] == "VERIFIED_CONVERSION" for c in candidates),
                 ambiguous_same_stem_candidates=sum(c["reason"] == "SAME_STEM_COMPARISON" and c["decision"] == "REVIEW" for c in candidates),
                 perceptual_near_pairs=len(near_rows),
                 label_connected=sum(bool(json.loads(m["yolo_label_path"]) or json.loads(m["xml_label_path"])) for m in metadata),
                 yolo_label_images=sum(bool(json.loads(m["yolo_label_path"])) for m in metadata),
                 xml_label_images=sum(bool(json.loads(m["xml_label_path"])) for m in metadata),
                 no_label_images=sum(not m["label_source"] for m in metadata),
                 label_conflict_images=len(conflicts), reference_bbox_total=sum(int(r["object_count"]) for r in reference),
                 review_bbox_total=sum(m["bbox_count"] for m in metadata),
                 exported_resolved_label_files=len(list((REVIEW / "labels").glob("*.txt"))),
                 machine_counts=dict(Counter(m["machine"] or "unknown" for m in metadata)),
                 capture_date_counts=dict(sorted(Counter(m["capture_date"] or "unknown" for m in metadata).items())),
                 source_counts=dict(Counter(m["representative_reason"] for m in metadata)))
    (AUDIT / "statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (REVIEW / "README.md").write_text(
        "# 고유 X-ray 512장 검토 자료\n\n"
        "기존 `outputs/eda_image_metrics.csv`의 라벨 매칭 512개 SHA-256 목록을 재검증한 자료입니다. "
        "전체 원본 고유 이미지 수와 혼동하지 마세요. `metadata.csv`에서 U번호를 원본 경로로 역추적할 수 있습니다.\n\n"
        "- `images`: 대표 원본의 바이트 동일 복사본(원래 확장자 유지)\n"
        "- `visualization`: 빨간 bbox와 ID. `LABEL CONFLICT`는 사람의 판정이 필요합니다.\n"
        "- `labels`: 충돌이 없는 경우에만 YOLO 형식으로 저장. 빈 파일은 zero-bbox 음성 샘플입니다.\n"
        "- `contact_sheets`: 원본 및 bbox 표시 이미지를 30장씩 빠르게 검토\n\n"
        "대표 라벨은 기존 정책(XML 우선, 없을 때 YOLO)을 표시하되, 다른 라벨과 충돌하면 "
        "학습용 라벨 파일을 생성하지 않았습니다. `outputs/unique_512_audit/label_conflicts.csv`에서 판정하세요. "
        "`review_status`를 OK/EXCLUDE/CHECK로 직접 바꿀 수 있습니다.\n",
        encoding="utf-8")
    assert len(metadata) == len(list((REVIEW / "images").iterdir())) == len(list((REVIEW / "visualization").iterdir()))
    assert len({m["unique_id"] for m in metadata}) == len(metadata)
    assert len({m["representative_path"] for m in metadata}) == len(metadata)
    assert not mismatch
    # Rehash every source image used by the review dataset; no source write occurs.
    assert all(sha256(Path(m["representative_path"])) == m["sha256"] for m in metadata)
    source_hashes_after = {p.relative_to(DATASET).as_posix(): sha256(p)
                           for p in DATASET.rglob("*") if p.is_file()}
    assert source_hashes_before == source_hashes_after, "Source dataset changed during audit"
    (AUDIT / "source_integrity.json").write_text(
        json.dumps({"files": len(source_hashes_before), "unchanged": True,
                    "aggregate_sha256": hashlib.sha256(json.dumps(source_hashes_after, sort_keys=True).encode()).hexdigest()},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
