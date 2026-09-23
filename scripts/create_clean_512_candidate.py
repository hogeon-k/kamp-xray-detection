"""Copy the selected BMPs and only same-stem canonical TXT labels for review."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
SELECTED = ROOT / "outputs" / "selected_512_original_bmp"
MANIFEST = SELECTED / "manifest.csv"
METADATA = ROOT / "outputs" / "unique_512_review" / "metadata.csv"
CONFLICTS = ROOT / "outputs" / "unique_512_audit" / "label_conflicts.csv"
CANONICAL = DATASET / "라벨링 6종 세트" / "labels"
OUT = ROOT / "outputs" / "clean_512_candidate"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def save(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def font(size: int = 15) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def validate_label(path: Path, width: int, height: int, uid: str) -> tuple[list[tuple], list[dict]]:
    boxes = []
    errors = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError as exc:
        return [], [dict(unique_id=uid, filename=path.name, line_number="", code="INVALID_ENCODING", details=str(exc))]
    for number, line in enumerate(lines, 1):
        if not line.strip():
            errors.append(dict(unique_id=uid, filename=path.name, line_number=number,
                               code="BLANK_LINE", details="blank line in TXT"))
            continue
        values = line.split()
        if len(values) != 5:
            errors.append(dict(unique_id=uid, filename=path.name, line_number=number,
                               code="COLUMN_COUNT", details=f"found {len(values)} columns"))
            continue
        try:
            class_id = int(values[0])
            x, y, w, h = map(float, values[1:])
        except ValueError:
            errors.append(dict(unique_id=uid, filename=path.name, line_number=number,
                               code="INVALID_NUMBER", details=line))
            continue
        if class_id < 0 or not all(math.isfinite(v) for v in (x, y, w, h)):
            errors.append(dict(unique_id=uid, filename=path.name, line_number=number,
                               code="INVALID_CLASS_OR_NONFINITE", details=line))
            continue
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            errors.append(dict(unique_id=uid, filename=path.name, line_number=number,
                               code="OUT_OF_NORMALIZED_RANGE", details=line))
            continue
        x1, y1, x2, y2 = (x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height
        if not (x1 >= -1e-7 and y1 >= -1e-7 and x2 <= width + 1e-7 and y2 <= height + 1e-7):
            errors.append(dict(unique_id=uid, filename=path.name, line_number=number,
                               code="BBOX_OUTSIDE_IMAGE", details=line))
            continue
        boxes.append((class_id, max(0, x1), max(0, y1), min(width, x2), min(height, y2)))
    return boxes, errors


def render(source: Path, target: Path, uid: str, boxes: list[tuple], note: str) -> None:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    for class_id, x1, y1, x2, y2 in boxes:
        draw.rectangle((x1, y1, x2, y2), outline=(255, 35, 25), width=3)
        caption = "defect" if class_id == 0 else f"class {class_id}"
        draw.text((x1 + 1, max(0, y1 - 17)), caption, fill=(255, 35, 25),
                  font=font(), stroke_width=1, stroke_fill="black")
    draw.rectangle((0, 0, min(image.width, 450), 25), fill="black")
    draw.text((6, 5), f"{uid} | {note} | bbox={len(boxes)}", fill="white", font=font())
    image.save(target, quality=94)


def make_sheets(items: list[dict], source_key: str, prefix: str, folder: Path) -> None:
    for start in range(0, len(items), 30):
        sheet = Image.new("RGB", (1000, 900), "white")
        draw = ImageDraw.Draw(sheet)
        for local, row in enumerate(items[start:start + 30]):
            with Image.open(row[source_key]) as opened:
                thumb = opened.convert("RGB")
                thumb.thumbnail((190, 125))
            x, y = local % 5 * 200, local // 5 * 150
            sheet.paste(thumb, (x + (200 - thumb.width) // 2, y))
            draw.text((x + 5, y + 128), f"{row['unique_id']} {row['filename'][:17]}", fill="black", font=font(13))
        sheet.save(folder / f"{prefix}_{start // 30 + 1:03d}.jpg", quality=90)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT}")
    inputs = [MANIFEST, METADATA, CONFLICTS]
    manifest = load(MANIFEST)
    old_metadata = {row["unique_id"]: row for row in load(METADATA)}
    conflict_ids = {row["unique_id"] for row in load(CONFLICTS)}
    canonical_files = sorted(CANONICAL.glob("*.txt"))
    canonical_by_stem = {path.stem.lower(): path for path in canonical_files}
    if len(manifest) != 512 or len(old_metadata) != 512 or len(conflict_ids) != 23:
        raise ValueError("Selected 512 inputs or historical audit counts changed")
    if len({row["unique_id"] for row in manifest}) != 512:
        raise ValueError("Duplicate unique_id in manifest")
    if len({row["copied_filename"].lower() for row in manifest}) != 512:
        raise ValueError("Duplicate image filename in manifest")
    if len(canonical_by_stem) != len(canonical_files):
        raise ValueError("Duplicate canonical TXT stem")

    protected_before = {p.relative_to(DATASET).as_posix(): digest(p) for p in DATASET.rglob("*") if p.is_file()}
    selected_before = {p.name: digest(p) for p in SELECTED.iterdir() if p.is_file()}
    other_before = {str(path): digest(path) for path in inputs}
    plan = []
    errors = []
    for row in manifest:
        uid = row["unique_id"]
        source = Path(row["copied_path"])
        if source.parent.resolve() != SELECTED.resolve() or source.suffix.lower() != ".bmp" or not source.is_file():
            raise ValueError(f"Invalid selected BMP path: {source}")
        if digest(source) != row["sha256"]:
            raise ValueError(f"Selected BMP SHA mismatch: {source}")
        if uid not in old_metadata or old_metadata[uid]["representative_path"] != row["representative_path"]:
            raise ValueError(f"Historical metadata mismatch: {uid}")
        label = canonical_by_stem.get(source.stem.lower())
        if label is None:
            errors.append(dict(unique_id=uid, filename=source.name, line_number="",
                               code="MISSING_CANONICAL_LABEL", details="No same-stem TXT in canonical pool"))
        plan.append((row, source, label))
    selected_stems = {source.stem.lower() for _, source, _ in plan}
    orphan_canonical = [p for p in canonical_files if p.stem.lower() not in selected_stems]

    for folder in (OUT, OUT / "images", OUT / "labels", OUT / "visualization", OUT / "contact_sheets"):
        folder.mkdir(parents=True, exist_ok=False)
    metadata = []
    class_counts = Counter()
    box_distribution = Counter()
    zero_bbox = 0
    format_error_count = 0
    for row, source, label in plan:
        uid = row["unique_id"]
        image_copy = OUT / "images" / source.name
        shutil.copyfile(source, image_copy)
        with Image.open(image_copy) as opened:
            width, height = opened.size
        boxes = []
        label_copy = ""
        label_hash = ""
        note = "MISSING CANONICAL TXT" if label is None else "canonical TXT"
        if label is not None:
            copied_label = OUT / "labels" / label.name
            shutil.copyfile(label, copied_label)
            label_copy = str(copied_label)
            label_hash = digest(label)
            boxes, label_errors = validate_label(label, width, height, uid)
            errors.extend(label_errors)
            format_error_count += len(label_errors)
            if label_errors:
                note = "LABEL FORMAT ERROR"
            if not boxes and not label_errors:
                zero_bbox += 1
            for class_id, *_ in boxes:
                class_counts[class_id] += 1
            if not label_errors:
                box_distribution[len(boxes)] += 1
        visualization = OUT / "visualization" / f"{source.stem}.jpg"
        render(image_copy, visualization, uid, boxes, note)
        metadata.append(dict(unique_id=uid, filename=source.name, image_path=str(image_copy),
                             label_path=label_copy, canonical_label_source=str(label) if label else "",
                             visualization_path=str(visualization), machine=row["machine"], serial=row["serial"],
                             capture_date=row["capture_date"], width=width, height=height,
                             bbox_count=len(boxes) if label else "", class_ids=json.dumps(sorted({box[0] for box in boxes})),
                             sha256_image=digest(source), sha256_label=label_hash,
                             previous_label_conflict=uid in conflict_ids,
                             label_status="PRESENT" if label else "MISSING_CANONICAL_LABEL",
                             review_status="UNREVIEWED"))
    make_sheets(metadata, "image_path", "original", OUT / "contact_sheets")
    make_sheets(metadata, "visualization_path", "bbox", OUT / "contact_sheets")
    save(OUT / "metadata.csv", metadata, list(metadata[0]))
    save(OUT / "validation_errors.csv", errors,
         ["unique_id", "filename", "line_number", "code", "details"])
    save(OUT / "orphan_canonical_labels.csv", [dict(path=str(p), filename=p.name, sha256=digest(p)) for p in orphan_canonical],
         ["path", "filename", "sha256"])

    copied_images = list((OUT / "images").glob("*"))
    copied_labels = list((OUT / "labels").glob("*"))
    visualizations = list((OUT / "visualization").glob("*.jpg"))
    image_mismatches = sum(digest(Path(item["image_path"])) != item["sha256_image"] for item in metadata)
    label_mismatches = sum(digest(Path(item["label_path"])) != item["sha256_label"] for item in metadata if item["label_path"])
    missing = [item for item in metadata if item["label_status"] == "MISSING_CANONICAL_LABEL"]
    stems_match = {p.stem.lower() for p in copied_labels} == {Path(item["filename"]).stem.lower() for item in metadata if item["label_path"]}
    report = dict(status="INCOMPLETE_MISSING_CANONICAL_LABELS" if missing else "READY_FOR_HUMAN_REVIEW",
                  clean_dataset_confirmed=False,
                  images=len(copied_images), labels=len(copied_labels), metadata_rows=len(metadata),
                  visualizations=len(visualizations), matched_pairs=len(metadata) - len(missing),
                  missing_canonical_labels=len(missing), missing_unique_ids=[item["unique_id"] for item in missing],
                  orphan_canonical_labels=len(orphan_canonical), image_label_stems_match=stems_match,
                  duplicate_unique_ids=len(metadata) - len({item["unique_id"] for item in metadata}),
                  duplicate_image_paths=len(metadata) - len({item["image_path"] for item in metadata}),
                  canonical_bbox_total=sum(class_counts.values()), zero_bbox_candidates=zero_bbox,
                  class_ids=sorted(class_counts), class_bbox_counts={str(k): v for k, v in sorted(class_counts.items())},
                  boxes_per_image_distribution={str(k): v for k, v in sorted(box_distribution.items())},
                  label_format_errors=format_error_count, previous_conflict_images=sum(item["previous_label_conflict"] for item in metadata),
                  image_sha256_mismatches=image_mismatches, label_sha256_mismatches=label_mismatches,
                  split_created=False, augmentation_applied=False, restoration_applied=False)
    (OUT / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Canonical TXT 기반 512장 candidate 검토 자료\n\n"
        "이미지는 `selected_512_original_bmp/manifest.csv`의 BMP만 복사했고, 라벨은 "
        "`dataset/라벨링 6종 세트/labels`의 같은 stem TXT만 복사했습니다. XML 및 다른 TXT는 사용하지 않았습니다. "
        "train/val/test 분할, augmentation, Telea 복원은 수행하지 않았습니다.\n\n"
        f"현재 원본 BMP 512장 중 canonical TXT가 있는 이미지는 {report['matched_pairs']}장이고, "
        f"{len(missing)}장은 canonical TXT가 없습니다. 따라서 512쌍의 clean dataset으로 확정되지 않았습니다. "
        "누락된 ID는 `validation_report.json`과 `validation_errors.csv`에 기록했습니다. "
        "이 12장의 OpenLabeling/test1 TXT를 임의로 대체하지 않았습니다.\n\n"
        "`visualization`에서 bbox와 `MISSING CANONICAL TXT` 표시를 확인하고, "
        "누락 이미지에는 GT bbox를 새로 그리지 않았습니다. 원본 BMP 자체의 컬러 마킹은 보일 수 있습니다. "
        "`contact_sheets`의 original/bbox sheet를 함께 보세요. "
        "기존 충돌 23장은 `metadata.csv`의 `previous_label_conflict`로 표시했습니다. "
        "U0275, U0503, U0508에는 다른 폴더의 YOLO TXT도 있으나, 여기서는 canonical TXT만 사용했습니다. "
        "`review_status`는 모두 UNREVIEWED입니다.\n",
        encoding="utf-8")

    assert len(copied_images) == 512 and all(p.suffix.lower() == ".bmp" for p in copied_images)
    assert len(copied_labels) == len(metadata) - len(missing)
    assert len(metadata) == len(visualizations) == 512
    assert stems_match and image_mismatches == label_mismatches == 0
    assert all(digest(path) == baseline for path, baseline in ((p, selected_before[p.name]) for p in SELECTED.iterdir() if p.is_file()))
    assert all(digest(path) == value for path, value in ((Path(p), h) for p, h in other_before.items()))
    dataset_after = {p.relative_to(DATASET).as_posix(): digest(p) for p in DATASET.rglob("*") if p.is_file()}
    assert protected_before == dataset_after, "Source dataset changed"
    report["protected_dataset_files_unchanged"] = len(protected_before)
    report["selected_files_unchanged"] = len(selected_before)
    (OUT / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
