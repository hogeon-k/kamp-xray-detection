"""Create the final unsplit 512-image BMP/YOLO dataset from verified sources."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import statistics
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from audit_missing_12_labels import box_equivalent
from create_clean_512_candidate import validate_label
from create_label_review import parse_xml, parse_yolo


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
SELECTED = ROOT / "outputs" / "selected_512_original_bmp"
MANIFEST = SELECTED / "manifest.csv"
OLD_METADATA = ROOT / "outputs" / "unique_512_review" / "metadata.csv"
FALLBACK_AUDIT = ROOT / "outputs" / "missing_12_label_audit" / "missing_12_label_sources.csv"
CONFLICTS = ROOT / "outputs" / "unique_512_audit" / "label_conflicts.csv"
CANONICAL = DATASET / "라벨링 6종 세트" / "labels"
OPENLABELING_YOLO = DATASET / "OpenLabeling-master" / "main" / "output" / "YOLO_darknet"
TEST1_YOLO = DATASET / "test1" / "yolov3" / "labels"
XML_DIR = DATASET / "OpenLabeling-master" / "main" / "output" / "PASCAL_VOC"
OUT = ROOT / "outputs" / "clean_512"
EXPECTED_FALLBACK = {
    "U0145", "U0181", "U0239", "U0247", "U0269", "U0281",
    "U0289", "U0321", "U0371", "U0385", "U0469", "U0507",
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def save_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def visualize(source: Path, output: Path, uid: str, boxes: list[tuple], source_type: str) -> None:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    for cls, x1, y1, x2, y2 in boxes:
        draw.rectangle((x1, y1, x2, y2), outline=(255, 45, 30), width=3)
        name = "defect" if cls == 0 else f"class {cls}"
        draw.text((x1 + 1, max(0, y1 - 18)), name, fill=(255, 45, 30),
                  font=font(14), stroke_width=1, stroke_fill="black")
    draw.rectangle((0, 0, min(image.width, 490), 28), fill="black")
    draw.text((5, 5), f"{uid} | bbox={len(boxes)} | {source_type}", fill="white", font=font())
    image.save(output, quality=94)


def make_sheets(items: list[dict], prefix: str, folder: Path) -> None:
    for start in range(0, len(items), 30):
        batch = items[start:start + 30]
        sheet = Image.new("RGB", (1000, ((len(batch) + 4) // 5) * 150), "white")
        draw = ImageDraw.Draw(sheet)
        for local, row in enumerate(batch):
            with Image.open(row["visualization_path"]) as opened:
                thumb = opened.convert("RGB")
                thumb.thumbnail((190, 125))
            x, y = local % 5 * 200, local // 5 * 150
            sheet.paste(thumb, (x + (200 - thumb.width) // 2, y))
            draw.text((x + 5, y + 127),
                      f"{row['unique_id']} bbox={row['bbox_count']} {row['label_source_type']}",
                      fill="black", font=font(11))
        sheet.save(folder / f"{prefix}_{start // 30 + 1:03d}.jpg", quality=90)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing clean_512: {OUT}")
    manifest = load_csv(MANIFEST)
    metadata = {row["unique_id"]: row for row in load_csv(OLD_METADATA)}
    fallback_audit = {row["unique_id"]: row for row in load_csv(FALLBACK_AUDIT)}
    conflict_ids = {row["unique_id"] for row in load_csv(CONFLICTS)}
    if len(manifest) != 512 or len(metadata) != 512 or len(fallback_audit) != 12 or len(conflict_ids) != 23:
        raise ValueError("Input cohort sizes changed")
    if set(fallback_audit) != EXPECTED_FALLBACK:
        raise ValueError("Fallback IDs differ from the verified 12")
    if len({row["unique_id"] for row in manifest}) != 512 or len({row["copied_filename"].lower() for row in manifest}) != 512:
        raise ValueError("Duplicate unique ID or image filename")

    dataset_before = {p.relative_to(DATASET).as_posix(): sha256(p) for p in DATASET.rglob("*") if p.is_file()}
    selected_before = {p.name: sha256(p) for p in SELECTED.iterdir() if p.is_file()}
    input_paths = [MANIFEST, OLD_METADATA, FALLBACK_AUDIT, CONFLICTS]
    input_before = {str(path): sha256(path) for path in input_paths}

    # Finish every source, format and fallback check before copying a single file.
    plan = []
    validation_errors = []
    for entry in manifest:
        uid = entry["unique_id"]
        image = Path(entry["copied_path"])
        original = Path(entry["representative_path"])
        if image.parent.resolve() != SELECTED.resolve() or not image.is_file() or image.suffix.lower() != ".bmp":
            raise ValueError(f"Invalid selected BMP: {uid}")
        if not original.is_file() or metadata[uid]["representative_path"] != str(original):
            raise ValueError(f"Invalid representative BMP: {uid}")
        if sha256(image) != entry["sha256"] or sha256(original) != entry["sha256"]:
            raise ValueError(f"BMP hash mismatch: {uid}")
        stem = image.stem
        canonical = CANONICAL / f"{stem}.txt"
        open_yolo = OPENLABELING_YOLO / f"{stem}.txt"
        test_yolo = TEST1_YOLO / f"{stem}.txt"
        xml = XML_DIR / f"{stem}.xml"
        if canonical.is_file():
            if uid in EXPECTED_FALLBACK:
                raise ValueError(f"Previously missing canonical TXT now exists: {uid}")
            source = canonical
            source_type = "CANONICAL_POOL"
            fallback_verified = False
            detail = "라벨링 6종 세트/labels 동일 stem TXT"
        else:
            if uid not in EXPECTED_FALLBACK or not (open_yolo.is_file() and test_yolo.is_file() and xml.is_file()):
                raise ValueError(f"Unverified or missing fallback source: {uid}")
            open_hash, test_hash = sha256(open_yolo), sha256(test_yolo)
            if open_hash != test_hash or open_yolo.read_bytes() != test_yolo.read_bytes():
                raise ValueError(f"Fallback YOLO copies differ: {uid}")
            audit_row = fallback_audit[uid]
            discovered_yolo = {Path(p).resolve() for p in json.loads(audit_row["discovered_yolo_paths"])}
            discovered_xml = {Path(p).resolve() for p in json.loads(audit_row["discovered_xml_paths"])}
            if not {open_yolo.resolve(), test_yolo.resolve()} <= discovered_yolo or xml.resolve() not in discovered_xml:
                raise ValueError(f"Fallback paths do not match prior audit: {uid}")
            if audit_row["canonical_txt_exists"].lower() != "false" or audit_row["representative_path"] != str(original):
                raise ValueError(f"Fallback provenance mismatch: {uid}")
            source = open_yolo
            source_type = "VERIFIED_FALLBACK"
            fallback_verified = True
            detail = "OpenLabeling YOLO_darknet; 동일한 test1 YOLO와 XML bbox 재검증"
        with Image.open(image) as opened:
            width, height = opened.size
        boxes, errors = validate_label(source, width, height, uid)
        validation_errors.extend(errors)
        if fallback_verified:
            alternative_boxes, alternative_errors = validate_label(test_yolo, width, height, uid)
            validation_errors.extend(alternative_errors)
            xml_boxes = parse_xml(xml)
            yolo_boxes = parse_yolo(source, width, height)
            if len(xml_boxes) != len(yolo_boxes) or not box_equivalent(xml_boxes, yolo_boxes, tolerance=1e-5):
                raise ValueError(f"Fallback XML/YOLO bbox or class mismatch: {uid}")
            if len(boxes) != len(alternative_boxes) or boxes != alternative_boxes:
                raise ValueError(f"Fallback YOLO bbox or class mismatch: {uid}")
        plan.append(dict(uid=uid, manifest=entry, image=image, original=original, source=source,
                         source_type=source_type, detail=detail, width=width, height=height,
                         boxes=boxes, canonical=canonical, open_yolo=open_yolo, test_yolo=test_yolo,
                         xml=xml, fallback_verified=fallback_verified,
                         image_hash=sha256(image), label_hash=sha256(source)))
    if validation_errors:
        # No source is copied when validation fails. An error-only report is left
        # in the new folder so the exact failing TXT/line can be reviewed.
        OUT.mkdir(parents=True, exist_ok=False)
        save_csv(OUT / "validation_errors.csv", validation_errors,
                 ["unique_id", "filename", "line_number", "code", "details"])
        (OUT / "validation_report.json").write_text(json.dumps(
            {"status": "INCOMPLETE_INVALID_LABEL", "errors": len(validation_errors),
             "clean_dataset_confirmed": False}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise ValueError(f"Invalid YOLO labels: {len(validation_errors)}; no files copied")

    counts = Counter(item["source_type"] for item in plan)
    if counts != {"CANONICAL_POOL": 500, "VERIFIED_FALLBACK": 12}:
        raise ValueError(f"Unexpected source composition: {counts}")
    if {item["uid"] for item in plan if item["fallback_verified"]} != EXPECTED_FALLBACK:
        raise ValueError("Fallback set changed")

    for path in (OUT, OUT / "images", OUT / "labels", OUT / "visualization", OUT / "contact_sheets"):
        path.mkdir(parents=True, exist_ok=False)
    final_metadata = []
    provenance = []
    class_counts = Counter()
    box_distribution = Counter()
    for item in plan:
        image_out = OUT / "images" / item["image"].name
        label_out = OUT / "labels" / f"{item['image'].stem}.txt"
        visualization = OUT / "visualization" / f"{item['image'].stem}.jpg"
        shutil.copyfile(item["image"], image_out)
        shutil.copyfile(item["source"], label_out)
        visualize(image_out, visualization, item["uid"], item["boxes"], item["source_type"])
        classes = sorted({box[0] for box in item["boxes"]})
        box_distribution[len(item["boxes"])] += 1
        for box in item["boxes"]:
            class_counts[box[0]] += 1
        entry = item["manifest"]
        final_metadata.append(dict(unique_id=item["uid"], filename=image_out.name,
                                   image_path=str(image_out), label_path=str(label_out),
                                   visualization_path=str(visualization),
                                   representative_original_path=str(item["original"]),
                                   original_label_path=str(item["source"]),
                                   label_source_type=item["source_type"], label_source_detail=item["detail"],
                                   machine=entry["machine"], serial=entry["serial"], capture_date=entry["capture_date"],
                                   width=item["width"], height=item["height"],
                                   bbox_count=len(item["boxes"]), class_ids=json.dumps(classes),
                                   sha256_image=item["image_hash"], sha256_label=item["label_hash"],
                                   previous_label_conflict=item["uid"] in conflict_ids,
                                   verified_fallback=item["fallback_verified"], review_status="UNREVIEWED"))
        note = ("canonical pool TXT 없음. 두 YOLO 사본 SHA-256/내용 동일 및 XML bbox 일치 확인 후 "
                "VERIFIED_FALLBACK 사용" if item["fallback_verified"] else "동일 stem canonical TXT 사용")
        provenance.append(dict(unique_id=item["uid"], image_filename=image_out.name,
                               final_label_path=str(label_out), final_label_source_type=item["source_type"],
                               final_label_original_path=str(item["source"]),
                               canonical_txt_path=str(item["canonical"]),
                               openlabeling_yolo_path=str(item["open_yolo"]),
                               test1_yolo_path=str(item["test_yolo"]), xml_path=str(item["xml"]),
                               canonical_exists=item["canonical"].is_file(),
                               fallback_verified=item["fallback_verified"],
                               final_label_sha256=item["label_hash"], notes=note))

    save_csv(OUT / "metadata.csv", final_metadata, list(final_metadata[0]))
    save_csv(OUT / "provenance.csv", provenance, list(provenance[0]))
    save_csv(OUT / "validation_errors.csv", [],
             ["unique_id", "filename", "line_number", "code", "details"])
    make_sheets(final_metadata, "all_bbox", OUT / "contact_sheets")
    make_sheets([row for row in final_metadata if row["verified_fallback"]], "verified_fallback", OUT / "contact_sheets")
    make_sheets([row for row in final_metadata if row["previous_label_conflict"]], "previous_conflict", OUT / "contact_sheets")

    images = list((OUT / "images").glob("*"))
    labels = list((OUT / "labels").glob("*"))
    visualizations = list((OUT / "visualization").glob("*.jpg"))
    image_copy_mismatch = sum(sha256(Path(row["image_path"])) != row["sha256_image"] for row in final_metadata)
    label_copy_mismatch = sum(sha256(Path(row["label_path"])) != row["sha256_label"] for row in final_metadata)
    image_stems = {path.stem.lower() for path in images}
    label_stems = {path.stem.lower() for path in labels}
    bbox_counts = [int(row["bbox_count"]) for row in final_metadata]
    report = dict(status="COMPLETE", clean_dataset_confirmed=True,
                  images=len(images), labels=len(labels), metadata_rows=len(final_metadata),
                  provenance_rows=len(provenance), visualizations=len(visualizations),
                  source_type_counts=dict(counts), total_bbox=sum(bbox_counts),
                  zero_bbox_images=box_distribution[0], class_ids=sorted(class_counts),
                  class_bbox_counts={str(k): v for k, v in sorted(class_counts.items())},
                  bbox_per_image_distribution={str(k): v for k, v in sorted(box_distribution.items())},
                  bbox_min=min(bbox_counts), bbox_max=max(bbox_counts),
                  bbox_mean=statistics.mean(bbox_counts), bbox_median=statistics.median(bbox_counts),
                  invalid_yolo_labels=0, image_label_stem_mismatch=len(image_stems ^ label_stems),
                  missing_labels=len(image_stems - label_stems), orphan_labels=len(label_stems - image_stems),
                  duplicate_unique_ids=len(final_metadata) - len({row["unique_id"] for row in final_metadata}),
                  duplicate_final_images=len(final_metadata) - len({row["image_path"] for row in final_metadata}),
                  sha256_copy_mismatches=image_copy_mismatch + label_copy_mismatch,
                  previous_conflict_images=sum(row["previous_label_conflict"] for row in final_metadata),
                  fallback_verified_images=sum(row["verified_fallback"] for row in final_metadata),
                  train_val_test_split=False, augmentation=False, telea_restoration=False, training_started=False)
    if not (len(images) == len(labels) == len(final_metadata) == len(provenance) == len(visualizations) == 512
            and counts == {"CANONICAL_POOL": 500, "VERIFIED_FALLBACK": 12}
            and image_stems == label_stems and report["sha256_copy_mismatches"] == 0
            and report["previous_conflict_images"] == 23 and report["fallback_verified_images"] == 12
            and report["duplicate_unique_ids"] == report["duplicate_final_images"] == 0):
        report["status"] = "INCOMPLETE_VALIDATION_FAILED"
        report["clean_dataset_confirmed"] = False
    selected_after = {p.name: sha256(p) for p in SELECTED.iterdir() if p.is_file()}
    dataset_after = {p.relative_to(DATASET).as_posix(): sha256(p) for p in DATASET.rglob("*") if p.is_file()}
    report["dataset_files_unchanged"] = len(dataset_before) if dataset_before == dataset_after else 0
    report["selected_files_unchanged"] = len(selected_before) if selected_before == selected_after else 0
    report["protected_inputs_unchanged"] = all(sha256(Path(path)) == value for path, value in input_before.items())
    if not (dataset_before == dataset_after and selected_before == selected_after and report["protected_inputs_unchanged"]):
        report["status"] = "INCOMPLETE_PROTECTED_INPUT_CHANGED"
        report["clean_dataset_confirmed"] = False
    (OUT / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# clean_512 학습 기준 데이터셋\n\n"
        "512장은 전체 원본의 고유 이미지 수가 아니라 라벨이 연결된 고유 학습 후보 수입니다. "
        "이미지는 `test1/yolov3/X선이물검출기(06.23_09.22)` 아래 대표 원본 BMP를 "
        "`selected_512_original_bmp`에서 바이트 그대로 복사했습니다.\n\n"
        "동일 stem의 `라벨링 6종 세트/labels` TXT가 있는 500장은 CANONICAL_POOL입니다. "
        "나머지 12장은 canonical TXT가 없어 OpenLabeling YOLO_darknet TXT와 test1 YOLO TXT의 "
        "SHA-256 및 바이트 동일성, VOC XML과 bbox/class 일치를 재검증하고 OpenLabeling TXT를 "
        "VERIFIED_FALLBACK으로 복사했습니다. XML은 최종 GT 파일로 사용하지 않았습니다.\n\n"
        f"실제 최종 class_id는 {sorted(class_counts)}이며, bbox는 {sum(bbox_counts)}개입니다. "
        "기존 충돌 23장은 metadata의 `previous_label_conflict`로 보존했고, 정책 외 라벨 선택은 하지 않았습니다. "
        "`provenance.csv`에서 최종 GT의 원래 파일을 추적할 수 있습니다. "
        "시각화와 contact sheet는 사람의 최종 확인용이며, review_status는 모두 UNREVIEWED입니다.\n\n"
        "train/val/test 분할 전 상태입니다. augmentation, Telea restoration, YOLO 학습은 수행하지 않았습니다.\n",
        encoding="utf-8")
    if not report["clean_dataset_confirmed"]:
        raise RuntimeError(f"Final validation failed; see {OUT / 'validation_report.json'}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
