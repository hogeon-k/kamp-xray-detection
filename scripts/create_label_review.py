"""Create read-only human review materials for the existing 512-image audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
METADATA = ROOT / "outputs" / "unique_512_review" / "metadata.csv"
CONFLICTS = ROOT / "outputs" / "unique_512_audit" / "label_conflicts.csv"
OUTPUT = ROOT / "outputs" / "label_review"
XML_COLOR = (255, 65, 45)
YOLO_COLOR = (0, 170, 245)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def font(size: int = 20) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/malgun.ttf")):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def parse_xml(path: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    boxes = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        if bbox is None:
            raise ValueError(f"XML bbox missing: {path}")
        coords = tuple(float(bbox.findtext(key, "nan")) for key in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append(dict(cls=obj.findtext("name", "").strip().lower(), coords=coords))
    return boxes


def parse_yolo(path: Path, width: int, height: int) -> list[dict]:
    boxes = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"YOLO column count: {path}")
        cls = int(parts[0])
        x, y, w, h = map(float, parts[1:])
        coords = ((x - w / 2) * width, (y - h / 2) * height,
                  (x + w / 2) * width, (y + h / 2) * height)
        boxes.append(dict(cls="defect" if cls == 0 else f"class_{cls}", coords=coords))
    return boxes


def check_boxes(boxes: list[dict], width: int, height: int, label: str) -> None:
    for box in boxes:
        x1, y1, x2, y2 = box["coords"]
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Out-of-range {label} bbox: {box}")


def iou(a: dict, b: dict) -> float:
    x1, y1, x2, y2 = a["coords"]
    u1, v1, u2, v2 = b["coords"]
    intersection = max(0.0, min(x2, u2) - max(x1, u1)) * max(0.0, min(y2, v2) - max(y1, v1))
    area_a = (x2 - x1) * (y2 - y1)
    area_b = (u2 - u1) * (v2 - v1)
    return intersection / (area_a + area_b - intersection) if intersection else 0.0


def match_boxes(xml_boxes: list[dict], yolo_boxes: list[dict]) -> list[tuple[int, int, float]]:
    """Maximum total IoU, allowing unmatched boxes. No label selection is made."""
    scores = [[iou(a, b) for b in yolo_boxes] for a in xml_boxes]

    @lru_cache(None)
    def solve(index: int, used: int) -> tuple[float, tuple[tuple[int, int, float], ...]]:
        if index == len(xml_boxes):
            return 0.0, ()
        best = solve(index + 1, used)
        for j, score in enumerate(scores[index]):
            if score <= 0 or used & (1 << j):
                continue
            tail_value, tail_pairs = solve(index + 1, used | (1 << j))
            candidate = (score + tail_value, ((index, j, score),) + tail_pairs)
            if candidate[0] > best[0]:
                best = candidate
        return best

    return list(solve(0, 0)[1])


def measures(xml_boxes: list[dict], yolo_boxes: list[dict]) -> tuple[dict, list[dict]]:
    matches = match_boxes(xml_boxes, yolo_boxes)
    pairs = []
    for i, j, score in matches:
        xml = xml_boxes[i]
        yolo = yolo_boxes[j]
        x1, y1, x2, y2 = xml["coords"]
        u1, v1, u2, v2 = yolo["coords"]
        pairs.append(dict(xml_index=i + 1, yolo_index=j + 1, iou=score,
                          center_distance=math.hypot((x1 + x2 - u1 - u2) / 2, (y1 + y2 - v1 - v2) / 2),
                          width_difference=(x2 - x1) - (u2 - u1),
                          height_difference=(y2 - y1) - (v2 - v1),
                          xml_class=xml["cls"], yolo_class=yolo["cls"],
                          class_diff=xml["cls"] != yolo["cls"]))
    values = [p["iou"] for p in pairs]
    aggregate = dict(xml_bbox_count=len(xml_boxes), yolo_bbox_count=len(yolo_boxes),
                     bbox_count_difference=len(xml_boxes) - len(yolo_boxes),
                     matched_count=len(pairs), unmatched_xml_count=len(xml_boxes) - len(pairs),
                     unmatched_yolo_count=len(yolo_boxes) - len(pairs),
                     mean_iou=sum(values) / len(values) if values else None,
                     min_iou=min(values) if values else None,
                     max_iou=max(values) if values else None,
                     mean_center_distance=sum(p["center_distance"] for p in pairs) / len(pairs) if pairs else None,
                     mean_abs_width_difference=sum(abs(p["width_difference"]) for p in pairs) / len(pairs) if pairs else None,
                     mean_abs_height_difference=sum(abs(p["height_difference"]) for p in pairs) / len(pairs) if pairs else None,
                     class_difference_count=sum(p["class_diff"] for p in pairs))
    return aggregate, pairs


def priority(metric: dict) -> str:
    # Review ordering only; never a decision about the correct label.
    if (metric["bbox_count_difference"] or metric["unmatched_xml_count"] or
            metric["unmatched_yolo_count"] or metric["class_difference_count"] or
            metric["mean_iou"] is None or metric["mean_iou"] < 0.5):
        return "HIGH"
    if metric["mean_iou"] < 0.85:
        return "MEDIUM"
    return "LOW"


def draw_box(draw: ImageDraw.ImageDraw, box: dict, color: tuple, scale: int,
             dashed: bool, index: int) -> None:
    coords = tuple(round(v * scale) for v in box["coords"])
    x1, y1, x2, y2 = coords
    if dashed:
        for x in range(x1, x2, 18):
            draw.line((x, y1, min(x + 10, x2), y1), fill=color, width=4)
            draw.line((x, y2, min(x + 10, x2), y2), fill=color, width=4)
        for y in range(y1, y2, 18):
            draw.line((x1, y, x1, min(y + 10, y2)), fill=color, width=4)
            draw.line((x2, y, x2, min(y + 10, y2)), fill=color, width=4)
    else:
        draw.rectangle(coords, outline=color, width=4)
    draw.text((x1 + 2, max(0, y1 - 22)), f"{index}:{box['cls']}", fill=color, font=font(18),
              stroke_width=2, stroke_fill="black")


def render(base: Image.Image, xml_boxes: list[dict], yolo_boxes: list[dict], path: Path,
           header: list[str], scale: int = 2) -> None:
    upscaled = base.resize((base.width * scale, base.height * scale), Image.Resampling.NEAREST)
    margin = 36 + 30 * len(header)
    canvas = Image.new("RGB", (upscaled.width, upscaled.height + margin), "#101820")
    canvas.paste(upscaled, (0, margin))
    draw = ImageDraw.Draw(canvas)
    for line_index, line in enumerate(header):
        draw.text((15, 9 + line_index * 30), line, fill="white", font=font(20))
    overlay = ImageDraw.Draw(upscaled)
    for i, box in enumerate(xml_boxes, 1):
        draw_box(overlay, box, XML_COLOR, scale, dashed=False, index=i)
    for i, box in enumerate(yolo_boxes, 1):
        draw_box(overlay, box, YOLO_COLOR, scale, dashed=True, index=i)
    canvas.paste(upscaled, (0, margin))
    canvas.save(path)


def sheet(paths: list[tuple[str, Path, str]], target: Path, columns: int = 4) -> None:
    tile_w, tile_h = 340, 315
    rows = math.ceil(len(paths) / columns)
    image = Image.new("RGB", (columns * tile_w, rows * tile_h), "white")
    draw = ImageDraw.Draw(image)
    for index, (uid, path, caption) in enumerate(paths):
        with Image.open(path) as source:
            thumb = source.convert("RGB")
            thumb.thumbnail((tile_w - 12, tile_h - 38))
        x = (index % columns) * tile_w
        y = (index // columns) * tile_h
        image.paste(thumb, (x + (tile_w - thumb.width) // 2, y))
        draw.text((x + 8, y + tile_h - 32), f"{uid} {caption}", fill="black", font=font(17))
    image.save(target, quality=92)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing review package: {OUTPUT}")
    baseline = {"metadata.csv": digest(METADATA), "label_conflicts.csv": digest(CONFLICTS)}
    dataset_before = {p.relative_to(DATASET).as_posix(): digest(p)
                      for p in DATASET.rglob("*") if p.is_file()}
    metadata = read_csv(METADATA)
    conflicts = read_csv(CONFLICTS)
    by_id = {row["unique_id"]: row for row in metadata}
    if len(metadata) != 512 or len(by_id) != 512 or len(conflicts) != 23:
        raise ValueError("Historical review counts changed; investigate before creating this package")
    conflict_ids = {row["unique_id"] for row in conflicts}
    zero = [row for row in metadata if int(row["bbox_count"]) == 0]
    zero_ids = {row["unique_id"] for row in zero}
    if len(zero) != 14 or len(zero_ids) != 14:
        raise ValueError("Zero-bbox count changed")
    overlap = conflict_ids & zero_ids
    for name in ("conflicts", "zero_bbox", "contact_sheets"):
        (OUTPUT / name).mkdir(parents=True, exist_ok=False)

    metric_rows = []
    pair_rows = []
    review_rows = []
    alternate_rows = []
    conflict_tiles = []
    for conflict in conflicts:
        uid = conflict["unique_id"]
        row = by_id[uid]
        image_path = Path(row["review_image_path"])
        original_path = Path(row["representative_path"])
        if not image_path.is_file() or not original_path.is_file():
            raise FileNotFoundError(uid)
        with Image.open(image_path) as source:
            base = source.convert("RGB")
        with Image.open(original_path) as source:
            if source.convert("RGB").tobytes() != base.tobytes():
                raise ValueError(f"Review image differs from original: {uid}")
        paths = [Path(value) for value in json.loads(conflict["label_paths"])]
        xml_paths = [path for path in paths if path.suffix.lower() == ".xml"]
        yolo_paths = [path for path in paths if path.suffix.lower() == ".txt"]
        if len(xml_paths) != 1 or not yolo_paths:
            raise ValueError(f"Expected one XML and at least one YOLO label: {uid}")
        preferred_yolo = next((p for p in yolo_paths if "라벨링 6종 세트" in str(p)), yolo_paths[0])
        xml_path = xml_paths[0]
        xml_boxes = parse_xml(xml_path)
        yolo_boxes = parse_yolo(preferred_yolo, base.width, base.height)
        check_boxes(xml_boxes, base.width, base.height, str(xml_path))
        check_boxes(yolo_boxes, base.width, base.height, str(preferred_yolo))
        alternate_status = []
        for alternate in yolo_paths:
            alternate_boxes = parse_yolo(alternate, base.width, base.height)
            check_boxes(alternate_boxes, base.width, base.height, str(alternate))
            identical = alternate_boxes == yolo_boxes
            alternate_status.append(f"{alternate.name}:{'SAME' if identical else 'DIFFERENT'}")
            alternate_rows.append(dict(unique_id=uid, primary_yolo_path=str(preferred_yolo),
                                       alternate_yolo_path=str(alternate), identical=identical,
                                       bbox_count=len(alternate_boxes)))
        metric, pairs = measures(xml_boxes, yolo_boxes)
        metric["priority"] = priority(metric)
        mean_text = f"{metric['mean_iou']:.3f}" if metric["mean_iou"] is not None else "N/A"
        note = (f"XML={len(xml_boxes)} YOLO={len(yolo_boxes)} matched={metric['matched_count']} "
                f"unmatched={metric['unmatched_xml_count']}/{metric['unmatched_yolo_count']} "
                f"mean IoU={mean_text} class diff={metric['class_difference_count']}")
        header = [f"{uid} | {row['original_filename']}", note,
                  "RED solid = XML    BLUE dashed = YOLO"]
        folder = OUTPUT / "conflicts"
        base.save(folder / f"{uid}_original.png")
        render(base, xml_boxes, [], folder / f"{uid}_xml.png", header[:1] + [f"XML boxes={len(xml_boxes)}", header[2]])
        render(base, [], yolo_boxes, folder / f"{uid}_yolo.png", header[:1] + [f"YOLO boxes={len(yolo_boxes)}", header[2]])
        render(base, xml_boxes, yolo_boxes, folder / f"{uid}_compare.png", header)
        conflict_tiles.append((uid, folder / f"{uid}_compare.png", f"XML={len(xml_boxes)} YOLO={len(yolo_boxes)}"))
        metric_rows.append(dict(unique_id=uid, image_path=str(original_path), xml_label_path=str(xml_path),
                                yolo_label_path=str(preferred_yolo), **metric,
                                notes="; ".join(alternate_status)))
        for pair in pairs:
            pair_rows.append(dict(unique_id=uid, **pair))
        review_rows.append(dict(unique_id=uid, original_path=str(original_path), xml_label_path=str(xml_path),
                                yolo_label_path=str(preferred_yolo),
                                all_yolo_label_paths=json.dumps([str(p) for p in yolo_paths], ensure_ascii=False),
                                xml_bbox_count=len(xml_boxes), yolo_bbox_count=len(yolo_boxes),
                                mean_iou=metric["mean_iou"], priority=metric["priority"],
                                review_decision="UNREVIEWED", review_note="", reviewer=""))
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    review_rows.sort(key=lambda row: (order[row["priority"]],
                                     row["mean_iou"] if row["mean_iou"] is not None else -1,
                                     row["unique_id"]))
    metric_rows.sort(key=lambda row: row["unique_id"])
    write_csv(OUTPUT / "conflict_metrics.csv", metric_rows,
              ["unique_id", "image_path", "xml_label_path", "yolo_label_path", "xml_bbox_count",
               "yolo_bbox_count", "bbox_count_difference", "matched_count", "unmatched_xml_count",
               "unmatched_yolo_count", "mean_iou", "min_iou", "max_iou", "mean_center_distance",
               "mean_abs_width_difference", "mean_abs_height_difference", "class_difference_count", "priority", "notes"])
    write_csv(OUTPUT / "bbox_pair_metrics.csv", pair_rows,
              ["unique_id", "xml_index", "yolo_index", "iou", "center_distance",
               "width_difference", "height_difference", "xml_class", "yolo_class", "class_diff"])
    write_csv(OUTPUT / "alternate_yolo_labels.csv", alternate_rows,
              ["unique_id", "primary_yolo_path", "alternate_yolo_path", "identical", "bbox_count"])
    write_csv(OUTPUT / "conflict_review.csv", review_rows,
              ["unique_id", "original_path", "xml_label_path", "yolo_label_path", "all_yolo_label_paths",
               "xml_bbox_count", "yolo_bbox_count", "mean_iou", "priority",
               "review_decision", "review_note", "reviewer"])

    zero_rows = []
    zero_tiles = []
    for row in zero:
        uid = row["unique_id"]
        original_path = Path(row["representative_path"])
        with Image.open(original_path) as source:
            base = source.convert("RGB")
        folder = OUTPUT / "zero_bbox"
        base.save(folder / f"{uid}_original.png")
        related_conflict = next((item for item in metric_rows if item["unique_id"] == uid), None)
        conflicting_yolo_count = related_conflict["yolo_bbox_count"] if related_conflict else 0
        lines = [f"{uid} | {row['original_filename']}",
                 f"Machine={row['machine']}  Date={row['capture_date']}  bbox=0",
                 f"Label source={Path(row['label_source']).name}"]
        if related_conflict:
            lines.append(f"LABEL CONFLICT: XML=0, YOLO={conflicting_yolo_count}; review before NEGATIVE_OK")
        render(base, [], [], folder / f"{uid}_context.png", lines)
        zero_tiles.append((uid, folder / f"{uid}_context.png",
                           f"XML=0 YOLO={conflicting_yolo_count}" if related_conflict else "bbox=0"))
        zero_rows.append(dict(unique_id=uid, original_path=str(original_path),
                              label_path=row["label_source"], label_source=row["label_source"],
                              machine=row["machine"], capture_date=row["capture_date"],
                              label_conflict=bool(related_conflict),
                              conflicting_yolo_bbox_count=conflicting_yolo_count,
                              review_decision="UNREVIEWED", review_note="", reviewer=""))
    write_csv(OUTPUT / "zero_bbox_review.csv", zero_rows,
              ["unique_id", "original_path", "label_path", "label_source", "machine", "capture_date",
               "label_conflict", "conflicting_yolo_bbox_count",
               "review_decision", "review_note", "reviewer"])
    sheet(conflict_tiles, OUTPUT / "contact_sheets" / "conflicts_sheet.jpg")
    sheet(zero_tiles, OUTPUT / "contact_sheets" / "zero_bbox_sheet.jpg")

    master_rows = []
    for row in metadata:
        uid = row["unique_id"]
        category = "CONFLICT" if uid in conflict_ids else "ZERO_BBOX" if uid in zero_ids else "NORMAL"
        master_rows.append(dict(unique_id=uid, representative_path=row["representative_path"],
                                bbox_count=row["bbox_count"], label_conflict=row["label_conflict"],
                                review_category=category, is_zero_bbox=uid in zero_ids,
                                review_decision="UNREVIEWED",
                                review_note="", reviewer=""))
    write_csv(OUTPUT / "review_master.csv", master_rows,
              ["unique_id", "representative_path", "bbox_count", "label_conflict",
               "review_category", "is_zero_bbox", "review_decision", "review_note", "reviewer"])
    priority_counts = {name: sum(row["priority"] == name for row in metric_rows)
                       for name in ("HIGH", "MEDIUM", "LOW")}
    summary = dict(conflict_images=len(conflicts), zero_bbox_images=len(zero),
                   conflict_zero_overlap=len(overlap), normal_images=len(metadata) - len(conflict_ids | zero_ids),
                   priority_counts=priority_counts,
                   different_bbox_count=sum(row["bbox_count_difference"] != 0 for row in metric_rows),
                   images_with_unmatched_bbox=sum(bool(row["unmatched_xml_count"] or row["unmatched_yolo_count"])
                                                  for row in metric_rows),
                   unmatched_xml_boxes=sum(row["unmatched_xml_count"] for row in metric_rows),
                   unmatched_yolo_boxes=sum(row["unmatched_yolo_count"] for row in metric_rows),
                   lowest_mean_iou=[row["unique_id"] for row in sorted(
                       (item for item in metric_rows if item["mean_iou"] is not None),
                       key=lambda r: (r["mean_iou"], r["unique_id"]))[:5]],
                   mean_iou_unavailable_images=sum(row["mean_iou"] is None for row in metric_rows),
                   alternate_yolo_differences=sum(row["identical"] is False for row in alternate_rows),
                   decision_fields_filled=0)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "README.md").write_text(
        "# 512장 라벨 품질 수동 검토\n\n"
        "이 폴더는 기존 metadata·audit·dataset을 읽기만 하여 만들었습니다. 어떤 라벨도 자동 확정하거나 수정하지 않았습니다.\n\n"
        "1. `contact_sheets/conflicts_sheet.jpg`에서 충돌 23장을 훑어보세요.\n"
        "2. `conflict_review.csv`는 HIGH, MEDIUM, LOW 순으로 정렬되어 있습니다. `conflicts/Uxxxx_compare.png`에서 "
        "빨간 실선(XML)과 파란 점선(YOLO)을 비교하세요. `conflict_metrics.csv`와 `bbox_pair_metrics.csv`는 "
        "IoU 기반 최대 합계 매칭의 참고 수치입니다. IoU가 0인 박스는 unmatched로 남겨둡니다.\n"
        "3. `review_decision`에 `XML_OK`, `YOLO_OK`, `MANUAL_FIX`, `EXCLUDE`, `CHECK` 중 하나를 직접 입력하세요. "
        "`review_note`와 `reviewer`도 기록하세요. 자동으로 정답을 고르지 않습니다.\n"
        "4. bbox 0개 이미지 14장은 **전부 충돌 23장에도 포함됩니다**. XML은 비어 있지만 YOLO에는 박스가 있습니다. "
        "`contact_sheets/zero_bbox_sheet.jpg`와 각 `zero_bbox/Uxxxx_context.png`에서 실제 이물질이 없는지 "
        "충돌 화면과 함께 확인하세요. `zero_bbox_review.csv`의 판정 후보는 "
        " `NEGATIVE_OK`, `LABEL_MISSING`, `EXCLUDE`, `CHECK`입니다.\n"
        "5. `review_master.csv`는 512장의 전체 현황입니다. 중복 소속 14장은 `review_category=CONFLICT`이고 "
        "`is_zero_bbox=True`로 표시됩니다. 개별 CSV의 수동 판정을 반영한 뒤, "
        "모든 미확정 항목을 확인하고 별도 단계에서 clean dataset을 만드세요. 이번 단계에서는 만들지 않습니다.\n\n"
        "충돌 중 YOLO 파일이 여럿인 경우 `alternate_yolo_labels.csv`에서 동일 여부를 확인하세요. "
        "현재 비교 화면의 YOLO는 기존 `라벨링 6종 세트/labels`를 우선 표시합니다. "
        "우선순위는 검토 순서일 뿐 어느 라벨이 옳다는 뜻이 아닙니다.\n",
        encoding="utf-8")

    assert len(metric_rows) == len(review_rows) == len(conflicts) == 23
    assert len(zero_rows) == len(zero_tiles) == 14
    assert len(master_rows) == 512
    assert len(list((OUTPUT / "conflicts").glob("*.png"))) == 23 * 4
    assert len(list((OUTPUT / "zero_bbox").glob("*.png"))) == 14 * 2
    assert all(Path(row["original_path"]).exists() for row in review_rows + zero_rows)
    assert all(row["review_decision"] == "UNREVIEWED" for row in review_rows + zero_rows + master_rows)
    assert digest(METADATA) == baseline["metadata.csv"]
    assert digest(CONFLICTS) == baseline["label_conflicts.csv"]
    dataset_after = {p.relative_to(DATASET).as_posix(): digest(p)
                     for p in DATASET.rglob("*") if p.is_file()}
    assert dataset_before == dataset_after, "Source dataset changed"
    integrity = dict(dataset_files=len(dataset_before), dataset_unchanged=True,
                     metadata_sha256=baseline["metadata.csv"],
                     label_conflicts_sha256=baseline["label_conflicts.csv"])
    (OUTPUT / "integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(json.dumps(summary | integrity, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
