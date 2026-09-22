from __future__ import annotations

import csv
import hashlib
import itertools
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(r"C:\workspace\Kamp_Xray")
DATASET = ROOT / "dataset"
POC = ROOT / "outputs" / "inpainting_poc_20"
OUT = ROOT / "outputs" / "yolo_poc_20"
SEED = 42


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_voc(path: Path, expected_width: int, expected_height: int) -> list[tuple[float, float, float, float]]:
    root = ET.parse(path).getroot()
    xml_width = int(root.findtext("size/width", "0"))
    xml_height = int(root.findtext("size/height", "0"))
    if (xml_width, xml_height) != (expected_width, expected_height):
        raise ValueError(f"VOC size mismatch: {path} has {xml_width}x{xml_height}")
    boxes = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"VOC object without bndbox: {path}")
        boxes.append(tuple(float(box.findtext(name, "nan")) for name in ("xmin", "ymin", "xmax", "ymax")))
    return boxes


def parse_yolo(path: Path, width: int, height: int) -> list[tuple[float, float, float, float]]:
    boxes = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5 or parts[0] != "0":
            raise ValueError(f"Unexpected YOLO row at {path}:{line_no}: {line}")
        _, xc, yc, bw, bh = map(float, parts)
        boxes.append(((xc - bw / 2) * width, (yc - bh / 2) * height,
                      (xc + bw / 2) * width, (yc + bh / 2) * height))
    return boxes


def choose_splits(items: list[dict]) -> dict[str, str]:
    # The shuffle is only a deterministic tie-breaker; score drives the split.
    rng = random.Random(SEED)
    candidates = list(range(len(items)))
    rng.shuffle(candidates)
    rank = {idx: pos for pos, idx in enumerate(candidates)}
    total_boxes = sum(len(item["boxes"]) for item in items)
    target_small = total_boxes * 3 / len(items)
    best = None
    best_pair = None
    all_indices = set(range(len(items)))

    for val_tuple in itertools.combinations(range(len(items)), 3):
        if {items[i]["machine"] for i in val_tuple} != {"1", "2", "3"}:
            continue
        remaining = sorted(all_indices - set(val_tuple))
        for test_tuple in itertools.combinations(remaining, 3):
            if {items[i]["machine"] for i in test_tuple} != {"1", "2", "3"}:
                continue
            train_tuple = tuple(sorted(all_indices - set(val_tuple) - set(test_tuple)))
            val_boxes = sum(len(items[i]["boxes"]) for i in val_tuple)
            test_boxes = sum(len(items[i]["boxes"]) for i in test_tuple)
            train_boxes = total_boxes - val_boxes - test_boxes
            date_to_splits: dict[str, set[str]] = {}
            for split, idxs in (("train", train_tuple), ("val", val_tuple), ("test", test_tuple)):
                for idx in idxs:
                    date_to_splits.setdefault(items[idx]["date"], set()).add(split)
            shared_date_penalty = sum(len(splits) - 1 for splits in date_to_splits.values())
            score = (
                shared_date_penalty * 100,
                abs(val_boxes - target_small) + abs(test_boxes - target_small),
                abs(val_boxes - test_boxes),
                abs(train_boxes - total_boxes * 14 / 20),
                tuple(sorted(rank[i] for i in val_tuple + test_tuple)),
            )
            if best is None or score < best:
                best = score
                best_pair = (set(val_tuple), set(test_tuple))

    if best_pair is None:
        raise RuntimeError("No valid 14/3/3 split found")
    val, test = best_pair
    return {
        items[i]["asset_id"]: ("val" if i in val else "test" if i in test else "train")
        for i in range(len(items))
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT}")

    selected = read_csv(POC / "selected_20.csv")
    methods = read_csv(POC / "method_summary.csv")
    image_metrics = read_csv(ROOT / "outputs" / "eda_image_metrics.csv")
    metrics_by_asset = {row["asset_id"]: row for row in image_metrics}

    if len(selected) != 20 or len({row["asset_id"] for row in selected}) != 20:
        raise ValueError("selected_20.csv must contain exactly 20 unique assets")
    best = min(methods, key=lambda row: int(row["rank"]))
    if (best["method"], best["dilation_px"], best["radius"]) != ("TELEA", "0", "3"):
        raise ValueError(f"Unexpected baseline method: {best}")

    restored_files = sorted((POC / "restored").glob("*_telea_d0_r3.png"))
    if len(restored_files) != 20:
        raise ValueError(f"Expected 20 TELEA d0 r3 images, found {len(restored_files)}")

    items = []
    protected_hashes: dict[Path, str] = {}
    for row in sorted(selected, key=lambda value: int(value["selection_order"])):
        asset_id = row["asset_id"]
        order = int(row["selection_order"])
        restored = POC / "restored" / f"{order:02d}_{asset_id[:12]}_telea_d0_r3.png"
        if not restored.is_file():
            raise FileNotFoundError(restored)
        original = Path(row["source_path"])
        if not original.is_file():
            raise FileNotFoundError(original)
        if sha256(original) != row["source_sha256"] or row["source_sha256"] != asset_id:
            raise ValueError(f"Original SHA-256 mismatch: {original}")
        with Image.open(original) as image:
            original_size = image.size
        with Image.open(restored) as image:
            restored_size = image.size
        expected_size = (int(row["width"]), int(row["height"]))
        if original_size != expected_size or restored_size != expected_size:
            raise ValueError(
                f"Resize/crop/padding detected for {asset_id}: original={original_size}, "
                f"restored={restored_size}, expected={expected_size}"
            )

        metric = metrics_by_asset.get(asset_id)
        if metric is None:
            raise ValueError(f"No representative GT found for {asset_id}")
        label_path = DATASET / Path(metric["label_path"])
        label_format = metric["label_format"]
        if label_format == "VOC":
            boxes = parse_voc(label_path, *expected_size)
        elif label_format == "YOLO":
            boxes = parse_yolo(label_path, *expected_size)
        else:
            raise ValueError(f"Unsupported representative label format: {label_format}")
        if not boxes:
            raise ValueError(f"Zero-bbox asset requires review and cannot be included: {asset_id}")
        if len(boxes) != int(row["object_count"]):
            raise ValueError(f"GT count mismatch for {asset_id}")
        for xmin, ymin, xmax, ymax in boxes:
            if not (0 <= xmin < xmax <= expected_size[0] and 0 <= ymin < ymax <= expected_size[1]):
                raise ValueError(f"Invalid GT bbox for {asset_id}: {(xmin, ymin, xmax, ymax)}")

        protected_hashes[original] = sha256(original)
        protected_hashes[restored] = sha256(restored)
        items.append({
            "asset_id": asset_id,
            "original": original,
            "restored": restored,
            "source_label": label_path,
            "label_source": label_format,
            "machine": row["machine"],
            "date": row["date"],
            "width": expected_size[0],
            "height": expected_size[1],
            "boxes": boxes,
            "image_stem": row["image_stem"],
            "original_sha256": row["source_sha256"],
            "restored_sha256": protected_hashes[restored],
        })

    split_by_asset = choose_splits(items)
    for item in items:
        item["split"] = split_by_asset[item["asset_id"]]

    for category in ("images", "labels", "previews"):
        for split in ("train", "val", "test"):
            (OUT / category / split).mkdir(parents=True, exist_ok=False)

    used_names: set[str] = set()
    manifest_rows = []
    validation_rows = []
    for item in items:
        filename = f"{item['image_stem']}.png"
        if filename.casefold() in used_names:
            filename = f"M{item['machine']}_{item['date'].replace('-', '')}_{item['image_stem']}.png"
        if filename.casefold() in used_names:
            raise ValueError(f"Unresolved output filename collision: {filename}")
        used_names.add(filename.casefold())
        split = item["split"]
        image_out = OUT / "images" / split / filename
        label_out = OUT / "labels" / split / f"{Path(filename).stem}.txt"
        shutil.copy2(item["restored"], image_out)
        if sha256(image_out) != item["restored_sha256"]:
            raise ValueError(f"Copied image hash mismatch: {image_out}")

        lines = []
        max_error_for_image = 0.0
        for bbox_index, (xmin, ymin, xmax, ymax) in enumerate(item["boxes"], 1):
            width, height = item["width"], item["height"]
            xc = ((xmin + xmax) / 2) / width
            yc = ((ymin + ymax) / 2) / height
            bw = (xmax - xmin) / width
            bh = (ymax - ymin) / height
            normalized = (xc, yc, bw, bh)
            if not all(0 <= value <= 1 for value in normalized) or bw <= 0 or bh <= 0:
                raise ValueError(f"Invalid normalized bbox for {item['asset_id']}: {normalized}")
            rendered = tuple(float(f"{value:.10f}") for value in normalized)
            lines.append("0 " + " ".join(f"{value:.10f}" for value in normalized))
            rxc, ryc, rbw, rbh = rendered
            inverse = (
                (rxc - rbw / 2) * width,
                (ryc - rbh / 2) * height,
                (rxc + rbw / 2) * width,
                (ryc + rbh / 2) * height,
            )
            errors = [abs(a - b) for a, b in zip((xmin, ymin, xmax, ymax), inverse)]
            max_error = max(errors)
            max_error_for_image = max(max_error_for_image, max_error)
            validation_rows.append({
                "asset_id": item["asset_id"], "output_image_path": str(image_out),
                "source_label_path": str(item["source_label"]), "label_source": item["label_source"],
                "bbox_index": bbox_index, "class": "defect", "class_id": 0,
                "width": width, "height": height,
                "xmin": f"{xmin:.10f}", "ymin": f"{ymin:.10f}",
                "xmax": f"{xmax:.10f}", "ymax": f"{ymax:.10f}",
                "x_center_norm": f"{rendered[0]:.10f}", "y_center_norm": f"{rendered[1]:.10f}",
                "bbox_width_norm": f"{rendered[2]:.10f}", "bbox_height_norm": f"{rendered[3]:.10f}",
                "roundtrip_max_error_px": f"{max_error:.10f}",
                "validation_status": "passed" if max_error <= 1.0 else "review_required",
            })
        label_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        review_status = "verified" if max_error_for_image <= 1.0 else "review_required"

        manifest_rows.append({
            "asset_id": item["asset_id"], "original_image_path": str(item["original"]),
            "restored_image_path": str(item["restored"]), "output_image_path": str(image_out),
            "output_label_path": str(label_out), "source_label_path": str(item["source_label"]),
            "label_source": item["label_source"], "machine": item["machine"], "date": item["date"],
            "width": item["width"], "height": item["height"], "bbox_count": len(item["boxes"]),
            "split": split, "review_status": review_status,
            "original_sha256": item["original_sha256"], "restored_sha256": item["restored_sha256"],
            "output_sha256": sha256(image_out),
        })

    # Draw previews after all clean training images and labels are finalized.
    for split in ("train", "val", "test"):
        preview_items = sorted((item for item in items if item["split"] == split), key=lambda x: x["asset_id"])[:3]
        for item in preview_items:
            manifest = next(row for row in manifest_rows if row["asset_id"] == item["asset_id"])
            with Image.open(manifest["output_image_path"]) as source:
                preview = source.convert("RGB")
            draw = ImageDraw.Draw(preview)
            for xmin, ymin, xmax, ymax in item["boxes"]:
                draw.rectangle((xmin, ymin, xmax, ymax), outline=(0, 255, 255), width=2)
                draw.text((xmin, max(0, ymin - 11)), "defect", fill=(0, 255, 255))
            preview.save(OUT / "previews" / split / Path(manifest["output_image_path"]).name)

    write_csv(OUT / "manifest.csv", manifest_rows, [
        "asset_id", "original_image_path", "restored_image_path", "output_image_path", "output_label_path",
        "source_label_path", "label_source", "machine", "date", "width", "height", "bbox_count", "split",
        "review_status", "original_sha256", "restored_sha256", "output_sha256",
    ])
    write_csv(OUT / "label_validation.csv", validation_rows, [
        "asset_id", "output_image_path", "source_label_path", "label_source", "bbox_index", "class", "class_id",
        "width", "height", "xmin", "ymin", "xmax", "ymax", "x_center_norm", "y_center_norm",
        "bbox_width_norm", "bbox_height_norm", "roundtrip_max_error_px", "validation_status",
    ])

    summary_rows = []
    for split in ("train", "val", "test", "total"):
        selected_items = items if split == "total" else [item for item in items if item["split"] == split]
        counts = Counter(item["machine"] for item in selected_items)
        summary_rows.append({
            "split": split, "image_count": len(selected_items),
            "bbox_count": sum(len(item["boxes"]) for item in selected_items),
            "machine_1_count": counts["1"], "machine_2_count": counts["2"], "machine_3_count": counts["3"],
        })
    write_csv(OUT / "split_summary.csv", summary_rows, [
        "split", "image_count", "bbox_count", "machine_1_count", "machine_2_count", "machine_3_count",
    ])

    (OUT / "dataset.yaml").write_text(
        "path: C:/workspace/Kamp_Xray/outputs/yolo_poc_20\n"
        "train: images/train\nval: images/val\ntest: images/test\n\n"
        "names:\n  0: defect\n",
        encoding="utf-8",
    )

    bbox_counts = [len(item["boxes"]) for item in items]
    total_boxes = sum(bbox_counts)
    split_lines = "\n".join(
        f"| {row['split']} | {row['image_count']} | {row['bbox_count']} | {row['machine_1_count']} | "
        f"{row['machine_2_count']} | {row['machine_3_count']} |" for row in summary_rows
    )
    readme = f"""# YOLO TELEA PoC dataset (20 images)

This is a small pipeline-validation split, not a statistically reliable evaluation split.

## Provenance

- Source images: TELEA, dilation=0 px, radius=3 only
- GT priority: representative Pascal VOC XML, then verified YOLO TXT
- Class mapping: `defect = 0`
- Split sizes: train=14, val=3, test=3
- Deterministic split seed: {SEED}
- Color-overlay components were not used as labels.
- Original images and existing inpainting outputs were read only.

## Summary

| split | images | bboxes | machine 1 | machine 2 | machine 3 |
| --- | ---: | ---: | ---: | ---: | ---: |
{split_lines}

- Total images: {len(items)}
- Total bbox: {total_boxes}
- Mean bbox/image: {total_boxes / len(items):.3f}
- Min bbox/image: {min(bbox_counts)}
- Max bbox/image: {max(bbox_counts)}
- Review required: {sum(row['review_status'] != 'verified' for row in manifest_rows)}
- Maximum GT -> YOLO -> pixel round-trip error: {max(float(row['roundtrip_max_error_px']) for row in validation_rows):.10f} px

`previews/` contains cyan-box visual checks. The clean images under `images/` contain no drawn boxes.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    # Final integrity checks, including proof that protected inputs were not modified.
    if any(sha256(path) != digest for path, digest in protected_hashes.items()):
        raise RuntimeError("A protected source file changed during generation")
    if any(row["review_status"] != "verified" for row in manifest_rows):
        raise RuntimeError("Review-required labels remain")
    for split in ("train", "val", "test"):
        images = {path.stem for path in (OUT / "images" / split).glob("*.png")}
        labels = {path.stem for path in (OUT / "labels" / split).glob("*.txt")}
        if images != labels:
            raise RuntimeError(f"Image/label mismatch in {split}")
    if len(list((OUT / "images").glob("*/*.png"))) != 20:
        raise RuntimeError("Final image count is not 20")

    by_split = {row["split"]: row for row in summary_rows}
    print("YOLO PoC dataset created")
    print(f"\nOutput:\n{OUT}")
    print("\nImages:")
    print(f"Train: {by_split['train']['image_count']}")
    print(f"Val: {by_split['val']['image_count']}")
    print(f"Test: {by_split['test']['image_count']}")
    print(f"Total: {by_split['total']['image_count']}")
    print("\nGT boxes:")
    print(f"Train: {by_split['train']['bbox_count']}")
    print(f"Val: {by_split['val']['bbox_count']}")
    print(f"Test: {by_split['test']['bbox_count']}")
    print(f"Total: {by_split['total']['bbox_count']}")
    print("\nReview required:\n0")
    print(f"\nDataset YAML:\n{OUT / 'dataset.yaml'}")
    print("\nReady for YOLO training:\nYES")


if __name__ == "__main__":
    main()
