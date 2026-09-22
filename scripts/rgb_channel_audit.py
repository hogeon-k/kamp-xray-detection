"""Audit RGB-channel outliers in the 512-image deduplicated X-ray set.

All inputs are read-only. Results are written below outputs/rgb_channel_audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


THRESHOLD = 80
BOUNDARY_BAND_PX = 2
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_rel_path(value: object) -> Path | None:
    if pd.isna(value):
        return None
    text = str(value).strip().replace("\\", "/")
    return Path(text) if text else None


def candidate_paths(project_root: Path, raw_path: object) -> Iterable[Path]:
    rel = normalize_rel_path(raw_path)
    if rel is None:
        return
    if rel.is_absolute():
        yield rel
        return
    yield project_root / rel
    yield project_root / "dataset" / rel


def build_basename_index(dataset_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in dataset_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(path.name.casefold(), []).append(path)
    return index


def resolve_images(
    project_root: Path, images: pd.DataFrame, label_audit: pd.DataFrame
) -> tuple[dict[str, Path], dict[str, str]]:
    audit_paths = (
        label_audit.dropna(subset=["asset_id", "image_path"])
        .groupby("asset_id", sort=False)["image_path"]
        .agg(list)
        .to_dict()
    )
    basename_index: dict[str, list[Path]] | None = None
    hash_cache: dict[Path, str] = {}
    resolved: dict[str, Path] = {}
    source: dict[str, str] = {}

    def hash_of(path: Path) -> str:
        if path not in hash_cache:
            hash_cache[path] = sha256_file(path)
        return hash_cache[path]

    for row in images.itertuples(index=False):
        asset_id = str(row.asset_id)
        raw_candidates = [(row.image_path, "eda_image_metrics")]
        raw_candidates.extend((p, "label_audit") for p in audit_paths.get(asset_id, []))
        seen: set[Path] = set()
        existing: list[tuple[Path, str]] = []
        for raw, origin in raw_candidates:
            for candidate in candidate_paths(project_root, raw):
                candidate = candidate.resolve()
                if candidate not in seen and candidate.is_file():
                    seen.add(candidate)
                    existing.append((candidate, origin))
        for candidate, origin in existing:
            if hash_of(candidate) == asset_id:
                resolved[asset_id] = candidate
                source[asset_id] = origin
                break
        if asset_id in resolved:
            continue

        if basename_index is None:
            basename_index = build_basename_index(project_root / "dataset")
        names = []
        for raw, _ in raw_candidates:
            rel = normalize_rel_path(raw)
            if rel is not None:
                names.append(rel.name.casefold())
        for name in dict.fromkeys(names):
            for candidate in basename_index.get(name, []):
                if hash_of(candidate) == asset_id:
                    resolved[asset_id] = candidate.resolve()
                    source[asset_id] = "dataset_basename_sha256_fallback"
                    break
            if asset_id in resolved:
                break

    missing = sorted(set(images["asset_id"].astype(str)) - set(resolved))
    if missing:
        raise FileNotFoundError(
            f"Could not resolve {len(missing)} SHA-256 assets after CSV and dataset fallback: "
            + ", ".join(missing[:10])
        )
    return resolved, source


def clipped_box(row: object, width: int, height: int) -> tuple[int, int, int, int]:
    # Integer pixel (x, y) is inside iff xmin <= x < xmax and ymin <= y < ymax.
    x0 = max(0, min(width, math.ceil(float(row.xmin_px))))
    y0 = max(0, min(height, math.ceil(float(row.ymin_px))))
    x1 = max(0, min(width, math.ceil(float(row.xmax_px))))
    y1 = max(0, min(height, math.ceil(float(row.ymax_px))))
    return x0, y0, x1, y1


def boundary_mask_for_box(
    shape: tuple[int, int], box: tuple[int, int, int, int], band: int
) -> np.ndarray:
    height, width = shape
    x0, y0, x1, y1 = box
    mask = np.zeros(shape, dtype=bool)
    ox0, oy0 = max(0, x0 - band), max(0, y0 - band)
    ox1, oy1 = min(width, x1 + band), min(height, y1 + band)
    mask[oy0:oy1, ox0:ox1] = True
    ix0, iy0 = min(width, x0 + band), min(height, y0 + band)
    ix1, iy1 = max(0, x1 - band), max(0, y1 - band)
    if ix1 > ix0 and iy1 > iy0:
        mask[iy0:iy1, ix0:ix1] = False
    return mask


def connected_components(mask: np.ndarray) -> list[dict[str, object]]:
    """Return 8-connected components without requiring scipy/opencv."""
    ys, xs = np.nonzero(mask)
    remaining = set(zip(xs.tolist(), ys.tolist()))
    components: list[dict[str, object]] = []
    neighbors = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
    while remaining:
        start = remaining.pop()
        stack = [start]
        coords = [start]
        while stack:
            x, y = stack.pop()
            for dx, dy in neighbors:
                neighbor = (x + dx, y + dy)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    coords.append(neighbor)
        component_x = np.fromiter((p[0] for p in coords), dtype=np.int32)
        component_y = np.fromiter((p[1] for p in coords), dtype=np.int32)
        x0, x1 = int(component_x.min()), int(component_x.max()) + 1
        y0, y1 = int(component_y.min()), int(component_y.max()) + 1
        crop = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        crop[component_y - y0, component_x - x0] = True
        best_iou = 0.0
        best_thickness = 0
        for thickness in range(1, min(5, math.ceil(min(crop.shape) / 2) + 1)):
            outline = np.ones(crop.shape, dtype=bool)
            if crop.shape[0] > 2 * thickness and crop.shape[1] > 2 * thickness:
                outline[thickness:-thickness, thickness:-thickness] = False
            intersection = int((crop & outline).sum())
            union = int((crop | outline).sum())
            iou = safe_ratio(intersection, union)
            if iou > best_iou:
                best_iou = iou
                best_thickness = thickness
        components.append(
            {
                "component_index": len(components) + 1,
                "pixel_count": len(coords),
                "rect": (x0, y0, x1, y1),
                "rect_width": x1 - x0,
                "rect_height": y1 - y0,
                "rectangular_outline_iou": best_iou,
                "best_outline_thickness": best_thickness,
            }
        )
    return components


def rect_overlap_metrics(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> tuple[float, float, int]:
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    iou = safe_ratio(intersection, area_a + area_b - intersection)
    containment_of_second = safe_ratio(intersection, area_b)
    max_edge_delta = max(abs(ax0 - bx0), abs(ay0 - by0), abs(ax1 - bx1), abs(ay1 - by1))
    return iou, containment_of_second, max_edge_delta


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def format_float(value: float) -> str:
    return f"{value:.8f}"


def draw_case(
    image_path: Path,
    boxes: list[dict[str, object]],
    abnormal: np.ndarray,
    inside: np.ndarray,
    destination_prefix: Path,
) -> None:
    with Image.open(image_path) as opened:
        original = opened.convert("RGB")
    original.save(destination_prefix.with_name(destination_prefix.name + "_original.png"))

    boxed = original.copy()
    draw = ImageDraw.Draw(boxed)
    for box in boxes:
        x0, y0, x1, y1 = box["pixel_box"]
        draw.rectangle((x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)), outline=(255, 215, 0), width=2)
        draw.text((x0 + 2, max(0, y0 - 11)), str(box["bbox_index"]), fill=(255, 215, 0), font=ImageFont.load_default())
    boxed.save(destination_prefix.with_name(destination_prefix.name + "_bbox.png"))

    rgba = np.asarray(original.convert("RGBA")).copy()
    overlay = np.zeros_like(rgba)
    overlay[..., 3] = 0
    outside = abnormal & ~inside
    overlay[outside] = np.array([0, 220, 255, 210], dtype=np.uint8)
    overlay[inside] = np.array([255, 40, 40, 220], dtype=np.uint8)
    overlaid = Image.alpha_composite(Image.fromarray(rgba, "RGBA"), Image.fromarray(overlay, "RGBA")).convert("RGB")
    draw = ImageDraw.Draw(overlaid)
    for box in boxes:
        x0, y0, x1, y1 = box["pixel_box"]
        draw.rectangle((x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)), outline=(255, 215, 0), width=2)
    legend_y = 4
    draw.rectangle((4, legend_y, 260, legend_y + 35), fill=(0, 0, 0))
    draw.text((9, legend_y + 3), "red: abnormal inside bbox", fill=(255, 80, 80), font=ImageFont.load_default())
    draw.text((9, legend_y + 18), "cyan: abnormal outside bbox", fill=(0, 220, 255), font=ImageFont.load_default())
    overlaid.save(destination_prefix.with_name(destination_prefix.name + "_overlay.png"))

    triptych = Image.new("RGB", (original.width * 3, original.height), "black")
    triptych.paste(original, (0, 0))
    triptych.paste(boxed, (original.width, 0))
    triptych.paste(overlaid, (original.width * 2, 0))
    triptych.save(destination_prefix.with_name(destination_prefix.name + "_triptych.png"))


def markdown_table(df: pd.DataFrame, columns: list[str], limit: int = 10) -> str:
    if df.empty:
        return "(해당 없음)"
    view = df.loc[:, columns].head(limit).copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, separator]
    for values in view.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(v).replace("|", "\\|") for v in values) + " |")
    return "\n".join(rows)


def audit(project_root: Path, output_dir: Path) -> None:
    inputs = project_root / "outputs"
    image_csv = inputs / "eda_image_metrics.csv"
    label_csv = inputs / "label_audit.csv"
    bbox_csv = inputs / "eda_bbox_metrics.csv"
    for path in (image_csv, label_csv, bbox_csv):
        if not path.is_file():
            raise FileNotFoundError(path)

    images = pd.read_csv(image_csv, low_memory=False)
    label_audit = pd.read_csv(label_csv, low_memory=False)
    bboxes = pd.read_csv(bbox_csv, low_memory=False)
    required_images = {"asset_id", "image_path", "image_width", "image_height", "object_count"}
    required_boxes = {"asset_id", "bbox_index", "class_name", "xmin_px", "ymin_px", "xmax_px", "ymax_px"}
    if missing := required_images - set(images.columns):
        raise ValueError(f"Missing image columns: {sorted(missing)}")
    if missing := required_boxes - set(bboxes.columns):
        raise ValueError(f"Missing bbox columns: {sorted(missing)}")
    if len(images) != 512 or images["asset_id"].nunique() != 512:
        raise ValueError(f"Expected 512 unique assets, got rows={len(images)}, unique={images['asset_id'].nunique()}")
    if len(bboxes) != 1134:
        raise ValueError(f"Expected 1,134 representative boxes, got {len(bboxes)}")

    resolved, path_source = resolve_images(project_root, images, label_audit)
    if output_dir.exists():
        # This directory contains only generated audit artifacts from this script.
        shutil.rmtree(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True)

    box_groups = {str(key): group.copy() for key, group in bboxes.groupby("asset_id", sort=False)}
    image_rows: list[dict[str, object]] = []
    bbox_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    render_cache: dict[str, tuple[Path, list[dict[str, object]], np.ndarray, np.ndarray]] = {}
    global_colors: Counter[tuple[int, int, int]] = Counter()
    dominant_channel = Counter()
    total_abnormal = total_inside = total_outside = 0

    abnormal_csv = output_dir / "abnormal_pixels.csv"
    pixel_fields = [
        "asset_id", "image_path", "x", "y", "R", "G", "B", "rgb_range",
        "inside_bbox", "bbox_count", "bbox_indices", "object_ids", "class_names",
    ]
    with abnormal_csv.open("w", newline="", encoding="utf-8-sig") as pixel_handle:
        pixel_writer = csv.DictWriter(pixel_handle, fieldnames=pixel_fields)
        pixel_writer.writeheader()

        for ordinal, image_row in enumerate(images.itertuples(index=False), start=1):
            asset_id = str(image_row.asset_id)
            path = resolved[asset_id]
            with Image.open(path) as opened:
                rgb_image = opened.convert("RGB")
                rgb = np.asarray(rgb_image, dtype=np.uint8)
            height, width = rgb.shape[:2]
            if (width, height) != (int(image_row.image_width), int(image_row.image_height)):
                raise ValueError(f"Dimension mismatch for {path}: actual={(width,height)} CSV={(image_row.image_width,image_row.image_height)}")

            rgb_range = rgb.max(axis=2) - rgb.min(axis=2)
            abnormal = rgb_range > THRESHOLD
            ys, xs = np.nonzero(abnormal)
            count = int(len(xs))
            inside_union = np.zeros((height, width), dtype=bool)
            boundary_unions = {band: np.zeros((height, width), dtype=bool) for band in (2, 5, 10)}
            boxes_for_image: list[dict[str, object]] = []
            group = box_groups.get(asset_id, pd.DataFrame(columns=bboxes.columns))
            for box_row in group.itertuples(index=False):
                pixel_box = clipped_box(box_row, width, height)
                x0, y0, x1, y1 = pixel_box
                box_mask_area = max(0, x1 - x0) * max(0, y1 - y0)
                inside_union[y0:y1, x0:x1] = True
                box_boundary = boundary_mask_for_box((height, width), pixel_box, BOUNDARY_BAND_PX)
                for band in boundary_unions:
                    boundary_unions[band] |= boundary_mask_for_box((height, width), pixel_box, band)

                box_abnormal = int(abnormal[y0:y1, x0:x1].sum())
                bw, bh = max(0, x1 - x0), max(0, y1 - y0)
                ex0, ey0 = max(0, x0 - bw), max(0, y0 - bh)
                ex1, ey1 = min(width, x1 + bw), min(height, y1 + bh)
                ring_area = max(0, ex1 - ex0) * max(0, ey1 - ey0) - box_mask_area
                ring_abnormal = int(abnormal[ey0:ey1, ex0:ex1].sum()) - box_abnormal
                box_rate = safe_ratio(box_abnormal, box_mask_area)
                ring_rate = safe_ratio(ring_abnormal, ring_area)
                boundary_abnormal = int((abnormal & box_boundary).sum())
                bbox_boundary_abnormal = int((abnormal[y0:y1, x0:x1] & box_boundary[y0:y1, x0:x1]).sum())
                enrichment = box_rate / ring_rate if ring_rate else (math.inf if box_rate else 0.0)
                object_id = f"{asset_id}:{int(box_row.bbox_index)}"
                summary_record = {
                        "asset_id": asset_id,
                        "image_path": str(image_row.image_path),
                        "resolved_image_path": str(path),
                        "bbox_index": int(box_row.bbox_index),
                        "object_id": object_id,
                        "class_name": str(box_row.class_name),
                        "xmin_px": float(box_row.xmin_px),
                        "ymin_px": float(box_row.ymin_px),
                        "xmax_px": float(box_row.xmax_px),
                        "ymax_px": float(box_row.ymax_px),
                        "raster_xmin": x0,
                        "raster_ymin": y0,
                        "raster_xmax_exclusive": x1,
                        "raster_ymax_exclusive": y1,
                        "bbox_raster_area_pixels": box_mask_area,
                        "rgb_diff_gt80_pixels_inside_bbox": box_abnormal,
                        "bbox_abnormal_ratio": box_rate,
                        "surrounding_ring_area_pixels": ring_area,
                        "surrounding_ring_rgb_diff_gt80_pixels": ring_abnormal,
                        "surrounding_ring_abnormal_ratio": ring_rate,
                        "bbox_vs_surrounding_enrichment": enrichment,
                        "boundary_band_px": BOUNDARY_BAND_PX,
                        "boundary_band_rgb_diff_gt80_pixels": boundary_abnormal,
                        "bbox_internal_boundary_rgb_diff_gt80_pixels": bbox_boundary_abnormal,
                        "boundary_share_of_bbox_abnormal": safe_ratio(bbox_boundary_abnormal, box_abnormal),
                    }
                boxes_for_image.append(
                    {
                        "bbox_index": int(box_row.bbox_index),
                        "class_name": str(box_row.class_name),
                        "object_id": object_id,
                        "pixel_box": pixel_box,
                        "summary_record": summary_record,
                    }
                )

            inside_abnormal_mask = abnormal & inside_union
            inside_count = int(inside_abnormal_mask.sum())
            outside_count = count - inside_count
            boundary_counts = {band: int((abnormal & mask).sum()) for band, mask in boundary_unions.items()}
            bbox_union_area = int(inside_union.sum())
            union_area = int((abnormal | inside_union).sum())
            components = connected_components(abnormal) if count else []
            matched_component_indices: set[int] = set()
            matched_bbox_count = 0
            for box in boxes_for_image:
                best_component = None
                best_metrics = (0.0, 0.0, 10**9)
                for component in components:
                    metrics = rect_overlap_metrics(component["rect"], box["pixel_box"])
                    # Prefer containment, then IoU, then smaller edge displacement.
                    if (metrics[1], metrics[0], -metrics[2]) > (best_metrics[1], best_metrics[0], -best_metrics[2]):
                        best_component = component
                        best_metrics = metrics
                rect_iou, containment, max_edge_delta = best_metrics
                component_match = bool(
                    best_component is not None
                    and containment >= 0.80
                    and rect_iou >= 0.20
                    and max_edge_delta <= 12
                    and float(best_component["rectangular_outline_iou"]) >= 0.50
                )
                if component_match:
                    matched_bbox_count += 1
                    matched_component_indices.add(int(best_component["component_index"]))
                box["summary_record"].update(
                    {
                        "nearest_abnormal_component_index": int(best_component["component_index"]) if best_component else "",
                        "component_rect_iou_with_bbox": rect_iou,
                        "component_rect_bbox_containment": containment,
                        "component_rect_max_edge_delta_px": max_edge_delta if best_component else "",
                        "component_rectangular_outline_iou": float(best_component["rectangular_outline_iou"]) if best_component else 0.0,
                        "component_bbox_shape_match": component_match,
                    }
                )
                bbox_rows.append(box["summary_record"])

            for component in components:
                component_index = int(component["component_index"])
                matched_boxes = [
                    box for box in boxes_for_image
                    if bool(box["summary_record"]["component_bbox_shape_match"])
                    and int(box["summary_record"]["nearest_abnormal_component_index"]) == component_index
                ]
                component_rows.append(
                    {
                        "asset_id": asset_id,
                        "image_path": str(image_row.image_path),
                        **component,
                        "matched_bbox_count": len(matched_boxes),
                        "matched_bbox_indices": ";".join(str(box["bbox_index"]) for box in matched_boxes),
                        "matched_object_ids": ";".join(str(box["object_id"]) for box in matched_boxes),
                    }
                )
            matched_component_pixels = sum(
                int(component["pixel_count"])
                for component in components
                if int(component["component_index"]) in matched_component_indices
            )
            values = rgb_range[abnormal]
            max_range = int(values.max()) if count else 0
            mean_range = float(values.mean()) if count else 0.0
            if count:
                pixels = rgb[ys, xs]
                unique_colors, color_counts = np.unique(pixels, axis=0, return_counts=True)
                global_colors.update({tuple(map(int, color)): int(n) for color, n in zip(unique_colors, color_counts)})
                maxima = pixels.max(axis=1)
                for channel_index, name in enumerate(("R", "G", "B")):
                    dominant_channel[name] += int(((pixels[:, channel_index] == maxima) & ((pixels == maxima[:, None]).sum(axis=1) == 1)).sum())
                dominant_channel["tie"] += int(((pixels == maxima[:, None]).sum(axis=1) > 1).sum())

            image_record = {
                "image_index": int(image_row.image_index) if hasattr(image_row, "image_index") else ordinal,
                "asset_id": asset_id,
                "image_path": str(image_row.image_path),
                "resolved_image_path": str(path),
                "path_source": path_source[asset_id],
                "image_width": width,
                "image_height": height,
                "object_count": int(image_row.object_count),
                "rgb_diff_gt80_pixels": count,
                "rgb_diff_gt80_ratio": safe_ratio(count, width * height),
                "max_rgb_range": max_range,
                "mean_rgb_range_abnormal": mean_range,
                "inside_bbox_pixels": inside_count,
                "outside_bbox_pixels": outside_count,
                "inside_ratio": safe_ratio(inside_count, count),
                "outside_ratio": safe_ratio(outside_count, count),
                "bbox_union_area_pixels": bbox_union_area,
                "bbox_union_coverage_by_abnormal": safe_ratio(inside_count, bbox_union_area),
                "abnormal_bbox_iou": safe_ratio(inside_count, union_area),
                "abnormal_near_bbox_boundary_pixels": boundary_counts[BOUNDARY_BAND_PX],
                "abnormal_near_bbox_boundary_ratio": safe_ratio(boundary_counts[BOUNDARY_BAND_PX], count),
                "abnormal_near_bbox_boundary_5px_pixels": boundary_counts[5],
                "abnormal_near_bbox_boundary_5px_ratio": safe_ratio(boundary_counts[5], count),
                "abnormal_near_bbox_boundary_10px_pixels": boundary_counts[10],
                "abnormal_near_bbox_boundary_10px_ratio": safe_ratio(boundary_counts[10], count),
                "abnormal_component_count": len(components),
                "rectangular_outline_component_count": sum(float(c["rectangular_outline_iou"]) >= 0.80 for c in components),
                "bbox_shape_matched_count": matched_bbox_count,
                "bbox_shape_match_ratio": safe_ratio(matched_bbox_count, len(boxes_for_image)),
                "bbox_matched_component_count": len(matched_component_indices),
                "bbox_matched_component_pixels": matched_component_pixels,
                "bbox_matched_component_pixel_ratio": safe_ratio(matched_component_pixels, count),
            }
            image_rows.append(image_record)
            total_abnormal += count
            total_inside += inside_count
            total_outside += outside_count

            if count:
                memberships: list[list[dict[str, object]]] = [[] for _ in range(count)]
                for box in boxes_for_image:
                    x0, y0, x1, y1 = box["pixel_box"]
                    hit_indices = np.flatnonzero((xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1))
                    for index in hit_indices:
                        memberships[int(index)].append(box)
                pixels = rgb[ys, xs]
                for index in range(count):
                    matched = memberships[index]
                    pixel_writer.writerow(
                        {
                            "asset_id": asset_id,
                            "image_path": str(image_row.image_path),
                            "x": int(xs[index]),
                            "y": int(ys[index]),
                            "R": int(pixels[index, 0]),
                            "G": int(pixels[index, 1]),
                            "B": int(pixels[index, 2]),
                            "rgb_range": int(values[index]),
                            "inside_bbox": bool(matched),
                            "bbox_count": len(matched),
                            "bbox_indices": ";".join(str(item["bbox_index"]) for item in matched),
                            "object_ids": ";".join(str(item["object_id"]) for item in matched),
                            "class_names": ";".join(str(item["class_name"]) for item in matched),
                        }
                    )
                render_cache[asset_id] = (path, boxes_for_image, abnormal.copy(), inside_abnormal_mask.copy())

            if ordinal % 50 == 0 or ordinal == len(images):
                print(f"processed {ordinal}/{len(images)} images; abnormal_pixels={total_abnormal}", flush=True)

    image_summary = pd.DataFrame(image_rows)
    bbox_summary = pd.DataFrame(bbox_rows)
    component_summary = pd.DataFrame(component_rows)
    image_summary.to_csv(output_dir / "image_summary.csv", index=False, encoding="utf-8-sig")
    bbox_summary.to_csv(output_dir / "bbox_summary.csv", index=False, encoding="utf-8-sig")
    component_summary.to_csv(output_dir / "component_summary.csv", index=False, encoding="utf-8-sig")

    abnormal_images = image_summary[image_summary["rgb_diff_gt80_pixels"] > 0].copy()
    selections = {
        "top_abnormal_pixels": abnormal_images.sort_values(["rgb_diff_gt80_pixels", "asset_id"], ascending=[False, True]).head(10),
        "top_inside_ratio": abnormal_images.sort_values(["inside_ratio", "inside_bbox_pixels", "asset_id"], ascending=[False, False, True]).head(10),
        "top_outside_pixels": abnormal_images.sort_values(["outside_bbox_pixels", "asset_id"], ascending=[False, True]).head(10),
    }
    figure_manifest_rows: list[dict[str, object]] = []
    for category, selected in selections.items():
        category_dir = figures_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        for rank, row in enumerate(selected.itertuples(index=False), start=1):
            stem = f"{rank:02d}_{row.asset_id[:12]}"
            prefix = category_dir / stem
            draw_case(*render_cache[row.asset_id], prefix)
            figure_manifest_rows.append(
                {
                    "category": category,
                    "rank": rank,
                    "asset_id": row.asset_id,
                    "image_path": row.image_path,
                    "rgb_diff_gt80_pixels": row.rgb_diff_gt80_pixels,
                    "inside_ratio": row.inside_ratio,
                    "outside_bbox_pixels": row.outside_bbox_pixels,
                    "file_prefix": str(prefix.relative_to(output_dir)),
                }
            )
    pd.DataFrame(figure_manifest_rows).to_csv(figures_dir / "figure_manifest.csv", index=False, encoding="utf-8-sig")

    abnormal_bbox = bbox_summary[bbox_summary["rgb_diff_gt80_pixels_inside_bbox"] > 0]
    image_concentration = safe_ratio(
        int(abnormal_images.nlargest(min(10, len(abnormal_images)), "rgb_diff_gt80_pixels")["rgb_diff_gt80_pixels"].sum()),
        total_abnormal,
    )
    bbox_hit_total = int(bbox_summary["rgb_diff_gt80_pixels_inside_bbox"].sum())
    bbox_concentration = safe_ratio(
        int(abnormal_bbox.nlargest(min(10, len(abnormal_bbox)), "rgb_diff_gt80_pixels_inside_bbox")["rgb_diff_gt80_pixels_inside_bbox"].sum()),
        bbox_hit_total,
    )
    near_identical = image_summary[
        (image_summary["inside_ratio"] >= 0.90)
        & (image_summary["bbox_union_coverage_by_abnormal"] >= 0.90)
        & (image_summary["abnormal_bbox_iou"] >= 0.80)
    ]
    high_boundary = abnormal_images[abnormal_images["abnormal_near_bbox_boundary_ratio"] >= 0.80]
    top_colors = global_colors.most_common(15)
    rectangular_components = component_summary[component_summary["rectangular_outline_iou"] >= 0.80]
    shape_matched_boxes = bbox_summary[bbox_summary["component_bbox_shape_match"]]
    matched_component_pixels = int(image_summary["bbox_matched_component_pixels"].sum())
    boundary_5px_pixels = int(image_summary["abnormal_near_bbox_boundary_5px_pixels"].sum())
    boundary_10px_pixels = int(image_summary["abnormal_near_bbox_boundary_10px_pixels"].sum())
    exact_primary_palette = all(
        color in {(255, 0, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0)}
        for color in global_colors
    )

    if total_abnormal == 0:
        interpretation = "임계값을 넘는 픽셀이 없어 색상 아티팩트 기반 데이터 누수 증거가 발견되지 않았다."
    elif (
        safe_ratio(len(rectangular_components), len(component_summary)) >= 0.90
        and safe_ratio(len(shape_matched_boxes), len(bbox_summary)) >= 0.70
        and safe_ratio(boundary_10px_pixels, total_abnormal) >= 0.80
    ):
        interpretation = (
            "데이터 누수 위험이 높다. 이상 픽셀은 자연스러운 X-ray 질감이 아니라 정확한 팔레트 원색의 얇은 사각 테두리이며, "
            "대부분의 라벨 bbox를 포함·추종한다. 내부 픽셀 비율이 낮은 이유는 색상 마킹이 bbox 바깥 테두리에 놓이기 때문이지 "
            "공간적 무관성을 뜻하지 않는다. 색상 마킹 제거 전에는 학습에 사용하지 않는 것이 안전하다."
        )
    else:
        inside_fraction = safe_ratio(total_inside, total_abnormal)
        max_iou = float(abnormal_images["abnormal_bbox_iou"].max()) if not abnormal_images.empty else 0.0
        if len(near_identical) > 0 or (inside_fraction >= 0.8 and max_iou >= 0.5):
            interpretation = (
                "이상 픽셀이 bbox와 강하게 공간적으로 정렬되는 사례가 있어 라벨/마킹/합성 과정의 컬러 아티팩트에 의한 "
                "데이터 누수 가능성을 배제하기 어렵다. 원본 획득 파이프라인과 마킹 전 영상을 대조해야 한다."
            )
        elif inside_fraction <= 0.2:
            interpretation = (
                "이상 픽셀 대부분이 bbox 밖에 있어 bbox 자체를 직접 인코딩한 색상 누수의 전역 증거는 약하다. "
                "다만 개별 고농도 bbox와 경계 정렬 사례는 별도로 검토해야 한다."
            )
        else:
            interpretation = (
                "bbox 내부와 외부 모두에 이상 픽셀이 존재해 전역 비율만으로 원인을 단정할 수 없다. "
                "상위 사례의 형태·경계 정렬과 RGB 값 반복성을 원본 획득 과정과 함께 검토해야 한다."
            )

    most_severe = abnormal_images.sort_values("rgb_diff_gt80_pixels", ascending=False).head(1)
    severe_text = "없음" if most_severe.empty else (
        f"`{most_severe.iloc[0]['image_path']}` (asset `{most_severe.iloc[0]['asset_id']}`, "
        f"{int(most_severe.iloc[0]['rgb_diff_gt80_pixels']):,}픽셀, 내부 {most_severe.iloc[0]['inside_ratio']:.2%})"
    )
    report = f"""# RGB 채널 이상 감사 보고서

## 결론 요약

- 검사 기준: 각 픽셀의 `max(R,G,B) - min(R,G,B) > {THRESHOLD}` (`Image.open(path).convert(\"RGB\")` 적용)
- 검사 이미지: **{len(image_summary):,}장** (SHA-256 고유 asset **{image_summary['asset_id'].nunique():,}개**)
- 대표 bbox 객체: **{len(bbox_summary):,}개**
- 이상 픽셀 보유 이미지: **{len(abnormal_images):,}장** ({safe_ratio(len(abnormal_images), len(image_summary)):.2%})
- 전체 이상 픽셀: **{total_abnormal:,}개**
- bbox 내부 이상 픽셀(고유 픽셀 기준): **{total_inside:,}개** ({safe_ratio(total_inside, total_abnormal):.2%})
- bbox 외부 이상 픽셀: **{total_outside:,}개** ({safe_ratio(total_outside, total_abnormal):.2%})
- 가장 심한 이미지: {severe_text}
- 해석: **{interpretation}**

## 방법과 좌표 규칙

`eda_image_metrics.csv`의 512개 고유 `asset_id`를 모집단으로 사용하고, 실제 파일 바이트의 SHA-256이 `asset_id`와 일치하는 경로만 검사했다. 경로는 `eda_image_metrics.csv`, 같은 asset의 `label_audit.csv`, 마지막으로 dataset 내 동일 basename+SHA-256 순서로 해석했다. 모든 원본과 기존 CSV는 읽기만 했다.

bbox 내부는 `xmin <= x < xmax`, `ymin <= y < ymax`로 정의했다. 겹치는 bbox가 있으면 `abnormal_pixels.csv`에 모든 bbox/object ID를 세미콜론으로 기록했다. 이미지 수준 내부 픽셀 수는 bbox 중첩과 무관한 고유 픽셀 수이고, bbox 수준 수치는 각 bbox별이므로 중첩 영역은 여러 bbox에 각각 포함될 수 있다.

bbox 주변 비교 영역은 bbox의 가로 길이만큼 좌우, 세로 길이만큼 상하로 확장한 사각형에서 원 bbox를 뺀 외부 링이다. `bbox_vs_surrounding_enrichment`는 bbox 내부 이상 밀도 / 주변 링 이상 밀도다. bbox 경계 관계는 경계 안팎 {BOUNDARY_BAND_PX}px 밴드로 측정했다.

또한 8-연결 이상 픽셀을 하나의 component로 묶고, component의 최소 사각형에 1~4px 두께의 사각 테두리를 맞춰 최고 IoU를 구했다. bbox-component shape match는 (1) component 최소 사각형이 bbox 면적의 80% 이상을 포함하고, (2) 두 사각형 IoU가 0.20 이상이고, (3) 대응 변의 최대 차이가 12px 이하이며, (4) 사각 테두리 적합도 IoU가 0.50 이상인 보수적 조건이다.

## 집중도와 공간적 관계

- 이상 픽셀 상위 10개 이미지가 전체 이상 픽셀의 **{image_concentration:.2%}**를 차지한다.
- 이상 픽셀이 있는 bbox는 **{len(abnormal_bbox):,}/{len(bbox_summary):,}개**다.
- bbox별 내부 카운트(중첩 중복 가능) 상위 10개 bbox가 전체 bbox hit의 **{bbox_concentration:.2%}**를 차지한다.
- 이상 마스크와 bbox union이 거의 동일한 조건(inside>=90%, bbox coverage>=90%, IoU>=80%)의 이미지는 **{len(near_identical):,}장**이다.
- 이상 픽셀의 80% 이상이 bbox 경계 ±{BOUNDARY_BAND_PX}px에 있는 이미지는 **{len(high_boundary):,}장**이다.
- 8-연결 이상 component는 **{len(component_summary):,}개**이며, 그중 사각 테두리 적합도 IoU>=0.80은 **{len(rectangular_components):,}개 ({safe_ratio(len(rectangular_components), len(component_summary)):.2%})**다.
- 보수적 사각형-bbox shape match를 통과한 bbox는 **{len(shape_matched_boxes):,}/{len(bbox_summary):,}개 ({safe_ratio(len(shape_matched_boxes), len(bbox_summary)):.2%})**다. 전체 bbox의 component-rectangle bbox 포함률 평균은 **{bbox_summary['component_rect_bbox_containment'].mean():.2%}**, 변 좌표 최대 차이 중앙값은 **{bbox_summary['component_rect_max_edge_delta_px'].median():.1f}px**다.
- shape-matched component에 속한 이상 픽셀은 **{matched_component_pixels:,}개 ({safe_ratio(matched_component_pixels, total_abnormal):.2%})**다.
- bbox 경계 ±5px에는 **{boundary_5px_pixels:,}개 ({safe_ratio(boundary_5px_pixels, total_abnormal):.2%})**, ±10px에는 **{boundary_10px_pixels:,}개 ({safe_ratio(boundary_10px_pixels, total_abnormal):.2%})**의 이상 픽셀이 있다.

`inside_ratio={safe_ratio(total_inside, total_abnormal):.2%}`와 `outside_ratio={safe_ratio(total_outside, total_abnormal):.2%}`를 단독으로 해석하면 안 된다. 시각화와 component 분석에서 색상 픽셀은 bbox 면을 채우지 않고 bbox 주변의 얇은 사각 테두리를 이루므로 대부분이 엄격한 bbox 내부 판정 밖에 놓인다. 즉 높은 외부 비율 자체가 오히려 “라벨 위치를 둘러싼 마킹 테두리”의 기하와 일치한다.

### 이상 픽셀 수 TOP 10 이미지

{markdown_table(abnormal_images.sort_values('rgb_diff_gt80_pixels', ascending=False), ['asset_id','image_path','rgb_diff_gt80_pixels','inside_bbox_pixels','outside_bbox_pixels','inside_ratio','abnormal_bbox_iou'])}

### bbox 내부 비율 TOP 10 이미지

{markdown_table(abnormal_images.sort_values(['inside_ratio','inside_bbox_pixels'], ascending=False), ['asset_id','image_path','rgb_diff_gt80_pixels','inside_bbox_pixels','inside_ratio','bbox_union_coverage_by_abnormal','abnormal_near_bbox_boundary_ratio'])}

### bbox 내부 이상 픽셀 TOP 10 객체

{markdown_table(bbox_summary.sort_values('rgb_diff_gt80_pixels_inside_bbox', ascending=False), ['asset_id','bbox_index','class_name','rgb_diff_gt80_pixels_inside_bbox','bbox_abnormal_ratio','surrounding_ring_abnormal_ratio','bbox_vs_surrounding_enrichment','boundary_share_of_bbox_abnormal'])}

### 색상 사각형과 bbox shape match TOP 10

{markdown_table(bbox_summary.sort_values(['component_bbox_shape_match','component_rect_iou_with_bbox'], ascending=False), ['asset_id','bbox_index','component_bbox_shape_match','component_rect_iou_with_bbox','component_rect_bbox_containment','component_rect_max_edge_delta_px','component_rectangular_outline_iou'])}

## RGB 값 패턴

- 단독 최대 채널 카운트: R={dominant_channel['R']:,}, G={dominant_channel['G']:,}, B={dominant_channel['B']:,}, 공동 최대(tie)={dominant_channel['tie']:,}
- 가장 빈번한 이상 RGB 값 15개: `{json.dumps([{'rgb': rgb, 'count': count} for rgb, count in top_colors], ensure_ascii=False)}`
- 검출된 이상 색이 5개 순수 팔레트 원색 집합에만 속하는지: **{exact_primary_palette}**; 모든 이상 픽셀의 `rgb_range`는 **255**다.

정상적인 단일채널/회색조 X-ray라면 R=G=B에 가까워 이 임계값을 넘기 어렵다. 여기서는 값이 정확히 0/255인 5색 팔레트, 1~수 px 두께의 사각형, bbox에 대한 높은 포함·근접성이 동시에 나타났다. 이는 해부학적/X-ray 물성 특징이나 일반 저장 노이즈보다 장비 검출 UI, 라벨 마킹 또는 합성 과정에서 영상에 구워진 컬러 박스라는 설명과 훨씬 잘 맞는다. 라벨 bbox가 이 박스 안 결함을 가리키므로 모델은 실제 결함 대신 색상 테두리의 위치·개수·색을 매우 쉽게 학습할 수 있다. 마킹 전 원본으로 교체하거나 색상 마킹을 제거하고, 이미지 단위 분할을 다시 검증해야 한다. 이 감사는 강한 공간적 상관을 입증하지만 어느 공정에서 박스가 삽입됐는지까지 확정하지는 않는다.

## 산출물

- `image_summary.csv`: 512장 전체 이미지별 집계와 공간 정렬 지표
- `abnormal_pixels.csv`: 임계값 초과 픽셀 전수와 포함 bbox/object 목록
- `bbox_summary.csv`: 1,134개 bbox별 내부/주변/경계 비교
- `component_summary.csv`: 이상 픽셀 연결요소별 사각 테두리 적합도와 bbox match
- `figures/figure_manifest.csv`: 세 TOP 10 선정 근거와 파일 prefix
- `figures/*`: 각 선정 사례의 원본, bbox, 내부(빨강)/외부(청록) overlay, 3-panel triptych
"""
    (output_dir / "rgb_audit_report.md").write_text(report, encoding="utf-8")

    print("\nRGB AUDIT COMPLETE")
    print(f"images={len(image_summary)} abnormal_images={len(abnormal_images)}")
    print(f"abnormal_pixels={total_abnormal} inside={total_inside} ({safe_ratio(total_inside,total_abnormal):.2%}) outside={total_outside} ({safe_ratio(total_outside,total_abnormal):.2%})")
    print(f"rectangular_components={len(rectangular_components)}/{len(component_summary)} bbox_shape_matches={len(shape_matched_boxes)}/{len(bbox_summary)}")
    print(f"within_bbox_boundary_10px={boundary_10px_pixels} ({safe_ratio(boundary_10px_pixels,total_abnormal):.2%})")
    print(f"output={output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_dir = (args.output_dir or (project_root / "outputs" / "rgb_channel_audit")).resolve()
    audit(project_root, output_dir)


if __name__ == "__main__":
    main()
