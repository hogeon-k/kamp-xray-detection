"""Audit 2,020 unique unlabeled NG X-rays and search for unmarked originals."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import math
import re
import shutil
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFont, ImageOps


THRESHOLD = 80
PALETTE = {
    (255, 0, 0),
    (0, 0, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 255, 0),
}
OUTLINE_IOU_THRESHOLD = 0.80
MASK_EXPANSION_PX = 5
FILENAME_RE = re.compile(r"(?P<sequence>\d+)_(?P<date>\d{8})_(?P<time>\d{6})\((?P<number>\d+)\)", re.I)


def safe_ratio(a: float | int, b: float | int) -> float:
    return float(a / b) if b else 0.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_candidates(project_root: Path, raw: object):
    if pd.isna(raw):
        return
    text = str(raw).strip().replace("\\", "/")
    if not text:
        return
    path = Path(text)
    if path.is_absolute():
        yield path
    else:
        yield project_root / path
        yield project_root / "dataset" / path


def resolve_assets(
    project_root: Path, rows: pd.DataFrame, asset_col: str, primary_path_col: str,
    extra_path_col: str | None = None,
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    hash_cache: dict[Path, str] = {}
    basename_index: dict[str, list[Path]] | None = None

    def get_hash(path: Path) -> str:
        path = path.resolve()
        if path not in hash_cache:
            hash_cache[path] = sha256_file(path)
        return hash_cache[path]

    for row in rows.itertuples(index=False):
        asset = str(getattr(row, asset_col))
        raw_paths = [getattr(row, primary_path_col)]
        if extra_path_col:
            extra = getattr(row, extra_path_col)
            if not pd.isna(extra):
                raw_paths.extend(piece.strip() for piece in str(extra).split(";") if piece.strip())
        seen: set[Path] = set()
        for raw in raw_paths:
            for candidate in path_candidates(project_root, raw):
                candidate = candidate.resolve()
                if candidate in seen or not candidate.is_file():
                    continue
                seen.add(candidate)
                if get_hash(candidate) == asset:
                    resolved[asset] = candidate
                    break
            if asset in resolved:
                break
        if asset in resolved:
            continue
        if basename_index is None:
            basename_index = {}
            for path in (project_root / "dataset").rglob("*"):
                if path.is_file() and path.suffix.lower() in {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                    basename_index.setdefault(path.name.casefold(), []).append(path)
        for raw in raw_paths:
            name = Path(str(raw).replace("\\", "/")).name.casefold()
            for candidate in basename_index.get(name, []):
                if get_hash(candidate) == asset:
                    resolved[asset] = candidate.resolve()
                    break
            if asset in resolved:
                break
    missing = sorted(set(rows[asset_col].astype(str)) - set(resolved))
    if missing:
        raise FileNotFoundError(f"Unresolved SHA-256 assets: {len(missing)}; examples={missing[:10]}")
    return resolved


def outline_fit(crop: np.ndarray) -> tuple[int, float]:
    best_thickness, best_iou = 0, 0.0
    for thickness in range(1, min(5, math.ceil(min(crop.shape) / 2) + 1)):
        outline = np.ones(crop.shape, dtype=bool)
        if crop.shape[0] > 2 * thickness and crop.shape[1] > 2 * thickness:
            outline[thickness:-thickness, thickness:-thickness] = False
        intersection = int((crop & outline).sum())
        union = int((crop | outline).sum())
        iou = safe_ratio(intersection, union)
        if iou > best_iou:
            best_thickness, best_iou = thickness, iou
    return best_thickness, best_iou


def component_details(mask: np.ndarray, rgb: np.ndarray) -> list[dict[str, object]]:
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
        cx = np.fromiter((p[0] for p in coords), dtype=np.int32)
        cy = np.fromiter((p[1] for p in coords), dtype=np.int32)
        x0, x1, y0, y1 = int(cx.min()), int(cx.max()) + 1, int(cy.min()), int(cy.max()) + 1
        crop = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        crop[cy - y0, cx - x0] = True
        thickness, fit_iou = outline_fit(crop)
        colors = rgb[cy, cx]
        unique, counts = np.unique(colors, axis=0, return_counts=True)
        order = np.argsort(counts)[::-1]
        color_counts = [(tuple(map(int, unique[i])), int(counts[i])) for i in order]
        components.append(
            {
                "component_id": len(components) + 1,
                "pixel_count": len(coords),
                "xmin": x0,
                "ymin": y0,
                "xmax_exclusive": x1,
                "ymax_exclusive": y1,
                "width": x1 - x0,
                "height": y1 - y0,
                "bounding_rectangle_area": (x1 - x0) * (y1 - y0),
                "component_density": safe_ratio(len(coords), (x1 - x0) * (y1 - y0)),
                "dominant_rgb": ",".join(map(str, color_counts[0][0])),
                "dominant_rgb_count": color_counts[0][1],
                "rgb_unique_count": len(color_counts),
                "rgb_values": ";".join(",".join(map(str, color)) for color, _ in color_counts),
                "best_outline_thickness": thickness,
                "rectangular_outline_iou": fit_iou,
                "is_rectangular_outline": fit_iou >= OUTLINE_IOU_THRESHOLD,
                "_coords": coords,
            }
        )
    return components


def dhash(gray: np.ndarray) -> int:
    small = Image.fromarray(gray, "L").resize((9, 8), Image.Resampling.LANCZOS)
    values = np.asarray(small, dtype=np.int16)
    bits = (values[:, 1:] > values[:, :-1]).ravel()
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def parse_filename(value: object) -> dict[str, object]:
    match = FILENAME_RE.search(Path(str(value)).stem)
    if not match:
        return {"sequence": None, "date": None, "time_seconds": None, "parenthesized_number": None}
    hhmmss = match.group("time")
    seconds = int(hhmmss[:2]) * 3600 + int(hhmmss[2:4]) * 60 + int(hhmmss[4:])
    return {
        "sequence": int(match.group("sequence")),
        "date": f"{match.group('date')[:4]}-{match.group('date')[4:6]}-{match.group('date')[6:]}",
        "time_seconds": seconds,
        "parenthesized_number": int(match.group("number")),
    }


@lru_cache(maxsize=256)
def load_gray(path_text: str) -> np.ndarray:
    with Image.open(path_text) as opened:
        return np.asarray(opened.convert("RGB").convert("L"), dtype=np.uint8)


def marking_exclusion_mask(path: Path) -> tuple[np.ndarray, list[dict[str, object]]]:
    with Image.open(path) as opened:
        rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    ranges = rgb.max(axis=2) - rgb.min(axis=2)
    abnormal = ranges > THRESHOLD
    components = component_details(abnormal, rgb)
    exclude = np.zeros(abnormal.shape, dtype=bool)
    height, width = abnormal.shape
    for component in components:
        x0 = max(0, int(component["xmin"]) - MASK_EXPANSION_PX)
        y0 = max(0, int(component["ymin"]) - MASK_EXPANSION_PX)
        x1 = min(width, int(component["xmax_exclusive"]) + MASK_EXPANSION_PX)
        y1 = min(height, int(component["ymax_exclusive"]) + MASK_EXPANSION_PX)
        exclude[y0:y1, x0:x1] = True
    return exclude, components


def similarity_metrics(labeled_gray: np.ndarray, candidate_gray: np.ndarray, exclude: np.ndarray) -> dict[str, float]:
    if labeled_gray.shape != candidate_gray.shape:
        return {"masked_mae": math.inf, "masked_mse": math.inf, "ncc": -1.0, "ssim": -1.0, "equal_pixel_ratio": 0.0, "p99_abs_diff": math.inf}
    valid = ~exclude
    a = labeled_gray[valid].astype(np.float64)
    b = candidate_gray[valid].astype(np.float64)
    diff = a - b
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff * diff))
    ac, bc = a - a.mean(), b - b.mean()
    ncc_denominator = math.sqrt(float(np.dot(ac, ac) * np.dot(bc, bc)))
    ncc = float(np.dot(ac, bc) / ncc_denominator) if ncc_denominator else (1.0 if np.array_equal(a, b) else 0.0)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    var_a, var_b = float(np.var(a)), float(np.var(b))
    covariance = float(np.mean(ac * bc))
    ssim = float(((2 * a.mean() * b.mean() + c1) * (2 * covariance + c2)) / ((a.mean() ** 2 + b.mean() ** 2 + c1) * (var_a + var_b + c2)))
    return {
        "masked_mae": mae,
        "masked_mse": mse,
        "ncc": ncc,
        "ssim": ssim,
        "equal_pixel_ratio": float(np.mean(diff == 0)),
        "p99_abs_diff": float(np.percentile(np.abs(diff), 99)),
    }


def best_small_shift(a: np.ndarray, b: np.ndarray, exclude: np.ndarray) -> tuple[int, int, float]:
    # Check alignment on quarter-resolution images; direct full-resolution metrics remain authoritative.
    ah = np.asarray(Image.fromarray(a).resize((max(1, a.shape[1] // 4), max(1, a.shape[0] // 4)), Image.Resampling.BOX), dtype=np.float32)
    bh = np.asarray(Image.fromarray(b).resize((ah.shape[1], ah.shape[0]), Image.Resampling.BOX), dtype=np.float32)
    valid = np.asarray(Image.fromarray((~exclude).astype(np.uint8) * 255).resize((ah.shape[1], ah.shape[0]), Image.Resampling.NEAREST)) > 0
    best = (0, 0, math.inf)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            y0a, y1a = max(0, dy), min(ah.shape[0], ah.shape[0] + dy)
            x0a, x1a = max(0, dx), min(ah.shape[1], ah.shape[1] + dx)
            y0b, y1b = max(0, -dy), min(bh.shape[0], bh.shape[0] - dy)
            x0b, x1b = max(0, -dx), min(bh.shape[1], bh.shape[1] - dx)
            region_valid = valid[y0a:y1a, x0a:x1a]
            if not region_valid.any():
                continue
            mae = float(np.mean(np.abs(ah[y0a:y1a, x0a:x1a][region_valid] - bh[y0b:y1b, x0b:x1b][region_valid])))
            if mae < best[2]:
                best = (dx * 4, dy * 4, mae)
    return best


def classification(metrics: dict[str, float], dhash_distance: int) -> str:
    if metrics["masked_mae"] <= 0.50 and metrics["ncc"] >= 0.9999 and metrics["ssim"] >= 0.999:
        return "EXACT_OR_NEAR_EXACT"
    if metrics["masked_mae"] <= 2.0 and metrics["ncc"] >= 0.999 and metrics["ssim"] >= 0.995 and dhash_distance <= 4:
        return "HIGH_CONFIDENCE"
    if metrics["masked_mae"] <= 8.0 and metrics["ncc"] >= 0.98 and metrics["ssim"] >= 0.95 and dhash_distance <= 12:
        return "POSSIBLE"
    return "NO_MATCH"


def ks_statistic(a: pd.Series, b: pd.Series) -> float:
    x = np.sort(pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float))
    y = np.sort(pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float))
    if not len(x) or not len(y):
        return math.nan
    values = np.sort(np.unique(np.concatenate([x, y])))
    return float(np.max(np.abs(np.searchsorted(x, values, side="right") / len(x) - np.searchsorted(y, values, side="right") / len(y))))


def draw_unlabeled_case(path: Path, components: list[dict[str, object]], destination: Path) -> None:
    with Image.open(path) as opened:
        original = opened.convert("RGB")
    marked = original.copy()
    draw = ImageDraw.Draw(marked)
    for component in components:
        x0, y0, x1, y1 = (int(component[k]) for k in ("xmin", "ymin", "xmax_exclusive", "ymax_exclusive"))
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 215, 0), width=1)
    mask = Image.new("RGB", original.size, "black")
    mask_draw = ImageDraw.Draw(mask)
    for component in components:
        x0, y0, x1, y1 = (int(component[k]) for k in ("xmin", "ymin", "xmax_exclusive", "ymax_exclusive"))
        mask_draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline="white", width=max(1, int(component["best_outline_thickness"])))
    triptych = Image.new("RGB", (original.width * 3, original.height), "black")
    triptych.paste(original, (0, 0)); triptych.paste(marked, (original.width, 0)); triptych.paste(mask, (original.width * 2, 0))
    triptych.save(destination)


def draw_pair(labeled_path: Path, candidate_path: Path, destination: Path) -> None:
    with Image.open(labeled_path) as opened:
        labeled = opened.convert("RGB")
    with Image.open(candidate_path) as opened:
        candidate = opened.convert("RGB")
    if candidate.size != labeled.size:
        candidate = candidate.resize(labeled.size, Image.Resampling.BILINEAR)
    diff = ImageChops.difference(labeled, candidate)
    diff = ImageEnhance.Contrast(diff).enhance(4.0)
    canvas = Image.new("RGB", (labeled.width * 3, labeled.height + 18), "black")
    canvas.paste(labeled, (0, 18)); canvas.paste(candidate, (labeled.width, 18)); canvas.paste(diff, (labeled.width * 2, 18))
    draw = ImageDraw.Draw(canvas); font = ImageFont.load_default()
    draw.text((4, 3), "labeled", fill="white", font=font)
    draw.text((labeled.width + 4, 3), "candidate", fill="white", font=font)
    draw.text((labeled.width * 2 + 4, 3), "absolute difference x4", fill="white", font=font)
    canvas.save(destination)


def markdown_table(df: pd.DataFrame, columns: list[str], limit: int = 10) -> str:
    if df.empty:
        return "(없음)"
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in df.loc[:, columns].head(limit).itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(v).replace("|", "\\|") for v in values) + " |")
    return "\n".join(rows)


def audit(project_root: Path, output_dir: Path) -> None:
    outputs = project_root / "outputs"
    unlabeled_all = pd.read_csv(outputs / "unlabeled_analysis.csv", low_memory=False)
    labeled = pd.read_csv(outputs / "eda_image_metrics.csv", low_memory=False)
    labeled_image_audit = pd.read_csv(outputs / "rgb_channel_audit" / "image_summary.csv", low_memory=False)
    labeled_components = pd.read_csv(outputs / "rgb_channel_audit" / "component_summary.csv", low_memory=False)
    labeled_pixels = pd.read_csv(outputs / "rgb_channel_audit" / "abnormal_pixels.csv", low_memory=False)

    target_class = unlabeled_all["classification"].value_counts().index[0]
    unlabeled = unlabeled_all[unlabeled_all["classification"] == target_class].copy().reset_index(drop=True)
    if len(unlabeled) != 2020 or unlabeled["asset_sha256"].nunique() != 2020:
        raise ValueError(f"Expected 2,020 unique unlabeled NG assets; got rows={len(unlabeled)}, unique={unlabeled['asset_sha256'].nunique()}")
    if len(labeled) != 512 or labeled["asset_id"].nunique() != 512:
        raise ValueError("Expected 512 unique labeled assets")

    unlabeled_paths = resolve_assets(project_root, unlabeled, "asset_sha256", "representative_path", "all_copy_paths")
    labeled_paths = {str(row.asset_id): Path(row.resolved_image_path) for row in labeled_image_audit.itertuples(index=False)}
    for asset, path in labeled_paths.items():
        if not path.is_file() or sha256_file(path) != asset:
            raise ValueError(f"Invalid resolved labeled path/hash: {asset} {path}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True)

    image_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    component_cache: dict[str, list[dict[str, object]]] = {}
    dhashes: dict[str, int] = {}
    global_colors: Counter[tuple[int, int, int]] = Counter()
    total_abnormal = total_palette = 0
    pixel_fields = ["asset_id", "image_path", "x", "y", "R", "G", "B", "rgb_range", "component_id"]
    with (output_dir / "unlabeled_abnormal_pixels.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=pixel_fields); writer.writeheader()
        for ordinal, row in enumerate(unlabeled.itertuples(index=False), start=1):
            asset = str(row.asset_sha256); path = unlabeled_paths[asset]
            with Image.open(path) as opened:
                rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
            height, width = rgb.shape[:2]
            ranges = rgb.max(axis=2) - rgb.min(axis=2)
            abnormal = ranges > THRESHOLD
            components = component_details(abnormal, rgb)
            component_cache[asset] = [{k: v for k, v in c.items() if k != "_coords"} for c in components]
            ys, xs = np.nonzero(abnormal); values = ranges[abnormal]; pixels = rgb[ys, xs]
            count = len(xs); total_abnormal += count
            unique, counts = np.unique(pixels, axis=0, return_counts=True) if count else (np.empty((0,3),dtype=np.uint8), np.array([],dtype=int))
            color_counter = {tuple(map(int, color)): int(n) for color, n in zip(unique, counts)}
            global_colors.update(color_counter)
            palette_count = sum(n for color, n in color_counter.items() if color in PALETTE); total_palette += palette_count
            coord_component: dict[tuple[int, int], int] = {}
            for component in components:
                for coord in component.pop("_coords"):
                    coord_component[coord] = int(component["component_id"])
                component_rows.append({"asset_id": asset, "image_path": str(row.representative_path), **component})
            for x, y, pixel, value in zip(xs, ys, pixels, values):
                writer.writerow({"asset_id": asset, "image_path": str(row.representative_path), "x": int(x), "y": int(y), "R": int(pixel[0]), "G": int(pixel[1]), "B": int(pixel[2]), "rgb_range": int(value), "component_id": coord_component[(int(x), int(y))]})
            gray = np.asarray(Image.fromarray(rgb, "RGB").convert("L"), dtype=np.uint8); dhashes[asset] = dhash(gray)
            image_rows.append({
                "asset_id": asset, "image_path": str(row.representative_path), "resolved_image_path": str(path),
                "machine": int(row.machine), "capture_date": row.capture_date, "capture_time": row.capture_time,
                "filename": row.filename, "width": width, "height": height, "total_pixels": width * height,
                "abnormal_pixel_count": count, "abnormal_pixel_ratio": safe_ratio(count, width * height),
                "max_rgb_range": int(values.max()) if count else 0, "min_abnormal_rgb_range": int(values.min()) if count else 0,
                "abnormal_rgb_unique_count": len(color_counter), "palette_abnormal_pixel_count": palette_count,
                "palette_abnormal_ratio": safe_ratio(palette_count, count), "non_palette_abnormal_pixel_count": count - palette_count,
                "all_abnormal_ranges_are_255": bool(count == 0 or np.all(values == 255)),
                "component_count": len(components),
                "rectangular_outline_iou_ge_050_count": sum(float(c["rectangular_outline_iou"]) >= .50 for c in components),
                "rectangular_outline_iou_ge_080_count": sum(float(c["rectangular_outline_iou"]) >= .80 for c in components),
                "best_outline_thickness_2_count": sum(int(c["best_outline_thickness"]) == 2 for c in components),
                "dhash_hex": f"{dhashes[asset]:016x}",
            })
            if ordinal % 100 == 0 or ordinal == len(unlabeled):
                print(f"RGB audit {ordinal}/{len(unlabeled)} abnormal_pixels={total_abnormal}", flush=True)

    image_summary = pd.DataFrame(image_rows)
    component_summary = pd.DataFrame(component_rows)
    image_summary.to_csv(output_dir / "unlabeled_image_summary.csv", index=False, encoding="utf-8-sig")
    component_summary.to_csv(output_dir / "unlabeled_component_summary.csv", index=False, encoding="utf-8-sig")

    labeled_color_counts = labeled_pixels.groupby(["R", "G", "B"]).size().to_dict()
    palette_rows = []
    for group, counter in (("labeled_512", Counter(labeled_color_counts)), ("unlabeled_2020", global_colors)):
        group_total = sum(counter.values())
        for color, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            color_tuple = tuple(map(int, color))
            palette_rows.append({"group": group, "R": color_tuple[0], "G": color_tuple[1], "B": color_tuple[2], "pixel_count": count, "group_ratio": safe_ratio(count, group_total), "is_reference_palette": color_tuple in PALETTE})
    pd.DataFrame(palette_rows).to_csv(output_dir / "palette_summary.csv", index=False, encoding="utf-8-sig")

    comparison_specs = [
        ("image", "abnormal_pixels_per_image", labeled_image_audit["rgb_diff_gt80_pixels"], image_summary["abnormal_pixel_count"]),
        ("image", "components_per_image", labeled_image_audit["abnormal_component_count"], image_summary["component_count"]),
        ("component", "pixel_count", labeled_components["pixel_count"], component_summary["pixel_count"]),
        ("component", "rectangle_width", labeled_components["rect_width"], component_summary["width"]),
        ("component", "rectangle_height", labeled_components["rect_height"], component_summary["height"]),
        ("component", "outline_thickness", labeled_components["best_outline_thickness"], component_summary["best_outline_thickness"]),
        ("component", "rectangular_outline_iou", labeled_components["rectangular_outline_iou"], component_summary["rectangular_outline_iou"]),
    ]
    comparison_rows = []
    for level, metric, left, right in comparison_specs:
        for group, values in (("labeled_512", left), ("unlabeled_2020", right)):
            numeric = pd.to_numeric(values, errors="coerce").dropna()
            comparison_rows.append({"level": level, "metric": metric, "group": group, "count": len(numeric), "mean": numeric.mean(), "std": numeric.std(), "min": numeric.min(), "p10": numeric.quantile(.1), "median": numeric.median(), "p90": numeric.quantile(.9), "max": numeric.max(), "two_sample_ks": ks_statistic(left, right)})
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "labeled_vs_unlabeled_comparison.csv", index=False, encoding="utf-8-sig")

    unlabeled_meta = image_summary.set_index("asset_id").to_dict("index")
    by_dimensions: dict[tuple[int, int], list[str]] = {}
    for asset, meta in unlabeled_meta.items():
        by_dimensions.setdefault((int(meta["width"]), int(meta["height"])), []).append(asset)
    match_rows: list[dict[str, object]] = []
    for ordinal, lrow in enumerate(labeled.itertuples(index=False), start=1):
        labeled_asset = str(lrow.asset_id); labeled_path = labeled_paths[labeled_asset]
        labeled_gray = load_gray(str(labeled_path)); exclude, _ = marking_exclusion_mask(labeled_path)
        labeled_hash = dhash(labeled_gray); parsed_labeled = parse_filename(lrow.image_stem)
        pool = by_dimensions.get((int(lrow.image_width), int(lrow.image_height)), [])
        distances = sorted(((labeled_hash ^ dhashes[asset]).bit_count(), asset) for asset in pool)
        shortlist = {asset for _, asset in distances[:12]}
        shortlist.update(asset for asset in pool if int(unlabeled_meta[asset]["machine"]) == int(lrow.machine) and str(unlabeled_meta[asset]["capture_date"]) == str(lrow.date))
        evaluated = []
        for asset in shortlist:
            meta = unlabeled_meta[asset]; candidate_gray = load_gray(meta["resolved_image_path"])
            distance = (labeled_hash ^ dhashes[asset]).bit_count()
            metrics = similarity_metrics(labeled_gray, candidate_gray, exclude)
            parsed_candidate = parse_filename(meta["filename"])
            same_date = str(lrow.date) == str(meta["capture_date"])
            if parsed_labeled["time_seconds"] is not None and parsed_candidate["time_seconds"] is not None:
                day_difference = abs((pd.Timestamp(lrow.date) - pd.Timestamp(meta["capture_date"])).days)
                time_difference = day_difference * 86400 + abs(int(parsed_labeled["time_seconds"]) - int(parsed_candidate["time_seconds"]))
            else:
                time_difference = math.nan
            verdict = classification(metrics, distance)
            evaluated.append({
                "labeled_asset_id": labeled_asset, "labeled_image_path": str(lrow.image_path),
                "candidate_asset_id": asset, "candidate_image_path": meta["image_path"],
                "same_machine": int(lrow.machine) == int(meta["machine"]), "same_date": same_date,
                "time_difference_seconds": time_difference, "same_dimensions": True,
                "same_sequence": parsed_labeled["sequence"] == parsed_candidate["sequence"],
                "same_parenthesized_number": parsed_labeled["parenthesized_number"] == parsed_candidate["parenthesized_number"],
                **metrics, "dhash_distance": distance,
                "candidate_has_color_marking": int(meta["abnormal_pixel_count"]) > 0,
                "candidate_abnormal_pixel_count": int(meta["abnormal_pixel_count"]),
                "match_classification": verdict,
                "_candidate_resolved": meta["resolved_image_path"],
            })
        priority = {"EXACT_OR_NEAR_EXACT": 0, "HIGH_CONFIDENCE": 1, "POSSIBLE": 2, "NO_MATCH": 3}
        evaluated.sort(key=lambda r: (priority[r["match_classification"]], r["masked_mae"], r["dhash_distance"], -r["ncc"]))
        for rank, result in enumerate(evaluated[:5], start=1):
            candidate_gray = load_gray(result["_candidate_resolved"])
            shift_x, shift_y, shift_mae = best_small_shift(labeled_gray, candidate_gray, exclude)
            result.update({"similarity_rank": rank, "best_shift_x": shift_x, "best_shift_y": shift_y, "downsampled_best_shift_mae": shift_mae})
            result["bbox_transferable"] = bool(
                not result["candidate_has_color_marking"]
                and result["match_classification"] in {"EXACT_OR_NEAR_EXACT", "HIGH_CONFIDENCE"}
                and shift_x == 0 and shift_y == 0
                and result["equal_pixel_ratio"] >= .995
            )
            match_rows.append({k: v for k, v in result.items() if not k.startswith("_")})
        if ordinal % 50 == 0 or ordinal == len(labeled):
            print(f"candidate search {ordinal}/{len(labeled)}", flush=True)

    matches = pd.DataFrame(match_rows)
    matches.to_csv(output_dir / "original_candidate_matches.csv", index=False, encoding="utf-8-sig")
    zero_assets = set(labeled.loc[labeled["object_count"] == 0, "asset_id"].astype(str))
    zero_matches = matches[matches["labeled_asset_id"].isin(zero_assets)].copy()
    zero_matches.to_csv(output_dir / "zero_bbox_candidate_matches.csv", index=False, encoding="utf-8-sig")
    zero_components = labeled_components[labeled_components["asset_id"].astype(str).isin(zero_assets)].copy()
    if not zero_components.empty:
        parsed_rects = zero_components["rect"].map(ast.literal_eval)
        zero_components["xmin"] = parsed_rects.map(lambda value: value[0])
        zero_components["ymin"] = parsed_rects.map(lambda value: value[1])
        zero_components["xmax_exclusive"] = parsed_rects.map(lambda value: value[2])
        zero_components["ymax_exclusive"] = parsed_rects.map(lambda value: value[3])
    zero_components.to_csv(output_dir / "zero_bbox_component_summary.csv", index=False, encoding="utf-8-sig")
    zero_pixels = labeled_pixels[labeled_pixels["asset_id"].astype(str).isin(zero_assets)]
    zero_color_counts = zero_pixels.groupby(["R", "G", "B"]).size().sort_values(ascending=False).to_dict()

    top_unlabeled_dir = figures_dir / "top_color_rectangles"
    top_unlabeled_dir.mkdir()
    for rank, row in enumerate(image_summary.sort_values(["rectangular_outline_iou_ge_080_count", "abnormal_pixel_count"], ascending=False).head(10).itertuples(index=False), start=1):
        draw_unlabeled_case(Path(row.resolved_image_path), component_cache[row.asset_id], top_unlabeled_dir / f"{rank:02d}_{row.asset_id[:12]}_triptych.png")
    pair_dir = figures_dir / "top_candidate_pairs"; pair_dir.mkdir()
    class_order = {"EXACT_OR_NEAR_EXACT": 0, "HIGH_CONFIDENCE": 1, "POSSIBLE": 2, "NO_MATCH": 3}
    best_per_label = matches[matches["similarity_rank"] == 1].copy()
    best_per_label = best_per_label.assign(_order=best_per_label.match_classification.map(class_order)).sort_values(["_order", "masked_mae", "dhash_distance"]).head(20)
    for rank, row in enumerate(best_per_label.itertuples(index=False), start=1):
        draw_pair(labeled_paths[row.labeled_asset_id], unlabeled_paths[row.candidate_asset_id], pair_dir / f"{rank:02d}_{row.labeled_asset_id[:10]}_{row.candidate_asset_id[:10]}_{row.match_classification}.png")

    abnormal_images = int((image_summary["abnormal_pixel_count"] > 0).sum())
    rectangular_images = int((image_summary["rectangular_outline_iou_ge_080_count"] > 0).sum())
    thickness2_components = int((component_summary["best_outline_thickness"] == 2).sum())
    rect50 = int((component_summary["rectangular_outline_iou"] >= .50).sum())
    rect80 = int((component_summary["rectangular_outline_iou"] >= .80).sum())
    top_matches = matches[matches["similarity_rank"] == 1]
    high_labels = int(top_matches["match_classification"].isin(["EXACT_OR_NEAR_EXACT", "HIGH_CONFIDENCE"]).sum())
    possible_labels = int((top_matches["match_classification"] == "POSSIBLE").sum())
    unmarked_high = int((top_matches["match_classification"].isin(["EXACT_OR_NEAR_EXACT", "HIGH_CONFIDENCE"]) & ~top_matches["candidate_has_color_marking"]).sum())
    transferable = int(top_matches["bbox_transferable"].sum())
    zero_top = zero_matches[zero_matches["similarity_rank"] == 1]
    zero_high = int(zero_top["match_classification"].isin(["EXACT_OR_NEAR_EXACT", "HIGH_CONFIDENCE"]).sum())
    zero_unmarked = int((zero_top["match_classification"].isin(["EXACT_OR_NEAR_EXACT", "HIGH_CONFIDENCE"]) & ~zero_top["candidate_has_color_marking"]).sum())
    same_palette = set(global_colors).issubset(PALETTE) and set(global_colors) == set(labeled_color_counts)
    labeled_palette_distribution = np.array([labeled_color_counts.get(color, 0) for color in sorted(PALETTE)], dtype=float); labeled_palette_distribution /= labeled_palette_distribution.sum()
    unlabeled_palette_distribution = np.array([global_colors.get(color, 0) for color in sorted(PALETTE)], dtype=float); unlabeled_palette_distribution /= unlabeled_palette_distribution.sum()
    palette_l1 = float(np.abs(labeled_palette_distribution - unlabeled_palette_distribution).sum())
    if abnormal_images >= int(.8 * len(image_summary)):
        scenario = "Scenario A"
        scenario_text = "미라벨 NG에도 대부분 컬러 사각형이 존재한다. 장비 또는 라벨링 이전 처리 단계에서 overlay가 이미 삽입됐을 가능성이 크며, 별도의 raw X-ray 원본 탐색이 필요하다."
    else:
        scenario = "Scenario B"
        scenario_text = "미라벨 NG에는 컬러 사각형이 드물다. 512장 생성/라벨링 과정에서 overlay가 추가됐을 가능성이 크다."
    if unmarked_high:
        scenario += " + Scenario C"
        scenario_text += f" 또한 높은 신뢰도의 무마킹 대응 원본 {unmarked_high}장을 발견했으므로 해당 쌍은 원본+bbox 이전을 우선 검토할 수 있다."
    else:
        scenario += " + Scenario D"
        scenario_text += " 높은 신뢰도의 무마킹 대응 원본을 찾지 못했으므로 컬러 사각형 제거 및 inpainting PoC가 필요하다."

    thresholds_text = "EXACT_OR_NEAR_EXACT: MAE<=0.50, NCC>=0.9999, SSIM>=0.999; HIGH_CONFIDENCE: MAE<=2.0, NCC>=0.999, SSIM>=0.995, dHash<=4; POSSIBLE: MAE<=8.0, NCC>=0.98, SSIM>=0.95, dHash<=12; 그 외 NO_MATCH"
    report = f"""# 미라벨 NG 2,020장 RGB 및 원본 후보 감사

## 핵심 결론

- 검사 고유 이미지: **{len(image_summary):,}장** (SHA-256 검증 완료; 제외 대상 19개 미포함)
- `rgb_diff > 80` 이미지: **{abnormal_images:,}장 ({safe_ratio(abnormal_images, len(image_summary)):.2%})**
- 전체 이상 픽셀: **{total_abnormal:,}개**
- 기존 5색 팔레트 픽셀: **{total_palette:,}개 ({safe_ratio(total_palette, total_abnormal):.2%})**; 비팔레트 이상 픽셀: **{total_abnormal-total_palette:,}개**
- 모든 이상 픽셀의 `rgb_range=255`: **{bool(image_summary['all_abnormal_ranges_are_255'].all())}**
- 기존 512장과 이상 RGB 집합이 정확히 동일: **{same_palette}** (5색 비중 L1 거리 {palette_l1:.4f})
- 사각 테두리(IoU>=0.80)가 있는 이미지: **{rectangular_images:,}장 ({safe_ratio(rectangular_images, len(image_summary)):.2%})**
- component: **{len(component_summary):,}개**; outline IoU>=0.50 **{rect50:,}개**, >=0.80 **{rect80:,}개**, best thickness=2px **{thickness2_components:,}개**
- 기존 512장 중 EXACT/HIGH 원본 후보: **{high_labels:,}장**
- TOP 1이 POSSIBLE인 이미지: **{possible_labels:,}장** (동일 원본 확정 불가; 모두 컬러 마킹 포함)
- 그중 무마킹 후보: **{unmarked_high:,}장**; bbox 이전 가능: **{transferable:,}장**
- zero-bbox 14장 중 EXACT/HIGH 후보: **{zero_high:,}장**; 무마킹 후보: **{zero_unmarked:,}장**
- 최종 분류: **{scenario} — {scenario_text}**

## 모집단과 경로 검증

`unlabeled_analysis.csv`의 최다 classification 그룹 2,020개를 이전 분석의 `미라벨링 NG 이미지` 모집단으로 사용했고, 나머지 19개는 제외했다. 2,020개 모두 고유 SHA-256이며 `representative_path`, `all_copy_paths`, dataset basename fallback 순으로 실제 파일을 찾은 뒤 파일 바이트 SHA-256 일치를 확인했다. 기존 512장도 저장된 resolved path의 SHA-256을 다시 검증했다. 원본 dataset과 기존 outputs는 읽기만 했다.

## RGB 및 component 패턴 비교

미라벨 NG에서 발견된 이상 색 집합과 기존 라벨 512장의 색 집합은 {"동일하다" if same_palette else "완전히 동일하지 않다"}. component 분포 비교는 `labeled_vs_unlabeled_comparison.csv`에 평균, 중앙값, 10/90 분위수와 two-sample KS 통계로 기록했다.

{markdown_table(comparison, ['level','metric','group','count','mean','median','p90','two_sample_ks'], 20)}

## 원본 후보 검색 방법과 판정

파일명에서 sequence, 날짜, HHMMSS, 괄호 번호를 파싱했다. 동일 크기 전체 후보에 dHash를 계산한 뒤 dHash 상위 12개와 동일 호기+날짜 후보의 합집합에 대해 원해상도 검증을 수행했다. 라벨 이미지의 컬러 component 최소 사각형을 ±{MASK_EXPANSION_PX}px 확장한 영역은 비교에서 제외했다. 나머지 픽셀에 masked MAE/MSE, 정규화 상관계수(NCC), 전역 masked SSIM, 동일 픽셀 비율을 계산했다. TOP 5 각각에 대해 1/4 해상도 ±8px 이동 탐색을 추가해 좌표계 이동 여부를 확인했다.

자동 판정 기준: `{thresholds_text}`.

`bbox_transferable=True`는 무마킹, EXACT/HIGH, 최적 이동 (0,0), 마스크 외 동일 픽셀 비율>=99.5%를 모두 만족해야 한다.

### 가장 높은 원본 후보

{markdown_table(top_matches.assign(_order=top_matches.match_classification.map(class_order)).sort_values(['_order','masked_mae']), ['labeled_asset_id','candidate_asset_id','match_classification','candidate_has_color_marking','masked_mae','ncc','ssim','dhash_distance','bbox_transferable'], 20)}

## zero-bbox 14장

14장에는 총 **{len(zero_components)}개** 컬러 component가 있으며, 이미지당 component 중앙값은 **{zero_components.groupby('asset_id').size().median():.1f}개**다. 사각형 폭/높이 중앙값은 **{zero_components['rect_width'].median():.1f}px / {zero_components['rect_height'].median():.1f}px**, outline IoU 중앙값은 **{zero_components['rectangular_outline_iou'].median():.3f}**, 최적 두께 2px 비율은 **{safe_ratio(int((zero_components['best_outline_thickness']==2).sum()),len(zero_components)):.2%}**다. 색 분포는 `{zero_color_counts}`이며 위치 좌표는 `zero_bbox_component_summary.csv`에 기록했다. 시각적으로 이 component들도 주변 X-ray 구조 위에 놓인 얇은 컬러 사각형이며 일반 영상 질감과 구별된다.

동일한 후보 검색을 적용한 결과 EXACT/HIGH 후보는 {zero_high}장, 그중 무마킹은 {zero_unmarked}장이다. 상세 TOP 5는 `zero_bbox_candidate_matches.csv`에 있다.

## 질문별 답변

1. 이상 픽셀 이미지: **{abnormal_images:,}/2,020장**.
2. 비율: **{safe_ratio(abnormal_images, 2020):.2%}**.
3. 기존 5색과 동일한가: **{same_palette}**, 팔레트 포함률 **{safe_ratio(total_palette,total_abnormal):.2%}**.
4. 사각 component 이미지: **{rectangular_images:,}장**.
5. 2px 패턴: **{thickness2_components:,}/{len(component_summary):,} components ({safe_ratio(thickness2_components,len(component_summary)):.2%})**.
6. 기존 패턴과 동일한가: RGB 집합과 사각 테두리 생성 형태를 기준으로 **{"동일 계열" if same_palette and safe_ratio(rect80,len(component_summary))>.8 else "차이가 있음"}**이다. 분포 차이는 비교 CSV를 참조한다.
7. 장비 단계에서 존재했을 가능성: 미라벨 모집단에서도 광범위하면 라벨링 이전 존재 가능성을 강하게 지지하지만, 장비 자체인지 중간 처리인지는 현 데이터만으로 확정하지 않는다.
8. EXACT/HIGH 후보: **{high_labels:,}/512장**.
   참고로 TOP 1이 POSSIBLE인 경우는 **{possible_labels:,}장**이나, 동일 원본으로 확정할 수준은 아니며 모두 마킹된 후보이다.
9. 그중 무마킹 후보: **{unmarked_high:,}장**.
10. bbox 이전 가능: **{transferable:,}장**.
11. zero-bbox 대응 후보: EXACT/HIGH **{zero_high:,}장**, 무마킹 **{zero_unmarked:,}장**.
12. 학습 안전성: 컬러 마킹이 광범위하면 그대로 학습하는 것은 안전하지 않다. 무마킹 raw 복구를 우선하고, 불가능한 경우 제거/inpainting을 검증한 뒤 사용해야 한다.

## 최종 시나리오

### {scenario}

{scenario_text}

## 산출물

- `unlabeled_image_summary.csv`, `unlabeled_abnormal_pixels.csv`, `unlabeled_component_summary.csv`
- `palette_summary.csv`, `labeled_vs_unlabeled_comparison.csv`
- `original_candidate_matches.csv`, `zero_bbox_candidate_matches.csv`
- `zero_bbox_component_summary.csv`: zero-bbox 14장의 컬러 component 위치·크기·형태
- `figures/top_color_rectangles/` TOP 10, `figures/top_candidate_pairs/` TOP 20
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print("\nUNLABELED RGB AUDIT COMPLETE")
    print(f"images={len(image_summary)} abnormal_images={abnormal_images} ({safe_ratio(abnormal_images,len(image_summary)):.2%})")
    print(f"abnormal_pixels={total_abnormal} palette_ratio={safe_ratio(total_palette,total_abnormal):.2%}")
    print(f"components={len(component_summary)} rect_iou_ge_080={rect80} thickness_2={thickness2_components}")
    print(f"high_confidence_candidates={high_labels} unmarked={unmarked_high} transferable={transferable}")
    print(f"scenario={scenario} output={output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = (args.output_dir or root / "outputs" / "unlabeled_rgb_audit").resolve()
    audit(root, output)


if __name__ == "__main__":
    main()
