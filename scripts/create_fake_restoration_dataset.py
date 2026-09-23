"""Create a separate augmented copy of the existing manifest-based YOLO PoC."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw

from create_yolo_poc_20 import read_csv, sha256, parse_yolo
from fake_restoration import RestorationSettings, augment_fake_restoration, inpaint_rgb

ROOT = Path(__file__).resolve().parents[1]


def load_rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def draw_debug(rgb, real, fake, boxes, title, destination):
    overlay = rgb.copy()
    for mask, color in ((real, (255, 80, 80)), (fake, (70, 255, 70))):
        active = mask > 0
        overlay[active] = (overlay[active]*0.25 + np.array(color)*0.75).astype(np.uint8)
    canvas = Image.new("RGB", (rgb.shape[1], rgb.shape[0]+48), "#151515")
    canvas.paste(Image.fromarray(overlay), (0, 48))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), title, fill="white")
    draw.text((8, 23), "GT: cyan | Real: red | Fake: green", fill="white")
    for x0, y0, x1, y1 in boxes:
        draw.rectangle((x0, y0+48, x1, y1+48), outline="cyan", width=2)
    canvas.save(destination)


def generate(source, poc, output, debug, *, seed=42, safety_margin=10,
             scale_range=(1., 1.), max_attempts=500, debug_samples=20):
    source, poc, output, debug = [Path(p).resolve() for p in (source, poc, output, debug)]
    # Outputs must be new, separate trees, including when paths resolve via symlinks.
    for target in (output, debug):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite: {target}")
        if any(target == p or p in target.parents or target in p.parents for p in (source, poc, ROOT/'dataset')):
            raise ValueError(f"Output overlaps protected input tree: {target}")
    if output in debug.parents or debug in output.parents or output == debug:
        raise ValueError("Dataset and debug directories must be separate")
    if seed < 0 or safety_margin < 0 or not np.isfinite(safety_margin) or debug_samples < 0 or max_attempts < 1:
        raise ValueError("Invalid seed, safety margin, debug count or attempt limit")
    if not 0.8 <= scale_range[0] <= scale_range[1] <= 1.2:
        raise ValueError("Scale range must be within [0.8, 1.2]")
    manifest_path = source/'manifest.csv'
    rows = sorted(read_csv(manifest_path), key=lambda row: (row['split'], row['asset_id']))
    if not rows or len({row['asset_id'] for row in rows}) != len(rows):
        raise ValueError("Manifest must contain unique assets")
    protected = {manifest_path: sha256(manifest_path)}
    items, names = [], set()
    for row in rows:
        split = row['split']
        if split not in {'train', 'val', 'test'}:
            raise ValueError(f"Unknown split: {split}")
        image_path = source/'images'/split/Path(row['output_image_path']).name
        label_path = source/'labels'/split/Path(row['output_label_path']).name
        if (split, image_path.name.casefold()) in names:
            raise ValueError("Duplicate image destination")
        names.add((split, image_path.name.casefold()))
        restored = Path(row['restored_image_path'])
        match = re.fullmatch(r'(.+)_(telea|ns)_d(\d+)_r(\d+)\.png', restored.name)
        if not match:
            raise ValueError(f"Cannot establish restoration method from {restored}")
        sample, method, dilation, radius = match.groups()
        settings = RestorationSettings(method.upper(), float(radius), int(dilation))
        mask_path = poc/'masks'/f'{sample}_d{dilation}.png'
        original = Path(row['original_image_path'])
        for path in (image_path, label_path, mask_path, original, restored):
            protected[path] = sha256(path)
        if protected[original] != row['original_sha256'] or protected[image_path] != row['output_sha256'] or protected[restored] != row['restored_sha256']:
            raise ValueError(f"Input provenance mismatch: {image_path}")
        rgb = load_rgb(image_path)
        with Image.open(mask_path) as image:
            real = np.array(image.convert('L'))
        if real.shape != rgb.shape[:2] or not set(np.unique(real)).issubset({0, 255}):
            raise ValueError(f"Invalid saved restoration mask: {mask_path}")
        # Verify method, radius, mask and color conversion against actual baseline pixels.
        if not np.array_equal(inpaint_rgb(load_rgb(original), real, settings), rgb):
            raise ValueError(f"Restoration replay differs from baseline: {image_path}")
        boxes = parse_yolo(label_path, rgb.shape[1], rgb.shape[0])
        if len(boxes) != int(row['bbox_count']):
            raise ValueError(f"GT count mismatch: {label_path}")
        items.append((row, image_path, label_path, rgb, real, boxes, settings, mask_path))
    # No writes until provenance and all baseline restorations have been verified.
    output.mkdir(parents=True, exist_ok=False)
    debug.mkdir(parents=True, exist_ok=False)
    for split in ('train', 'val', 'test'):
        for kind in ('images', 'labels'):
            (output/kind/split).mkdir(parents=True)
    rng = np.random.default_rng(seed)
    records, augmented_manifest = [], []
    for index, (row, image_path, label_path, rgb, real, boxes, settings, mask_path) in enumerate(items):
        result, fake, audit = augment_fake_restoration(
            rgb, real, boxes, split=row['split'], rng=rng, settings=settings,
            safety_margin=safety_margin, scale_range=scale_range, max_attempts=max_attempts)
        image_out = output/'images'/row['split']/image_path.name
        label_out = output/'labels'/row['split']/label_path.name
        if audit['applied_count']:
            Image.fromarray(result).save(image_out)
        else:
            shutil.copy2(image_path, image_out)
        shutil.copy2(label_path, label_out)
        if sha256(label_out) != protected[label_path]:
            raise RuntimeError('Label copy mismatch')
        if row['split'] != 'train' and sha256(image_out) != protected[image_path]:
            raise RuntimeError('Evaluation image changed')
        if not np.array_equal(result[fake == 0], rgb[fake == 0]):
            raise RuntimeError('Inpainting changed pixels outside fake mask')
        Image.fromarray(fake).save(debug/f"{row['asset_id']}_fake_mask.png")
        if index < debug_samples:
            draw_debug(result, real, fake, boxes,
                       f"{row['split']} | fake={audit['applied_count']} | {row['asset_id'][:12]}",
                       debug/f"{index+1:02d}_{row['asset_id'][:12]}_debug.png")
        records.append(dict(asset_id=row['asset_id'], split=row['split'], image=str(image_out),
                            real_mask=str(mask_path), method=settings.method, radius=settings.radius,
                            dilation=settings.dilation, seed=seed, **audit))
        augmented_manifest.append(dict(row, output_image_path=str(image_out), output_label_path=str(label_out),
                                       output_sha256=sha256(image_out)))
    if any(sha256(path) != digest for path, digest in protected.items()):
        raise RuntimeError('Protected input changed during generation')
    train_records = [r for r in records if r['split'] == 'train']
    summary = dict(total_processed_images=len(records), train_images=len(train_records),
                   fake_applied_images=sum(r['applied_count'] > 0 for r in records),
                   train_applied_distribution={str(k):sum(r['applied_count'] == k for r in train_records) for k in range(4)},
                   train_requested_distribution={str(k):sum(r['requested_count'] == k for r in train_records) for k in range(4)},
                   all_images_applied_distribution={str(k):sum(r['applied_count'] == k for r in records) for k in range(4)},
                   skipped_fake_regions=sum(r['skipped_count'] for r in records),
                   **{f'overlap_{kind}_count':sum(r[f'overlap_{kind}_count'] for r in records) for kind in ('gt','safety','real')},
                   seed=seed, safety_margin_px=safety_margin, scale_range=list(scale_range),
                   count_probabilities=[.25,.35,.25,.15], max_attempts=max_attempts,
                   protected_input_hashes_verified=len(protected), baseline_replays_verified=len(items),
                   opencv_version=cv2.__version__, numpy_version=np.__version__,
                   annotation_assumption='All foreign objects are covered by supplied GT boxes.')
    (debug/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (debug/'per_image.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    for path, data in ((debug/'per_image.csv', records), (output/'manifest.csv', augmented_manifest)):
        with path.open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows({**r, **({'placements':json.dumps(r['placements'])} if 'placements' in r else {})} for r in data)
    (output/'dataset.yaml').write_text(
        f'path: {json.dumps(output.as_posix())}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: defect\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'outputs/yolo_poc_20')
    parser.add_argument('--poc', type=Path, default=ROOT/'outputs/inpainting_poc_20')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/yolo_poc_20_fake_restoration')
    parser.add_argument('--debug-dir', type=Path, default=ROOT/'outputs/fake_restoration_debug')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--safety-margin', type=float, default=10)
    parser.add_argument('--scale-range', type=float, nargs=2, default=(1., 1.))
    parser.add_argument('--max-attempts', type=int, default=500)
    parser.add_argument('--debug-samples', type=int, default=20)
    args = parser.parse_args()
    generate(args.source, args.poc, args.output, args.debug_dir, seed=args.seed,
             safety_margin=args.safety_margin, scale_range=args.scale_range,
             max_attempts=args.max_attempts, debug_samples=args.debug_samples)


if __name__ == '__main__':
    main()
