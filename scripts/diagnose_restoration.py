"""Read-only Real/Fake restoration diagnostics; never calls inpainting."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('mean_absolute_difference', 'max_absolute_difference', 'changed_pixel_ratio',
           'ssim', 'laplacian_variance_before', 'laplacian_variance_after',
           'laplacian_variance_change_pct', 'gradient_magnitude_before',
           'gradient_magnitude_after', 'gradient_magnitude_change_pct',
           'mask_width', 'mask_height', 'mask_area')


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert('RGB'))


def load_mask(path, shape):
    with Image.open(path) as image:
        mask = np.array(image.convert('L'))
    if mask.shape != shape or not np.isin(mask, [0, 255]).all():
        raise ValueError(f'Invalid binary mask: {path}')
    return mask > 0


def change_pct(before, after):
    return None if before == 0 else 100.0 * (after-before)/before


def feature_maps(before, after):
    """Full-image grayscale derivatives and local Gaussian SSIM map.

    SSIM: 11x11 Gaussian, sigma=1.5, population covariance, L=255,
    K1=.01, K2=.03, reflected border. Aggregation happens only on mask pixels.
    """
    a, b = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float64) for x in (before, after)]
    def blur(x):
        return cv2.GaussianBlur(x, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT_101)
    ma, mb = blur(a), blur(b)
    va, vb = np.maximum(0, blur(a*a)-ma*ma), np.maximum(0, blur(b*b)-mb*mb)
    cov = blur(a*b)-ma*mb
    ssim = ((2*ma*mb+2.55**2)*(2*cov+7.65**2) /
            ((ma*ma+mb*mb+2.55**2)*(va+vb+7.65**2)))
    maps = {'ssim': ssim}
    for name, gray in (('before', a), ('after', b)):
        maps['laplacian_'+name] = cv2.Laplacian(gray, cv2.CV_64F, ksize=3, borderType=cv2.BORDER_REFLECT_101)
        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
        maps['gradient_'+name] = np.hypot(dx, dy)
    return maps


def region_metrics(before, after, mask, maps=None):
    if before.shape != after.shape or before.dtype != np.uint8 or after.dtype != np.uint8:
        raise ValueError('Expected matching uint8 RGB images')
    if mask.shape != before.shape[:2] or not np.any(mask):
        raise ValueError('Expected nonempty matching mask')
    mask = mask > 0
    maps = feature_maps(before, after) if maps is None else maps
    ys, xs = np.nonzero(mask)
    diff = np.abs(after.astype(np.int16)-before.astype(np.int16))[mask]
    values = dict(mean_absolute_difference=float(diff.mean()), max_absolute_difference=int(diff.max()),
                  changed_pixel_ratio=float(np.any(diff != 0, axis=1).mean()),
                  ssim=float(maps['ssim'][mask].mean()),
                  mask_x=int(xs.min()), mask_y=int(ys.min()),
                  mask_width=int(xs.max()-xs.min()+1), mask_height=int(ys.max()-ys.min()+1),
                  mask_area=int(mask.sum()))
    for metric, map_name, operation in (('laplacian_variance', 'laplacian', np.var),
                                         ('gradient_magnitude', 'gradient', np.mean)):
        a, b = [float(operation(maps[map_name+'_'+side][mask])) for side in ('before', 'after')]
        values.update({metric+'_before':a, metric+'_after':b, metric+'_change_pct':change_pct(a, b)})
    return values


def real_regions(mask):
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return [labels == i for i in range(1, count)]


def fake_regions(mask, placements):
    regions, covered = [], np.zeros(mask.shape, bool)
    h, w = mask.shape
    for p in placements:
        x, y, pw, ph = [int(p[k]) for k in ('x', 'y', 'width', 'height')]
        if not (0 <= x < x+pw <= w and 0 <= y < y+ph <= h):
            raise ValueError('Fake placement outside image')
        region = np.zeros(mask.shape, bool)
        region[y:y+ph, x:x+pw] = mask[y:y+ph, x:x+pw]
        if not region.any() or (region & covered).any():
            raise ValueError('Empty or ambiguous fake placement')
        covered |= region
        regions.append(region)
    if not np.array_equal(covered, mask):
        raise ValueError('Fake mask and placement metadata disagree')
    return regions


def save_views(directory, before, after, mask, gain):
    directory.mkdir(parents=True, exist_ok=False)
    diff = np.abs(after.astype(np.int16)-before.astype(np.int16))
    amplified = np.clip(diff.astype(np.float64)*gain, 0, 255).astype(np.uint8)
    for name, array in (('before', before), ('after', after), ('abs_diff_amplified', amplified),
                        ('mask', mask.astype(np.uint8)*255)):
        Image.fromarray(array).save(directory/(name+'.png'))


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def distribution_summary(groups):
    rows = []
    for scope in ('all', 'train'):
        for metric in METRICS:
            row = dict(scope=scope, metric=metric)
            for group in ('real', 'fake'):
                selected = [r for r in groups[group] if scope == 'all' or r['split'] == scope]
                values = np.array([r[metric] for r in selected if r[metric] is not None], dtype=float)
                row[group+'_regions'] = len(selected)
                row[group+'_valid_count'] = len(values)
                row[group+'_undefined_count'] = len(selected)-len(values)
                for stat in ('mean', 'std', 'min', 'p05', 'p25', 'median', 'p75', 'p95', 'max'):
                    value = None
                    if values.size:
                        if stat in ('mean', 'std', 'min', 'max'):
                            value = float(getattr(np, stat)(values))
                        else:
                            value = float(np.percentile(values, {'p05':5, 'p25':25, 'median':50, 'p75':75, 'p95':95}[stat]))
                    row[group+'_'+stat] = value
            for stat in ('mean', 'median'):
                a, b = row['real_'+stat], row['fake_'+stat]
                row['fake_minus_real_'+stat] = None if a is None or b is None else b-a
            rows.append(row)
    return rows


DEFINITIONS = """# Restoration diagnostics

Read-only analysis. No restoration/augmentation code is executed or modified.
Real before = original color-marked image; after = saved real restoration.
Fake before = saved real restoration; after = saved fake-augmented image.

real.csv: one row per 8-connected actual restoration mask component.
fake.csv: one row per recorded fake placement, using saved fake mask pixels.
Empty fake masks produce no fake region rows; samples.csv includes every image/group.
summary.csv: equally weighted REGION distributions (population std, linear percentiles),
both all available regions and train-only comparisons. Regions are not independent images.

MAE/max: absolute RGB channel differences in 0..255, restricted to mask pixels.
Changed pixel ratio: fraction of mask pixels with ANY RGB channel changed, threshold=0.
SSIM: grayscale local SSIM map, Gaussian 11x11 sigma=1.5, K1=.01, K2=.03,
L=255, population covariance, BORDER_REFLECT_101. Average at mask pixels only.
Thin/single-pixel masks retain surrounding image context in SSIM windows.
Laplacian: grayscale, ksize=3; variance at mask pixels (population variance).
Gradient: grayscale Sobel ksize=3, mean sqrt(dx^2+dy^2) at mask pixels.
Derivatives are computed BEFORE cropping/masking, using reflected image borders.
Change percent = 100*(after-before)/before. Zero baseline -> empty CSV cell (undefined).
mask_width/height: tight mask bounding rectangle in pixels; mask_area: nonzero pixel count.

images/<group>/<asset>/ contains full before/after/amplified diff and mask, including
zero-fake samples. regions/<group>/<asset>/<region>/ contains the same views cropped
with context padding. Metrics use original-resolution masks, not these padded crops.
Diff visualization = clip(abs(after-before)*diff_gain, 0, 255), fixed gain across groups;
no per-image auto normalization. Raw metrics never use amplified differences.

Real includes color-mark removal, so larger Real changes are not by themselves evidence
of stronger inpainting artifacts. These descriptive diagnostics do not measure YOLO
shortcut learning or recover the unknown unmarked original signal.
"""


def diagnose(manifest, debug, output, *, diff_gain=8., padding=16):
    manifest, debug, output = [Path(p).resolve() for p in (manifest, debug, output)]
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    if not np.isfinite(diff_gain) or diff_gain <= 0 or padding < 0:
        raise ValueError('diff_gain must be positive/finite and padding nonnegative')
    for protected_root in (ROOT/'dataset', ROOT/'scripts', ROOT/'outputs/yolo_poc_20',
                           ROOT/'outputs/inpainting_poc_20', manifest.parent, debug):
        if output == protected_root or protected_root in output.parents or output in protected_root.parents:
            raise ValueError(f'Output overlaps input/code directory: {output}')
    protected = {}
    def protect(path):
        path = Path(path).resolve()
        if path == output or output in path.parents:
            raise ValueError('Output contains an input')
        protected[path] = sha256(path)
        return path
    protect(manifest)
    audit_path = protect(debug/'per_image.json')
    with manifest.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    audits = json.loads(audit_path.read_text(encoding='utf-8'))
    by_asset = {r['asset_id']: r for r in audits}
    if (not rows or len(by_asset) != len(audits) or len({r['asset_id'] for r in rows}) != len(rows)
            or {r['asset_id'] for r in rows} != set(by_asset)):
        raise ValueError('Manifest and audit must have matching unique assets')
    for name in ('fake_restoration.py', 'create_fake_restoration_dataset.py'):
        protect(ROOT/'scripts'/name)
    items = []
    for row in sorted(rows, key=lambda r:r['asset_id']):
        audit = by_asset[row['asset_id']]
        if row['split'] != audit['split'] or Path(row['output_image_path']).resolve() != Path(audit['image']).resolve():
            raise ValueError('Audit/manifest mismatch')
        paths = [protect(row[k]) for k in ('original_image_path', 'restored_image_path', 'output_image_path')]
        for path, key in zip(paths, ('original_sha256', 'restored_sha256', 'output_sha256')):
            if protected[path] != row[key]:
                raise ValueError(f'Image provenance mismatch: {path}')
        protect(row['output_label_path'])
        real_path = protect(audit['real_mask'])
        fake_path = protect(debug/(row['asset_id']+'_fake_mask.png'))
        original, restored, augmented = [load_rgb(p) for p in paths]
        if original.shape != restored.shape or restored.shape != augmented.shape:
            raise ValueError('Image size mismatch')
        real = load_mask(real_path, original.shape[:2])
        fake = load_mask(fake_path, original.shape[:2])
        regions = fake_regions(fake, audit['placements'])
        if len(regions) != audit['applied_count'] or (row['split'] != 'train' and fake.any()):
            raise ValueError('Invalid fake count/split')
        for before, after, mask in ((original, restored, real), (restored, augmented, fake)):
            if not np.array_equal(before[~mask], after[~mask]):
                raise ValueError('Image changed outside recorded restoration mask')
        items.append((row, audit, original, restored, augmented, real, fake, regions, paths))
    output.mkdir(parents=True, exist_ok=False)
    groups, samples = {'real':[], 'fake':[]}, []
    for row, audit, original, restored, augmented, real, fake, fake_parts, paths in items:
        for group, before, after, mask, regions, before_path, after_path in (
                ('real', original, restored, real, real_regions(real), paths[0], paths[1]),
                ('fake', restored, augmented, fake, fake_parts, paths[1], paths[2])):
            asset = row['asset_id']
            sample_dir = output/'images'/group/asset
            save_views(sample_dir, before, after, mask, diff_gain)
            samples.append(dict(asset_id=asset, split=row['split'], group=group, region_count=len(regions),
                                mask_area=int(mask.sum()), before_path=str(before_path), after_path=str(after_path),
                                views=str(sample_dir)))
            maps = feature_maps(before, after) if regions else None
            for i, region in enumerate(regions, 1):
                values = region_metrics(before, after, region, maps)
                x, y, w, h = [values[k] for k in ('mask_x', 'mask_y', 'mask_width', 'mask_height')]
                roi = np.s_[max(0,y-padding):min(before.shape[0],y+h+padding),
                            max(0,x-padding):min(before.shape[1],x+w+padding)]
                views = output/'regions'/group/asset/f'{i:03d}'
                save_views(views, before[roi], after[roi], region[roi], diff_gain)
                groups[group].append(dict(asset_id=asset, split=row['split'], group=group, region_id=i,
                    method=audit['method'], radius=audit['radius'], seed=audit['seed'],
                    before_path=str(before_path), after_path=str(after_path), views=str(views), **values))
    fields = ['asset_id','split','group','region_id','method','radius','seed','before_path','after_path','views',
              'mean_absolute_difference','max_absolute_difference','changed_pixel_ratio','ssim',
              'mask_x','mask_y','mask_width','mask_height','mask_area',
              *METRICS[4:10]]
    for group in groups:
        write_csv(output/(group+'.csv'), groups[group], fields)
    summary = distribution_summary(groups)
    write_csv(output/'summary.csv', summary, list(summary[0]))
    write_csv(output/'samples.csv', samples, list(samples[0]))
    if any(sha256(path) != digest for path, digest in protected.items()):
        raise RuntimeError('Protected input changed during diagnostics')
    metadata = dict(images=len(items), real_regions=len(groups['real']), fake_regions=len(groups['fake']),
                    diff_gain=diff_gain, padding=padding, protected_files_verified=len(protected),
                    input_hashes={str(p):v for p,v in protected.items()},
                    opencv_version=cv2.__version__, numpy_version=np.__version__)
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (output/'README.md').write_text(DEFINITIONS, encoding='utf-8')
    print(json.dumps({k:v for k,v in metadata.items() if k != 'input_hashes'}, indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'outputs/yolo_poc_20_fake_restoration/manifest.csv')
    parser.add_argument('--debug-dir', type=Path, default=ROOT/'outputs/fake_restoration_debug')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/restoration_diagnostics')
    parser.add_argument('--diff-gain', type=float, default=8.)
    parser.add_argument('--padding', type=int, default=16)
    args = parser.parse_args()
    diagnose(args.manifest, args.debug_dir, args.output, diff_gain=args.diff_gain, padding=args.padding)


if __name__ == '__main__':
    main()
