"""Read-only validation and reproducible session-group split; never overwrite outputs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ('train', 'val', 'test')

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, columns=None):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns or list(rows[0]), extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)

def snapshot(paths):
    return {str(p.resolve()): sha256(p) for root in paths for p in sorted(Path(root).rglob('*')) if p.is_file()}

def verify_snapshot(before):
    errors = [p for p, digest in before.items() if not Path(p).is_file() or sha256(p) != digest]
    if errors:
        raise ValueError(f'Protected files changed: {errors[:10]}')
    return len(before)

def labels(path):
    boxes = []
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5 or values[0] != '0':
            raise ValueError(f'Invalid class/label: {path}: {line}')
        coords = list(map(float, values[1:]))
        if not all(np.isfinite(v) and 0 <= v <= 1 for v in coords) or min(coords[2:]) <= 0:
            raise ValueError(f'Invalid normalized bbox: {path}')
        boxes.append(coords)
    return boxes

def validate_source(source):
    rows = read_csv(source / 'metadata.csv')
    provenance = read_csv(source / 'provenance.csv')
    images = {p.stem: p for p in (source / 'images').iterdir() if p.is_file()}
    txts = {p.stem: p for p in (source / 'labels').iterdir() if p.is_file()}
    if len(images) != 512 or len(txts) != 512 or images.keys() != txts.keys() or len(rows) != 512 or len(provenance) != 512:
        raise ValueError('Source count/stem/metadata/provenance validation failed')
    if len({r['unique_id'] for r in rows}) != 512 or {Path(r['filename']).stem for r in rows} != images.keys():
        raise ValueError('Metadata IDs or filenames do not match source')
    prov = {r['unique_id']: r for r in provenance}
    total = 0
    for r in rows:
        stem = Path(r['filename']).stem
        image, label = images[stem], txts[stem]
        boxes = labels(label)
        if not boxes or len(boxes) != int(r['bbox_count']):
            raise ValueError(f'BBox count mismatch: {stem}')
        total += len(boxes)
        r['image_sha256'], r['label_sha256'] = sha256(image), sha256(label)
        if r['image_sha256'] != r['sha256_image'] or r['label_sha256'] != r['sha256_label'] or r['label_sha256'] != prov[r['unique_id']]['final_label_sha256']:
            raise ValueError(f'Source recorded hash mismatch: {stem}')
        with Image.open(image) as im:
            rgb = im.convert('RGB')
            if im.size != (int(r['width']), int(r['height'])):
                raise ValueError(f'Size mismatch: {stem}')
            r['pixel_sha256'] = hashlib.sha256(str(im.size).encode() + rgb.tobytes()).hexdigest()
        match = re.fullmatch(r'\d+_(\d{8})_(\d{6})\(\d+\)', stem)
        if not match or match[1] != r['capture_date'].replace('-', '') or not all(r[k] for k in ('machine', 'serial', 'capture_date')):
            raise ValueError(f'Filename/session metadata mismatch: {stem}')
        r['group_id'] = '|'.join(r[k] for k in ('machine', 'serial', 'capture_date'))
    if total != 1163 or Counter(r['label_source_type'] for r in rows) != {'CANONICAL_POOL': 500, 'VERIFIED_FALLBACK': 12}:
        raise ValueError('Source bbox/source-type totals mismatch')
    return rows

def group_split(rows, seed=42):
    # Merge session groups only if exact file or decoded RGB equality requires it.
    parent = {r['group_id']: r['group_id'] for r in rows}
    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for key in ('image_sha256', 'pixel_sha256'):
        seen = {}
        for r in rows:
            g = find(r['group_id'])
            if r[key] in seen:
                other = find(seen[r[key]])
                parent[max(g, other)] = min(g, other)
            seen[r[key]] = g
    for r in rows:
        r['session_id'] = r['group_id']
        r['group_id'] = find(r['group_id'])
    groups = sorted({r['group_id'] for r in rows})
    machines = sorted({r['machine'] for r in rows})
    bins = sorted({int(r['bbox_count']) for r in rows})
    features = np.zeros((len(groups), 2 + len(machines) + len(bins)))
    for r in rows:
        i = groups.index(r['group_id'])
        features[i, 0] += 1
        features[i, 1] += int(r['bbox_count'])
        features[i, 2 + machines.index(r['machine'])] += 1
        features[i, 2 + len(machines) + bins.index(int(r['bbox_count']))] += 1
    target = np.array([.70, .15, .15])[:, None] * features.sum(axis=0)
    weights = np.array([8, 2] + [4] * len(machines) + [1] * len(bins))
    def score(counts):
        return float((((counts - target) / np.maximum(target, 5)) ** 2 * weights).sum())
    rng = np.random.default_rng(seed)
    best_score, best = float('inf'), None
    for restart in range(40):
        assignment = rng.choice(3, len(groups), p=[.7,.15,.15])
        counts = np.array([features[assignment == s].sum(axis=0) for s in range(3)])
        value = score(counts)
        for step in range(3500):
            i = int(rng.integers(len(groups)))
            old, new = int(assignment[i]), int(rng.integers(3))
            j = int(rng.integers(len(groups))) if rng.random() < .5 else i
            other = int(assignment[j])
            if j != i:
                new = other
            if old == new:
                continue
            delta = features[i] - (features[j] if j != i else 0)
            candidate = counts.copy()
            candidate[old] -= delta
            candidate[new] += delta
            new_value = score(candidate)
            temperature = .03 * (1 - step / 3500) ** 3 + .00001
            if new_value < value or rng.random() < np.exp(min(0, (value - new_value) / temperature)):
                assignment[i] = new
                if j != i:
                    assignment[j] = old
                counts, value = candidate, new_value
                if value < best_score and all(counts[:, 0] > 0):
                    best_score, best = value, assignment.copy()
    mapping = dict(zip(groups, (SPLITS[int(s)] for s in best)))
    for r in rows:
        r['split'] = mapping[r['group_id']]
    return best_score

def leakage(rows, image_key='image_sha256'):
    result = {}
    for key in ('unique_id', image_key, 'pixel_sha256', 'group_id', 'session_id'):
        values = defaultdict(set)
        for r in rows:
            values[r[key]].add(r['split'])
        result[key + '_cross_split'] = sum(len(v) > 1 for v in values.values())
    if any(result.values()):
        raise ValueError(f'Leakage: {result}')
    return result

def summarize(rows):
    return {s: {'images': len(part), 'bbox': sum(int(r['bbox_count']) for r in part),
                'machine': dict(sorted(Counter(r['machine'] for r in part).items())),
                'bbox_count': dict(sorted(Counter(r['bbox_count'] for r in part).items())),
                'groups': len({r['group_id'] for r in part})}
            for s in SPLITS for part in [[r for r in rows if r['split'] == s]]}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'outputs/clean_512')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/clean_512_split')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = validate_source(args.source)
    print('SOURCE VALIDATED: 512 images/labels, 1163 bbox, class 0, hashes match', flush=True)
    protected = snapshot([ROOT / 'dataset', ROOT / 'outputs'])
    print(f'Protected baseline: {len(protected)} files', flush=True)
    score = group_split(rows, args.seed)
    checks = leakage(rows)
    summary = {'seed': args.seed, 'grouping': 'machine + serial + capture_date (whole calendar day); exact SHA/RGB duplicate groups merged',
               'session_groups': len({r['session_id'] for r in rows}), 'group_size_distribution': dict(Counter(Counter(r['session_id'] for r in rows).values())),
               'optimization': 'seeded group-level annealing: image/machine/bbox-total/bbox-count balance; no restoration features',
               'balance_score': score, 'splits': summarize(rows), 'leakage': checks, 'protected_files': len(protected)}
    verify_snapshot(protected)
    args.output.mkdir(parents=True)
    write_csv(args.output / 'split_manifest.csv', rows, ['unique_id','filename','split','machine','serial','capture_date','bbox_count','image_sha256','label_sha256','pixel_sha256','group_id','session_id','label_source_type'])
    for s in SPLITS:
        (args.output / (s + '_ids.txt')).write_text(''.join(r['unique_id'] + '\n' for r in rows if r['split'] == s), encoding='utf-8')
    write_json(args.output / 'protected_baseline.json', protected)
    write_json(args.output / 'split_summary.json', summary)
    (args.output / 'README.md').write_text('# clean_512 grouped split\n\nWhole machine/serial/date groups prevent same-day session leakage. Filename date agrees with metadata for all 512 images. 45 observed sessions (1–36 images) permit the requested ratio without inventing time-gap thresholds. Seeded group-only optimization balances counts, machine and bbox distributions. Identical decoded RGB and file hashes cannot cross splits. Identical label hashes are allowed. No image or label is rewritten here.\n\nProtected baseline hashes every pre-existing dataset/ and outputs/ file. Outputs refuse overwrite. Restoration must use the committed manifest; no resplitting.\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
