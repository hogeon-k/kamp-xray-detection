"""Frozen Test FN/FP audit. Reuses predictions/features; never trains or infers.

Run with Python 3 + numpy + Pillow: python -B -X utf8 scripts/error_analysis.py
Existing outputs are protected; use --out for a new run. Optional --review supplies
case-id keyed visual-review notes. Causes are diagnostic, not established physics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'outputs/yolo_512_telea'
CACHE = ROOT / 'outputs/foreign_object_condition_analysis'
POLICY = ROOT / 'outputs/safety_threshold_policy'
FEATURES = ROOT / 'outputs/fn_root_cause_analysis/features/test_gt_features.csv'
FONT = 'C:/Windows/Fonts/malgun.ttf'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection/union if union else 0.


def candidate_diagnosis(gt_box, candidate, threshold):
    """Diagnose exactly the displayed raw candidate, independent of eligibility."""
    if candidate is None:
        return None, 'unknown'
    score = iou(gt_box, candidate['box'])
    if candidate['confidence'] < threshold:
        return score, ('low_confidence_localization_success' if score >= .5
                       else 'low_confidence_localization_failure')
    return score, 'localization_failure' if score < .5 else 'unknown'


def match(gt, predictions, threshold):
    # Exact ordering/tie behavior of evaluate_edge_reinspection.matches.
    eligible = [dict(p, cache_index=i) for i, p in enumerate(predictions)
                if p['confidence'] >= threshold and p['class_id'] == 0]
    pairs = sorted(((iou(g, p['box']), i, j) for i, g in enumerate(gt)
                    for j, p in enumerate(eligible)), reverse=True)
    used_gt, used_pred, found = set(), set(), {}
    for score, i, j in pairs:
        if score < .5:
            break
        if i not in used_gt and j not in used_pred:
            used_gt.add(i); used_pred.add(j)
            found[i] = dict(eligible[j], match_iou=score)
    return found, [p for j, p in enumerate(eligible) if j not in used_pred], eligible


def boxes(row, size):
    w, h = size
    result = []
    for line in (DATA/'labels/test'/(Path(row['filename']).stem+'.txt')).read_text().splitlines():
        cls, x, y, bw, bh = map(float, line.split())
        assert cls == 0 and bw > 0 and bh > 0
        result.append([(x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h])
    return result


def geometry(box, size, mask):
    w, h = size
    x1, y1, x2, y2 = box
    bw, bh = x2-x1, y2-y1
    a, b, c, d = max(0, math.floor(x1)), max(0, math.floor(y1)), min(w, math.ceil(x2)), min(h, math.ceil(y2))
    area_pixels = max(0, c-a)*max(0, d-b)
    overlap = int(mask[b:d, a:c].sum()) if mask is not None else None
    distance = None
    if mask is not None:
        ys, xs = np.where(mask)
        if len(xs):
            # Minimum distance from mask pixel centres to continuous bbox rectangle.
            dx = np.maximum(np.maximum(x1-(xs+.5), (xs+.5)-x2), 0)
            dy = np.maximum(np.maximum(y1-(ys+.5), (ys+.5)-y2), 0)
            distance = float(np.hypot(dx, dy).min())
    edge = min(x1, y1, w-x2, h-y2)
    return dict(bbox_width=bw, bbox_height=bh, bbox_area=bw*bh,
                bbox_area_ratio=bw*bh/(w*h), bbox_width_ratio=bw/w, bbox_height_ratio=bh/h,
                aspect_ratio=bw/bh, image_width=w, image_height=h,
                distance_to_edge=edge, relative_distance_to_edge=edge/min(w, h),
                image_edge_near=edge <= .05*min(w, h),
                restoration_overlap_pixels=overlap,
                restoration_overlap=overlap/area_pixels if overlap is not None and area_pixels else None,
                restoration_distance=distance, restoration_near=distance <= 5 if distance is not None else False,
                restoration_mask_available=mask is not None)


def project_audit(out):
    """Read all project code, reports, CSVs, and JSON metadata, save inventory.

    Runtime/vendor trees excluded. Binary files are inventoried, not decoded.
    """
    inventory, texts, csvs = [], [], []
    excluded = {'.git', '.venv', '.cache', '__pycache__'}
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or excluded.intersection(p.relative_to(ROOT).parts) or out in p.parents:
            continue
        rel = str(p.relative_to(ROOT))
        inventory.append(dict(path=rel, bytes=p.stat().st_size, suffix=p.suffix.lower()))
        if p.suffix.lower() not in {'.py', '.md', '.json', '.yaml', '.yml', '.csv'}:
            continue
        content = p.read_text(encoding='utf-8-sig', errors='replace')
        item = dict(path=rel, sha256=sha(p), lines=len(content.splitlines()))
        if p.suffix.lower() == '.csv':
            parsed = list(csv.reader(content.splitlines()))
            csvs.append(dict(path=rel, rows=max(0, len(parsed)-1), columns=json.dumps(parsed[0] if parsed else [], ensure_ascii=False)))
        if p.suffix.lower() == '.json':
            try:
                parsed = json.loads(content)
                item['top_level_keys'] = list(parsed)[:40] if isinstance(parsed, dict) else 'list'
            except json.JSONDecodeError:
                item['parse_status'] = 'invalid_json'
        texts.append(item)
    write_csv(out/'summary/project_inventory.csv', inventory)
    write_csv(out/'summary/csv_inventory.csv', csvs)
    dump(out/'summary/project_text_audit.json', texts)


def draw_box(draw, box, color, label, width=2):
    draw.rectangle(tuple(box), outline=color, width=width)
    x, y = max(0, box[0]), max(0, box[1]-19)
    font = ImageFont.truetype(FONT, 13)
    draw.rectangle(draw.textbbox((x, y), label, font=font), fill='black')
    draw.text((x, y), label, font=font, fill=color)


def visualize(case, record, out):
    image = record['image'].convert('RGB')
    annotated = image.copy(); d = ImageDraw.Draw(annotated)
    for i, box in enumerate(record['gt']):
        draw_box(d, box, '#00ffff', f'GT {i}')
    for p in record['eligible']:
        draw_box(d, p['box'], '#55ff55', f"P {p['confidence']:.3f}")
    focus = json.loads(case['gt_bbox'] if case['error_type'] == 'FN' else case['prediction_bbox'])
    draw_box(d, focus, '#ff4545', case['case_id'], 3)
    if case['error_type'] == 'FN' and case['prediction_bbox']:
        draw_box(d, json.loads(case['prediction_bbox']), '#ffa500', f"candidate {case['confidence']:.3f}")
    allfocus = [focus]
    if case['gt_bbox']:
        allfocus.append(json.loads(case['gt_bbox']))
    if case['prediction_bbox']:
        allfocus.append(json.loads(case['prediction_bbox']))
    w, h = image.size
    crop = (max(0, math.floor(min(b[0] for b in allfocus))-24),
            max(0, math.floor(min(b[1] for b in allfocus))-24),
            min(w, math.ceil(max(b[2] for b in allfocus))+24),
            min(h, math.ceil(max(b[3] for b in allfocus))+24))
    original = record['original'].convert('RGB')
    overlay = np.array(image).copy()
    if record['mask'] is not None:
        overlay[record['mask']] = (overlay[record['mask']]*.45 + np.array([255, 0, 255])*.55).astype(np.uint8)
    canvas = Image.new('RGB', (1400, 720), 'white'); cd = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 16)
    # Keep enlarged boxes thin and labels outside the ROI so tiny defects remain visible.
    local = image.copy(); ld = ImageDraw.Draw(local)
    for box in record['gt']:
        ld.rectangle(box, outline='#00ffff', width=1)
    for p in record['eligible']:
        ld.rectangle(p['box'], outline='#55ff55', width=1)
    ld.rectangle(focus, outline='#ff4545', width=1)
    if case['error_type']=='FN' and case['prediction_bbox']:
        ld.rectangle(json.loads(case['prediction_bbox']), outline='#ffa500', width=1)
    panels = [(annotated, '전체 Test 입력 / GT 청록, 예측 초록, 오류 빨강'),
              (local.crop(crop), '확대 / FN 후보 주황 (정답 매칭 아님)'),
              (original.crop(crop), '복원 전 원본 확대'),
              (Image.fromarray(overlay).crop(crop), 'Test 입력 확대 / 복원 마스크 자홍')]
    for k, (panel, title) in enumerate(panels):
        x = 10+k*350
        scale = min(340/panel.width, 455/panel.height)
        panel = panel.resize((round(panel.width*scale), round(panel.height*scale)), Image.Resampling.NEAREST if k else Image.Resampling.BILINEAR)
        canvas.paste(panel, (x, 95))
        cd.text((x, 65), title, font=ImageFont.truetype(FONT, 13), fill='black')
    cd.text((10, 8), f"{case['case_id']} | {case['image_name']} | {case['cause']}", font=font, fill='black')
    cd.text((10, 32), f"bbox area={case['bbox_area']:.2f} px²  | image edge={case['distance_to_edge']:.2f}px | restoration distance={case['restoration_distance']}", font=font, fill='black')
    candidate_score = case['candidate_gt_iou']
    candidate_label = f'{candidate_score:.4f}' if candidate_score is not None else 'NA (no candidate)'
    cd.text((10, 565), f"candidate–GT IoU={candidate_label} | accepted prediction IoU={case['best_eligible_iou']:.4f} | confidence={case['confidence']}", font=font, fill='black')
    cd.text((10, 593), f"조건: {case['condition_flags']}", font=font, fill='black')
    cd.text((10, 625), 'FN의 prediction bbox는 진단용 후보 / FP의 GT bbox는 가장 겹치는 GT. 모두 TP 매칭이 아닙니다.', font=font, fill='black')
    cd.text((10, 654), '물리적 원인, 제품 구조, 라벨 정확성은 사람 검토 필요. 원본에는 색상 표식이 포함될 수 있습니다.', font=font, fill='black')
    path = out/case['error_type'].lower()/(case['case_id']+'.png')
    canvas.save(path)
    return canvas


def bars(path, title, entries, rate=False):
    font = ImageFont.truetype(FONT, 18)
    left = max(350, max((font.getbbox(name)[2]+40 for name,_,_ in entries), default=350))
    canvas = Image.new('RGB', (left+750, 110+max(1, len(entries))*65), 'white')
    d = ImageDraw.Draw(canvas)
    d.text((20, 15), title, font=font, fill='black')
    maximum = max((v for _, v, _ in entries), default=1) or 1
    for i, (name, value, label) in enumerate(entries):
        y = 70+i*65
        d.text((20, y), name, font=font, fill='black')
        d.rectangle((left, y, left+600*value/(1 if rate else maximum), y+28), fill='#4675ad')
        d.text((left+10+600*value/(1 if rate else maximum), y), label, font=font, fill='black')
    canvas.save(path)


def main(out, review_path, update=False):
    if out.exists() and not update:
        raise FileExistsError(f'Existing results protected. Choose a new --out: {out}')
    policy = json.loads((POLICY/'locked_policy.json').read_text(encoding='utf-8'))
    reference = json.loads((POLICY/'summary.json').read_text(encoding='utf-8'))
    provenance = json.loads((CACHE/'prediction_provenance.json').read_text(encoding='utf-8'))
    model = Path(provenance['model'])
    assert sha(model) == provenance['model_sha256'] == reference['provenance']['model_sha256']
    assert policy == reference['lock'] and policy['selection_split'] == 'val'
    threshold, high = policy['selected']['T_low'], policy['selected']['T_high']
    assert provenance['conf'] <= threshold and reference['GT_matching_IoU'] == .5
    predictions = json.loads((CACHE/'predictions.json').read_text(encoding='utf-8'))
    manifest = read_csv(DATA/'manifest.csv'); test = [r for r in manifest if r['split'] == 'test']
    features = {(r['unique_id'], int(r['bbox_id'])): r for r in read_csv(FEATURES)}
    reviews = json.loads(review_path.read_text(encoding='utf-8')) if review_path else {}
    source_paths = [model, CACHE/'predictions.json', CACHE/'prediction_provenance.json',
                    POLICY/'locked_policy.json', POLICY/'summary.json', DATA/'manifest.csv', DATA/'dataset.yaml', FEATURES]
    for r in test:
        stem = Path(r['filename']).stem
        source_paths += [DATA/'images/test'/r['restored_filename'], DATA/'labels/test'/(stem+'.txt'),
                         DATA/'restoration_masks/test'/(stem+'.png'), ROOT/'outputs/clean_512/images'/r['filename']]
    before = {str(p): sha(p) for p in source_paths}
    # Check against the original prediction run's input snapshot, not only this run.
    old_hashes = json.loads((CACHE/'input_hashes.json').read_text(encoding='utf-8'))
    feature_hashes = json.loads((ROOT/'outputs/fn_root_cause_analysis/input_hashes.json').read_text(encoding='utf-8'))
    for p in source_paths:
        for snapshot in (old_hashes, feature_hashes):
            if str(p) in snapshot:
                assert before[str(p)] == snapshot[str(p)], f'Stale inputs: {p}'
    for r in test:
        assert sha(DATA/'images/test'/r['restored_filename']) == r['restored_sha256']
        assert sha(DATA/'labels/test'/(Path(r['filename']).stem+'.txt')) == r['label_sha256']
    small = policy['small_low_contrast_train_cutoffs']['small_area_q25']
    contrast = policy['small_low_contrast_train_cutoffs']['low_contrast_q25']
    cases, gtrows, detrows, records = [], [], [], {}
    counts = Counter()
    for r in test:
        uid = r['unique_id']; image = Image.open(DATA/'images/test'/r['restored_filename']).copy()
        gt = boxes(r, image.size); raw = [dict(p, cache_index=i) for i,p in enumerate(predictions[uid])]
        assert len(gt) == int(r['bbox_count'])
        found, fps, eligible = match(gt, raw, threshold)
        baseline, basefps, baseeligible = match(gt, raw, high)
        assert found == baseline and fps == basefps and eligible == baseeligible
        maskpath = DATA/'restoration_masks/test'/(Path(r['filename']).stem+'.png')
        mask = np.array(Image.open(maskpath)) > 0 if maskpath.exists() else None
        assert mask is None or mask.shape == (image.height, image.width)
        records[uid] = dict(image=image, original=Image.open(ROOT/'outputs/clean_512/images'/r['filename']).copy(),
                            mask=mask, gt=gt, eligible=eligible)
        for i, box in enumerate(gt):
            f = features[(uid, i)]; status = 'TP' if i in found else 'FN'
            assert f['status'] == status
            geo = geometry(box, image.size, mask)
            assert math.isclose(geo['bbox_area'], float(f['bbox_area']), rel_tol=1e-8)
            norm = float(f['normalized_contrast']) if f['normalized_contrast'] else None
            overlaps = [(iou(box, p['box']), p) for p in eligible]
            best_score, best = max(overlaps, key=lambda x:x[0], default=(0., None))
            below = [p for p in raw if p['class_id'] == 0 and iou(box, p['box']) >= .5]
            low_candidate = max(below, key=lambda p:p['confidence'], default=None)
            # Spatial diagnostic: an accepted prediction overlaps this unmatched
            # GT by IoU>=.1 but misses .5. Visual review must confirm object identity.
            localization = status == 'FN' and .1 <= best_score < .5
            row = dict(unique_id=uid, image_name=r['restored_filename'], bbox_id=i, status=status,
                       machine=r['machine'], **geo, normalized_contrast=norm,
                       small_object=geo['bbox_area'] <= small+1e-8,
                       low_contrast=norm <= contrast if norm is not None else None,
                       product_edge=f['product_edge'] == 'True',
                       product_contour_distance=float(f['contour_distance']),
                       low_confidence_candidate=low_candidate is not None and low_candidate['confidence'] < threshold,
                       best_eligible_iou=best_score,
                       confidence=found[i]['confidence'] if i in found else None,
                       localization_candidate=localization)
            gtrows.append(row); counts[status] += 1
            if status == 'TP':
                continue
            candidate = best if localization else low_candidate
            if candidate is None:
                raw_overlaps = [(iou(box,p['box']),p) for p in raw if p['class_id']==0]
                raw_score, raw_best = max(raw_overlaps,key=lambda x:x[0],default=(0.,None))
                candidate = raw_best if raw_score > 0 else None
            candidate_iou, cause = candidate_diagnosis(box, candidate, threshold)
            candidate_overlaps = [(iou(g, candidate['box']), j) for j,g in enumerate(gt)] if candidate else []
            nearest_iou, nearest_gt = max(candidate_overlaps, default=(None, None))
            flags = [key for key in ('small_object','low_contrast','product_edge','image_edge_near','restoration_near','low_confidence_candidate','localization_candidate') if row[key]]
            case = dict(row, case_id=f'{uid}_FN_gt{i}', error_type='FN', cause=cause,
                        cause_basis='spatial_matching_diagnostic' if localization else 'confidence_diagnostic',
                        gt_bbox=json.dumps(box), prediction_bbox=json.dumps(candidate['box']) if candidate else '',
                        prediction_role='unmatched_spatial_candidate' if localization else ('below_threshold_candidate' if low_candidate else 'raw_best_overlap_candidate'),
                        prediction_index=candidate.get('cache_index') if candidate else None,
                        confidence=candidate['confidence'] if candidate else None,
                        gt_area_ratio=geo['bbox_area_ratio'], best_gt_iou=nearest_iou,
                        nearest_gt_id=nearest_gt, candidate_gt_iou=candidate_iou,
                        candidate_confidence_threshold=threshold,
                        low_confidence_bbox=json.dumps(low_candidate['box']) if low_candidate and low_candidate['confidence'] < threshold else '',
                        low_confidence_gt_iou=iou(box,low_candidate['box']) if low_candidate and low_candidate['confidence'] < threshold else None,
                        low_confidence_confidence=low_candidate['confidence'] if low_candidate and low_candidate['confidence'] < threshold else None,
                        condition_flags=';'.join(flags), review_required=True,
                        notes='Observable failure mechanism; physical cause unconfirmed.')
            if candidate:
                pg = geometry(candidate['box'], image.size, mask)
                case.update({f'prediction_{k}':v for k,v in pg.items()})
            case.update({f'gt_{k}':v for k,v in geo.items()})
            cases.append(case)
        fp_indices = {p['cache_index'] for p in fps}
        for p in eligible:
            overlap = [(iou(g, p['box']), i) for i, g in enumerate(gt)]
            best_score, best_gt = max(overlap, default=(0., None))
            geo = geometry(p['box'], image.size, mask)
            duplicate = best_score >= .5 and best_gt in found
            near_unmatched = .1 <= best_score < .5 and best_gt not in found
            status = 'FP' if p['cache_index'] in fp_indices else 'TP'
            detrows.append(dict(unique_id=uid, prediction_index=p['cache_index'], status=status,
                                confidence=p['confidence'], prediction_bbox=json.dumps(p['box']), **geo, best_gt_iou=best_score))
            if status != 'FP':
                continue
            counts['FP'] += 1
            cause = 'duplicate_detection' if duplicate else ('localization_failure' if near_unmatched else 'unknown')
            associated = geometry(gt[best_gt], image.size, mask) if best_gt is not None else {}
            flags = [key for key in ('image_edge_near','restoration_near') if geo[key]]
            if near_unmatched: flags.append('localization_candidate')
            case = dict(unique_id=uid, image_name=r['restored_filename'], bbox_id='',
                        case_id=f"{uid}_FP_pred{p['cache_index']}", error_type='FP', cause=cause,
                        cause_basis='spatial_matching_diagnostic', **geo,
                        gt_bbox=json.dumps(gt[best_gt]) if best_gt is not None and best_score > 0 else '',
                        prediction_bbox=json.dumps(p['box']), prediction_role='unmatched_accepted_prediction',
                        prediction_index=p['cache_index'], confidence=p['confidence'],
                        gt_area_ratio=associated.get('bbox_area_ratio') if best_score > 0 else None,
                        associated_gt_id=best_gt if best_score > 0 else None,
                        best_gt_iou=best_score, best_eligible_iou=best_score,
                        candidate_gt_iou=best_score, candidate_confidence_threshold=threshold,
                        prediction_to_gt_area_ratio=geo['bbox_area']/associated['bbox_area'] if best_score > 0 else None,
                        condition_flags=';'.join(flags), review_required=True,
                        notes='FP is unmatched at IoU .5; normal structure/artifact cannot be inferred from FP label.')
            case.update({f'prediction_{k}':v for k,v in geo.items()})
            case.update({f'gt_{k}':v for k,v in associated.items()} if best_score > 0 else {})
            cases.append(case)
    expected = next(r for r in reference['test_results'] if r['policy'] == 'Final_policy')
    assert all(counts[k] == expected[k] for k in ('TP','FN','FP'))
    assert len(gtrows) == 160 and len(test) == 76
    unknown_review_ids = set(reviews)-{c['case_id'] for c in cases}
    assert not unknown_review_ids, unknown_review_ids
    for c in cases:
        c['physical_cause'] = 'unknown'
        if c['case_id'] in reviews:
            note = reviews[c['case_id']]
            c['visual_review_status'] = 'assistant_visual_reviewed'
            c['notes'] += ' ' + note['notes']
            # A contradictory visual review conservatively returns unknown.
            if not note.get('supports_spatial_diagnostic', True):
                c['cause'] = 'unknown'
        else:
            c['visual_review_status'] = 'pending'
    out.mkdir(parents=True, exist_ok=update)
    for folder in ('fn','fp','summary'):
        (out/folder).mkdir(exist_ok=update)
    project_audit(out)
    frequencies = []
    for error in ('FN','FP'):
        group = [c for c in cases if c['error_type'] == error]
        for cause, n in Counter(c['cause'] for c in group).most_common():
            frequencies.append(dict(error_type=error, cause=cause, count=n, total=len(group), ratio=n/len(group)))
        bars(out/f'summary/{error.lower()}_cause_frequency.png', f'{error} 관찰된 오류 기전 (물리적 원인 미확정)',
             [(r['cause'], r['count'], f"{r['count']}/{r['total']}") for r in frequencies if r['error_type']==error])
    conditions = []
    for row in gtrows+detrows:
        row['restoration_intersects'] = row['restoration_overlap_pixels'] > 0 if row['restoration_overlap_pixels'] is not None else None
    for key in ('small_object','low_contrast','image_edge_near','product_edge','restoration_near','restoration_intersects'):
        for value in (True, False):
            group = [r for r in gtrows if r[key] is value]
            fn = sum(r['status']=='FN' for r in group)
            conditions.append(dict(condition=key, group=str(value), GT=len(group), FN=fn, rate=fn/len(group) if group else None))
    for key, boundaries in [('bbox_area', [0,81+1e-8,160,math.inf]),
                            ('aspect_ratio', [0,.75,1.3333333333,math.inf])]:
        for lo, hi in zip(boundaries, boundaries[1:]):
            group = [r for r in gtrows if lo < r[key] <= hi]
            fn = sum(r['status']=='FN' for r in group)
            conditions.append(dict(condition=key, group=f'({lo:g},{hi:g}]', GT=len(group), FN=fn, rate=fn/len(group) if group else None))
    confidence_groups = []
    for lo, hi in ((threshold,.5),(.5,.7),(.7,1.00000001)):
        group = [r for r in detrows if lo <= r['confidence'] < hi]
        fp = sum(r['status']=='FP' for r in group)
        confidence_groups.append(dict(lower=lo, upper=min(hi,1), detections=len(group), FP=fp,
                                      FP_fraction=fp/len(group) if group else None, fraction_of_all_FP=fp/counts['FP']))
    fp_conditions = []
    for key in ('image_edge_near','restoration_near','restoration_intersects'):
        for value in (True,False):
            group = [r for r in detrows if r[key] is value]
            fp = sum(r['status']=='FP' for r in group)
            fp_conditions.append(dict(condition=key, group=str(value), detections=len(group), FP=fp, FP_fraction=fp/len(group) if group else None))
    write_csv(out/'summary/cause_frequency.csv', frequencies)
    write_csv(out/'summary/gt_condition_rates.csv', conditions)
    write_csv(out/'summary/fp_confidence_bins.csv', confidence_groups)
    write_csv(out/'summary/fp_condition_rates.csv', fp_conditions)
    write_csv(out/'summary/all_test_gt.csv', gtrows)
    write_csv(out/'summary/all_test_predictions.csv', detrows)
    write_csv(out/'error_cases.csv', cases)
    write_csv(out/'error_classification.csv', cases)
    write_csv(out/'summary/fn_candidate_diagnostics.csv', [
        {k:c[k] for k in ('case_id','prediction_index','gt_bbox','prediction_bbox','confidence',
                          'candidate_confidence_threshold','candidate_gt_iou','best_eligible_iou','cause',
                          'low_confidence_bbox','low_confidence_gt_iou','low_confidence_confidence')}
        for c in cases if c['error_type']=='FN'])
    write_csv(out/'summary/review_required.csv', cases)
    bars(out/'summary/fn_condition_rates.png', '조건별 FN / 전체 GT (조건 중복 가능)',
         [(r['condition']+' '+r['group'],r['rate'] or 0, f"{r['FN']}/{r['GT']}" if r['GT'] else '산출 불가') for r in conditions[:12]], True)
    bars(out/'summary/fp_confidence_bins.png', 'confidence 구간별 FP / 전체 예측',
         [(f"[{r['lower']:.2f},{r['upper']:.2f})",r['FP_fraction'] or 0,f"{r['FP']}/{r['detections']}") for r in confidence_groups], True)
    for error in ('FN','FP'):
        selected = [c for c in cases if c['error_type'] == error]
        contact = Image.new('RGB',(1400,360*len(selected)), 'white')
        for i,c in enumerate(selected):
            canvas = visualize(c,records[c['unique_id']],out)
            contact.paste(canvas.resize((700,360)),((i%2)*700,(i//2)*360))
        contact = contact.crop((0,0,1400,math.ceil(len(selected)/2)*360))
        contact.save(out/f'summary/{error.lower()}_contact_sheet.png')
    assert all(sha(Path(p)) == digest for p,digest in before.items())
    dump(out/'summary/input_hashes.json',before)
    dump(out/'summary/verification.json', dict(
        FN=counts['FN'], FP=counts['FP'], TP=counts['TP'], source_hashes_unchanged=True,
        candidate_iou_verified=all(c['candidate_gt_iou'] == iou(json.loads(c['gt_bbox']),json.loads(c['prediction_bbox']))
                                   for c in cases if c['gt_bbox'] and c['prediction_bbox']),
        script_sha256=sha(Path(__file__)), no_training=True, no_inference=True))
    summary = dict(images=len(test),GT=len(gtrows),**counts,
                   FN_images=len({c['unique_id'] for c in cases if c['error_type']=='FN'}),
                   FP_images=len({c['unique_id'] for c in cases if c['error_type']=='FP'}),
                   provenance=provenance, policy=policy, applied_threshold=threshold, automatic_NG_threshold=high,
                   matching_iou=.5, no_inference=True,no_training=True,input_hashes_verified=True,
                   cause_frequency=frequencies,conditions=conditions,fp_confidence=confidence_groups,
                   automatic_rules=dict(small_area_px2=small,low_normalized_contrast=contrast,
                                        image_edge_fraction=.05,product_edge_px=48,restoration_distance_px=5,
                                        localization_candidate_iou_range=[.1,.5]),
                   human_only_causes=['overlap_structure','structure_confusion','high_density_region',
                                      'edge_pattern','restoration_artifact','restoration_related'],
                   low_confidence_matching_FN=sum(r['status']=='FN' and r['low_confidence_candidate'] for r in gtrows),
                   FP_localization_pairs=[dict(case_id=c['case_id'],unique_id=c['unique_id'],gt_id=c.get('associated_gt_id'),
                                               IoU=c['best_gt_iou'],prediction_to_gt_area_ratio=c.get('prediction_to_gt_area_ratio'))
                                          for c in cases if c['error_type']=='FP'],
                   review_required=[c['case_id'] for c in cases],visual_review_source=str(review_path) if review_path else None)
    dump(out/'summary/summary.json',summary)
    if review_path:
        dump(out/'summary/visual_review_notes.json',reviews)
    report(out,summary,cases)
    print(json.dumps(dict(counts=counts,frequency=frequencies,conditions=conditions,fp_confidence=confidence_groups),ensure_ascii=False,indent=2))


def report(out,s,cases):
    cond = {(r['condition'],r['group']):r for r in s['conditions']}
    lines = ['# Test FN/FP 오류 분석','',
             f"76장 / GT160 / TP{s['TP']} / FN{s['FN']} ({s['FN_images']}장) / FP{s['FP']} ({s['FP_images']}장).",'',
             f"최종 모델: `{s['provenance']['model']}`. 잠긴 Two_threshold 정책 T_low={s['applied_threshold']}, T_high={s['automatic_NG_threshold']}, GT IoU=0.5. Test middle band=0으로 두 기준 결과가 동일. NMS IoU=0.7의 기존 conf floor0.001 예측을 그대로 재사용. 학습·추론 없음.",'',
             '## 원인과 조건의 구분','',
             'cause는 일대일 IoU 매칭/점수에서 관찰된 오류 기전이며 물리적 원인 확정이 아닙니다. FN spatial candidate는 accepted prediction IoU가 [0.1,0.5)인 경우이며 같은 물체인지 시각 확인이 필요합니다. FP localization_failure는 이 범위로 미검출 GT와 겹치는 경우입니다. 중복은 이미 TP 매칭된 GT에 IoU≥0.5인 추가 예측입니다. 물체 크기/주변 대비/edge/복원 근접은 중복 가능한 condition_flags에 따로 보존합니다. 한 GT의 FN과 동일 물체의 FP가 동시에 발생할 수 있으므로 FN+FP를 서로 다른 물체 개수로 합하지 않습니다.','',
             'FN의 candidate_gt_iou는 시각화 prediction_bbox와 해당 gt_bbox의 직접 IoU이며 confidence 필터를 적용하지 않습니다. best_eligible_iou는 threshold 이상 예측 중 해당 GT와의 최고 IoU로 별도 저장합니다. best_gt_iou는 표시 candidate와 모든 GT 사이의 최고 IoU입니다. 기존 candidate 선택 우선순위(accepted spatial candidate → IoU≥0.5 최고 confidence 후보 → raw 최고 IoU 후보)는 유지합니다. 선택된 candidate의 confidence<T_low이고 IoU≥0.5면 low_confidence_localization_success, IoU<0.5면 low_confidence_localization_failure입니다. accepted 후보가 선택된 다른 FN의 저점수 IoU≥0.5 후보도 별도 열에 보존합니다.','',
             'small_object는 Train bbox 면적 하위25% ≤81px², low_contrast는 기존 Train normalized contrast 하위25% ≤0.0698918 기준입니다. 대비는 GT bbox 전체 평균과 GT 제외 제품 내부 ring 평균 차이/ring 평균으로서 실제 결함 자체의 대비와 다를 수 있습니다. GT 특성은 기존 계산을 검증 후 재사용했습니다. bbox 실수 면적과 mask raster 면적은 다릅니다. 복원 overlap은 floor/ceil raster bbox 내부 mask 픽셀 비율, distance는 mask 픽셀 중심과 실수 bbox 사이 최소 거리이며 5px 이하를 근접으로 표시합니다. image edge는 bbox와 이미지 경계 간 최소 거리≤이미지 짧은 변5%, 제품 edge는 기존 48px 중심 거리 판정을 사용합니다. FP 제품 경계/대비를 GT 값으로 대입하지 않았습니다.','',
             '구조 겹침, 정상 구조 오인, 고밀도, 선형 구조 및 복원 흔적의 인과성은 영상 전문가/라벨 검토가 필요합니다. 마스크와 가까운 것만으로 복원 원인으로 분류하지 않습니다. 모든 사례는 review_required=True이며 assistant visual review와 사람의 승인을 구분합니다.','',
             '## 원인별 빈도 (상호 배타적인 주 기전)','', '|유형|기전|개수|비율|','|---|---|---:|---:|']
    for r in s['cause_frequency']:
        lines.append(f"|{r['error_type']}|{r['cause']}|{r['count']}/{r['total']}|{r['ratio']:.1%}|")
    lines += ['', '## 중복 가능한 FN 조건별 빈도와 발생률','','|조건|FN/GT|FN 발생률|','|---|---:|---:|']
    # Individual diagnostics refer to exactly the same bbox used by the PNG.
    diagnostics = ['## FN candidate 직접 계산', '', '|사례|confidence|candidate–GT IoU|분류|', '|---|---:|---:|---|']
    for c in cases:
        if c['error_type']=='FN':
            score = f"{c['candidate_gt_iou']:.6f}" if c['candidate_gt_iou'] is not None else 'NA'
            conf = f"{c['confidence']:.6f}" if c['confidence'] is not None else 'NA'
            diagnostics.append(f"|{c['case_id']}|{conf}|{score}|{c['cause']}|")
    lines[len(lines)-5:len(lines)-5] = diagnostics+['']
    for r in s['conditions']:
        lines.append(f"|{r['condition']} {r['group']}|{r['FN']}/{r['GT']}|{r['rate']:.2%}|" if r['GT'] else f"|{r['condition']} {r['group']}|0/0|산출 불가|")
    lines += ['', '### FN 집중 조건']
    fnfreq = [r for r in s['cause_frequency'] if r['error_type']=='FN']
    lines += [f"- 가장 많은 원인: {fnfreq[0]['cause']} {fnfreq[0]['count']}/12 (관찰된 기전).",
              '- 두 번째 원인: '+(f"{fnfreq[1]['cause']} {fnfreq[1]['count']}/12." if len(fnfreq)>1 else '없음.'),
              '- 특징: 아래 Train 기준 조건별 발생률로 기술하며 조건과 인과 원인을 구분합니다.']
    for key in ('small_object','low_contrast','product_edge','restoration_near'):
        a,b=cond[(key,'True')],cond[(key,'False')]
        av=f"{a['FN']}/{a['GT']} ({a['rate']:.2%})" if a['GT'] else '분모0'
        bv=f"{b['FN']}/{b['GT']} ({b['rate']:.2%})" if b['GT'] else '분모0'
        lines.append(f'- {key}: 해당 {av}, 비해당 {bv}.')
    lines += ['- 대표 사례: U0371 GT0/1/2 (28–35px²), U0147 GT0/1, U0488 GT0 (최고 eligible IoU 약0.4985).','',
              '### FP 집중 조건',
              '- 가장 많은 원인: '+', '.join(f"{r['cause']} {r['count']}/8" for r in s['cause_frequency'] if r['error_type']=='FP')+'.',
              '- 두 번째 원인: 별도 확정된 기전 없음. 정상 구조 오인으로 자동 해석하지 않음.',
              '- 특징: GT와 겹치지만 IoU 기준 미달인 경우가 많아 FN/FP가 같은 이미지에 동반되는지 개별 확인 필요.']
    for r in s['fp_confidence']:
        lines.append(f"- confidence [{r['lower']:.2f},{r['upper']:.2f}): FP{r['FP']}/{r['detections']} 예측.")
    lines += [f"- FN 중 IoU≥0.5 저점수 후보 존재: {s['low_confidence_matching_FN']}/12. 주 기전과 중복되는 조건이며 별도 합산하지 않습니다.",
              '- 복원 5px 이내 GT는160/160으로 비근접 비교군이 없습니다. 복원 때문에 FN/FP가 집중된다고 결론 내릴 수 없습니다.']
    pairs=s['FP_localization_pairs']
    ratios=[r['prediction_to_gt_area_ratio'] for r in pairs]
    lines += [f"- FP {len(pairs)}/8개가 FN이 있는 동일 {len({r['unique_id'] for r in pairs})}개 이미지에 동반되며 각각 미검출 GT와 겹칩니다. IoU 범위 {min(r['IoU'] for r in pairs):.4f}–{max(r['IoU'] for r in pairs):.4f}.",
              f"- Prediction 면적이 GT보다 큰 FP {sum(r>1 for r in ratios)}/8개. 면적비 범위 {min(ratios):.3f}–{max(ratios):.3f}. U0488은 면적 증가보다 중심 이동이 시각적으로 관찰됩니다.",
              '- 크기/종횡비 구간은 기술 통계용입니다. 면적81px² 경계는 기존 Train 기준이고, 추가 면적160px²와 종횡비0.75,4/3은 비교표 구간이며 모델/정책 선택에 사용하지 않습니다.']
    lines += ['- 대표 사례: U0147, U0303, U0371, U0488의 FP 전체 (개별 파일과 확대판 참조).','',
              '### 평가보고서용 결론']
    small_a,small_b=cond[('small_object','True')],cond[('small_object','False')]
    low_a,low_b=cond[('low_contrast','True')],cond[('low_contrast','False')]
    lines += [f"기존 최종 모델과 잠긴 정책을 적용한 Test76장에서 FN12개, FP8개를 확인했다. Train 기준 소형 bbox의 FN 발생률은 {small_a['FN']}/{small_a['GT']} ({small_a['rate']:.2%}), 그 외는 {small_b['FN']}/{small_b['GT']} ({small_b['rate']:.2%})이고, 낮은 bbox 주변 대비군은 {low_a['FN']}/{low_a['GT']} ({low_a['rate']:.2%}), 그 외는 {low_b['FN']}/{low_b['GT']} ({low_b['rate']:.2%})였다. FP의 GT 겹침과 FN의 점수·위치 오차를 함께 기록했으며, 같은 물체의 위치·크기 오차가 FN과 FP를 동시에 만들 수 있다. 복원 마스크 근접과 구조적 패턴은 기록된 조건일 뿐 원인으로 확정하지 않았으며, 소수 이미지에 묶인 오류이므로 물리적 원인과 라벨 타당성은 사람이 검토해야 한다.",'',
              '## 재현과 검증','',
              'Python 3 + numpy + Pillow로 `python -B -X utf8 scripts/error_analysis.py --out outputs/error_analysis_run2 --review outputs/error_analysis/summary/visual_review_notes.json`. 기존 출력 덮어쓰기 차단. 원본 예측 생성 시점의 입력 SHA와 비교, Test manifest image/label SHA 확인, GT features와 재매칭 상태 일치, 기존 잠긴 Test 집계와 TP/FN/FP 일치, 분석 전후 SHA 불변 검증. 프로젝트 코드·보고서·CSV·JSON을 읽은 목록은 project_text_audit.json, 전체 파일 목록은 project_inventory.csv, CSV 스키마/행수는 csv_inventory.csv. 기존 학습 CSV의 mAP/PR 지표와 고정 confidence/IoU 객체 매칭 집계를 혼동하지 않았습니다.','',
              '## 사람이 직접 검토해야 할 사례','']
    lines += [f"- {c['case_id']} — {c['image_name']} — {c['cause']}" for c in cases]
    (out/'summary/conclusions.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/error_analysis')
    parser.add_argument('--review', type=Path)
    parser.add_argument('--update', action='store_true', help='Regenerate analysis artifacts in an existing --out directory')
    args = parser.parse_args()
    main(args.out.resolve(),args.review,args.update)
