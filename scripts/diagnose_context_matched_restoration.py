"""Same-image, context-matched diagnostic pairs; never changes training augmentation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import rankdata, wilcoxon

from create_yolo_poc_20 import parse_yolo
from fake_restoration import RestorationSettings, bbox_mask, inpaint_rgb
from diagnose_after_restoration_similarity import (
    ROOT, FEATURES, KEY_METRICS, CLASSIFIER_FEATURES, after_maps,
    classifier_diagnostic, load_mask, load_rgb, make_ring, measure_region,
    read_csv, real_regions, fake_regions, sha256, write_csv,
)

CONTEXT = ('ring_mean', 'ring_std', 'ring_laplacian', 'ring_gradient', 'local_contrast')


def context_vector(maps, ys, xs):
    """Pre-fake ring context only; contrast=P90-P10, never uses inner artifact features."""
    if not len(ys):
        raise ValueError('Empty context ring')
    gray, lap, grad = maps
    intensity = gray[ys, xs]
    return np.array([intensity.mean(), intensity.std(), lap[ys, xs].var(),
                     grad[ys, xs].mean(), np.percentile(intensity, 90)-np.percentile(intensity, 10)])


def crop_mask(mask):
    ys, xs = np.nonzero(mask)
    x, y = int(xs.min()), int(ys.min())
    return x, y, mask[y:ys.max()+1, x:xs.max()+1].copy()


def ring_offsets(template, width):
    padded = np.pad(template, width)
    yy, xx = np.nonzero(make_ring(padded, width))
    return yy-width, xx-width


def ring_coordinates(x, y, offsets, shape):
    yy, xx = offsets[0]+y, offsets[1]+x
    valid = (yy >= 0) & (yy < shape[0]) & (xx >= 0) & (xx < shape[1])
    return yy[valid], xx[valid]


def window_sums(mask, h, w):
    integral = cv2.integral(mask.astype(np.uint8), sdepth=cv2.CV_32S)
    return integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]


def evaluate_candidates(maps, template, forbidden, selected, rng, *, ring_width=5,
                        candidates=512, background_low=5, background_high=250, valid_fraction=.98):
    """Uniformly sample unique safe translations; evaluate >=200 valid candidates or skip."""
    h, w = template.shape
    height, width = forbidden.shape
    if h > height or w > width:
        return np.empty((0,2),int), np.empty((0,len(CONTEXT))), dict(safe_positions=0)
    valid_xray = (maps[0] > background_low) & (maps[0] < background_high)
    safe = (window_sums(forbidden | selected,h,w) == 0)
    safe &= window_sums(valid_xray,h,w) >= valid_fraction*h*w
    yy, xx = np.nonzero(safe)
    offsets = ring_offsets(template,ring_width)
    positions, vectors = [], []
    for index in rng.permutation(len(xx)):
        x, y = int(xx[index]), int(yy[index])
        ry, rx = ring_coordinates(x,y,offsets,forbidden.shape)
        if not len(ry) or float(valid_xray[ry,rx].mean()) < valid_fraction:
            continue
        vector = context_vector(maps,ry,rx)
        if not np.isfinite(vector).all():
            raise RuntimeError('Nonfinite context')
        positions.append((x,y)); vectors.append(vector)
        if len(positions) >= candidates:
            break
    return np.array(positions,int).reshape(-1,2), np.array(vectors,float).reshape(-1,len(CONTEXT)), dict(safe_positions=len(xx))


def standardized_distances(target, candidates):
    center, scale = candidates.mean(axis=0), candidates.std(axis=0)
    # A constant feature receives unit scale; no epsilon division that explodes.
    scale = np.where(scale <= 1e-12, 1., scale)
    distances = np.linalg.norm((candidates-target)/scale,axis=1)
    return distances, center, scale


def translated_mask(template, x, y, shape):
    h,w = template.shape
    if not (0 <= x and 0 <= y and x+w <= shape[1] and y+h <= shape[0]):
        raise ValueError('Translated mask exceeds image')
    result = np.zeros(shape,bool)
    result[y:y+h,x:x+w] = template
    return result


def validate_fake(mask, template, x, y, gt, safety, real, selected):
    h,w = template.shape
    if not np.array_equal(mask[y:y+h,x:x+w],template) or int(mask.sum()) != int(template.sum()):
        raise RuntimeError('Mask shape changed')
    for name, forbidden in (('gt',gt),('safety',safety),('real',real),('selected_fake',selected)):
        if np.any(mask & forbidden):
            raise RuntimeError(f'Fake overlaps {name}')


def paired_statistics(feature_pairs):
    rows = []
    for metric in FEATURES:
        values = [(p['real'][metric],p['fake'][metric]) for p in feature_pairs
                  if p['real'][metric] is not None and p['fake'][metric] is not None]
        real, fake = np.array(values,float).reshape(-1,2).T
        differences = fake-real
        # Round numerical dust before ranking; zeros are omitted by Wilcox convention.
        ranked_diff = np.round(differences,12)
        nz = ranked_diff != 0
        n = len(values)
        if np.any(nz):
            test = wilcoxon(ranked_diff,alternative='two-sided',zero_method='wilcox',method='approx')
            ranks = rankdata(np.abs(ranked_diff[nz]))
            rank_biserial = float(np.dot(ranks,np.sign(ranked_diff[nz]))/ranks.sum())
            statistic, pvalue = float(test.statistic),float(test.pvalue)
        else:
            statistic, pvalue, rank_biserial = (0.,1.,0.) if n else (None,None,None)
        pooled_std = float(np.std(np.concatenate((real,fake)))) if n else 0.
        standardized_abs = (float(np.abs(differences).mean()/pooled_std) if pooled_std > 1e-12
                            else 0. if n and np.all(differences == 0) else None)
        rows.append(dict(metric=metric,key_metric=metric in KEY_METRICS,count=n,
            undefined_pairs=len(feature_pairs)-n, real_median=float(np.median(real)) if n else None,
            fake_median=float(np.median(fake)) if n else None,
            median_paired_difference=float(np.median(differences)) if n else None,
            mean_paired_difference=float(differences.mean()) if n else None,
            median_absolute_difference=float(np.median(np.abs(differences))) if n else None,
            mean_absolute_difference=float(np.abs(differences).mean()) if n else None,
            standardized_mean_absolute_difference=standardized_abs,
            wilcoxon_statistic=statistic,p_value=pvalue,paired_rank_biserial=rank_biserial,p_value_bh=None))
    ordered = sorted([r for r in rows if r['p_value'] is not None],key=lambda r:r['p_value'])
    adjusted = 1.
    for i in range(len(ordered)-1,-1,-1):
        adjusted = min(adjusted,ordered[i]['p_value']*len(ordered)/(i+1))
        ordered[i]['p_value_bh'] = adjusted
    return rows


def context_summary(records):
    result = []
    for source in sorted({r['comparison'] for r in records}):
        subset = [r for r in records if r['comparison']==source]
        for name in (*CONTEXT,'context_distance'):
            if name == 'context_distance':
                diffs = np.array([r['context_distance'] for r in subset])
                standardized = diffs
            else:
                diffs = np.array([abs(r['fake_'+name]-r['real_'+name]) for r in subset])
                standardized = np.array([abs(r['fake_'+name]-r['real_'+name])/r['scale_'+name] for r in subset])
            result.append(dict(comparison=source,metric=name,count=len(subset),
                median_absolute_difference=float(np.median(diffs)),mean_absolute_difference=float(diffs.mean()),
                median_standardized_distance=float(np.median(standardized)),mean_standardized_distance=float(standardized.mean())))
    return result


def make_context_record(comparison, image_id, region_id, real, fake, scale):
    return dict(comparison=comparison,image_id=image_id,real_region_id=region_id,
                context_distance=float(np.linalg.norm((fake-real)/scale)),
                **{'real_'+f:float(real[i]) for i,f in enumerate(CONTEXT)},
                **{'fake_'+f:float(fake[i]) for i,f in enumerate(CONTEXT)},
                **{'scale_'+f:float(scale[i]) for i,f in enumerate(CONTEXT)})


def write_rows(path, rows, empty_fields=('pair_id','reason')):
    write_csv(path,rows,list(rows[0]) if rows else list(empty_fields))


def finite_tree(value):
    if isinstance(value,dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value,(list,tuple)):
        return all(finite_tree(v) for v in value)
    if isinstance(value,(float,np.floating)):
        return bool(np.isfinite(value))
    return True


def save_pair_views(output, pair, baseline, restored, real_mask, fake_mask, boxes, margin, ring_width, debug):
    name = pair['pair_id']
    for group,image,mask in (('real',baseline,real_mask),('fake',restored,fake_mask)):
        Image.fromarray(image).save(output/'paired_images'/f'{name}_{group}.png')
        Image.fromarray(mask.astype(np.uint8)*255).save(output/'masks'/f'{name}_{group}.png')
        ys,xs = np.nonzero(mask | make_ring(mask,ring_width))
        roi = np.s_[max(0,ys.min()-8):min(image.shape[0],ys.max()+9),max(0,xs.min()-8):min(image.shape[1],xs.max()+9)]
        Image.fromarray(image[roi]).save(output/'regions'/f'{name}_{group}.png')
    if not debug:
        return
    display = restored.copy()
    for mask,color in ((make_ring(real_mask,ring_width),[0,190,255]),
                       (make_ring(fake_mask,ring_width),[220,150,0]),
                       (real_mask,[255,70,70]),(fake_mask,[70,255,70])):
        display[mask] = (.25*display[mask]+.75*np.array(color)).astype(np.uint8)
    canvas = Image.fromarray(display)
    draw = ImageDraw.Draw(canvas)
    for x0,y0,x1,y1 in boxes:
        draw.rectangle((max(0,x0-margin),max(0,y0-margin),min(canvas.width-1,x1+margin),min(canvas.height-1,y1+margin)),outline='yellow',width=1)
        draw.rectangle((x0,y0,x1,y1),outline='cyan',width=2)
    panel = Image.new('RGB',(max(canvas.width,680),canvas.height+50),'#171717')
    panel.paste(canvas,(0,50)); draw = ImageDraw.Draw(panel)
    draw.text((8,5),f"{name} | context distance={pair['context_distance']:.3f}",fill='white')
    draw.text((8,25),'Real:red Fake:green GT:cyan Margin:yellow Rings:blue/orange',fill='white')
    panel.save(output/'regions'/f'{name}_overlay.png')


def run_diagnostic(manifest, debug_dir, prior, output, *, seed=None, ring_width=5,
                   safety_margin=None, candidates=512, distance_threshold=2.5,
                   background_low=5, background_high=250, valid_fraction=.98):
    manifest,debug_dir,prior,output = [Path(p).resolve() for p in (manifest,debug_dir,prior,output)]
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    for existing in (ROOT/'outputs').iterdir():
        if existing.is_dir() and existing.resolve() in output.parents:
            raise ValueError('New output cannot be nested inside an existing output tree')
    if (candidates < 200 or ring_width < 1 or not np.isfinite(distance_threshold) or distance_threshold < 0
        or not 0 <= background_low < background_high <= 255 or not 0 < valid_fraction <= 1
        or (seed is not None and seed < 0)):
        raise ValueError('Invalid candidate count, threshold, ring, background or seed option')
    roots = [ROOT/'dataset',ROOT/'scripts',ROOT/'outputs',manifest.parent,debug_dir,prior]
    for p in roots:
        if p == ROOT/'outputs':
            continue  # The new output may be a new child, but cannot contain an old file.
        if output == p or p in output.parents or output in p.parents:
            raise ValueError(f'Output overlaps protected directory: {p}')
    protected = {p.resolve():sha256(p) for root in roots for p in root.rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts}
    if any(output == p or output in p.parents for p in protected):
        raise ValueError('Output contains protected input')
    git_before = subprocess.run(['git','diff','--binary'],cwd=ROOT,capture_output=True,check=True).stdout
    stored_summary = json.loads((debug_dir/'summary.json').read_text(encoding='utf-8'))
    metadata = json.loads((debug_dir/'per_image.json').read_text(encoding='utf-8'))
    by_asset = {a['asset_id']:a for a in metadata}
    source_rows = sorted(read_csv(manifest),key=lambda r:r['asset_id'])
    if not source_rows or len(by_asset)!=len(metadata) or len({r['asset_id'] for r in source_rows})!=len(source_rows) or set(by_asset)!={r['asset_id'] for r in source_rows}:
        raise ValueError('Duplicate/mismatched image IDs')
    seed = stored_summary['seed'] if seed is None else seed
    safety_margin = stored_summary['safety_margin_px'] if safety_margin is None else safety_margin
    if not np.isfinite(safety_margin) or safety_margin < 0:
        raise ValueError('Invalid safety margin')
    rng = np.random.default_rng(seed)
    pairs,unmatched,candidate_rows,contexts,features,differences = [],[],[],[],[],[]
    paired_features,control_features,view_items,scalers = [],[],[],[]
    all_region_ids = set()
    for source in source_rows:
        asset = source['asset_id']; audit = by_asset[asset]
        if audit['split'] != source['split'] or Path(audit['image']).resolve() != Path(source['output_image_path']).resolve():
            raise ValueError('Source/audit mismatch')
        for key,digest_key in (('original_image_path','original_sha256'),('restored_image_path','restored_sha256'),('output_image_path','output_sha256')):
            p = Path(source[key]).resolve(); protected.setdefault(p,sha256(p))
            if protected[p] != source[digest_key]:
                raise ValueError(f'Input hash mismatch: {p}')
        baseline = load_rgb(source['restored_image_path'])
        real_mask_path = Path(audit['real_mask']).resolve()
        protected.setdefault(real_mask_path,sha256(real_mask_path))
        real = load_mask(real_mask_path,baseline.shape[:2])
        regions = real_regions(real)
        settings = RestorationSettings(audit['method'],float(audit['radius']),int(audit['dilation']))
        parsed = re.fullmatch(r'.+_(telea|ns)_d(\d+)_r(\d+)\.png',Path(source['restored_image_path']).name)
        if parsed is None or (parsed[1].upper(),int(parsed[2]),float(parsed[3])) != (settings.method,settings.dilation,settings.radius):
            raise ValueError('Restoration filename and metadata settings disagree')
        if not real_mask_path.name.endswith(f'_d{settings.dilation}.png'):
            raise ValueError('Saved mask dilation disagrees with settings')
        label = Path(source['output_label_path']).resolve(); protected.setdefault(label,sha256(label))
        boxes = parse_yolo(label,baseline.shape[1],baseline.shape[0])
        if len(boxes) != int(source['bbox_count']):
            raise ValueError('GT count mismatch')
        gt, safety = bbox_mask(real.shape,boxes), bbox_mask(real.shape,boxes,safety_margin)
        # Guard Real rings plus derivative stencil so a diagnostic fake cannot change Real context.
        guard_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*(ring_width+2)+1,)*2)
        real_guard = cv2.dilate(real.astype(np.uint8),guard_kernel)>0
        forbidden = safety | real_guard
        selected = np.zeros(real.shape,bool)
        maps = after_maps(baseline)
        old_fake = load_mask(debug_dir/(asset+'_fake_mask.png'),real.shape)
        old_parts = fake_regions(old_fake,audit['placements'])
        for rid,real_region in enumerate(regions,1):
            all_region_ids.add((asset,rid))
            rx,ry,template = crop_mask(real_region); h,w = template.shape
            real_ring = make_ring(real_region,ring_width)
            if not real_ring.any():
                unmatched.append(dict(image_id=asset,real_region_id=rid,real_x=rx,real_y=ry,width=w,height=h,
                    area=int(template.sum()),evaluated_candidates=0,safe_positions=0,
                    reason='empty_real_context_ring',best_context_distance=None))
                continue
            target = context_vector(maps,*np.nonzero(real_ring))
            positions,vectors,search_audit = evaluate_candidates(maps,template,forbidden,selected,rng,
                ring_width=ring_width,candidates=candidates,background_low=background_low,
                background_high=background_high,valid_fraction=valid_fraction)
            base = dict(image_id=asset,real_region_id=rid,real_x=rx,real_y=ry,width=w,height=h,
                        area=int(template.sum()),evaluated_candidates=len(vectors),**search_audit)
            if len(vectors) < 200:
                unmatched.append(dict(**base,reason='fewer_than_200_valid_candidates',best_context_distance=None))
                continue
            distances,center,scale = standardized_distances(target,vectors)
            best = int(np.argmin(distances))
            random_control = int(rng.integers(len(vectors)))  # Independent of distance/target outcome.
            scalers.append(dict(image_id=asset,real_region_id=rid,center=center.tolist(),scale=scale.tolist()))
            for j,((cx,cy),v) in enumerate(zip(positions,vectors)):
                candidate_rows.append(dict(image_id=asset,real_region_id=rid,x=int(cx),y=int(cy),
                    context_distance=float(distances[j]),best=j==best,random_control=j==random_control,
                    accepted=bool(j==best and distances[best]<=distance_threshold),
                    **{f:float(v[k]) for k,f in enumerate(CONTEXT)}))
            historical = []
            for p,mask in zip(audit['placements'],old_parts):
                if int(p['template_id'])+1 == rid:
                    historical.append(context_vector(maps,*np.nonzero(make_ring(mask,ring_width))))
            for v in historical:
                contexts.append(make_context_record('historical_random_all_evaluable',asset,rid,target,v,scale))
            if distances[best] > distance_threshold:
                unmatched.append(dict(**base,reason='context_distance_above_threshold',best_context_distance=float(distances[best])))
                continue
            x,y = map(int,positions[best]); control_x,control_y = map(int,positions[random_control])
            fake = translated_mask(template,x,y,real.shape)
            control_mask = translated_mask(template,control_x,control_y,real.shape)
            validate_fake(fake,template,x,y,gt,safety,real,selected)
            validate_fake(control_mask,template,control_x,control_y,gt,safety,real,selected)
            fake_image = inpaint_rgb(baseline,fake,settings)
            control_image = inpaint_rgb(baseline,control_mask,settings)
            if not np.array_equal(fake_image[~fake],baseline[~fake]) or not np.array_equal(control_image[~control_mask],baseline[~control_mask]):
                raise RuntimeError('Inpainting changed pixels outside mask')
            selected[y:y+h,x:x+w] = True
            pair_id = f'pair_{len(pairs)+1:04d}'
            matched_context = make_context_record('context_matched',asset,rid,target,vectors[best],scale)
            contexts.append(matched_context)
            contexts.append(make_context_record('same_pair_random_control',asset,rid,target,vectors[random_control],scale))
            for v in historical:
                contexts.append(make_context_record('historical_random_successful_templates',asset,rid,target,v,scale))
                contexts.append(make_context_record('matched_for_historical_successful_templates',asset,rid,target,vectors[best],scale))
            pair = dict(pair_id=pair_id,**base,fake_x=x,fake_y=y,
                random_control_x=control_x,random_control_y=control_y,
                context_distance=float(distances[best]),random_control_distance=float(distances[random_control]),
                method=settings.method,radius=settings.radius,dilation=settings.dilation,seed=seed,
                augmentation_seed=audit['seed'],safety_margin=safety_margin,ring_width=ring_width,
                **{k:v for k,v in matched_context.items() if k.startswith(('real_ring','fake_ring','real_local','fake_local','scale_'))})
            pairs.append(pair)
            shared = dict(pair_id=pair_id,asset_id=asset,region_id=rid,split=source['split'])
            real_features = dict(**shared,group='real',**measure_region(maps,real_region,real_ring))
            fake_features = dict(**shared,group='fake',**measure_region(after_maps(fake_image),fake,make_ring(fake,ring_width)))
            control_row = dict(**shared,group='fake',**measure_region(after_maps(control_image),control_mask,make_ring(control_mask,ring_width)))
            features.extend((real_features,fake_features))
            paired_features.append(dict(real=real_features,fake=fake_features))
            control_features.append(dict(real=real_features,fake=control_row))
            for metric in FEATURES:
                a,b = real_features[metric],fake_features[metric]
                delta = None if a is None or b is None else b-a
                differences.append(dict(pair_id=pair_id,image_id=asset,metric=metric,real=a,fake=b,
                                        signed_difference=delta,absolute_difference=None if delta is None else abs(delta)))
            view_items.append((pair,baseline,fake_image,real_region,fake,boxes))
    expected = {(r['asset_id'],int(r['region_id'])) for r in read_csv(prior/'real_after_features.csv')}
    if all_region_ids != expected or len(pairs)+len(unmatched) != len(expected):
        raise RuntimeError('Real region accounting mismatch')
    statistics = paired_statistics(paired_features)
    quality = context_summary(contexts)
    classifiers,predictions = {},[]
    print('WARNING: small, selected paired sample; classifiers/tests are exploratory. No YOLO shortcut conclusion.',flush=True)
    for comparison,pair_features in (('context_matched',paired_features),('same_pair_random_control',control_features)):
        rows = [r for p in pair_features for r in (p['real'],p['fake'])]
        for shape,columns in (('full_features',CLASSIFIER_FEATURES),('without_shape',tuple(f for f in CLASSIFIER_FEATURES if f not in ('mask_area','aspect_ratio')))):
            key = comparison+'_'+shape
            results,preds = classifier_diagnostic(rows,seed,columns)
            classifiers[key] = results
            predictions.extend(dict(analysis=key,**r) for r in preds)
    historical_classifier = json.loads((prior/'classifier_diagnostic.json').read_text(encoding='utf-8'))
    comparison_rows = []
    for shape,historical_key in (('full_features','all_full_features'),('without_shape','matched_images_without_shape')):
        new = classifiers['context_matched_'+shape]
        for reference,old in (('historical_'+historical_key,historical_classifier[historical_key]),
                              ('same_pair_random_control',classifiers['same_pair_random_control_'+shape])):
            old_auc,new_auc = old.get('roc_auc'),new.get('roc_auc')
            comparison_rows.append(dict(features=shape,reference=reference,reference_auc=old_auc,context_matched_auc=new_auc,
                auc_decrease=None if old_auc is None or new_auc is None else old_auc-new_auc,
                balanced_accuracy=new.get('balanced_accuracy'),precision=new.get('precision'),recall=new.get('recall'),
                cohort_note='same accepted pairs and CV groups' if reference=='same_pair_random_control' else 'different cohorts; descriptive only'))
    if not finite_tree([pairs,unmatched,contexts,features,differences,statistics,quality,classifiers,comparison_rows]):
        raise RuntimeError('NaN/inf found in outputs')
    output.mkdir(parents=True,exist_ok=False)
    for directory in ('paired_images','regions','masks','plots'):
        (output/directory).mkdir()
    for filename,rows in (('matched_pairs.csv',pairs),('unmatched_real_regions.csv',unmatched),
        ('context_feature_summary.csv',quality),('context_comparisons.csv',contexts),
        ('candidate_audit.csv',candidate_rows),('paired_after_features.csv',features),
        ('paired_feature_differences.csv',differences),('paired_statistics.csv',statistics),
        ('random_control_after_features.csv',[r for p in control_features for r in (p['real'],p['fake'])]),
        ('classifier_comparison.csv',comparison_rows),('classifier_oof_predictions.csv',predictions)):
        empty_fields = ('image_id','real_region_id','reason','best_context_distance') if filename=='unmatched_real_regions.csv' else (
            ('pair_id','image_id','real_region_id','real_x','real_y','fake_x','fake_y','width','height','area','context_distance',
             'real_ring_mean','fake_ring_mean','real_ring_std','fake_ring_std','real_ring_laplacian','fake_ring_laplacian',
             'real_ring_gradient','fake_ring_gradient') if filename=='matched_pairs.csv' else ('pair_id','reason'))
        write_rows(output/filename,rows,empty_fields)
    for index,args in enumerate(view_items):
        save_pair_views(output,*args,safety_margin,ring_width,debug=index<20)
    create_plots(output,pairs,paired_features)
    (output/'classifier_diagnostic.json').write_text(json.dumps(classifiers,indent=2,allow_nan=False),encoding='utf-8')
    (output/'context_scalers.json').write_text(json.dumps(scalers,indent=2),encoding='utf-8')
    if any(sha256(p)!=digest for p,digest in protected.items()):
        raise RuntimeError('Existing dataset, code or output file changed')
    current = {p.resolve() for root in roots for p in root.rglob('*')
               if p.is_file() and '__pycache__' not in p.parts and output not in p.resolve().parents}
    if current != set(protected):
        raise RuntimeError('Existing protected files added/removed')
    git_after = subprocess.run(['git','diff','--binary'],cwd=ROOT,capture_output=True,check=True).stdout
    if git_before != git_after:
        raise RuntimeError('Tracked git diff changed')
    metadata_out = dict(real_regions=len(expected),matched_pairs=len(pairs),unmatched=len(unmatched),
        seed=seed,ring_width=ring_width,safety_margin=safety_margin,candidates=candidates,
        distance_threshold=distance_threshold,background_low=background_low,background_high=background_high,
        valid_fraction=valid_fraction,scaling='per-real candidate pool population std, raw ring features',
        normality_assumption='GT is complete; extreme background exclusion is intensity-based, not segmentation',
        fake_gt_overlap=0,fake_safety_overlap=0,fake_real_overlap=0,fake_fake_overlap=0,
        boundary_violations=0,mask_shape_mismatches=0,nonfinite_values=0,
        undefined_artifact_values=sum(r[f] is None for r in features for f in FEATURES),
        protected_files_verified=len(protected),git_diff_unchanged=True,
        protected_hashes={str(p):v for p,v in protected.items()},
        numpy_version=np.__version__,opencv_version=cv2.__version__)
    (output/'metadata.json').write_text(json.dumps(metadata_out,indent=2,allow_nan=False),encoding='utf-8')
    (output/'README.md').write_text(DEFINITIONS,encoding='utf-8')
    (output/'REPORT.md').write_text(build_report(metadata_out,quality,statistics,comparison_rows),encoding='utf-8')
    print(json.dumps({k:metadata_out[k] for k in ('real_regions','matched_pairs','unmatched','seed','protected_files_verified')},indent=2))
    return pairs,unmatched,statistics,classifiers


def create_plots(output,pairs,feature_pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(7,4),layout='constrained')
    if pairs:
        ax.hist([p['random_control_distance'] for p in pairs],bins=12,alpha=.5,label='Same-pair random control')
        ax.hist([p['context_distance'] for p in pairs],bins=12,alpha=.7,label='Context matched')
        ax.legend()
    ax.set(xlabel='Standardized context distance',ylabel='Pairs',title='Accepted pairs: context distance')
    fig.savefig(output/'plots/context_distance.png',dpi=160); plt.close(fig)
    for name in ('ring_mean','ring_gradient','texture_ratio','intensity_zscore','boundary_discontinuity_mean'):
        if name in CONTEXT:
            xy = [(p['real_'+name],p['fake_'+name]) for p in pairs]
            title = 'Pre-fake context: '+name
        else:
            xy = [(p['real'][name],p['fake'][name]) for p in feature_pairs if p['real'][name] is not None and p['fake'][name] is not None]
            title = 'After-only: '+name
        fig,ax = plt.subplots(figsize=(5,5),layout='constrained')
        if xy:
            arr = np.array(xy,float)
            ax.scatter(arr[:,0],arr[:,1],alpha=.75)
            lo,hi = arr.min(),arr.max()
            pad = max((hi-lo)*.05,.01)
            ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],'--',color='gray',label='y=x')
            ax.set_xlim(lo-pad,hi+pad); ax.set_ylim(lo-pad,hi+pad); ax.legend()
        ax.set(xlabel='Real',ylabel='Matched fake',title=title); ax.grid(alpha=.2)
        fig.savefig(output/'plots'/f'{name}.png',dpi=160); plt.close(fig)
    fig,ax = plt.subplots(figsize=(10,5),layout='constrained')
    for i,name in enumerate(KEY_METRICS):
        values = [(p['real'][name],p['fake'][name]) for p in feature_pairs if p['real'][name] is not None and p['fake'][name] is not None]
        if values:
            arr = np.array(values); scale = max(float(arr.std()),1e-12)
            ax.scatter(np.full(len(arr),i)+np.linspace(-.08,.08,len(arr)),(arr[:,1]-arr[:,0])/scale,alpha=.7)
    ax.axhline(0,color='gray',linestyle='--')
    ax.set_xticks(range(len(KEY_METRICS)),KEY_METRICS,rotation=25,ha='right')
    ax.set(ylabel='(Fake - Real) / pooled feature std',title='Paired after-only differences (shape preserved)')
    fig.savefig(output/'plots/paired_feature_differences.png',dpi=160); plt.close(fig)


def build_report(metadata,quality,statistics,comparisons):
    def fmt(v):
        return 'N/A' if v is None else f'{v:.4f}'
    lines = ['# Context-Matched Fake Restoration Diagnostic', '',
        f"Real {metadata['real_regions']}개, matched {metadata['matched_pairs']}쌍, unmatched {metadata['unmatched']}개.",
        f"seed={metadata['seed']}, ring={metadata['ring_width']}px, safety margin={metadata['safety_margin']}px, 거리 임계값={metadata['distance_threshold']}.",
        '', '## 문맥 매칭 품질', '',
        '| 비교 | N | Context distance 평균 | 중앙값 |','|---|---:|---:|---:|']
    for r in quality:
        if r['metric']=='context_distance':
            lines.append(f"| {r['comparison']} | {r['count']} | {fmt(r['mean_absolute_difference'])} | {fmt(r['median_absolute_difference'])} |")
    q = {r['comparison']:r for r in quality if r['metric']=='context_distance'}
    for a,b in (('same_pair_random_control','context_matched'),('historical_random_successful_templates','matched_for_historical_successful_templates')):
        if a in q and b in q and q[a]['mean_absolute_difference']>0:
            reduction = 100*(1-q[b]['mean_absolute_difference']/q[a]['mean_absolute_difference'])
            lines.extend(['',f'{a} 대비 평균 context distance 감소: {reduction:.2f}% (동일 대응 집합 기준).'])
    lines.extend(['','## After-only 핵심 지표','','| 지표 | Real median | Fake median | Median(Fake−Real) | Paired rank-biserial |',
                  '|---|---:|---:|---:|---:|'])
    for r in statistics:
        if r['key_metric']:
            lines.append(f"| {r['metric']} | {fmt(r['real_median'])} | {fmt(r['fake_median'])} | {fmt(r['median_paired_difference'])} | {fmt(r['paired_rank_biserial'])} |")
    artifact = [r for r in statistics if r['metric'] in KEY_METRICS and r['metric'] not in ('mask_area','aspect_ratio') and r['standardized_mean_absolute_difference'] is not None]
    if artifact:
        largest=max(artifact,key=lambda r:r['standardized_mean_absolute_difference'])
        closest=min(artifact,key=lambda r:r['standardized_mean_absolute_difference'])
        lines.extend(['',f"단위 차이를 보정한 mean absolute paired difference 기준 가장 큰 핵심 artifact 지표: {largest['metric']} ({fmt(largest['standardized_mean_absolute_difference'])}).",
                      f"같은 기준으로 가장 가까운 핵심 artifact 지표: {closest['metric']} ({fmt(closest['standardized_mean_absolute_difference'])}).",
                      'Shape는 동일 mask를 이동했으므로 차이가 0이며 artifact 유사성 판단 순위에서 제외했다.'])
    lines.extend(['','## Classifier','','| 특징 | 비교 기준 | 이전/대조 AUC | Matched AUC | 감소 | BA | Precision | Recall |',
                  '|---|---|---:|---:|---:|---:|---:|---:|'])
    for r in comparisons:
        lines.append(f"| {r['features']} | {r['reference']} | {fmt(r['reference_auc'])} | {fmt(r['context_matched_auc'])} | {fmt(r['auc_decrease'])} | {fmt(r['balanced_accuracy'])} | {fmt(r['precision'])} | {fmt(r['recall'])} |")
    lines.extend(['','## 해석 및 검증','',
        '기존 Random 결과는 표본 구성과 pair 수가 다르다. same_pair_random_control은 같은 성공 pair, 같은 mask, 같은 안전 후보군과 같은 그룹 분할을 사용한 더 직접적인 비교다.',
        '매칭은 artifact after 특징을 보지 않고 pre-fake ring context만 사용한다. 거리가 줄어드는 것은 선택 기준에 따른 결과이며 독립 검증 성능으로 해석하지 않는다.',
        'GT가 완전하다는 전제이며 유효 X-ray 영역은 밝기 기반 극단 배경 제외 휴리스틱이다. 정상 위치를 찾지 못하거나 기준을 통과하지 못하면 unmatched로 제외한다.',
        '작은 성공 표본 및 선택 편향, 같은 이미지의 여러 pair 간 상관 때문에 Wilcoxon p-value와 classifier는 탐색용이다. AUC 하락은 완전 동일성의 증거가 아니며 높은 AUC도 YOLO shortcut 학습의 증거가 아니다.',
        f"기존 파일 {metadata['protected_files_verified']}개 SHA-256 및 git diff 무변경 확인. GT/safety/Real/fake 간 overlap, 경계 초과, mask shape 불일치, NaN/inf 모두 0건.",
        f"정의 불가 artifact 값은 {metadata['undefined_artifact_values']}개이며 CSV 공란으로 기록했다.",
        '이 출력은 진단 전용이다. 기존 학습 augmentation 및 라벨은 수정하지 않았다.'])
    return '\n'.join(lines)+'\n'


DEFINITIONS = """# Context-matched diagnostic definitions

Only a NEW diagnostic tree is written. There is no YOLO dataset YAML or new label.
Baseline image = saved REAL restoration, before existing random fake augmentation.
Real masks/settings and seed/safety margin come from existing metadata; filenames are
cross-checked. Existing inpaint_rgb is imported unchanged (same OpenCV method/radius).

Real masks: 8-connected components, identical template translated with no resize/rotation.
Candidates: unique integer same-image positions sampled uniformly from safe bounding
rectangles, default 512 evaluated. Fewer than 200 valid candidates => unmatched.
Whole template rectangle excludes GT+margin, real mask plus (ring width+2) guard, and
already selected fake rectangles. The guard preserves Real ring/derivative context.
Normality assumes complete GT. Extreme background heuristic: at least 98% of candidate
rectangle AND ring pixels must have grayscale strictly between 5 and 250 (CLI configurable).
This is not semantic product segmentation or detection of unlabelled foreign objects.

Ring = ellipse dilation minus mask (same make_ring as after-only); borders clipped.
Context uses the baseline image BEFORE applying candidate fake: ring mean, population std,
Laplacian variance, mean Sobel magnitude, P90-P10 ring contrast. No inner after artifact
feature enters matching. Standardization: per-Real safe candidate pool mean/std, zero-std
columns use scale=1. Distance is Euclidean in these standardized coordinates. All candidates,
scalers, best distance and rejection reasons are saved. Threshold default 2.5, fixed before
observing results. Greedy matching processes image IDs and real component IDs in sorted order.
Unmatched regions are never forced to match, and successful pairs are a selected subset.

Each accepted pair has an independent full-image fake generated from the same baseline;
other diagnostic fake regions are not accumulated in its pixels. Fake positions are still
nonoverlapping across all accepted pairs for that image. Real/fake clean full images and
patches are saved, with 20 overlays or all accepted pairs when fewer than 20 exist.

Same-pair random control: uniformly choose one of the SAME safe candidate positions,
without considering distance, then apply identical restoration. No additional labels.
Historical random controls: existing placement.template_id maps to its source Real component;
ring context is recomputed on the same PRE-fake baseline, using that Real's candidate scaler.
Historical successful-template comparisons repeat the matched context for each corresponding
historical fake to keep the comparison weights identical. Historical all-evaluable and accepted
subsets are separated; comparisons across different cohorts are not causal estimates.

After-only features reuse measure_region exactly. Signed difference=Fake-Real; absolute
difference=abs(signed). Wilcoxon two-sided, normal approximation, zeros omitted, differences
rounded to 12 decimals before ranking; all-zero => statistic=0,p=1,effect=0. Paired effect is
rank-biserial (positive=Fake larger). BH correction across all feature tests. Region pairs
from one image can be correlated, so p-values are exploratory. Standardized mean absolute
paired difference divides by pooled population feature std; identical constants => 0.

Classifier reuses fixed LogisticRegression and image-group StratifiedGroupKFold, with
train-fold-only imputation/scaling. Both full and shape-excluded versions, matched and
same-pair random, use identical image groups and balanced pair counts. Seed is inherited.
Historical AUC comparisons are separately labelled as different-cohort descriptive results.
NaN/inf output is forbidden; mathematically undefined ratios are null/empty with counts.

Files: matched_pairs, unmatched_real_regions, context_feature_summary, context_comparisons,
candidate_audit, paired_after_features, paired_feature_differences, paired_statistics,
random_control_after_features, classifier_comparison and classifier_oof_predictions CSVs;
context_scalers/classifier_diagnostic/metadata JSON; paired_images/regions/masks/plots.

Run from project root (Python 3.12 with the existing diagnostic dependencies):
python scripts/diagnose_context_matched_restoration.py --ring-width 5 --candidates 512 --distance-threshold 2.5
Reruns require a NEW --output directory. Existing output trees are never overwritten.
Tests: python -m unittest discover -s scripts -p test_context_matched_restoration.py -v

Current-machine fallback, without changing the broken project .venv:
```powershell
$env:MPLCONFIGDIR = 'C:/workspace/Kamp_Xray/.cache/after_similarity_mpl'
& 'C:/Users/kang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -c "import sys,runpy; sys.path.insert(0,r'C:/workspace/Kamp_Xray/.cache/after_similarity_runtime'); sys.path.append(r'C:/workspace/Kamp_Xray/.venv/Lib/site-packages'); sys.path.insert(0,'scripts'); runpy.run_path('scripts/diagnose_context_matched_restoration.py',run_name='__main__')"
```
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'outputs/yolo_poc_20_fake_restoration/manifest.csv')
    parser.add_argument('--debug-dir',type=Path,default=ROOT/'outputs/fake_restoration_debug')
    parser.add_argument('--prior',type=Path,default=ROOT/'outputs/restoration_after_similarity')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/context_matched_restoration')
    parser.add_argument('--seed',type=int,default=None)
    parser.add_argument('--ring-width',type=int,default=5)
    parser.add_argument('--safety-margin',type=float,default=None)
    parser.add_argument('--candidates',type=int,default=512)
    parser.add_argument('--distance-threshold',type=float,default=2.5)
    parser.add_argument('--background-low',type=float,default=5)
    parser.add_argument('--background-high',type=float,default=250)
    parser.add_argument('--valid-fraction',type=float,default=.98)
    args=parser.parse_args()
    run_diagnostic(args.manifest,args.debug_dir,args.prior,args.output,seed=args.seed,ring_width=args.ring_width,
        safety_margin=args.safety_margin,candidates=args.candidates,distance_threshold=args.distance_threshold,
        background_low=args.background_low,background_high=args.background_high,valid_fraction=args.valid_fraction)


if __name__=='__main__':
    main()
