"""After-only local restoration artifact diagnostic. All inputs are read-only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import ks_2samp, mannwhitneyu

from diagnose_restoration import ROOT, fake_regions, load_mask, load_rgb, real_regions, sha256, write_csv

KEY_METRICS = ('texture_ratio', 'gradient_ratio', 'intensity_zscore',
               'boundary_discontinuity_mean', 'mask_area', 'aspect_ratio')
FEATURES = ('inner_mean', 'inner_std', 'ring_mean', 'ring_std',
            'intensity_difference', 'absolute_intensity_difference', 'intensity_ratio', 'intensity_zscore',
            'inner_laplacian_variance', 'ring_laplacian_variance', 'texture_ratio',
            'inner_gradient_mean', 'ring_gradient_mean', 'gradient_ratio',
            'boundary_discontinuity_mean', 'boundary_discontinuity_median',
            'boundary_discontinuity_max', 'boundary_sobel_response',
            'mask_width', 'mask_height', 'mask_area', 'aspect_ratio')
# No coordinates, IDs, split, method, labels, or before-derived features enter the model.
CLASSIFIER_FEATURES = tuple(f for f in FEATURES if f not in ('mask_width', 'mask_height'))
EPS = 1e-12


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def ratio(a, b):
    return None if a is None or b is None or abs(b) <= EPS else float(a/b)


def make_ring(mask, width=5):
    if not isinstance(width, int) or width < 1 or mask.ndim != 2 or not mask.any():
        raise ValueError('Ring width must be a positive integer and mask nonempty/2D')
    mask = mask > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*width+1, 2*width+1))
    ring = (cv2.dilate(mask.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT,
                       borderValue=0) > 0) & ~mask
    if (ring & mask).any():
        raise RuntimeError('Mask/ring overlap')
    return ring


def after_maps(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    return gray, lap, np.hypot(gx, gy)


def boundary_values(gray, mask):
    """Absolute difference across every in-image 4-neighbor crossing edge once.

    Includes hole boundaries of hollow masks. Outside-image pixels are never synthesized.
    """
    boundary = np.zeros(mask.shape, bool)
    differences = []
    for first, second in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])):
        crossing = mask[first] != mask[second]
        differences.append(np.abs(gray[first]-gray[second])[crossing])
        boundary[first] |= crossing & mask[first]
        boundary[second] |= crossing & mask[second]
    return np.concatenate(differences), boundary


def measure_region(maps, mask, ring):
    gray, lap, gradient = maps
    mask, ring = mask > 0, ring > 0
    if mask.shape != gray.shape or ring.shape != gray.shape or not mask.any() or (mask & ring).any():
        raise ValueError('Invalid mask/ring')
    def stat(array, selected, fn):
        return float(fn(array[selected])) if selected.any() else None
    im, rm = stat(gray, mask, np.mean), stat(gray, ring, np.mean)
    ist, rst = stat(gray, mask, np.std), stat(gray, ring, np.std)
    il, rl = stat(lap, mask, np.var), stat(lap, ring, np.var)
    ig, rg = stat(gradient, mask, np.mean), stat(gradient, ring, np.mean)
    differences, boundary = boundary_values(gray, mask)
    ys, xs = np.nonzero(mask)
    w, h = int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)
    delta = None if rm is None else im-rm
    result = dict(inner_mean=im, inner_std=ist, ring_mean=rm, ring_std=rst,
        intensity_difference=delta, absolute_intensity_difference=None if delta is None else abs(delta),
        intensity_ratio=ratio(im, rm), intensity_zscore=ratio(delta, rst),
        inner_laplacian_variance=il, ring_laplacian_variance=rl, texture_ratio=ratio(il, rl),
        inner_gradient_mean=ig, ring_gradient_mean=rg, gradient_ratio=ratio(ig, rg),
        boundary_discontinuity_mean=float(differences.mean()) if differences.size else None,
        boundary_discontinuity_median=float(np.median(differences)) if differences.size else None,
        boundary_discontinuity_max=float(differences.max()) if differences.size else None,
        boundary_sobel_response=stat(gradient, boundary, np.mean),
        mask_width=w, mask_height=h, mask_area=int(mask.sum()), aspect_ratio=w/h,
        mask_x=int(xs.min()), mask_y=int(ys.min()), ring_area=int(ring.sum()),
        boundary_crossing_edges=int(differences.size), boundary_pixels=int(boundary.sum()))
    if any(v is not None and not np.isfinite(v) for v in result.values()):
        raise RuntimeError('Nonfinite region feature')
    return result


def cliffs_delta(real, fake):
    """Positive means Real tends to exceed Fake; ties contribute zero."""
    if not len(real) or not len(fake):
        return None
    return float(np.sign(np.asarray(real)[:, None]-np.asarray(fake)[None, :]).mean())


def scope_rows(rows, scope):
    if scope == 'all':
        return rows
    if scope == 'train':
        return [r for r in rows if r['split'] == 'train']
    shared = ({r['asset_id'] for r in rows if r['group'] == 'real'} &
              {r['asset_id'] for r in rows if r['group'] == 'fake'})
    return [r for r in rows if r['asset_id'] in shared]


def summarize(rows):
    summary = []
    for scope in ('all', 'train', 'matched_images'):
        for metric in FEATURES:
            record = dict(scope=scope, metric=metric, key_metric=metric in KEY_METRICS)
            values = {}
            for group in ('real', 'fake'):
                selected = [r for r in scope_rows(rows, scope) if r['group'] == group]
                v = np.array([r[metric] for r in selected if r[metric] is not None], float)
                values[group] = v
                record[group+'_count'] = len(v)
                record[group+'_undefined_count'] = len(selected)-len(v)
                for stat in ('mean', 'std', 'median', 'q1', 'q3', 'min', 'max'):
                    result = None
                    if v.size:
                        result = float(np.quantile(v, .25 if stat == 'q1' else .75)) if stat in ('q1','q3') else float(getattr(np,stat)(v))
                    record[group+'_'+stat] = result
            a, b = values['real'], values['fake']
            available = len(a) > 0 and len(b) > 0
            # Tie-corrected asymptotic two-sided U test: exploratory, clustered regions.
            test = mannwhitneyu(a, b, alternative='two-sided', method='asymptotic') if available else None
            record.update(mann_whitney_u=float(test.statistic) if test else None,
                          p_value=float(test.pvalue) if test else None,
                          cliffs_delta=cliffs_delta(a, b),
                          ks_distance=float(ks_2samp(a,b).statistic) if available else None,
                          fake_minus_real_median=float(np.median(b)-np.median(a)) if available else None,
                          p_value_bh=None)
            summary.append(record)
        valid = [r for r in summary if r['scope'] == scope and r['p_value'] is not None]
        ordered = sorted(valid, key=lambda r:r['p_value'])
        adjusted = 1.
        for i in range(len(ordered)-1, -1, -1):
            adjusted = min(adjusted, ordered[i]['p_value']*len(ordered)/(i+1))
            ordered[i]['p_value_bh'] = adjusted
    return summary


def classifier_diagnostic(rows, seed, features=CLASSIFIER_FEATURES):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    x = np.array([[np.nan if r[f] is None else r[f] for f in features] for r in rows], float)
    y = np.array([int(r['group'] == 'fake') for r in rows])
    groups = np.array([r['asset_id'] for r in rows])
    n_splits = min(5, *(len(set(groups[y == c])) for c in (0,1)))
    warning = 'Small sample: exploratory grouped CV only; low AUC does not establish equivalence.'
    if n_splits < 2:
        return dict(status='unavailable', reason='Fewer than two image groups per class', warning=warning), []
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    probabilities, fold_ids = np.zeros(len(rows)), np.full(len(rows), -1)
    folds = []
    def scores(truth, proba):
        pred = proba >= .5
        return dict(roc_auc=float(roc_auc_score(truth, proba)) if len(set(truth)) == 2 else None,
                    balanced_accuracy=float(balanced_accuracy_score(truth, pred)),
                    precision=float(precision_score(truth, pred, zero_division=0)),
                    recall=float(recall_score(truth, pred, zero_division=0)))
    for fold, (train, test) in enumerate(cv.split(x,y,groups)):
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError('Classifier image-group leakage')
        if len(set(y[train])) != 2:
            return dict(status='unavailable', reason='Single-class training fold', warning=warning), []
        # Every imputation and scaling statistic is fit on the training fold only.
        model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(),
            LogisticRegression(C=1., class_weight='balanced', max_iter=5000, random_state=seed))
        with threadpool_limits(limits=1):
            model.fit(x[train], y[train])
            probabilities[test] = model.predict_proba(x[test])[:,1]
        fold_ids[test] = fold
        folds.append(dict(fold=fold, train_groups=sorted(set(groups[train])), test_groups=sorted(set(groups[test])),
                          train_regions=len(train), test_regions=len(test), **scores(y[test], probabilities[test])))
    if (fold_ids < 0).any():
        raise RuntimeError('Missing out-of-fold prediction')
    predictions = [dict(asset_id=r['asset_id'], region_id=r['region_id'], group=r['group'],
                        fold=int(fold_ids[i]), fake_probability=float(probabilities[i]), truth=int(y[i]))
                   for i,r in enumerate(rows)]
    return dict(status='ok', model='LogisticRegression', features=list(features), positive_class='fake',
                seed=seed, regions=len(rows), original_image_groups=len(set(groups)), n_splits=n_splits,
                warning=warning, **scores(y, probabilities), folds=folds), predictions


def save_region(output, row, rgb, mask, ring, padding):
    ys, xs = np.nonzero(mask | ring)
    roi = np.s_[max(0,ys.min()-padding):min(mask.shape[0],ys.max()+padding+1),
                max(0,xs.min()-padding):min(mask.shape[1],xs.max()+padding+1)]
    patch = rgb[roi].copy()
    m, r = mask[roi], ring[roi]
    marked = patch.copy()
    marked[m] = np.round(.3*marked[m]+.7*np.array([255,75,75])).astype(np.uint8)
    marked[r] = np.round(.5*marked[r]+.5*np.array([0,230,255])).astype(np.uint8)
    dest = output/'regions'/row['group']/f"{row['asset_id']}_{row['region_id']:03d}"
    dest.mkdir(parents=True)
    for name, array in (('restored_patch',patch), ('mask_and_ring',marked),
                        ('mask',m.astype(np.uint8)*255), ('outer_ring',r.astype(np.uint8)*255)):
        Image.fromarray(array).save(dest/(name+'.png'))
    # Nearest-neighbor enlarged review panel; numeric features stay at native resolution.
    scale = max(1, min(4, 240//max(1,patch.shape[0])))
    a, b = [Image.fromarray(arr).resize((patch.shape[1]*scale,patch.shape[0]*scale), Image.Resampling.NEAREST)
            for arr in (patch, marked)]
    panel = Image.new('RGB',(max(420, a.width*2+12), a.height+52),'#171717')
    panel.paste(a,(0,52)); panel.paste(b,(a.width+12,52))
    draw = ImageDraw.Draw(panel)
    draw.text((8,6), f"{row['group']} | {row['asset_id'][:12]} | region {row['region_id']}",fill='white')
    draw.text((8,25), 'After only | red: mask | cyan: ring',fill='white')
    panel.save(dest/'panel.png')


def plot_distributions(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = output/'distributions'
    directory.mkdir()
    for metric in KEY_METRICS:
        fig, axes = plt.subplots(1,2,figsize=(10,4.5), layout='constrained')
        for ax, scope in zip(axes, ('all','matched_images')):
            selected = scope_rows(rows,scope)
            data = [[r[metric] for r in selected if r['group']==g and r[metric] is not None] for g in ('real','fake')]
            ax.boxplot(data, tick_labels=[f'Real (n={len(data[0])})',f'Fake (n={len(data[1])})'], showfliers=False)
            for i, values in enumerate(data,1):
                ax.scatter(np.full(len(values),i)+np.linspace(-.09,.09,len(values)), values,
                           s=18, alpha=.65, color=('#c84b47' if i==1 else '#2586b4'))
            ax.set_title(scope.replace('_',' ')); ax.set_ylabel(metric); ax.grid(axis='y',alpha=.2)
        fig.suptitle('After-only: '+metric, fontsize=13)
        fig.savefig(directory/(metric+'.png'),dpi=160)
        plt.close(fig)


def analyze(manifest, debug, prior, output, *, ring_width=5, seed=None, padding=8, classifier=True):
    manifest, debug, prior, output = [Path(p).resolve() for p in (manifest,debug,prior,output)]
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    if ring_width < 1 or padding < 0 or (seed is not None and seed < 0):
        raise ValueError('Invalid ring width, padding or seed')
    protected_roots = [ROOT/'dataset', ROOT/'scripts', ROOT/'outputs/inpainting_poc_20',
                       ROOT/'outputs/yolo_poc_20', manifest.parent, debug, prior]
    for p in protected_roots:
        if p == output or p in output.parents or output in p.parents:
            raise ValueError(f'Output overlaps protected directory: {p}')
    # Hash entire raw dataset, existing selected datasets, masks, diagnostics and code.
    protected = {p.resolve():sha256(p) for root in protected_roots for p in root.rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts}
    protected[manifest] = sha256(manifest)
    audit_path = debug/'per_image.json'
    audits = json.loads(audit_path.read_text(encoding='utf-8'))
    by_asset = {r['asset_id']:r for r in audits}
    manifest_rows = sorted(read_csv(manifest),key=lambda r:r['asset_id'])
    if not manifest_rows or len(by_asset) != len(audits) or len({r['asset_id'] for r in manifest_rows}) != len(manifest_rows) or set(by_asset) != {r['asset_id'] for r in manifest_rows}:
        raise ValueError('Manifest/audit assets mismatch')
    stored_seeds = sorted({a['seed'] for a in audits})
    if seed is None:
        if len(stored_seeds) != 1:
            raise ValueError('Multiple stored seeds: specify diagnostic --seed')
        seed = stored_seeds[0]
    rows, views = [], []
    for source in manifest_rows:
        asset = source['asset_id']; audit = by_asset[asset]
        after_path = Path(source['output_image_path']).resolve()
        if Path(audit['image']).resolve() != after_path or source['split'] != audit['split']:
            raise ValueError('Audit image/split mismatch')
        for key, digest_key in (('original_image_path','original_sha256'),('restored_image_path','restored_sha256'),
                                ('output_image_path','output_sha256')):
            path = Path(source[key]).resolve()
            protected.setdefault(path,sha256(path))
            if protected[path] != source[digest_key]:
                raise ValueError(f'Image provenance mismatch: {path}')
        # The ONLY image decoded for feature extraction is the final augmented after image.
        rgb = load_rgb(after_path)
        real_path = Path(audit['real_mask']).resolve()
        fake_path = debug/(asset+'_fake_mask.png')
        for p in (real_path, fake_path):
            protected.setdefault(p,sha256(p))
        real, fake = [load_mask(p,rgb.shape[:2]) for p in (real_path,fake_path)]
        parts = dict(real=real_regions(real), fake=fake_regions(fake,audit['placements']))
        if len(parts['fake']) != audit['applied_count'] or (source['split']!='train' and fake.any()) or (real & fake).any():
            raise ValueError('Invalid fake region count/split or real/fake overlap')
        maps = after_maps(rgb)
        for group in ('real','fake'):
            for i,mask in enumerate(parts[group],1):
                ring = make_ring(mask,ring_width)
                features = measure_region(maps,mask,ring)
                row = dict(asset_id=asset, split=source['split'], group=group, region_id=i,
                    augmentation_seed=audit['seed'], diagnostic_seed=seed, after_image=str(after_path),
                    method=audit['method'], radius=audit['radius'], ring_width=ring_width,
                    ring_other_restoration_pixels=int((ring & (real | fake)).sum()),
                    **features)
                row['undefined_features'] = ';'.join(f for f in FEATURES if row[f] is None)
                rows.append(row); views.append((row,rgb,mask,ring))
    for group in ('real','fake'):
        previous = read_csv(prior/(group+'.csv'))
        expected = {(r['asset_id'],int(r['region_id'])) for r in previous}
        actual = {(r['asset_id'],r['region_id']) for r in rows if r['group']==group}
        if expected != actual:
            raise ValueError(f'Region identities differ from previous diagnostic: {group}')
    summary = summarize(rows)
    classifier_results, predictions = {}, []
    if classifier:
        print('WARNING: small sample; grouped classifier and p-values are exploratory, not equivalence tests.', flush=True)
        for scope in ('all','matched_images'):
            for setting,features in (('full_features',CLASSIFIER_FEATURES),
                                     ('without_shape',tuple(f for f in CLASSIFIER_FEATURES if f not in ('mask_area','aspect_ratio')))):
                key = scope+'_'+setting
                result, pred = classifier_diagnostic(scope_rows(rows,scope),seed,features)
                classifier_results[key] = result
                predictions.extend(dict(analysis=key,**p) for p in pred)
    output.mkdir(parents=True,exist_ok=False)
    for group in ('real','fake'):
        write_csv(output/(group+'_after_features.csv'),[r for r in rows if r['group']==group],list(rows[0]))
    write_csv(output/'comparison_summary.csv',summary,list(summary[0]))
    if predictions:
        write_csv(output/'classifier_oof_predictions.csv',predictions,list(predictions[0]))
    (output/'classifier_diagnostic.json').write_text(json.dumps(classifier_results,indent=2,allow_nan=False),encoding='utf-8')
    for row,rgb,mask,ring in views:
        save_region(output,row,rgb,mask,ring,padding)
    plot_distributions(rows,output)
    if any(sha256(p)!=digest for p,digest in protected.items()):
        raise RuntimeError('Protected dataset/code/output file changed')
    # Also detect additions/deletions to the protected tree, excluding Python bytecode caches.
    current = {p.resolve() for root in protected_roots for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if not current.issubset(protected):
        raise RuntimeError('Unexpected file added to protected tree')
    metadata = dict(images=len(manifest_rows), real_regions=sum(r['group']=='real' for r in rows),
        fake_regions=sum(r['group']=='fake' for r in rows), ring_width=ring_width,
        augmentation_seeds=stored_seeds, diagnostic_seed=seed, nonfinite_numeric_features=0,
        undefined_features={g:{f:sum(r['group']==g and r[f] is None for r in rows) for f in FEATURES} for g in ('real','fake')},
        empty_rings=sum(r['ring_area']==0 for r in rows), mask_ring_overlap_pixels=0,
        protected_files_verified=len(protected), input_hashes={str(p):v for p,v in protected.items()},
        opencv_version=cv2.__version__, numpy_version=np.__version__,
        after_source='Final fake-augmented image for BOTH Real and Fake; no before image decoded.')
    import scipy, sklearn, matplotlib
    metadata.update(scipy_version=scipy.__version__, sklearn_version=sklearn.__version__, matplotlib_version=matplotlib.__version__)
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding='utf-8')
    (output/'README.md').write_text(DEFINITIONS,encoding='utf-8')
    print(json.dumps({k:metadata[k] for k in ('images','real_regions','fake_regions','protected_files_verified','empty_rings','diagnostic_seed')},indent=2))
    return rows,summary,classifier_results


DEFINITIONS = """# After-only Restoration Artifact Similarity

Both groups use pixels from the SAME final fake-augmented image. Before/color-marked
images are hashed for integrity but never decoded or used for features.
Real = saved real mask 8-connected components; Fake = saved placement masks.
Every region is saved, so representative views can be selected without random sampling.

Ring = ellipse-kernel dilation (2*N+1 square, default N=5) minus the region mask,
clipped to image bounds with zero morphology border. Ring includes holes in hollow masks.
Other restored pixels are NOT silently removed from the specified ring; the CSV records
ring_other_restoration_pixels to make such context contamination visible.

Intensity: OpenCV RGB-to-gray 0..255. Std and Laplacian variance use population moments.
Difference = inner mean - ring mean; absolute difference is its absolute value.
Intensity ratio = inner mean/ring mean; z-score = difference/ring std.
Laplacian ksize=3; gradient = hypot(Sobel x, Sobel y), ksize=3.
Derivative maps are computed on full after images with REFLECT_101 borders before masking.
Texture/gradient ratios divide inner statistics by ring statistics.
Boundary discontinuity = abs(gray inside - gray outside) on every in-image 4-neighbor
mask crossing edge, counted once. Includes hole boundaries; ignores out-of-image edges.
Boundary Sobel response = mean gradient on unique inner boundary pixels.
Width/height = tight bounding rectangle; area = mask pixel count; aspect = width/height.
Empty ring/boundary or denominator <= 1e-12 in magnitude -> undefined (empty CSV cell,
JSON null), NOT zero. No NaN/inf is exported; undefined counts are recorded separately.

Comparison scopes: all regions; train-only; matched_images (only image IDs with BOTH
classes). Count excludes undefined values; std is population std; quartiles are linear.
Mann-Whitney U: Real first, two-sided, tie-corrected asymptotic, continuity correction.
BH p-values correct across features within each scope. Regions within an image are
correlated: these p-values remain exploratory, not confirmatory independent-sample tests.
Cliff's delta > 0 means Real tends to be larger. KS distance supplements delta to detect
distribution differences even when rank effects cancel. Neither is an equivalence test.

Classifier: fixed Logistic Regression C=1, class_weight=balanced, threshold=.5, Fake=1;
StratifiedGroupKFold up to 5 splits with original asset ID as group. Imputer and scaler
fit on training folds only. No hyperparameter tuning. All and matched-image analyses,
with and without shape features; all use out-of-fold predictions. Every test/train image
ID is recorded to audit leakage. Fold AUC is null if a test fold contains only one class.
Small sample and repeated regions limit interpretation; low AUC does not prove identity.
Area, location/background and proximity to actual defects can distinguish groups without
reflecting restoration artifacts alone. Removing shape features is a sensitivity check.

This diagnostic never changes the restoration algorithm, data, masks, or earlier outputs.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'outputs/yolo_poc_20_fake_restoration/manifest.csv')
    parser.add_argument('--debug-dir',type=Path,default=ROOT/'outputs/fake_restoration_debug')
    parser.add_argument('--prior-diagnostics',type=Path,default=ROOT/'outputs/restoration_diagnostics')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/restoration_after_similarity')
    parser.add_argument('--ring-width',type=int,default=5)
    parser.add_argument('--seed',type=int,default=None,help='Default: inherit augmentation metadata seed')
    parser.add_argument('--padding',type=int,default=8)
    parser.add_argument('--skip-classifier',action='store_true')
    args = parser.parse_args()
    analyze(args.manifest,args.debug_dir,args.prior_diagnostics,args.output,ring_width=args.ring_width,
            seed=args.seed,padding=args.padding,classifier=not args.skip_classifier)


if __name__ == '__main__':
    main()
