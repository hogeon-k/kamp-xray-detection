"""Paired YOLO shortcut diagnostic on immutable restoration image pairs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw

from create_yolo_poc_20 import parse_yolo
from fake_restoration import bbox_mask

ROOT = Path(__file__).resolve().parents[1]
BASELINE_RUN = ROOT/'outputs/yolo_runs/telea_yolo11n_e200_img640_b8_adamw_lr0.001_wd0.0005_noaug_s42'
FAKE_AUG_RUN = ROOT/'outputs/yolo_runs/telea_fake_restoration_yolo11n_e200_img640_b8_adamw_lr0.001_wd0.0005_noaug_s42'
CONTEXT = ROOT/'outputs/context_matched_restoration'
SOURCE_DATASET = ROOT/'outputs/yolo_poc_20_fake_restoration'
DATASET_OUT = ROOT/'outputs/shortcut_diagnostic_dataset'
OUT = ROOT/'outputs/shortcut_diagnostic'
MODELS = ('baseline','fake_aug')
SIDES = ('original','fake')


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path,rows,fields):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def load_rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert('RGB'))


def load_mask(path,shape):
    with Image.open(path) as image:
        mask=np.array(image.convert('L'))
    if mask.shape!=shape or not np.isin(mask,[0,255]).all():
        raise ValueError(f'Invalid binary mask: {path}')
    return mask>0


def box_iou(a,b):
    ix=max(0.,min(a[2],b[2])-max(a[0],b[0]))
    iy=max(0.,min(a[3],b[3])-max(a[1],b[1]))
    intersection=ix*iy
    area_a=max(0.,a[2]-a[0])*max(0.,a[3]-a[1])
    area_b=max(0.,b[2]-b[0])*max(0.,b[3]-b[1])
    union=area_a+area_b-intersection
    return intersection/union if union>0 else 0.


def spatial_relation(box,mask,fake_bbox):
    h,w=mask.shape
    xc,yc=(box[0]+box[2])/2,(box[1]+box[3])/2
    center_in_bbox=bool(fake_bbox[0]<=xc<fake_bbox[2] and fake_bbox[1]<=yc<fake_bbox[3])
    center_in_mask=bool(0<=xc<w and 0<=yc<h and mask[int(yc),int(xc)])
    # Rasterized coverage captures hollow/skinny masks better than bbox area alone.
    x0=max(0,int(math.floor(box[0])));y0=max(0,int(math.floor(box[1])))
    x1=min(w,int(math.ceil(box[2])));y1=min(h,int(math.ceil(box[3])))
    covered=int(mask[y0:y1,x0:x1].sum()) if x1>x0 and y1>y0 else 0
    coverage=covered/int(mask.sum())
    return dict(iou_fake_bbox=box_iou(box,fake_bbox),fake_mask_coverage=coverage,
                center_in_fake_bbox=center_in_bbox,center_in_fake_mask=center_in_mask)


def annotate_predictions(predictions,boxes,mask,fake_bbox,*,conf,raw_conf,gt_iou,coverage_threshold):
    """Return prediction records; GT-matched detections are excluded from fake-FP."""
    records=[]
    for i,p in enumerate(predictions):
        box=tuple(float(v) for v in p['box'])
        confidence=float(p['conf'])
        if len(box)!=4 or not all(math.isfinite(v) for v in (*box,confidence)) or not 0<=confidence<=1:
            raise ValueError('Invalid YOLO detection')
        best=max((box_iou(box,g) for g in boxes),default=0.)
        relation=spatial_relation(box,mask,fake_bbox)
        near=relation['center_in_fake_bbox'] or relation['center_in_fake_mask'] or relation['fake_mask_coverage']>=coverage_threshold
        records.append(dict(prediction_id=i+1,box=box,conf=confidence,class_id=int(p['class_id']),
            max_gt_iou=best,gt_matched=best>=gt_iou,spatial_fake=bool(near),
            fake_fp_candidate=bool(near and best<gt_iou),
            fake_fp_final=bool(near and best<gt_iou and confidence>=conf),**relation))
    return records


def region_summary(predictions):
    fake=[p for p in predictions if p['fake_fp_candidate']]
    final=[p for p in fake if p['fake_fp_final']]
    return dict(max_confidence=max((p['conf'] for p in fake),default=0.),
                final_count=len(final),final_present=bool(final))


def gt_confidences(predictions,boxes,gt_iou):
    return [max((p['conf'] for p in predictions if box_iou(p['box'],gt)>=gt_iou),default=0.) for gt in boxes]


def paired_wilcoxon(baseline,augmented):
    if len(baseline)!=len(augmented):
        raise ValueError('Paired lengths differ')
    diff=np.asarray(augmented,float)-np.asarray(baseline,float)
    if not np.isfinite(diff).all():
        raise ValueError('Nonfinite paired difference')
    nonzero=np.round(diff,12)!=0
    if nonzero.any():
        stat,p=exact_signed_rank(np.round(diff[nonzero],12))
    else:
        stat,p=0.,1.
    return dict(n=len(diff),baseline_mean=float(np.mean(baseline)) if len(diff) else None,
                fake_aug_mean=float(np.mean(augmented)) if len(diff) else None,
                paired_mean_difference=float(diff.mean()) if len(diff) else None,
                paired_median_difference=float(np.median(diff)) if len(diff) else None,
                positive_difference_rate=float((diff>0).mean()) if len(diff) else None,
                wilcoxon_statistic=stat if len(diff) else None,p_value=p if len(diff) else None)


def exact_signed_rank(nonzero_differences):
    """Two-sided exact Wilcoxon sign-permutation p, including average-tie ranks."""
    differences=np.asarray(nonzero_differences,float)
    if not len(differences) or not np.isfinite(differences).all() or np.any(differences==0):
        raise ValueError('Expected finite nonzero paired differences')
    magnitudes=np.abs(differences)
    order=np.argsort(magnitudes,kind='stable')
    ranks2=np.zeros(len(differences),dtype=int)
    start=0
    while start<len(order):
        end=start+1
        while end<len(order) and magnitudes[order[end]]==magnitudes[order[start]]:
            end+=1
        # Twice the average of one-based ranks in [start+1, end].
        ranks2[order[start:end]]=start+1+end
        start=end
    total=int(ranks2.sum())
    observed=int(ranks2[differences>0].sum())
    counts=[0]*(total+1);counts[0]=1
    reachable=0
    for rank in ranks2:
        rank=int(rank)
        for position in range(reachable,-1,-1):
            if counts[position]:
                counts[position+rank]+=counts[position]
        reachable+=rank
    smaller=min(observed,total-observed)
    probability=min(1.,2*sum(counts[:smaller+1])/(2**len(differences)))
    return smaller/2,float(probability)


def mcnemar_exact(baseline,augmented):
    a=np.asarray(baseline,bool);b=np.asarray(augmented,bool)
    if a.shape!=b.shape:
        raise ValueError('Paired lengths differ')
    both=int((a&b).sum());baseline_only=int((a&~b).sum());aug_only=int((~a&b).sum());neither=int((~a&~b).sum())
    discordant=baseline_only+aug_only
    p=min(1.,2*sum(math.comb(discordant,i) for i in range(min(baseline_only,aug_only)+1))/(2**discordant)) if discordant else 1.
    return dict(both=both,baseline_only=baseline_only,fake_aug_only=aug_only,neither=neither,
                discordant=discordant,mcnemar_exact_p=float(p),risk_difference_fake_aug_minus_baseline=float(b.mean()-a.mean()) if len(a) else None)


def summarize_delta(values):
    a=np.array(values,float)
    return dict(mean=float(a.mean()),median=float(np.median(a)),q1=float(np.quantile(a,.25)),
                q3=float(np.quantile(a,.75)),max=float(a.max()),positive_ratio=float((a>0).mean()),
                above_005_ratio=float((a>.05).mean()),above_010_ratio=float((a>.10).mean()),
                above_025_ratio=float((a>.25).mean())) if len(a) else {}


def source_snapshot():
    """Hash all raw/training data, YOLO weights, and restoration/fake code."""
    roots=(ROOT/'dataset',ROOT/'outputs/yolo_poc_20',ROOT/'outputs/yolo_poc_20_fake_restoration',
           ROOT/'outputs/inpainting_poc_20',ROOT/'outputs/fake_restoration_debug',CONTEXT)
    paths={p.resolve() for root in roots for p in root.rglob('*') if p.is_file()}
    for run in (BASELINE_RUN,FAKE_AUG_RUN):
        paths.add((run/'weights/best.pt').resolve())
        paths.add((run/'summary.json').resolve())
        paths.add((run/'experiment_config.json').resolve())
    for script in ('fake_restoration.py','create_fake_restoration_dataset.py','diagnose_context_matched_restoration.py',
                   'diagnose_yolo_shortcut.py'):
        paths.add((ROOT/'scripts'/script).resolve())
    return {p:sha256(p) for p in sorted(paths)}


def validate_training_configs():
    configs=[json.loads((run/'experiment_config.json').read_text(encoding='utf-8')) for run in (BASELINE_RUN,FAKE_AUG_RUN)]
    ignored={'preprocessing','data','run_name','start_time','end_time','duration_seconds'}
    differences={k:(configs[0].get(k),configs[1].get(k)) for k in set(configs[0])|set(configs[1])
                 if k not in ignored and configs[0].get(k)!=configs[1].get(k)}
    if differences:
        raise ValueError(f'Model training settings differ: {differences}')
    return configs


def prepare_pairs(source_rows):
    manifest={r['asset_id']:r for r in read_csv(SOURCE_DATASET/'manifest.csv')}
    pairs=[]
    for row in sorted(source_rows,key=lambda r:r['pair_id']):
        pair_id,image_id=row['pair_id'],row['image_id']
        if image_id not in manifest:
            raise ValueError(f'Missing image manifest: {image_id}')
        record=manifest[image_id]
        original=CONTEXT/'paired_images'/f'{pair_id}_real.png'
        fake=CONTEXT/'paired_images'/f'{pair_id}_fake.png'
        mask_path=CONTEXT/'masks'/f'{pair_id}_fake.png'
        for p in (original,fake,mask_path):
            if not p.is_file():
                raise FileNotFoundError(p)
        a,b=load_rgb(original),load_rgb(fake)
        if a.shape!=b.shape:
            raise ValueError('Pair image sizes differ')
        mask=load_mask(mask_path,a.shape[:2])
        if not mask.any() or not np.array_equal(a[~mask],b[~mask]):
            raise ValueError(f'Pair differs outside fake mask: {pair_id}')
        restored=Path(record['restored_image_path'])
        if sha256(original)!=sha256(restored):
            raise ValueError('Original pair is not the real-restored baseline')
        label=Path(record['output_label_path'])
        if record['split'] not in ('train','val','test') or int(record['bbox_count'])<1:
            raise ValueError('Invalid source split/GT')
        boxes=parse_yolo(label,a.shape[1],a.shape[0])
        if len(boxes)!=int(record['bbox_count']):
            raise ValueError('GT count mismatch')
        yy,xx=np.nonzero(mask)
        bbox=(int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1))
        expected=(int(row['fake_x']),int(row['fake_y']),int(row['fake_x'])+int(row['width']),int(row['fake_y'])+int(row['height']))
        if bbox!=expected or int(mask.sum())!=int(row['area']):
            raise ValueError('Fake mask metadata mismatch')
        if any(mask[max(0,math.floor(y0)):min(mask.shape[0],math.ceil(y1)),
                    max(0,math.floor(x0)):min(mask.shape[1],math.ceil(x1))].any() for x0,y0,x1,y1 in boxes):
            raise ValueError('Fake mask overlaps GT')
        if (mask & bbox_mask(mask.shape,boxes,float(row['safety_margin']))).any():
            raise ValueError('Fake mask overlaps GT safety margin')
        if any(r['pair_id']==pair_id for r in pairs):
            raise ValueError('Duplicate pair ID')
        pairs.append(dict(pair_id=pair_id,image_id=image_id,source_split=record['split'],seed=int(row['seed']),
                          image_shape=a.shape,bbox=bbox,mask_area=int(mask.sum()),gt_boxes=boxes,
                          original_source=original,fake_source=fake,mask_source=mask_path,
                          original_hash=sha256(original),fake_hash=sha256(fake),label_hash=sha256(label)))
    if not pairs:
        raise ValueError('No diagnostic image pairs')
    return pairs


def inference_device(requested):
    import torch
    if str(requested) == '0' and not torch.cuda.is_available():
        print('CUDA device 0 사용 불가: 동일한 CPU 설정으로 두 모델을 추론합니다.',flush=True)
        return 'cpu'
    return str(requested)


def predict_pair(model,original,fake,settings):
    results=model.predict(source=[str(original),str(fake)],imgsz=settings['imgsz'],conf=settings['raw_conf'],
        iou=settings['iou'],device=settings['device'],augment=False,verbose=False,save=False,
        classes=[0],agnostic_nms=False,max_det=settings['max_det'],rect=False)
    if len(results)!=2:
        raise RuntimeError('YOLO returned the wrong number of paired predictions')
    converted=[]
    for r in results:
        boxes=r.boxes
        if len(boxes)>=settings['max_det']:
            raise RuntimeError('Raw predictions reached max_det; increase --max-det before measuring confidence')
        converted.append([dict(box=[float(v) for v in xyxy],conf=float(conf),class_id=int(cls))
                          for xyxy,conf,cls in zip(boxes.xyxy.cpu().numpy(),boxes.conf.cpu().numpy(),boxes.cls.cpu().numpy())])
    return converted


def metrics_for_scope(region_rows,gt_rows,model,scope,normal):
    regions=[r for r in region_rows if r['model']==model and (scope=='all' or r['source_split'] in
             (('val','test') if scope=='holdout' else (scope,)))]
    gt=[r for r in gt_rows if r['model']==model and (scope=='all' or r['source_split'] in
        (('val','test') if scope=='holdout' else (scope,)))]
    if not regions:
        return None
    fake_conf=np.array([r['fake_max_confidence'] for r in regions],float)
    deltas=np.array([r['delta_confidence'] for r in regions],float)
    split='test' if scope in ('all','holdout','test') else 'val' if scope=='val' else None
    return dict(model=model,scope=scope,diagnostic_pairs=len(regions),fake_regions=len(regions),
        source_images=len({r['image_id'] for r in regions}),normal_metrics_source=split,
        normal_precision=normal.get(split+'_precision') if split else None,
        normal_recall=normal.get(split+'_recall') if split else None,
        normal_map50=normal.get(split+'_map50') if split else None,
        normal_map50_95=normal.get(split+'_map50_95') if split else None,
        fr_fpr=float(np.mean([r['fake_fp_present'] for r in regions])),
        fake_fp_total=sum(r['fake_fp_count'] for r in regions),
        fake_fp_pair_images=sum(r['fake_fp_present'] for r in regions),
        fake_fp_unique_source_images=len({r['image_id'] for r in regions if r['fake_fp_present']}),
        mean_fake_region_max_conf=float(fake_conf.mean()),median_fake_region_max_conf=float(np.median(fake_conf)),
        mean_delta_confidence=float(deltas.mean()),median_delta_confidence=float(np.median(deltas)),
        delta_q1=float(np.quantile(deltas,.25)),delta_q3=float(np.quantile(deltas,.75)),
        delta_max=float(deltas.max()),positive_delta_rate=float((deltas>0).mean()),
        delta_gt_005_rate=float((deltas>.05).mean()),delta_gt_010_rate=float((deltas>.10).mean()),
        delta_gt_025_rate=float((deltas>.25).mean()),
        new_fake_fp_count=sum(r['new_fake_fp'] for r in regions),
        new_fake_fp_rate=float(np.mean([r['new_fake_fp'] for r in regions])),
        gt_count=len(gt),gt_detection_loss_count=sum(r['detection_lost'] for r in gt),
        gt_detection_gain_count=sum(r['detection_gained'] for r in gt),
        mean_gt_confidence_change=float(np.mean([r['delta_confidence'] for r in gt])) if gt else None)


def paired_comparison(region_rows,scope):
    selected=[r for r in region_rows if scope=='all' or r['source_split'] in ('val','test')]
    by_pair=defaultdict(dict)
    for r in selected:
        by_pair[r['pair_id']][r['model']]=r
    if not by_pair or any(set(v)!=set(MODELS) for v in by_pair.values()):
        raise ValueError('Models lack corresponding pair results')
    ordered=[by_pair[key] for key in sorted(by_pair)]
    results=[]
    for metric in ('fake_max_confidence','delta_confidence'):
        values=paired_wilcoxon([r['baseline'][metric] for r in ordered],
                               [r['fake_aug'][metric] for r in ordered])
        results.append(dict(scope=scope,metric=metric,test='Wilcoxon signed-rank',**values))
    discord=mcnemar_exact([r['baseline']['fake_fp_present'] for r in ordered],
                          [r['fake_aug']['fake_fp_present'] for r in ordered])
    results.append(dict(scope=scope,metric='fake_fp_present',test='McNemar exact',n=len(ordered),
                        baseline_mean=np.mean([r['baseline']['fake_fp_present'] for r in ordered]),
                        fake_aug_mean=np.mean([r['fake_aug']['fake_fp_present'] for r in ordered]),
                        paired_mean_difference=discord['risk_difference_fake_aug_minus_baseline'],
                        paired_median_difference=None,positive_difference_rate=None,wilcoxon_statistic=None,
                        p_value=discord['mcnemar_exact_p'],**discord))
    return results


def save_plots(output,region_rows,summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_dir=output/'plots';plot_dir.mkdir()
    holdout={r['model']:r for r in summaries if r['scope']=='holdout'}
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
    ax.bar(['Baseline','Fake-Aug'],[holdout[m]['fr_fpr'] for m in MODELS],color=['#c35b55','#418daf'])
    ax.set_ylim(0,1);ax.set_ylabel('FR-FPR');ax.set_title('Held-out val/test pairs')
    fig.savefig(plot_dir/'fr_fpr_comparison.png',dpi=160);plt.close(fig)
    for metric,name in (('delta_confidence','delta_confidence_distribution.png'),
                        ('fake_max_confidence','fake_region_confidence_comparison.png')):
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        for model,color in (('baseline','#c35b55'),('fake_aug','#418daf')):
            vals=[r[metric] for r in region_rows if r['model']==model and r['source_split'] in ('val','test')]
            ax.hist(vals,bins=np.linspace(-1 if metric=='delta_confidence' else 0,1,21),alpha=.5,label=model,color=color)
        ax.legend();ax.set(xlabel=metric,ylabel='Held-out regions',title=metric+' (val/test)')
        fig.savefig(plot_dir/name,dpi=160);plt.close(fig)
    pairwise=defaultdict(dict)
    for r in region_rows:
        if r['source_split'] in ('val','test'):
            pairwise[r['pair_id']][r['model']]=r
    fig,ax=plt.subplots(figsize=(5,5),layout='constrained')
    x=[v['baseline']['fake_max_confidence'] for v in pairwise.values()]
    y=[v['fake_aug']['fake_max_confidence'] for v in pairwise.values()]
    ax.scatter(x,y);ax.plot([0,1],[0,1],'--',color='gray')
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Baseline fake max confidence',
           ylabel='Fake-Aug fake max confidence',title='Same held-out fake regions')
    fig.savefig(plot_dir/'baseline_vs_fake_aug_scatter.png',dpi=160);plt.close(fig)


def draw_case_panel(dest,dataset,pair,predictions,conf):
    images={side:load_rgb(dataset/side/(pair['pair_id']+'.png')) for side in SIDES}
    # Caller copies clean images into the case directory; preserve original pixels.
    for side,rgb in images.items():
        Image.fromarray(rgb).save(dest/f'clean_{side}.png')
    panels=[]
    mask=load_mask(dataset/'masks'/(pair['pair_id']+'.png'),images['fake'].shape[:2])
    for model in MODELS:
        for side in SIDES:
            canvas=Image.fromarray(images[side]);draw=ImageDraw.Draw(canvas)
            ys,xs=np.nonzero(mask)
            draw.rectangle((int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())),outline='lime',width=2)
            for box in pair['gt_boxes']:
                draw.rectangle(box,outline='cyan',width=2)
            for p in predictions[(pair['pair_id'],model,side)]:
                if p['conf']<conf: continue
                color='red' if p['fake_fp_final'] else 'yellow' if p['gt_matched'] else 'orange'
                draw.rectangle(p['box'],outline=color,width=2)
                draw.text((p['box'][0],max(0,p['box'][1]-12)),f"{p['conf']:.2f}",fill=color)
            panels.append((f'{model} | {side}',canvas))
    w,h=images['fake'].shape[1],images['fake'].shape[0]
    board=Image.new('RGB',(2*w,2*(h+35)),'#101010')
    for i,(label,canvas) in enumerate(panels):
        col,row=i%2,i//2;board.paste(canvas,(col*w,row*(h+35)+35))
        ImageDraw.Draw(board).text((col*w+6,row*(h+35)+5),label,fill='white')
    board.save(dest/'overlay.png')


def choose_cases(region_rows):
    by_pair=defaultdict(dict)
    for r in region_rows:
        by_pair[r['pair_id']][r['model']]=r
    cases={}
    def choose(name,predicate,score):
        options=[v for v in by_pair.values() if predicate(v)]
        cases[name]=max(options,key=score)['baseline']['pair_id'] if options else None
    choose('A_baseline_fake_fp',lambda v:v['baseline']['fake_fp_present'],
           lambda v:v['baseline']['fake_max_confidence'])
    choose('B_disappears_with_fake_aug',lambda v:v['baseline']['fake_fp_present'] and not v['fake_aug']['fake_fp_present'],
           lambda v:v['baseline']['fake_max_confidence'])
    choose('C_both_fake_fp',lambda v:v['baseline']['fake_fp_present'] and v['fake_aug']['fake_fp_present'],
           lambda v:max(v['baseline']['fake_max_confidence'],v['fake_aug']['fake_max_confidence']))
    choose('D_new_in_fake_aug',lambda v:not v['baseline']['fake_fp_present'] and v['fake_aug']['fake_fp_present'],
           lambda v:v['fake_aug']['fake_max_confidence'])
    choose('E_largest_delta',lambda v:True,
           lambda v:max(v['baseline']['delta_confidence'],v['fake_aug']['delta_confidence']))
    return cases


def run_diagnostic(*,dataset_output=DATASET_OUT,output=OUT,conf=.25,raw_conf=.001,
                   imgsz=640,iou=.7,device='0',fake_coverage=.3,gt_iou=.5,seed=42,max_det=3000):
    dataset_output,output=Path(dataset_output).resolve(),Path(output).resolve()
    if dataset_output.exists() or output.exists():
        raise FileExistsError(f'Refusing to overwrite: {dataset_output if dataset_output.exists() else output}')
    existing=[p.resolve() for p in (ROOT/'outputs').iterdir() if p.is_dir()]
    for dest in (dataset_output,output):
        if dest==ROOT or ROOT in dest.parents and dest==ROOT/'outputs':
            raise ValueError('Output must be a new child directory')
        if any(p==dest or p in dest.parents or dest in p.parents for p in existing):
            raise ValueError('Output overlaps an existing result tree')
    if dataset_output==output or dataset_output in output.parents or output in dataset_output.parents:
        raise ValueError('Diagnostic output trees must be distinct')
    if not 0<raw_conf<=conf<=1 or not 0<iou<=1 or not 0<gt_iou<=1 or not 0<=fake_coverage<=1 or imgsz<32 or seed<0 or max_det<300:
        raise ValueError('Invalid inference, matching or seed option')
    if not (BASELINE_RUN/'weights/best.pt').is_file() or not (FAKE_AUG_RUN/'weights/best.pt').is_file():
        raise FileNotFoundError('YOLO best.pt is missing')
    configs=validate_training_configs()
    if configs[0].get('seed')!=configs[1].get('seed'):
        raise ValueError('Training seeds differ')
    before=source_snapshot()
    git_before=subprocess.run(['git','diff','--binary'],cwd=ROOT,capture_output=True,check=True).stdout
    pairs=prepare_pairs(read_csv(CONTEXT/'matched_pairs.csv'))
    if len({p['pair_id'] for p in pairs})!=len(pairs) or len({p['seed'] for p in pairs})!=1:
        raise ValueError('Duplicate pair or mixed pair seeds')
    if all(p['source_split']=='train' for p in pairs):
        raise ValueError('No held-out val/test pairs')
    import torch
    from ultralytics import YOLO
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if not torch.cuda.is_available():
        torch.set_num_threads(min(4,max(1,torch.get_num_threads())))
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    resolved_device=inference_device(device)
    settings=dict(imgsz=imgsz,raw_conf=raw_conf,iou=iou,device=resolved_device,
                  conf=conf,augment=False,classes=[0],agnostic_nms=False,max_det=max_det,rect=False)
    models={name:YOLO(str(run/'weights/best.pt')) for name,run in
            zip(MODELS,(BASELINE_RUN,FAKE_AUG_RUN))}
    # All provenance is checked before creating either output tree.
    for dest in (dataset_output,output):
        dest.mkdir(parents=True,exist_ok=False)
    for category in ('original','fake','masks'):
        (dataset_output/category).mkdir()
    metadata=[]
    for p in pairs:
        name=p['pair_id']+'.png'
        paths={side:dataset_output/side/name for side in SIDES}
        for side,source in (('original',p['original_source']),('fake',p['fake_source'])):
            shutil.copy2(source,paths[side])
        shutil.copy2(p['mask_source'],dataset_output/'masks'/name)
        if sha256(paths['original'])!=p['original_hash'] or sha256(paths['fake'])!=p['fake_hash']:
            raise RuntimeError('Copied pair hash mismatch')
        x0,y0,x1,y1=p['bbox']
        metadata.append(dict(pair_id=p['pair_id'],image_id=p['image_id'],source_split=p['source_split'],
            original_path=str(paths['original']),fake_path=str(paths['fake']),fake_mask_path=str(dataset_output/'masks'/name),
            fake_region_id=p['pair_id'],fake_x1=x0,fake_y1=y0,fake_x2=x1,fake_y2=y1,
            fake_mask_area=p['mask_area'],seed=p['seed'],gt_boxes_json=json.dumps(p['gt_boxes']),
            original_sha256=p['original_hash'],fake_sha256=p['fake_hash'],source_label_sha256=p['label_hash']))
    write_csv(dataset_output/'metadata.csv',metadata,list(metadata[0]))
    # Repeat one pair under the exact inference settings for each model.
    reproducibility={}
    first_name=pairs[0]['pair_id']+'.png'
    first_predictions={}
    for model_name in MODELS:
        first=predict_pair(models[model_name],dataset_output/'original'/first_name,
                           dataset_output/'fake'/first_name,settings)
        repeat=predict_pair(models[model_name],dataset_output/'original'/first_name,
                            dataset_output/'fake'/first_name,settings)
        if first!=repeat:
            raise RuntimeError(f'Repeated inference differed for {model_name} under seed {seed}')
        first_predictions[model_name]=first
        reproducibility[model_name]=dict(pair_id=pairs[0]['pair_id'],
            raw_predictions=[len(side) for side in first],exact_repeat_match=True)
    region_rows,image_rows,gt_rows,prediction_rows=[],[],[],[]
    all_predictions={}
    normal={name:json.loads((run/'summary.json').read_text(encoding='utf-8')) for name,run in
            zip(MODELS,(BASELINE_RUN,FAKE_AUG_RUN))}
    for index,p in enumerate(pairs,1):
        name=p['pair_id']+'.png';mask=load_mask(dataset_output/'masks'/name,p['image_shape'][:2])
        for model_name in MODELS:
            original_preds,fake_preds=(first_predictions[model_name] if index==1 else
                predict_pair(models[model_name],dataset_output/'original'/name,dataset_output/'fake'/name,settings))
            tagged={side:annotate_predictions(raw,p['gt_boxes'],mask,p['bbox'],
                conf=conf,raw_conf=raw_conf,gt_iou=gt_iou,coverage_threshold=fake_coverage)
                for side,raw in (('original',original_preds),('fake',fake_preds))}
            for side in SIDES:
                all_predictions[(p['pair_id'],model_name,side)]=tagged[side]
                for prediction in tagged[side]:
                    x0,y0,x1,y1=prediction['box']
                    prediction_rows.append(dict(pair_id=p['pair_id'],image_id=p['image_id'],source_split=p['source_split'],
                        model=model_name,side=side,prediction_id=prediction['prediction_id'],
                        x0=x0,y0=y0,x1=x1,y1=y1,confidence=prediction['conf'],class_id=prediction['class_id'],
                        max_gt_iou=prediction['max_gt_iou'],gt_matched=prediction['gt_matched'],
                        iou_fake_bbox=prediction['iou_fake_bbox'],fake_mask_coverage=prediction['fake_mask_coverage'],
                        center_in_fake_bbox=prediction['center_in_fake_bbox'],center_in_fake_mask=prediction['center_in_fake_mask'],
                        spatial_fake=prediction['spatial_fake'],fake_fp_candidate=prediction['fake_fp_candidate'],
                        fake_fp_final=prediction['fake_fp_final']))
            before_region,after_region=region_summary(tagged['original']),region_summary(tagged['fake'])
            original_conf,fake_conf=before_region['max_confidence'],after_region['max_confidence']
            new_fp=bool(after_region['final_present'] and not before_region['final_present'])
            region_rows.append(dict(pair_id=p['pair_id'],image_id=p['image_id'],source_split=p['source_split'],
                model=model_name,fake_region_id=p['pair_id'],original_max_confidence=original_conf,
                fake_max_confidence=fake_conf,delta_confidence=fake_conf-original_conf,
                original_fp_count=before_region['final_count'],fake_fp_count=after_region['final_count'],
                original_fp_present=before_region['final_present'],fake_fp_present=after_region['final_present'],
                new_fake_fp=new_fp,original_raw_detections=len(tagged['original']),fake_raw_detections=len(tagged['fake'])))
            gt_before=gt_confidences(tagged['original'],p['gt_boxes'],gt_iou)
            gt_after=gt_confidences(tagged['fake'],p['gt_boxes'],gt_iou)
            for gt_index,(a,b) in enumerate(zip(gt_before,gt_after),1):
                gt_rows.append(dict(pair_id=p['pair_id'],image_id=p['image_id'],source_split=p['source_split'],
                    model=model_name,gt_id=gt_index,original_confidence=a,fake_confidence=b,
                    delta_confidence=b-a,detected_before=a>=conf,detected_after=b>=conf,
                    detection_lost=bool(a>=conf and b<conf),detection_gained=bool(a<conf and b>=conf)))
            image_rows.append(dict(pair_id=p['pair_id'],image_id=p['image_id'],source_split=p['source_split'],
                model=model_name,gt_count=len(p['gt_boxes']),fake_region_count=1,
                fake_fp_count=after_region['final_count'],fake_fp_present=after_region['final_present'],
                new_fake_fp=new_fp,gt_loss_count=sum(a>=conf and b<conf for a,b in zip(gt_before,gt_after)),
                gt_gain_count=sum(a<conf and b>=conf for a,b in zip(gt_before,gt_after))))
        if index%10==0 or index==len(pairs):
            print(f'추론 진행: {index}/{len(pairs)} pairs',flush=True)
    if len(region_rows)!=len(pairs)*2 or len(image_rows)!=len(pairs)*2:
        raise RuntimeError('Original/Fake inference pair count mismatch')
    summaries=[row for scope in ('all','holdout','val','test','train') for model in MODELS
               if (row:=metrics_for_scope(region_rows,gt_rows,model,scope,normal[model])) is not None]
    comparison=[row for scope in ('all','holdout') for row in paired_comparison(region_rows,scope)]
    save_plots(output,region_rows,summaries)
    fields={'region_results.csv':region_rows,'image_results.csv':image_rows,'gt_results.csv':gt_rows,
            'prediction_results.csv':prediction_rows,'model_summary.csv':summaries,
            'summary.csv':summaries,'paired_comparison.csv':comparison}
    for name,rows in fields.items():
        write_csv(output/name,rows,list(rows[0]) if rows else ['pair_id','model','side','prediction_id'])
    cases=choose_cases(region_rows)
    (output/'cases').mkdir()
    by_id={p['pair_id']:p for p in pairs}
    for category,pair_id in cases.items():
        if pair_id is None: continue
        dest=output/'cases'/f'{category}_{pair_id}';dest.mkdir()
        draw_case_panel(dest,dataset_output,by_id[pair_id],all_predictions,conf)
    if any(sha256(path)!=digest for path,digest in before.items()):
        raise RuntimeError('Existing input/model/code hash changed')
    after_paths={p.resolve() for root in (ROOT/'dataset',ROOT/'outputs/yolo_poc_20',ROOT/'outputs/yolo_poc_20_fake_restoration',
        ROOT/'outputs/inpainting_poc_20',ROOT/'outputs/fake_restoration_debug',CONTEXT)
        for p in root.rglob('*') if p.is_file()}
    if not after_paths.issubset(before):
        raise RuntimeError('An existing data tree gained files')
    git_after=subprocess.run(['git','diff','--binary'],cwd=ROOT,capture_output=True,check=True).stdout
    if git_after!=git_before:
        raise RuntimeError('Tracked git diff changed during inference')
    for rows in (region_rows,image_rows,gt_rows,prediction_rows,summaries,comparison):
        for row in rows:
            if any(isinstance(v,float) and not math.isfinite(v) for v in row.values()):
                raise RuntimeError('NaN/inf in a result')
    import ultralytics
    diagnostic=dict(pairs=len(pairs),fake_regions=len(pairs),heldout_pairs=sum(p['source_split']!='train' for p in pairs),
        split_counts={s:sum(p['source_split']==s for p in pairs) for s in ('train','val','test')},
        settings=settings,requested_device=str(device),resolved_device=resolved_device,
        fake_coverage_threshold=fake_coverage,gt_iou_threshold=gt_iou,seed=seed,
        input_pair_seed=pairs[0]['seed'],model_sha256={name:before[(run/'weights/best.pt').resolve()]
            for name,run in zip(MODELS,(BASELINE_RUN,FAKE_AUG_RUN))},
        inference_repeatability=reproducibility,
        ultralytics_version=ultralytics.__version__,torch_version=torch.__version__,opencv_version=cv2.__version__,
        protected_files_verified=len(before),git_diff_unchanged=True,
        overlap_errors=0,missing_fake_metadata=0,nonfinite_values=0,
        normal_test_metrics={name:{k:v for k,v in data.items() if k.startswith(('test_','val_'))} for name,data in normal.items()},
        model_summary=summaries,paired_comparison=comparison,cases=cases,
        warning='현재 결과는 pipeline/shortcut diagnostic PoC이며 통계적으로 안정적인 최종 성능 결론이 아니다.')
    (output/'summary.json').write_text(json.dumps(diagnostic,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    (output/'protected_hashes.json').write_text(json.dumps({str(p):digest for p,digest in before.items()},indent=2),encoding='utf-8')
    (output/'README.md').write_text(DEFINITIONS,encoding='utf-8')
    (output/'REPORT.md').write_text(build_report(diagnostic),encoding='utf-8')
    print(json.dumps({k:diagnostic[k] for k in ('pairs','fake_regions','heldout_pairs','split_counts','resolved_device','protected_files_verified','warning')},ensure_ascii=False,indent=2))
    return diagnostic


def build_report(result):
    rows={(r['scope'],r['model']):r for r in result['model_summary']}
    paired={(r['scope'],r['metric']):r for r in result['paired_comparison']}
    def fmt(x):
        return 'N/A' if x is None else f'{x:.4f}'
    lines=['# YOLO Shortcut Diagnostic','',result['warning'],'',
        f"동일한 pair {result['pairs']}개, fake 영역 {result['fake_regions']}개. "+
        f"val/test {result['heldout_pairs']}개, split={result['split_counts']}.",
        f"두 모델 공통 추론 설정: {result['settings']}; fake coverage ≥{result['fake_coverage_threshold']}, "+
        f"GT IoU ≥{result['gt_iou_threshold']}; seed={result['seed']}.",
        '', '## 모델별 비교', '',
        '| 범위 | 모델 | FR-FPR | Fake max conf 평균 | Δconf 평균 | Δconf 중앙값 | New FP | GT loss/gain |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for scope in ('all','holdout','test','val','train'):
        for name in MODELS:
            r=rows.get((scope,name))
            if r:
                lines.append(f"| {scope} | {name} | {fmt(r['fr_fpr'])} | {fmt(r['mean_fake_region_max_conf'])} | "+
                    f"{fmt(r['mean_delta_confidence'])} | {fmt(r['median_delta_confidence'])} | "+
                    f"{r['new_fake_fp_count']} | {r['gt_detection_loss_count']}/{r['gt_detection_gain_count']} |")
    lines.extend(['','FR-FPR 분모는 fake 영역 수이고, fake FP 검출 개수와 별도로 센다. '+
                  'normal/test P·R·mAP는 기존 run summary.json 값이며 model_summary.csv에 기록했다.',
                  '', '## Paired 통계', '',
                  '| 범위 | 지표 | Fake-Aug − Baseline 평균 | 중앙값 | 검정 | p |',
                  '|---|---|---:|---:|---|---:|'])
    for scope in ('all','holdout'):
        for metric in ('fake_max_confidence','delta_confidence','fake_fp_present'):
            r=paired[(scope,metric)]
            lines.append(f"| {scope} | {metric} | {fmt(r['paired_mean_difference'])} | "+
                f"{fmt(r['paired_median_difference'])} | {r['test']} | {fmt(r['p_value'])} |")
    hold_base,hold_aug=rows[('holdout','baseline')],rows[('holdout','fake_aug')]
    fr_diff=hold_aug['fr_fpr']-hold_base['fr_fpr']
    if hold_base['fr_fpr']==0 and hold_aug['fr_fpr']==0 and max(abs(hold_base['mean_delta_confidence']),abs(hold_aug['mean_delta_confidence']))<.05:
        conclusion='현재 PoC에서 shortcut evidence는 약함.'
    elif fr_diff<0 and hold_aug['mean_delta_confidence']<hold_base['mean_delta_confidence']:
        conclusion='Fake-Aug의 흔적 반응 감소와 일치하는 수치가 있으나 표본이 작아 판단 보류.'
    else:
        conclusion='동일한 방향의 감소 근거가 부족하여 판단 보류.'
    lines.extend(['','## 해석','',conclusion,
        'train pair는 모델이 이미 본 원본 이미지를 사용하므로 해석에서 val/test를 우선한다. '+
        'val은 학습 중 모델 선택에 사용되었으므로 test 결과도 따로 확인한다.',
        'Raw 후보는 conf 하한 아래를 관측하지 못한다. 원본 동일 위치에 후보가 없으면 max confidence=0이다. '+
        '이 비교는 인과적 증명이 아니며 20장 PoC에서 최종 일반화 성능을 주장할 수 없다.',
        '', '## 사례 이미지', ''])
    for name,pair_id in result['cases'].items():
        lines.append(f"- {name}: {pair_id if pair_id else '해당 사례 없음'}")
    lines.extend(['','cases/<유형>_<pair_id>/에 clean_original.png, clean_fake.png, overlay.png를 저장했다.',
                  '', '## 검증', '',
                  f"기존 입력·모델·코드 파일 {result['protected_files_verified']}개의 SHA-256 보존, "+
                  f"Git diff 무변경. GT overlap, metadata 누락, NaN/inf: 0건. "+
                  '두 모델은 동일한 추론 설정을 사용했다.'])
    return '\n'.join(lines)+'\n'


DEFINITIONS = """# YOLO Shortcut Diagnostic

This is a read-only PoC comparison. The source paired images are matched diagnostic
outputs; no model, existing dataset, source label, restoration script, or prior output
is changed. The copied diagnostic dataset has no training labels/YAML.

Run: python scripts/diagnose_yolo_shortcut.py --conf 0.25 --raw-conf 0.001 --imgsz 640
Rerun with NEW --dataset-output and --output directories. CPU is used for BOTH models
when requested CUDA device 0 is unavailable. The actual device is recorded.
Current-machine Python 3.12 dependencies are isolated in .cache/shortcut_runtime and
.cache/after_similarity_runtime; the original .venv interpreter path is broken.

Dataset: paired_images/<pair>_real is the saved actual-restored baseline, and
<pair>_fake has only one diagnostic fake mask applied. Both are copied byte-for-byte.
GT annotation is the same source YOLO label for both. The mask is checked against
metadata and GT+safety margin. Source split and SHA-256 are in metadata.csv.

Inference: Ultralytics YOLO predict on both images as one pair; imgsz, raw conf,
NMS IoU, device, class filter, max_det, half, rect and augment flags are identical.
Raw detections are NMS-filtered at raw_conf (default .001). Final threshold is applied
after collection (default .25). Raw confidences below .001 are censored to zero.

Fake spatial relation: IoU(pred box, fake box), exact binary-mask pixel coverage
intersection/fake mask area, prediction center in fake bbox or mask. A fake response
exists when center is in bbox/mask OR coverage >= configured threshold (default .3).
Predictions with IoU >= .5 to ANY GT are first marked GT matches and excluded from
fake FP and fake-region confidence. Conf>=.25 defines final fake FP. A single region
with multiple boxes contributes one FR-FPR numerator but all boxes to FP total.

Delta confidence: max raw (GT-unmatched, spatially related) prediction confidence in
fake image minus max at the SAME region in original; no prediction means 0. New FP:
final-threshold response in fake and none in original. GT confidence is max confidence
over predictions with IoU>=.5 to each GT; loss/gain uses .25 threshold.

Summary scopes: all, holdout=val+test, val, test, train. Train images may have been seen
in training, so holdout/test get interpretation priority. Normal test metrics are read
from existing run summary.json, not recalculated from paired diagnostic images.
Wilcoxon compares model responses at identical fake regions; exact McNemar compares
binary FR-FP outcomes. Small, correlated PoC samples limit p-value interpretation.

Files: dataset metadata.csv and original/fake/masks; output region_results.csv,
image_results.csv, gt_results.csv, prediction_results.csv, model_summary.csv,
summary.csv, paired_comparison.csv, summary.json, protected_hashes.json, REPORT.md,
four plots and representative cases with clean images plus overlays.
"""


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-output',type=Path,default=DATASET_OUT)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--conf',type=float,default=.25)
    parser.add_argument('--raw-conf',type=float,default=.001)
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--iou',type=float,default=.7)
    parser.add_argument('--max-det',type=int,default=3000)
    parser.add_argument('--device',default='0')
    parser.add_argument('--fake-coverage',type=float,default=.3)
    parser.add_argument('--gt-iou',type=float,default=.5)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    run_diagnostic(dataset_output=args.dataset_output,output=args.output,conf=args.conf,
        raw_conf=args.raw_conf,imgsz=args.imgsz,iou=args.iou,device=args.device,
        fake_coverage=args.fake_coverage,gt_iou=args.gt_iou,seed=args.seed,max_det=args.max_det)


if __name__=='__main__':
    main()
