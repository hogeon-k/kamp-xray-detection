"""Evaluate product-contour edge crops with the frozen YOLO11n. Inputs are read only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'outputs/yolo_512_telea'
OLD = ROOT / 'outputs/foreign_object_condition_analysis'
MODEL = ROOT / 'outputs/yolo_runs/telea_512_yolo11n_e200_img1280_b4_adamw_lr0.001_wd0.0005_noaug_s42/weights/best.pt'
OUT = ROOT / 'outputs/edge_reinspection_poc'
WIDTHS = {'narrow': 24, 'medium': 40, 'wide': 64}
CONF, MATCH_IOU, NMS_IOU = .25, .5, .5
PRODUCT_EDGE_PX = 48  # fixed before Val comparison; independent of the ROI candidate
LEGACY_Q25 = .36407082305519806


def csvread(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csvwrite(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def product_mask(gray):
    """Largest dark foreground against border background; explicit failure checks."""
    h, w = gray.shape
    border = np.concatenate((gray[:12].ravel(), gray[-12:].ravel(), gray[:, :12].ravel(), gray[:, -12:].ravel()))
    bg = float(np.median(border))
    smooth = cv2.GaussianBlur(gray, (7, 7), 0)
    # Otsu on the image is capped by a conservative background contrast cutoff.
    otsu, _ = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    threshold = min(float(otsu), bg - 12)
    raw = (smooth < threshold).astype(np.uint8)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    if n < 2:
        return np.zeros_like(raw), {'valid': False, 'reason': 'no_component', 'background': bg}
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == idx).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [contour], -1, 1, -1)
    x, y, bw, bh = cv2.boundingRect(contour)
    area = int(filled.sum())
    contrast = bg - float(np.median(gray[filled > 0])) if area else 0
    touches = x < 4 or y < 4 or x + bw > w - 4 or y + bh > h - 4
    valid = (0.05 < area / (h*w) < .75 and bw > .15*w and bh > .15*h
             and not touches and contrast > 15)
    reason = 'ok' if valid else 'area_size_border_or_contrast'
    return filled, {'valid': bool(valid), 'reason': reason, 'background': bg,
                    'threshold': threshold, 'area_fraction': area/(h*w),
                    'contrast': contrast, 'bbox': [x, y, x+bw, y+bh]}


def edge_distance(mask):
    return cv2.distanceTransform(mask, cv2.DIST_L2, 3)


def crop_tiles(mask, margin=16):
    """Four overlapping product quadrants, clipping to the image."""
    ys, xs = np.where(mask > 0)
    if not len(xs): return []
    x1, x2, y1, y2 = xs.min(), xs.max()+1, ys.min(), ys.max()+1
    mx, my = (x1+x2)//2, (y1+y2)//2
    h, w = mask.shape
    return [tuple(map(int,(max(0, a-margin), max(0, b-margin), min(w, c+margin), min(h, d+margin))))
            for a,b,c,d in [(x1,y1,mx,my),(mx,y1,x2,my),(x1,my,mx,y2),(mx,my,x2,y2)]]


def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(aa+bb-inter) if aa+bb>inter else 0.


def nms(preds):
    selected=[]
    for p in sorted(preds,key=lambda q:q['confidence'],reverse=True):
        if all(iou(p['box'],q['box']) < NMS_IOU for q in selected): selected.append(p)
    return selected


def fuse(global_preds, edge_preds):
    """Preserve the baseline exactly; add only edge boxes not overlapping a global box."""
    return global_preds + [p for p in edge_preds
                           if all(iou(p['box'],q['box']) < NMS_IOU for q in global_preds)]


def matches(gt, preds):
    eligible=[p for p in preds if p['confidence'] >= CONF and p['class_id']==0]
    pairs=sorted(((iou(g['box'],p['box']),i,j) for i,g in enumerate(gt)
                  for j,p in enumerate(eligible)),reverse=True)
    used_gt=set(); used_pred=set(); found={}
    for score,i,j in pairs:
        if score < MATCH_IOU: break
        if i not in used_gt and j not in used_pred:
            used_gt.add(i); used_pred.add(j)
            found[i]={'confidence': eligible[j]['confidence'], 'iou': score,
                      'source': eligible[j]['source'], 'prediction_index':j}
    return found, len(eligible)-len(used_pred), [eligible[j] for j in range(len(eligible)) if j not in used_pred]


def label_boxes(row, shape):
    h,w=shape
    path=DATA/'labels'/row['split']/(Path(row['filename']).stem+'.txt')
    result=[]
    for line in path.read_text().splitlines():
        cls,x,y,bw,bh=map(float,line.split())
        assert cls==0
        result.append({'box':[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]})
    return result


def annotate_edge(gt, mask, dist, feature_by_box, uid):
    h,w=mask.shape
    for i,g in enumerate(gt):
        x1,y1,x2,y2=g['box']; cx=int(np.clip((x1+x2)/2,0,w-1)); cy=int(np.clip((y1+y2)/2,0,h-1))
        # Very small mask inaccuracies at the silhouette are tolerated (8 px).
        d=float(dist[cy,cx]); near=bool(mask[cy,cx] or cv2.dilate(mask,np.ones((17,17),np.uint8))[cy,cx])
        g['product_edge']=bool(near and d <= PRODUCT_EDGE_PX)
        f=feature_by_box.get((uid,str(i)))
        g['legacy_edge']=bool(f and f['segmentation_reliable']=='True' and f['normalized_edge_distance']
                              and float(f['normalized_edge_distance'])<=LEGACY_Q25)
        g['product_distance']=d if near else None


def summarize(split, candidate, method, records):
    groups={k:[] for k in ('overall','edge','non_edge','legacy_edge')}
    fps=defaultdict(int); images=len(records); confs=defaultdict(list); ious=defaultdict(list)
    for rec in records:
        gt=rec['gt']; pred=rec['pred'][method]; found,extra,unmatched=matches(gt,pred)
        fps['overall']+=extra
        h,w=rec['gray'].shape
        for p in unmatched:
            cx=int(np.clip((p['box'][0]+p['box'][2])/2,0,w-1));cy=int(np.clip((p['box'][1]+p['box'][3])/2,0,h-1))
            near=bool(rec['mask'][cy,cx])
            fps['edge' if near and rec['dist'][cy,cx]<=PRODUCT_EDGE_PX else 'non_edge']+=1
            if min(cx/w,cy/h,1-cx/w,1-cy/h)<=LEGACY_Q25:fps['legacy_edge']+=1
        for i,g in enumerate(gt):
            keys=['overall','edge' if g['product_edge'] else 'non_edge']
            if g['legacy_edge']: keys.append('legacy_edge')
            for key in keys:
                groups[key].append(i in found)
                if i in found:
                    confs[key].append(found[i]['confidence']); ious[key].append(found[i]['iou'])
    rows=[]
    for group,vals in groups.items():
        tp=sum(vals); count=len(vals)
        rows.append({'split':split,'candidate':candidate,'method':method,'group':group,'GT_count':count,
                     'TP':tp,'FN':count-tp,'recall':tp/count if count else '',
                     'FP':fps[group],'precision':tp/(tp+fps[group]) if tp+fps[group] else '',
                     'FP_per_image':fps[group]/images if images else '',
                     'mean_confidence':float(np.mean(confs[group])) if confs[group] else '',
                     'mean_iou':float(np.mean(ious[group])) if ious[group] else ''})
    return rows


def make_contact(records, name, out, limit=12):
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',15)
    selected=records[:limit]
    if not selected:
        empty=Image.new('RGB',(700,70),'white')
        ImageDraw.Draw(empty).text((12,20),f'{name}: 해당 사례 없음',font=font,fill='black')
        empty.save(out)
        return
    canvas=Image.new('RGB',(6*280,len(selected)*320),'white'); draw=ImageDraw.Draw(canvas)
    for ri,rec in enumerate(selected):
        image=cv2.cvtColor(rec['gray'],cv2.COLOR_GRAY2RGB)
        mask=rec['mask']; dist=rec['dist']; band=(mask>0)&(dist<=PRODUCT_EDGE_PX)
        overlay=image.copy(); overlay[band]=(overlay[band]*.5+np.array([255,210,0])*.5).astype('uint8')
        panels=[image,overlay,image.copy(),image.copy(),image.copy(),image.copy()]
        for pi,key in [(2,'global'),(3,'edge'),(4,'fusion')]:
            for p in rec['pred'][key]:
                if p['confidence']<CONF:continue
                x1,y1,x2,y2=map(round,p['box']);cv2.rectangle(panels[pi],(x1,y1),(x2,y2),(0,255,0),2)
        for pi in range(1,6):
            for g in rec['gt']:
                x1,y1,x2,y2=map(round,g['box']);cv2.rectangle(panels[pi],(x1,y1),(x2,y2),(255,0,0),1)
        titles=['원본','제품 edge + GT','Global + GT','Edge crop + GT','Fusion + GT','GT']
        for pi,panel in enumerate(panels):
            thumb=Image.fromarray(panel).resize((280,280))
            canvas.paste(thumb,(pi*280,ri*320+38))
            draw.text((pi*280+4,ri*320+3),f"{rec['uid']} {titles[pi]}",font=font,fill='black')
        # GT panel is intentionally clean except GT boxes.
        gtpanel=image.copy()
        for g in rec['gt']:
            x1,y1,x2,y2=map(round,g['box']);cv2.rectangle(gtpanel,(x1,y1),(x2,y2),(255,0,0),2)
        canvas.paste(Image.fromarray(gtpanel).resize((280,280)),(5*280,ri*320+38))
    canvas.save(out)


def run(args):
    from ultralytics import YOLO
    assert MODEL.is_file() and DATA.is_dir()
    OUT.mkdir(parents=True,exist_ok=True)
    for d in ('roi_debug','predictions','visualizations'):(OUT/d).mkdir(exist_ok=True)
    model_hash=sha(MODEL)
    old_meta=json.loads((OLD/'prediction_provenance.json').read_text(encoding='utf-8'))
    assert old_meta['model_sha256']==model_hash and old_meta['imgsz']==1280
    global_cache=json.loads((OLD/'predictions.json').read_text(encoding='utf-8'))
    manifest=csvread(DATA/'manifest.csv')
    features=csvread(OLD/'features.csv')
    feat={(r['unique_id'],r['bbox_id']):r for r in features}
    model=YOLO(str(MODEL))
    all_summaries=[]; all_edge=[]; diagnostics=[]; selection=None
    for split in ('val','test'):
        if split=='test' and args.val_only:break
        rows=[r for r in manifest if r['split']==split]
        prepared=[]; crops=[]; croprefs=[]
        for row in rows:
            uid=row['unique_id']; gray=cv2.imread(str(DATA/'images'/split/row['restored_filename']),0)
            if gray is None:raise FileNotFoundError(row['restored_filename'])
            mask,quality=product_mask(gray);dist=edge_distance(mask)
            gt=label_boxes(row,gray.shape);annotate_edge(gt,mask,dist,feat,uid)
            rec={'uid':uid,'row':row,'gray':gray,'mask':mask,'dist':dist,'quality':quality,
                 'gt':gt,'global':[dict(p,source='global') for p in global_cache[uid]],'raw_edge':[]}
            prepared.append(rec)
            diagnostics.append({'split':split,'unique_id':uid,**quality,'GT_count':len(gt),
                                'product_edge_GT':sum(g['product_edge'] for g in gt)})
            if quality['valid']:
                for tile in crop_tiles(mask):
                    x1,y1,x2,y2=tile
                    crops.append(cv2.cvtColor(gray[y1:y2,x1:x2],cv2.COLOR_GRAY2RGB))
                    croprefs.append((len(prepared)-1,tile))
        if crops:
            for crop,(idx,tile) in zip(crops,croprefs,strict=True):
                result=model.predict(source=crop,imgsz=1280,conf=.001,iou=.7,max_det=300,
                                     device=0,save=False,verbose=False,augment=False)[0]
                x1,y1,x2,y2=tile; boxes=result.boxes
                for b,c,k in zip(boxes.xyxy.cpu().numpy().tolist(),boxes.conf.cpu().numpy().tolist(),boxes.cls.cpu().numpy().tolist()):
                    # YOLO returns xyxy in original crop pixels. Translation restores full-image coordinates.
                    prepared[idx]['raw_edge'].append({'box':[b[0]+x1,b[1]+y1,b[2]+x1,b[3]+y1],
                                                       'confidence':float(c),'class_id':int(k),'source':'edge','tile':tile})
        candidates=WIDTHS if split=='val' else {selection:WIDTHS[selection]}
        per_candidate={}
        for name,width in candidates.items():
            records=[]
            for rec in prepared:
                h,w=rec['gray'].shape
                edge=[]
                for p in rec['raw_edge']:
                    cx=int(np.clip((p['box'][0]+p['box'][2])/2,0,w-1));cy=int(np.clip((p['box'][1]+p['box'][3])/2,0,h-1))
                    if rec['mask'][cy,cx] and rec['dist'][cy,cx]<=width:edge.append(p)
                edge=nms(edge)
                fused=fuse(rec['global'],edge)
                records.append({**rec,'pred':{'global':rec['global'],'edge':edge,'fusion':fused}})
            per_candidate[name]=records
            for method in ('global','edge','fusion'):
                all_summaries+=summarize(split,name,method,records)
        if split=='val':
            # Maximize edge TP, then overall TP, then minimize FP; all from Val only.
            def score(name):
                rows2=[r for r in all_summaries if r['split']=='val' and r['candidate']==name and r['method']=='fusion']
                edge=next(r for r in rows2 if r['group']=='edge');overall=next(r for r in rows2 if r['group']=='overall')
                return (edge['TP'],overall['TP'],-overall['FP'],-WIDTHS[name])
            selection=max(WIDTHS,key=score)
        chosen=selection
        records=per_candidate[chosen]
        if split=='val':
            wide_fp=[]
            for rec in per_candidate['wide']:
                _,_,unmatched=matches(rec['gt'],rec['pred']['fusion'])
                if any(p['source']=='edge' for p in unmatched):wide_fp.append(rec)
            make_contact(wide_fp,'wide_new_edge_fp',OUT/'visualizations'/'val_wide_new_edge_fp.png')
        serial=[]
        for rec in records:
            serial.append({'unique_id':rec['uid'],'quality':rec['quality'],
                           'gt':rec['gt'],'global':rec['pred']['global'],
                           'edge':rec['pred']['edge'],'fusion':rec['pred']['fusion']})
            gm,_fp,_=matches(rec['gt'],rec['pred']['global']);em,_,_=matches(rec['gt'],rec['pred']['edge']);fm,_,_=matches(rec['gt'],rec['pred']['fusion'])
            for i,g in enumerate(rec['gt']):
                all_edge.append({'split':split,'candidate':chosen,'unique_id':rec['uid'],'bbox_id':i,
                                 'product_edge':g['product_edge'],'legacy_edge':g['legacy_edge'],
                                 'product_distance':g['product_distance'],
                                 'global_status':'TP' if i in gm else 'FN',
                                 'edge_status':'TP' if i in em else 'FN',
                                 'fusion_status':'TP' if i in fm else 'FN',
                                 'global_confidence':gm.get(i,{}).get('confidence',''),
                                 'edge_confidence':em.get(i,{}).get('confidence',''),
                                 'fusion_confidence':fm.get(i,{}).get('confidence',''),
                                 'fusion_iou':fm.get(i,{}).get('iou','')})
        dump(OUT/'predictions'/f'{split}.json',serial)
        rescued=[];both=[];newfp=[]
        for rec in records:
            gm,_,_=matches(rec['gt'],rec['pred']['global']);em,_,_=matches(rec['gt'],rec['pred']['edge']);fm,_,_=matches(rec['gt'],rec['pred']['fusion'])
            if any(g['product_edge'] and i not in gm and i in em for i,g in enumerate(rec['gt'])):rescued.append(rec)
            if any(g['product_edge'] and i not in gm and i not in em for i,g in enumerate(rec['gt'])):both.append(rec)
            _,_,gfp=matches(rec['gt'],rec['pred']['global']);_,_,ffp=matches(rec['gt'],rec['pred']['fusion'])
            if any(p['source']=='edge' and all(iou(p['box'],q['box'])<.5 for q in gfp) for p in ffp):newfp.append(rec)
        for category,items in [('global_fn_to_edge_tp',rescued),('both_fn',both),('new_edge_fp',newfp),('fusion_final',records[:12])]:
            make_contact(items,category,OUT/'visualizations'/f'{split}_{category}.png')
        print(f'{split}: {len(rows)} images; selected={selection}; rescued_images={len(rescued)}',flush=True)
    csvwrite(OUT/'val_comparison.csv',[r for r in all_summaries if r['split']=='val'],list(all_summaries[0]))
    if not args.val_only:
        csvwrite(OUT/'test_comparison.csv',[r for r in all_summaries if r['split']=='test'],list(all_summaries[0]))
    csvwrite(OUT/'edge_results.csv',all_edge,list(all_edge[0]))
    csvwrite(OUT/'roi_debug'/'quality.csv',diagnostics,list(diagnostics[0]))
    for split in ('val','test'):
        sample=[r for r in diagnostics if r['split']==split][:12]
        if not sample:continue
        rows=[r for r in manifest if r['split']==split]
        imgs=[]
        for d in sample:
            row=next(r for r in rows if r['unique_id']==d['unique_id']); gray=cv2.imread(str(DATA/'images'/split/row['restored_filename']),0)
            mask,q=product_mask(gray);dist=edge_distance(mask)
            rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB);band=(mask>0)&(dist<=PRODUCT_EDGE_PX)
            rgb[band]=(rgb[band]*.5+np.array([255,210,0])*.5).astype('uint8')
            imgs.append(Image.fromarray(rgb).resize((280,210)))
        sheet=Image.new('RGB',(4*280,3*240),'white');draw=ImageDraw.Draw(sheet)
        for i,img in enumerate(imgs):
            x=(i%4)*280;y=(i//4)*240;sheet.paste(img,(x,y+25))
            draw.text((x+4,y+3),f"{sample[i]['unique_id']} {sample[i]['reason']}",fill='black')
        sheet.save(OUT/'roi_debug'/f'{split}_product_edge_sheet.png')
    summary={'model':str(MODEL),'model_sha256':model_hash,'data':str(DATA),
             'global_prediction_source':str(OLD/'predictions.json'),'selected_on_val':selection,
             'widths_px':WIDTHS,'product_edge_evaluation_width_px':PRODUCT_EDGE_PX,
             'matching_confidence':CONF,'matching_iou':MATCH_IOU,'fusion_nms_iou':NMS_IOU,
             'roi_quality':{s:{'images':sum(r['split']==s for r in diagnostics),
                               'valid':sum(r['split']==s and r['valid'] for r in diagnostics)} for s in ('val','test')},
             'results':all_summaries}
    summary['transitions']={s:{'global_FN_to_edge_TP':sum(r['split']==s and r['product_edge'] and r['global_status']=='FN' and r['edge_status']=='TP' for r in all_edge),
                               'remaining_product_edge_FN':sum(r['split']==s and r['product_edge'] and r['fusion_status']=='FN' for r in all_edge),
                               'remaining_legacy_edge_FN':sum(r['split']==s and r['legacy_edge'] and r['fusion_status']=='FN' for r in all_edge)}
                            for s in ('val','test')}
    dump(OUT/'summary.json',summary)
    (OUT/'README.md').write_text('''# 제품 가장자리 ROI 재검사 PoC

동결된 YOLO11n의 global 예측 캐시를 재사용하고, 각 영상의 제품 외곽선 주변 4개 겹친 crop을 같은 모델에 imgsz=1280으로 재추론했습니다. crop의 xyxy에는 crop 원점을 더해 원본 좌표로 복원했습니다. Conf ≥0.25, GT IoU ≥0.5 일대일 매칭입니다. Edge 내부는 confidence 순 IoU 0.5 NMS를 적용합니다. Fusion은 Global 검출을 그대로 보존하고 IoU ≥0.5로 겹치는 Edge 검출만 버립니다. 추론은 conf 0.001, NMS IoU 0.7, max_det 300, augment=False입니다.

제품 mask는 배경 border median과 Otsu 임계값으로 어두운 연결 성분을 찾고 가장 큰 성분의 외곽선을 채웁니다. 면적, 크기, 경계 접촉, 명암 조건을 통과한 경우에만 crop을 사용합니다. 실패 영상은 global만 남으며 roi_debug/quality.csv에 기록됩니다. 별도 segmentation GT가 없어 mask는 형태학적 추정입니다.

후보 narrow 24 px, medium 40 px, wide 64 px는 제품 외곽선 안쪽 거리 band에서 crop 검출 중심을 받아들이는 폭입니다. 제품 edge GT 평가는 후보와 독립적으로 48 px로 고정했습니다. legacy_edge는 기존 foreign_object_condition_analysis의 신뢰도 통과 GT와 Train q25 정규화 이미지 거리(0.364070823...) 기준입니다. 따라서 제품 edge와 legacy_edge는 다른 모집단이며 기존 Test 86.27%는 후자와 비교해야 합니다.

선택: Val fusion의 제품 edge TP, 전체 TP, 전체 FP, 폭 순서로 결정. Test는 선택된 후보 하나만 실행합니다. Group별 precision/FP는 해당 영역 GT의 TP와 해당 영역의 unmatched 검출 기준입니다. legacy_edge FP는 검출 중심의 정규화 이미지 거리 기준입니다. 평균 confidence와 IoU는 TP에 한합니다. edge_results.csv는 GT별 변화를 기록합니다. 시각화의 초록색은 검출, 빨간색은 GT, 노란 영역은 제품 edge입니다. 빈 contact sheet는 해당 유형의 사례가 없다는 뜻입니다. val_wide_new_edge_fp.png는 선택되지 않은 wide 후보에서 FP가 생긴 사례입니다.

재현: `.venv\\Scripts\\python.exe -B -X utf8 scripts\\evaluate_edge_reinspection.py` (원 Python 경로가 설치돼 있어야 함). 입력 이미지, 라벨, split, 모델은 읽기 전용입니다. 기존 작업 트리의 다른 변경 사항은 이 스크립트가 건드리지 않습니다.
''',encoding='utf-8')
    def row(split,method,group):
        return next(r for r in all_summaries if r['split']==split and r['candidate']==selection and r['method']==method and r['group']==group)
    with (OUT/'README.md').open('a',encoding='utf-8') as f:
        f.write('\n## 결과\n\n')
        for split in ('val','test'):
            if split=='test' and args.val_only:continue
            g=row(split,'global','overall');z=row(split,'fusion','overall')
            ge=row(split,'global','edge');ze=row(split,'fusion','edge')
            gl=row(split,'global','legacy_edge');zl=row(split,'fusion','legacy_edge')
            f.write(f"- {split}: 제품 edge {ge['TP']}/{ge['GT_count']} → {ze['TP']}/{ze['GT_count']} "
                    f"({ge['recall']*100:.2f}% → {ze['recall']*100:.2f}%). "
                    f"전체 recall {g['recall']*100:.2f}% → {z['recall']*100:.2f}%, "
                    f"precision {g['precision']*100:.2f}% → {z['precision']*100:.2f}%, "
                    f"FP {g['FP']} → {z['FP']}. "
                    f"기존 분석 edge 모집단 {gl['TP']}/{gl['GT_count']} → {zl['TP']}/{zl['GT_count']}. "
                    f"Global FN→Edge TP {summary['transitions'][split]['global_FN_to_edge_TP']}개.\n")
        f.write('\n**결론: 유의미한 개선 없음.** crop에서 기존 모델 confidence가 대부분 0.25 미만으로 떨어졌고, Val의 넓은 band는 TP 증가 없이 FP를 늘렸습니다. 이 PoC만으로 Edge 전용 모델의 효과를 입증할 수는 없습니다.\n')
    print(json.dumps({'selected':selection,'roi_quality':summary['roi_quality']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--val-only',action='store_true')
    run(parser.parse_args())
