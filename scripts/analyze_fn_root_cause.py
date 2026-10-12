"""Descriptive Test TP/FN analysis. Reuses frozen detections and evaluation definitions."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from evaluate_edge_reinspection import (ROOT, DATA, OLD, MODEL, CONF, MATCH_IOU,
    csvread, csvwrite, dump, sha, product_mask, label_boxes, matches, iou)

DEFAULT_OUT=ROOT/'outputs/fn_root_cause_analysis'
FEATURES=['bbox_width','bbox_height','bbox_area','bbox_area_ratio','defect_mean',
          'defect_median','defect_std','ring_mean','ring_median','absolute_contrast',
          'normalized_contrast','local_difference_mean','local_difference_max',
          'local_difference_p90','laplacian_variance','contour_distance',
          'bbox_to_contour_min_distance','gradient_mean']
MAIN=['bbox_area','absolute_contrast','normalized_contrast','local_difference_mean',
      'laplacian_variance','contour_distance']
EPS=1e-6


def raster(box,shape):
    h,w=shape
    x1,y1,x2,y2=box
    return (max(0,math.floor(x1)),max(0,math.floor(y1)),min(w,math.ceil(x2)),min(h,math.ceil(y2)))


def calculate(gray,background,mask,gt,index,contour,preds,found):
    h,w=gray.shape; box=gt[index]['box'];x1,y1,x2,y2=raster(box,gray.shape)
    bw,bh=box[2]-box[0],box[3]-box[1]
    padding=int(np.clip(round(max(bw,bh)*.5),5,15))
    ring=np.zeros_like(mask)
    ring[max(0,y1-padding):min(h,y2+padding),max(0,x1-padding):min(w,x2+padding)]=1
    # Exclude all GT rectangles, including the target, and external product background.
    for g in gt:
        a,b,c,d=raster(g['box'],gray.shape);ring[b:d,a:c]=0
    ring &= mask
    obj=gray[y1:y2,x1:x2].astype(float);sur=gray[ring>0].astype(float)
    difference=np.abs(gray.astype(np.float32)-background)[y1:y2,x1:x2]
    lap=cv2.Laplacian(gray.astype(np.float32),cv2.CV_32F,ksize=3)[y1:y2,x1:x2]
    gx=cv2.Sobel(gray.astype(np.float32),cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(gray.astype(np.float32),cv2.CV_32F,0,1,ksize=3)
    gradient=cv2.magnitude(gx,gy)[y1:y2,x1:x2]
    cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
    signed=cv2.pointPolygonTest(contour,(float(cx),float(cy)),True)
    xy=contour.reshape(-1,2).astype(float)
    dx=np.maximum(np.maximum(box[0]-xy[:,0],xy[:,0]-box[2]),0)
    dy=np.maximum(np.maximum(box[1]-xy[:,1],xy[:,1]-box[3]),0)
    minbox=float(np.hypot(dx,dy).min())
    rm=float(sur.mean()) if len(sur)>=12 else None
    abscontrast=abs(float(obj.mean())-rm) if rm is not None else None
    eligible=[p for p in preds if p['confidence']>=CONF]
    overlaps=[(iou(box,p['box']),p['confidence']) for p in eligible]
    best=max(overlaps,default=(0.,None))
    nearlow=[p['confidence'] for p in preds if iou(box,p['box'])>=MATCH_IOU]
    return dict(bbox_width=bw,bbox_height=bh,bbox_area=bw*bh,bbox_area_ratio=bw*bh/(h*w),
                defect_mean=float(obj.mean()),defect_median=float(np.median(obj)),defect_std=float(obj.std()),
                ring_mean=rm,ring_median=float(np.median(sur)) if rm is not None else None,
                absolute_contrast=abscontrast,normalized_contrast=abscontrast/(rm+EPS) if rm is not None else None,
                local_difference_mean=float(difference.mean()),local_difference_max=float(difference.max()),
                local_difference_p90=float(np.percentile(difference,90)),laplacian_variance=float(lap.var()),
                contour_distance=abs(float(signed)),contour_signed_distance=float(signed),
                bbox_to_contour_min_distance=minbox,gradient_mean=float(gradient.mean()),
                ring_padding=padding,ring_pixels=int(len(sur)),ring_valid=rm is not None,
                raster_x1=x1,raster_y1=y1,raster_x2=x2,raster_y2=y2,
                match_confidence=found.get(index,{}).get('confidence'),match_iou=found.get(index,{}).get('iou'),
                best_eligible_iou=best[0],best_eligible_confidence=best[1],
                max_confidence_at_iou50=max(nearlow,default=None))


def statistics(rows):
    result=[]
    for feature in FEATURES:
        for status in ('TP','FN'):
            values=np.array([r[feature] for r in rows if r['status']==status and r[feature] is not None],float)
            result.append(dict(feature=feature,status=status,n=len(values),mean=float(values.mean()),
                               median=float(np.median(values)),minimum=float(values.min()),maximum=float(values.max()),
                               std=float(values.std())))
    return result


def fmt(value):return 'NA' if value is None else f'{value:.3f}'


def sheets(fn,images,out):
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16)
    canvas=Image.new('RGB',(1320,len(fn)*285),'white');draw=ImageDraw.Draw(canvas)
    local=Image.new('RGB',(900,len(fn)*245),'white');ld=ImageDraw.Draw(local)
    for j,r in enumerate(fn):
        rec=images[r['unique_id']];gray=rec['gray'];bg=rec['background'];diff=np.abs(gray.astype(float)-bg)
        x1,y1,x2,y2=[r[k] for k in ('raster_x1','raster_y1','raster_x2','raster_y2')]
        lo=max(0,x1-24),max(0,y1-24);hi=min(gray.shape[1],x2+24),min(gray.shape[0],y2+24)
        full=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
        marked=full.copy();cv2.rectangle(marked,(x1,y1),(x2,y2),(255,0,0),2)
        for p in rec['preds']:
            if p['confidence']>=CONF:
                a,b,c,d=map(round,p['box']);cv2.rectangle(marked,(a,b),(c,d),(0,255,0),1)
        panels=[full,marked,marked[lo[1]:hi[1],lo[0]:hi[0]]]
        for k,p in enumerate(panels):
            scale=min(250/p.shape[1],230/p.shape[0])
            im=Image.fromarray(p).resize((round(p.shape[1]*scale),round(p.shape[0]*scale)),Image.Resampling.NEAREST if k==2 else Image.Resampling.BILINEAR)
            canvas.paste(im,(k*265,j*285+35))
        draw.text((5,j*285+4),f"{r['unique_id']} bbox{r['bbox_id']} FN: 일대일 매칭 없음",font=font,fill='black')
        caption=(f"area={r['bbox_area']:.1f} px² / ROI mean={r['defect_mean']:.2f}\n"
                 f"ring mean={fmt(r['ring_mean'])} / abs contrast={fmt(r['absolute_contrast'])}\n"
                 f"norm contrast={fmt(r['normalized_contrast'])}\n"
                 f"local diff mean={r['local_difference_mean']:.2f}\n"
                 f"contour distance={r['contour_distance']:.2f} px\n"
                 f"best IoU(conf≥0.25)={r['best_eligible_iou']:.3f}\n"
                 f"max conf(IoU≥0.5)={fmt(r['max_confidence_at_iou50'])}")
        draw.multiline_text((805,j*285+45),caption,font=font,fill='black',spacing=7)
        ld.text((5,j*245+3),f"{r['unique_id']} bbox{r['bbox_id']} / 차이 영상 0–60 고정 스케일",font=font,fill='black')
        for k,(im,title) in enumerate([(gray,'원본'),(bg,'Gaussian background σ=5'),(np.clip(diff*255/60,0,255),'abs local difference')]):
            crop=np.uint8(np.clip(im,0,255))[lo[1]:hi[1],lo[0]:hi[0]]
            rgb=cv2.cvtColor(crop,cv2.COLOR_GRAY2RGB)
            cv2.rectangle(rgb,(x1-lo[0],y1-lo[1]),(x2-lo[0],y2-lo[1]),(255,0,0),1)
            image=Image.fromarray(rgb).resize((220,200),Image.Resampling.NEAREST)
            local.paste(image,(k*300+10,j*245+40));ld.text((k*300+10,j*245+23),title,font=font,fill='black')
    canvas.save(out/'fn_contact_sheet.png');local.save(out/'fn_local_difference_sheet.png')


def plots(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    rng=np.random.default_rng(42)
    fig,axes=plt.subplots(2,3,figsize=(14,9))
    for ax,key in zip(axes.flat,MAIN):
        for x,status,color in [(0,'TP','#3676a6'),(1,'FN','#d74732')]:
            vals=[r[key] for r in rows if r['status']==status and r[key] is not None]
            ax.scatter(x+rng.uniform(-.13,.13,len(vals)),vals,s=23 if status=='TP' else 48,alpha=.6 if status=='TP' else 1,c=color)
            ax.plot([x-.2,x+.2],[np.median(vals)]*2,c='black',lw=2)
        ax.set_xticks([0,1],['TP (148)','FN (12)']);ax.set_title(key);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Test GT 특성 비교 — 점: 개별 GT / 검은 선: 중앙값')
    fig.tight_layout();fig.savefig(out/'tp_fn_feature_comparison.png',dpi=160);plt.close(fig)


def write_interpretation(out,summary):
    c={r['feature']:r for r in summary['comparisons']};fn=summary['FN_samples']
    combined=sum(r['bbox_area']<c['bbox_area']['tp_median'] and
                 r['normalized_contrast']<c['normalized_contrast']['tp_median'] for r in fn)
    lowconf=sum(r['max_confidence_at_iou50'] is not None and r['max_confidence_at_iou50']<CONF for r in fn)
    allrows=csvread(out/'features/test_gt_features.csv')
    machine=[]
    for key in sorted({r['machine'] for r in allrows}):
        group=[r for r in allrows if r['machine']==key]
        machine.append(dict(machine=key,GT=len(group),FN=sum(r['status']=='FN' for r in group)))
    conclusion=('잔여 FN은 제품 외곽선 근접성보다 낮은 주변 대비와 작은 bbox 크기에 더 연관되어 보이며, '
                'Gaussian local difference와 blur 지표에서는 일관된 공통 패턴이 약합니다.')
    summary['analysis']=dict(conclusion=conclusion,small_and_low_normalized_contrast_FN=combined,
                             matched_low_confidence_FN=lowconf,machine_counts=machine,
                             next_poc='Train/Val에서 local background difference + connected component + surrounding contrast 후보를 비교하고 FP/image로 선택; 작은 결함군은 별도 hard-example/scale 분석')
    dump(out/'summary.json',summary)
    with (out/'README.md').open('a',encoding='utf-8') as f:
        f.write('\n## 최종 해석과 다음 PoC\n\n')
        for key in ('bbox_area','absolute_contrast','normalized_contrast','local_difference_mean'):
            r=c[key];change=(r['fn_median']/r['tp_median']-1)*100
            f.write(f"- {key}: FN 중앙값은 TP 대비 {change:.1f}% 변화. TP 중앙값 미만 FN {r['fn_below_tp_median']}/{r['fn_count']}개.\n")
        f.write(f"\n크기와 normalized contrast가 모두 TP 중앙값 미만인 FN은 {combined}/12개입니다. "
                    "low-contrast / small-defect 가설은 부분적으로 지지됩니다. 다만 U0371의 3개 FN은 area 28–35 px²로 매우 작지만 local difference는 10.61–13.97로 높습니다. U0488은 area 121 px², normalized contrast 0.186으로 크기/대비 가설의 예외이며 conf≥0.25 검출의 최고 IoU가 0.4985로 기준에 근접해 위치 정밀도 문제도 의심됩니다.\n")
        f.write(f"\nContour 중심 거리는 TP 중앙값 {c['contour_distance']['tp_median']:.2f} px, FN {c['contour_distance']['fn_median']:.2f} px이며 제품 Edge FN은 3/12개입니다. Laplacian variance는 FN 중앙값이 더 높아 일관된 blur 증가 근거가 없습니다. ROI 기반 blur score가 광학 blur를 독립적으로 측정하지 않는 한계는 남습니다.\n")
        f.write(f"\nFN {lowconf}/12개는 IoU≥0.5 예측이 저장된 conf≥0.001 캐시에 존재하지만 confidence가 0.25 미만입니다. 나머지는 이 캐시에서 IoU≥0.5 예측이 없습니다. 낮은 confidence와 위치 오차가 함께 나타날 수 있으므로 이 분류는 원인 확정이 아닙니다.\n")
        f.write('\nMachine별 FN/GT: '+', '.join(f"{r['machine']}번 {r['FN']}/{r['GT']}" for r in machine)+'. 같은 제품/촬영 조건이 얽혀 있고 FN 12개가 8개 이미지에 묶여 있으므로 장비 원인으로 단정할 수 없습니다.\n')
        f.write('\n다음 단계는 Case A를 우선 후보로 두되, Train/Val에서 Gaussian local difference + connected component + surrounding contrast를 결합한 보조 검출 PoC를 비교하는 것입니다. 이번 Gaussian difference 하나만으로는 강한 구분력이 확인되지 않아 단일 difference threshold로 진행하기 어렵습니다. U0371 같은 작은 결함군은 hard example mining과 검출 해상도/scale 변화의 Val 비교를 함께 검토합니다. U0488은 bbox와 예측의 위치 오차를 먼저 점검합니다. Test로 임계값·sigma·padding을 선택하지 않았습니다.\n')
        f.write('\n'+conclusion+'\n')


def main(out):
    if out.exists():raise FileExistsError(f'기존 결과 보호: --out으로 새 경로 지정: {out}')
    protected=[DATA/'manifest.csv',DATA/'dataset.yaml',MODEL,OLD/'predictions.json']
    manifest=csvread(DATA/'manifest.csv');test=[r for r in manifest if r['split']=='test']
    protected += [p for folder in ('images','labels') for p in (DATA/folder/'test').glob('*') if p.is_file()]
    before={str(p):sha(p) for p in protected}
    provenance=json.loads((OLD/'prediction_provenance.json').read_text(encoding='utf-8'))
    assert provenance['model_sha256']==sha(MODEL) and provenance['imgsz']==1280
    cache=json.loads((OLD/'predictions.json').read_text(encoding='utf-8'))
    old_edge={(r['unique_id'],r['bbox_id']):r for r in csvread(ROOT/'outputs/edge_reinspection_poc/edge_results.csv') if r['split']=='test'}
    rows=[];fps=[];images={}
    for record in test:
        uid=record['unique_id'];gray=cv2.imread(str(DATA/'images'/'test'/record['restored_filename']),0)
        mask,quality=product_mask(gray);assert quality['valid'],(uid,quality)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);contour=max(contours,key=cv2.contourArea)
        background=cv2.GaussianBlur(gray.astype(np.float32),(0,0),5)
        gt=label_boxes(record,gray.shape);preds=[dict(p,source='global') for p in cache[uid]]
        found,_,unmatched=matches(gt,preds)
        images[uid]=dict(gray=gray,background=background,preds=preds)
        for i,g in enumerate(gt):
            edge=old_edge[(uid,str(i))]
            row=dict(unique_id=uid,filename=record['restored_filename'],bbox_id=i,status='TP' if i in found else 'FN',
                     machine=record['machine'],capture_date=record['capture_date'],product_edge=edge['product_edge']=='True',
                     legacy_edge=edge['legacy_edge']=='True',**calculate(gray,background,mask,gt,i,contour,preds,found))
            assert row['status']==edge['global_status']
            rows.append(row)
        for i,p in enumerate(unmatched):fps.append(dict(unique_id=uid,fp_id=i,confidence=p['confidence'],x1=p['box'][0],y1=p['box'][1],x2=p['box'][2],y2=p['box'][3]))
    assert len(rows)==160 and sum(r['status']=='FN' for r in rows)==12 and len(fps)==8
    out.mkdir(parents=True);(out/'features').mkdir();(out/'visualizations').mkdir()
    stats=statistics(rows);fn=[r for r in rows if r['status']=='FN'];tp=[r for r in rows if r['status']=='TP']
    for name,selected in [('test_gt_features',rows),('tp_features',tp),('fn_features',fn)]:csvwrite(out/'features'/f'{name}.csv',selected,list(rows[0]))
    csvwrite(out/'features/fp_predictions.csv',fps,list(fps[0]));csvwrite(out/'features/tp_fn_statistics.csv',stats,list(stats[0]))
    comparisons=[]
    for key in MAIN:
        tv=np.array([r[key] for r in tp if r[key] is not None]);fv=np.array([r[key] for r in fn if r[key] is not None])
        comparisons.append(dict(feature=key,tp_median=float(np.median(tv)),fn_median=float(np.median(fv)),
                                 fn_below_tp_median=int(sum(fv<np.median(tv))),fn_count=len(fv),
                                 fn_percentiles=[dict(unique_id=r['unique_id'],bbox_id=r['bbox_id'],
                                     percentile=float(np.mean(tv<=r[key])*100)) for r in fn if r[key] is not None]))
    sheets(fn,images,out/'visualizations');plots(rows,out/'visualizations')
    assert before=={str(p):sha(p) for p in protected}
    summary=dict(images=len(test),GT=160,TP=148,FN=12,FP=8,confidence=CONF,iou=MATCH_IOU,
                 model=str(MODEL),model_sha256=sha(MODEL),parameters=dict(ring_padding='clip(round(max(bbox_width,bbox_height)*0.5),5,15)',
                 gaussian_sigma=5,eps=EPS,seed=42),statistics=stats,comparisons=comparisons,
                 FN_samples=fn,product_edge_FN=sum(r['product_edge'] for r in fn),input_hashes_verified=True)
    dump(out/'summary.json',summary)
    dump(out/'input_hashes.json',before)
    text='''# Test 잔여 FN 원인 분석

## 목적과 기준
Edge ROI PoC의 제품 Edge Test Recall은 70/73(95.89%)로 유지됐고 전체 Recall 92.50%, Precision 94.87%, FP 8, 복구 0개였습니다. 이 분석은 Global YOLO Test 전체 GT 160개를 대상으로 TP 148개, FN 12개, FP 8개를 재확인합니다. 제품 Edge FN 3개는 전체 FN의 일부입니다. 기존 evaluate_edge_reinspection.py의 product_mask, label_boxes, matches와 기존 예측 캐시를 재사용했습니다. confidence≥0.25, IoU≥0.5 일대일 매칭이며 신규 추론이나 threshold 선택은 없습니다.

## Feature 정의
크기는 원 YOLO 부동소수점 bbox의 폭·높이·면적입니다. 영상 통계는 floor lower / ceil upper의 raster bbox 전체 픽셀 기준이며 bbox 내부에는 정상 제품 구조도 포함됩니다. 따라서 bbox 통계는 순수 결함 분할 통계가 아닙니다.

Ring padding은 bbox 최대 변의 절반을 반올림해 5–15 px로 제한하고 이미지 경계에서 clipping했습니다. 모든 GT bbox를 제외하고 기존 제품 mask 내부만 사용합니다. 12픽셀 미만이면 ring 통계는 미정의입니다. 절대 대비=abs(ROI mean−ring mean), 정규화 대비=절대 대비/(ring mean+1e-6).

Gaussian background는 원 grayscale에 sigma=5 px를 적용했습니다. abs(I−B)의 bbox 내부 mean/max/p90을 계산했습니다. Laplacian variance는 전체 영상에서 ksize=3 Laplacian을 계산한 뒤 bbox 내부 분산, gradient는 ksize=3 Sobel 크기의 평균입니다. 작은 bbox의 blur 지표는 제품 구조·대비·크기의 영향을 함께 받으므로 광학 blur의 독립 측정이 아닙니다.

Contour distance는 GT bbox 중심에서 제품 외곽선까지의 최단 유클리드 거리(px)입니다. signed distance도 별도 저장하며 내부는 양수입니다. bbox_to_contour_min_distance는 bbox 사각형과 contour 점 사이 최소 거리입니다. 기존 Edge 48 px 판정은 재사용된 product_edge 열에 저장합니다.

## TP/FN 비교
검정 p-value 없이 중앙값과 개별 GT의 TP 분포 내 위치를 기술합니다.

|Feature|TP median|FN median|TP median보다 작은 FN|
|---|---:|---:|---:|
'''
    for c in comparisons:text+=f"|{c['feature']}|{c['tp_median']:.4f}|{c['fn_median']:.4f}|{c['fn_below_tp_median']}/{c['fn_count']}|\n"
    text+='\n## FN 개별 특징\n\n|GT|area|abs contrast|norm contrast|local diff|contour distance|최고 conf(IoU≥0.5)|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in fn:text+=f"|{r['unique_id']} bbox{r['bbox_id']}|{r['bbox_area']:.1f}|{fmt(r['absolute_contrast'])}|{fmt(r['normalized_contrast'])}|{r['local_difference_mean']:.3f}|{r['contour_distance']:.2f}|{fmt(r['max_confidence_at_iou50'])}|\n"
    text+='''
## 해석 및 다음 단계
이 표는 Test에 대한 설명이며 인과 원인이나 detector threshold를 확정하지 않습니다. FN은 12개이며 일부 GT가 같은 이미지에 있어 표본이 독립적이지 않습니다. Gaussian background에도 제품 경계·홈 구조가 들어가고 bbox 평균은 결함을 희석할 수 있습니다. 다음 PoC의 규칙/파라미터는 Train/Val에서 선택하고 Test는 최종 검증에만 사용해야 합니다.

실제 결과에 따른 해석과 추천은 아래 최종 해석에 기록합니다.

## 재현 및 파일
`.venv\\Scripts\\python.exe -B -X utf8 scripts\\analyze_fn_root_cause.py`
기존 결과가 있으면 덮어쓰지 않으며 `--out outputs/fn_root_cause_analysis_run2`처럼 새 경로를 지정합니다. 데이터, 라벨, manifest, 모델, 예측 캐시 SHA256 전후 동일을 확인했습니다. features/tp_fn_statistics.csv에 mean/median/min/max/std, summary.json에 FN의 TP 분포 percentile을 저장했습니다. 시각화는 fn_contact_sheet.png → tp_fn_feature_comparison.png → fn_local_difference_sheet.png 순서로 확인합니다. Contact sheet의 빨강은 GT, 초록은 confidence 기준을 통과한 검출입니다. FN은 매칭 검출이 없다는 뜻이며 주변 검출이 아예 없다는 뜻은 아닙니다.
'''
    (out/'README.md').write_text(text,encoding='utf-8')
    write_interpretation(out,summary)
    print(json.dumps(dict(TP=148,FN=12,FP=8,comparisons=[{k:v for k,v in c.items() if k!='fn_percentiles'} for c in comparisons]),ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    main(parser.parse_args().out.resolve())
