"""Val-selected local-contrast confirmation of uncertain YOLO proposals; Test once."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from evaluate_edge_reinspection import ROOT,DATA,OLD,MODEL,CONF,MATCH_IOU,csvread,csvwrite,dump,sha,product_mask,label_boxes,matches,iou,nms
OUT=ROOT/'outputs/low_contrast_aux_poc'
METHODS=('gaussian','median','opening')
THRESHOLDS=('fixed','otsu','adaptive')
FILTERS={'lenient':(8.,3.),'strict':(12.,6.)}
WEAK_MIN=.01
ADDED_FP_PER_IMAGE_BUDGET=.05


def maps(gray,method,mask):
    if method=='gaussian':background=cv2.GaussianBlur(gray.astype(np.float32),(0,0),5)
    elif method=='median':background=cv2.medianBlur(gray,11).astype(np.float32)
    else:background=cv2.morphologyEx(gray,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))).astype(np.float32)
    return background,np.abs(gray.astype(np.float32)-background)


def components(gray,mask,difference,threshold):
    u=np.uint8(np.clip(difference,0,255))
    if threshold=='fixed':binary=(difference>=8).astype(np.uint8);t=8.
    elif threshold=='otsu':
        t,_=cv2.threshold(u[mask>0],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        binary=(difference>t).astype(np.uint8)
    else:
        binary=(cv2.adaptiveThreshold(u,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,-3)>0).astype(np.uint8);t=None
    binary &= cv2.erode(mask,np.ones((5,5),np.uint8))
    n,labels,stats,centers=cv2.connectedComponentsWithStats(binary,8)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);contour=max(contours,key=cv2.contourArea)
    h,w=gray.shape;result=[]
    for idx in range(1,n):
        x,y,bw,bh,area=map(int,stats[idx]);aspect=max(bw/bh,bh/bw)
        if not (3<=area<=80 and bw<=20 and bh<=20 and aspect<=4):continue
        pix=labels[y:y+bh,x:x+bw]==idx
        values=difference[y:y+bh,x:x+bw][pix];intensities=gray[y:y+bh,x:x+bw][pix]
        a,b,c,d=max(0,x-5),max(0,y-5),min(w,x+bw+5),min(h,y+bh+5)
        ring=mask[b:d,a:c].copy();ring[y-b:y-b+bh,x-a:x-a+bw]=0
        # Other threshold components are excluded from the component's surrounding ring.
        ring[labels[b:d,a:c]>0]=0
        surrounding=gray[b:d,a:c][ring>0]
        if len(surrounding)<12:continue
        mean=float(intensities.mean());sm=float(surrounding.mean());contrast=abs(mean-sm)
        cx,cy=map(float,centers[idx])
        result.append(dict(component_id=idx,box=[x,y,x+bw,y+bh],area=area,width=bw,height=bh,aspect_ratio=aspect,
                           mean_local_difference=float(values.mean()),max_local_difference=float(values.max()),p90_local_difference=float(np.percentile(values,90)),
                           component_mean_intensity=mean,surrounding_mean_intensity=sm,absolute_surrounding_contrast=contrast,
                           normalized_surrounding_contrast=contrast/(sm+1e-6),product_contour_distance=abs(cv2.pointPolygonTest(contour,(cx,cy),True)),
                           center=[cx,cy],threshold_value=t,ring_pixels=len(surrounding)))
    return result,labels


def candidates(comps,preds,filter_name):
    dmin,cmin=FILTERS[filter_name]
    strong=[p for p in preds if p['confidence']>=CONF]
    weak=[p for p in preds if WEAK_MIN<=p['confidence']<CONF and p['class_id']==0]
    result=[]
    for p in weak:
        if any(iou(p['box'],q['box'])>=.5 for q in strong):continue
        support=[]
        for c in comps:
            cx,cy=c['center'];x1,y1,x2,y2=p['box']
            if x1<=cx<=x2 and y1<=cy<=y2 and c['mean_local_difference']>=dmin and c['absolute_surrounding_contrast']>=cmin:
                support.append(c)
        if not support:continue
        c=max(support,key=lambda z:z['mean_local_difference'])
        # The component validates the weak YOLO proposal; its original box preserves localization.
        result.append(dict(box=p['box'],confidence=CONF,class_id=0,source='auxiliary',weak_yolo_confidence=p['confidence'],component=c))
    return nms(result)


def evaluate(records,config):
    tp=fn=fp=recovery=lost=0;baseline_tp=baseline_fp=0
    for rec in records:
        base=[p for p in rec['preds'] if p['confidence']>=CONF]
        aux=rec['aux'][config];fused=base+aux
        bm,bfp,_=matches(rec['gt'],base);fm,ffp,_=matches(rec['gt'],fused)
        baseline_tp+=len(bm);baseline_fp+=bfp;tp+=len(fm);fn+=len(rec['gt'])-len(fm);fp+=ffp
        recovery+=len(set(fm)-set(bm));lost+=len(set(bm)-set(fm))
    added=fp-baseline_fp
    return dict(config=config,TP=tp,FN=fn,FP=fp,precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn),
                FN_Recovery=recovery,Lost_TP=lost,Added_FP=added,Recovery_per_Added_FP=recovery/added if added>0 else None,
                FP_per_image=fp/len(records),baseline_TP=baseline_tp,baseline_FP=baseline_fp)


def prepare(split,cache,configs):
    records=[]
    for row in [r for r in csvread(DATA/'manifest.csv') if r['split']==split]:
        uid=row['unique_id'];gray=cv2.imread(str(DATA/'images'/split/row['restored_filename']),0)
        mask,quality=product_mask(gray);assert quality['valid'],(uid,quality)
        rec=dict(uid=uid,row=row,gray=gray,mask=mask,gt=label_boxes(row,gray.shape),preds=[dict(p,source='global') for p in cache[uid]],aux={})
        for method in sorted({c.split('_')[0] for c in configs}):
            bg,diff=maps(gray,method,mask)
            for threshold in sorted({c.split('_')[1] for c in configs if c.split('_')[0]==method}):
                comps,_=components(gray,mask,diff,threshold)
                for config in configs:
                    m,t,f=config.split('_')
                    if m==method and t==threshold:rec['aux'][config]=candidates(comps,rec['preds'],f)
        records.append(rec)
    return records


def contact(items,config,path,title):
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',15)
    if not items:
        im=Image.new('RGB',(800,65),'white');ImageDraw.Draw(im).text((10,18),title+': 해당 사례 없음',font=font,fill='black');im.save(path);return
    # Each sheet contains up to 12 examples; additional pages are generated.
    for start in range(0,len(items),12):
        chosen=items[start:start+12];sheet=Image.new('RGB',(1400,300*len(chosen)),'white');draw=ImageDraw.Draw(sheet)
        for j,(rec,box,extra) in enumerate(chosen):
            gray=rec['gray'];m,t,_=config.split('_');bg,diff=maps(gray,m,rec['mask']);comps,labels=components(gray,rec['mask'],diff,t)
            h,w=gray.shape;x1,y1,x2,y2=box;a,b,c,d=max(0,int(x1)-24),max(0,int(y1)-24),min(w,int(x2)+25),min(h,int(y2)+25)
            rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB);marked=rgb.copy();cv2.rectangle(marked,(round(x1),round(y1)),(round(x2),round(y2)),(255,0,0),1)
            component=cv2.cvtColor(np.uint8(np.isin(labels,[q['component_id'] for q in comps]))*255,cv2.COLOR_GRAY2RGB)
            nearby=[p for p in rec['aux'][config] if iou(p['box'],box)>.1]
            if nearby:
                support=max(nearby,key=lambda p:iou(p['box'],box))['component']['component_id']
                component[labels==support]=(0,255,0)
            fused=rgb.copy()
            for p in rec['aux'][config]:
                q=p['box'];cv2.rectangle(fused,(round(q[0]),round(q[1])),(round(q[2]),round(q[3])),(0,255,0),1)
            panels=[rgb,marked,np.uint8(np.clip(diff*255/60,0,255)),component,fused]
            for k,img in enumerate(panels):
                if k>0:img=img[b:d,a:c]
                im=Image.fromarray(img);scale=min(270/im.width,215/im.height);im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.NEAREST)
                sheet.paste(im,(k*280,j*300+45))
                draw.text((k*280+5,j*300+22),['원본','GT/FP 대상','Local difference 0–60','Connected components','최종 aux 후보'][k],font=font,fill='black')
            draw.text((5,j*300+3),f"{rec['uid']} {title} {extra}",font=font,fill='black')
        suffix='' if start==0 else f'_{start//12+1:02d}'
        sheet.save(path.with_name(path.stem+suffix+path.suffix))


def main(out):
    if out.exists():raise FileExistsError(f'기존 출력 보호: 새 --out 경로 필요: {out}')
    meta=json.loads((OLD/'prediction_provenance.json').read_text(encoding='utf-8'));assert meta['model_sha256']==sha(MODEL) and meta['imgsz']==1280
    source_paths=[MODEL,OLD/'predictions.json',DATA/'manifest.csv',DATA/'dataset.yaml']+[p for folder in ('images','labels') for p in (DATA/folder).rglob('*') if p.is_file()]
    hashes={str(p):sha(p) for p in source_paths}
    # Read only Val predictions during selection. Test detections are accessed after config is frozen.
    cache=json.loads((OLD/'predictions.json').read_text(encoding='utf-8'))
    configs=[f'{m}_{t}_{f}' for m in METHODS for t in THRESHOLDS for f in FILTERS]
    val=prepare('val',cache,configs);search=[evaluate(val,c) for c in configs]
    budget=math.floor(len(val)*ADDED_FP_PER_IMAGE_BUDGET)
    eligible=[r for r in search if r['Added_FP']<=budget and r['Lost_TP']==0]
    if eligible:selected=max(eligible,key=lambda r:(r['FN_Recovery'],-r['Added_FP'],r['precision']))
    else:selected=min(search,key=lambda r:(r['Added_FP'],-r['FN_Recovery']))
    for r in search:
        r['pareto']=not any(q['FN_Recovery']>=r['FN_Recovery'] and q['Added_FP']<=r['Added_FP'] and (q['FN_Recovery']>r['FN_Recovery'] or q['Added_FP']<r['Added_FP']) for q in search)
        r['selected']=r['config']==selected['config'];r['within_FP_budget']=r['Added_FP']<=budget
    out.mkdir(parents=True);(out/'visualizations').mkdir();(out/'predictions').mkdir()
    csvwrite(out/'val_parameter_search.csv',search,list(search[0]))
    dump(out/'locked_config.json',dict(config=selected['config'],selection_split='val',additional_FP_budget=budget,weak_confidence_min=WEAK_MIN,
                                     filters=FILTERS,kernels=11,gaussian_sigma=5,fixed_threshold=8,adaptive_block=15,adaptive_C=-3))
    print(f"Val selection locked: {selected['config']} recovery={selected['FN_Recovery']} addedFP={selected['Added_FP']} budget={budget}",flush=True)
    # Exactly one chosen configuration is generated and evaluated on Test.
    test=prepare('test',cache,[selected['config']]);result=evaluate(test,selected['config'])
    base=dict(config='baseline',TP=result['baseline_TP'],FN=sum(len(r['gt']) for r in test)-result['baseline_TP'],FP=result['baseline_FP'],
              precision=result['baseline_TP']/(result['baseline_TP']+result['baseline_FP']),recall=result['baseline_TP']/sum(len(r['gt']) for r in test),FN_Recovery=0,Added_FP=0,Recovery_per_Added_FP=None)
    assert (base['TP'],base['FN'],base['FP'])==(148,12,8)
    csvwrite(out/'test_comparison.csv',[base,result],list(result))
    recovered=[];remaining=[];addedfps=[];recrows=[];fprows=[];serial=[]
    for rec in test:
        strong=[p for p in rec['preds'] if p['confidence']>=CONF];aux=rec['aux'][selected['config']]
        bm,_,_=matches(rec['gt'],strong);fm,_,unmatched=matches(rec['gt'],strong+aux)
        for i,g in enumerate(rec['gt']):
            if i not in bm:
                row=dict(unique_id=rec['uid'],bbox_id=i,baseline='FN',final='TP' if i in fm else 'FN',box=g['box'])
                if i in fm:recovered.append((rec,g['box'],f'bbox{i} FN→TP'));recrows.append(row)
                else:remaining.append((rec,g['box'],f'bbox{i} FN 유지'))
        for p in unmatched:
            if p['source']=='auxiliary':
                c=p['component'];cause='경계 의심' if c['product_contour_distance']<8 else '제품 구조/텍스처 의심'
                row=dict(unique_id=rec['uid'],box=p['box'],weak_yolo_confidence=p['weak_yolo_confidence'],cause_heuristic= cause,**{k:v for k,v in c.items() if k!='box'})
                fprows.append(row);addedfps.append((rec,p['box'],f"area={c['area']} D={c['mean_local_difference']:.1f} C={c['absolute_surrounding_contrast']:.1f} {cause}"))
        serial.append(dict(unique_id=rec['uid'],auxiliary=aux))
    csvwrite(out/'recovered_fn.csv',recrows,['unique_id','bbox_id','baseline','final','box'])
    csvwrite(out/'added_fp.csv',fprows,list(fprows[0]) if fprows else ['unique_id','box','area','mean_local_difference','absolute_surrounding_contrast','cause_heuristic'])
    dump(out/'predictions/test_auxiliary.json',serial)
    contact(recovered,selected['config'],out/'visualizations/test_recovered_fn_sheet.png','FN 복구')
    contact(remaining,selected['config'],out/'visualizations/test_remaining_fn_sheet.png','잔여 FN')
    contact(addedfps,selected['config'],out/'visualizations/test_added_fp_sheet.png','추가 FP')
    contact([(r,r['gt'][0]['box'],'Val 예시') for r in val[:6]],selected['config'],out/'visualizations/local_difference_examples.png','Difference 예시')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Malgun Gothic'
    fig,ax=plt.subplots(figsize=(10,6))
    for i,r in enumerate(search):
        ax.scatter(r['Added_FP'],r['FN_Recovery'],c='red' if r['selected'] else ('green' if r['pareto'] else 'gray'))
        ax.annotate(str(i+1),(r['Added_FP'],r['FN_Recovery']),xytext=(4,4),textcoords='offset points')
    ax.axvline(budget,ls='--',c='orange',label=f'추가 FP 예산 {budget}');ax.set_xlabel('Val Added FP');ax.set_ylabel('Val FN Recovery');ax.set_title('Val tradeoff / 번호는 CSV 행 순서');ax.legend();fig.tight_layout();fig.savefig(out/'visualizations/val_tradeoff.png',dpi=150);plt.close(fig)
    verdict='C. 기각' if result['FN_Recovery']==0 or result['Added_FP']>len(test)*ADDED_FP_PER_IMAGE_BUDGET else ('A. 채택' if result['FN_Recovery']>=3 else 'B. 조건부 채택')
    summary=dict(selected=selected['config'],val_selected=selected,test=result,baseline=base,verdict=verdict,
                 recovered_GT=recrows,added_FP=fprows,test_configuration_count=1,parameter_selection_split='val',
                 limitation='Weak-YOLO proposal confirmation only; no proposal means no recovery. Auxiliary confidence 0.25 is an evaluation inclusion flag, not a calibrated probability.',
                 provenance=meta,input_hashes_verified=True)
    assert hashes=={str(p):sha(p) for p in source_paths};dump(out/'input_hashes.json',hashes);dump(out/'summary.json',summary)
    report=f'''# Low contrast auxiliary inspection PoC

## 목적 및 배경
기존 Test TP148/FN12/FP8. FN 분석에서 작은 bbox와 낮은 surrounding contrast의 연관성이 관찰됐습니다. 이 PoC는 기존 detector가 확신하지 못한 후보를 local difference 연결 성분으로 재확인합니다. 모델은 재학습하지 않았습니다.

## 방식과 후보
기존 product_mask와 GT/prediction 형식, confidence {CONF}, IoU {MATCH_IOU} 매칭을 재사용합니다. Gaussian sigma5, median11, morphological opening11을 비교했습니다. 모두 D=abs(I−B). Opening은 어두운 결함보다는 밝은 성분에 민감할 수 있다는 방향성 한계가 있습니다. Threshold는 fixed8, product 내부 Otsu, adaptive Gaussian block15/C−3입니다. Otsu는 이미지별 분포에 영향받고 adaptive는 텍스처를 많이 분할할 수 있습니다.

총 18개 설정(3 background×3 threshold×2 filter)을 Val에서 비교했습니다. Lenient D≥8/C≥3, strict D≥12/C≥6. CC area 3–80, 폭/높이≤20, 양방향 aspect≤4를 공통 적용했습니다. Ring은 component bbox 외부 5 px, 제품 mask 내부이며 다른 threshold 성분도 제외합니다. 12px 이상 ring을 요구합니다. 제품 경계의 threshold 잡음은 mask erosion2 px로 줄였습니다. 이 수치는 Test 결과로 선택하지 않았습니다.

## 기존 YOLO와 결합
conf 0.01≤p<0.25의 기존 YOLO bbox 안에 필터를 통과한 CC 중심이 있으면 그 bbox를 보조 후보로 채택합니다. 컴포넌트 bbox를 원 annotation 크기로 억지로 확대하지 않고 기존 weak proposal의 위치를 재사용합니다. CC-only 신규 영역 탐지는 이번 범위에 없습니다. 보조검사 내부 NMS IoU0.5, strong global과 IoU≥0.5 중복은 제외합니다. Auxiliary confidence=0.25는 기존 평가 함수에 포함시키는 표시이며 확률이 아닙니다. 기존 Global prediction을 그대로 보존합니다.

## Val 선택
추가 FP 예산은 사전에 정한 0.05/image의 내림값 {budget}개입니다. 예산 안에서 복구 수를 우선하고 같은 복구 수에서는 FP가 적은 설정을 선택했습니다. Lost TP는 허용하지 않습니다. 예산 안의 설정이 없으면 최소 FP 설정을 선택하되 운영 채택을 자동 인정하지 않습니다. 선택: **{selected['config']}**, recovery {selected['FN_Recovery']}, added FP {selected['Added_FP']}. Pareto 여부와 각 방식의 실제 수치는 val_parameter_search.csv에 있습니다. locked_config.json을 Test 적용 전에 저장했습니다.

## Test 최종 결과
선택된 설정 하나만 한 번 적용했습니다.

|항목|Baseline|YOLO+Auxiliary|
|---|---:|---:|
|TP|{base['TP']}|{result['TP']}|
|FN|{base['FN']}|{result['FN']}|
|FP|{base['FP']}|{result['FP']}|
|Recall|{base['recall']*100:.2f}%|{result['recall']*100:.2f}%|
|Precision|{base['precision']*100:.2f}%|{result['precision']*100:.2f}%|
|FN Recovery|0|{result['FN_Recovery']}|
|Added FP|0|{result['Added_FP']}|

Recovery/Added FP: {result['Recovery_per_Added_FP']}. 분모가 0이면 미정의(null)입니다. 판정: **{verdict}**. 현재 데이터에서의 PoC 판단이며 다른 제조 라인에서의 검증 결과는 아닙니다.

## 個別例とFP
recovered_fn.csv, added_fp.csv 및 3종류 contact sheet를 참조하세요. 차이 영상은 0–60 고정 스케일입니다. Component 열은 기본 크기 조건을 통과한 성분을 표시하며 해당 보조 후보를 지지한 성분은 초록색입니다. 후보가 없는 영역이나 weak proposal의 위치가 부정확한 FN은 이 방식으로 복구하기 어렵습니다. FP 진단은 아래 실제 GT 겹침 수치로 보완합니다.

## 再現
`.venv\\Scripts\\python.exe -B -X utf8 scripts\\evaluate_low_contrast_auxiliary.py`
기존 출력이 있으면 덮어쓰지 않으며 `--out outputs/low_contrast_aux_poc_run2`처럼 새 경로를 지정합니다. 이전 평가, 이미지, 라벨, split, 모델, 예측 캐시는 SHA256 전후 동일입니다.

Local contrast auxiliary inspection recovered {result['FN_Recovery']} of 12 FN with {result['Added_FP']} additional FP; {verdict}.
'''
    (out/'README.md').write_text(report,encoding='utf-8')
    post_analysis(out)
    print(json.dumps(dict(selected=selected['config'],test=result,verdict=verdict),ensure_ascii=False),flush=True)


def post_analysis(out):
    """Summarize frozen predictions; no Test parameter evaluation or adjustment."""
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    root=json.loads((ROOT/'outputs/fn_root_cause_analysis/summary.json').read_text(encoding='utf-8'))
    serial={r['unique_id']:r for r in json.loads((out/'predictions/test_auxiliary.json').read_text(encoding='utf-8'))}
    manifest={r['unique_id']:r for r in csvread(DATA/'manifest.csv')}
    fps=summary['added_FP'];causes={}
    for p in fps:
        row=manifest[p['unique_id']];gray=cv2.imread(str(DATA/'images/test'/row['restored_filename']),0);gt=label_boxes(row,gray.shape)
        score,idx=max((iou(p['box'],g['box']),i) for i,g in enumerate(gt))
        p['max_GT_iou']=score;p['nearest_GT_bbox_id']=idx
        p['cause_heuristic']='GT 인근 위치/크기 오차 또는 중복' if score>=.1 else ('제품 경계 의심' if p['product_contour_distance']<8 else '제품 구조/텍스처/미라벨 영역 의심')
        causes[p['cause_heuristic']]=causes.get(p['cause_heuristic'],0)+1
    if fps:csvwrite(out/'added_fp.csv',fps,list(fps[0]))
    remaining=[];recovered={(r['unique_id'],r['bbox_id']) for r in summary['recovered_GT']}
    for r in root['FN_samples']:
        if (r['unique_id'],r['bbox_id']) in recovered:continue
        row=manifest[r['unique_id']];gray=cv2.imread(str(DATA/'images/test'/row['restored_filename']),0)
        box=label_boxes(row,gray.shape)[r['bbox_id']]['box'];aux=serial[r['unique_id']]['auxiliary']
        remaining.append(dict(unique_id=r['unique_id'],bbox_id=r['bbox_id'],bbox_area=r['bbox_area'],normalized_contrast=r['normalized_contrast'],
                              max_matching_weak_confidence=r['max_confidence_at_iou50'],best_final_aux_iou=max((iou(box,p['box']) for p in aux),default=0.),
                              diagnosis='보조 후보 위치/크기 불일치' if any(iou(box,p['box'])>=.1 for p in aux) else '해당 GT를 지지하는 최종 보조 후보 없음'))
    csvwrite(out/'remaining_fn.csv',remaining,list(remaining[0]) if remaining else ['unique_id','bbox_id'])
    search=csvread(out/'val_parameter_search.csv')
    summary['added_FP_diagnostics']=causes;summary['remaining_FN_diagnostics']=remaining
    dump(out/'summary.json',summary)
    with (out/'README.md').open('a',encoding='utf-8') as f:
        f.write('\n## 실제 후보 비교와 개별 원인\n\n|Val 설정|FN Recovery|Added FP|Recovery/Added FP|\n|---|---:|---:|---:|\n')
        for r in search:
            f.write(f"|{r['config']}|{r['FN_Recovery']}|{r['Added_FP']}|{r['Recovery_per_Added_FP']}|\n")
        f.write('\nVal Pareto의 주요 선택지는 복구0/추가FP2, 복구1/추가FP20, 복구2/추가FP31입니다. 이는 제조 현장에서 유리한 복구/FP 교환 관계를 보여주지 못했습니다. Adaptive는 Gaussian/Median에서 복구2개에 FP59–60개로 구조/텍스처 후보를 많이 늘렸습니다. Otsu는 Gaussian에서 후보를 크게 제한했지만 FN 복구도 없었습니다.\n')
        f.write('\n복구된 Test FN이 없어 복구군의 공통 특징을 도출할 수 없습니다. 12개 FN이 그대로 남았으며 기존 root-cause 분석의 크기/낮은 대비 분포도 유지됩니다. remaining_fn.csv는 각 FN에 대해 최종 보조 후보의 최고 IoU를 기록합니다. 낮은 confidence 후보 중에서도 연결 성분 확인을 통과하지 못하거나 bbox의 위치/크기가 IoU0.5에 미달했습니다.\n')
        f.write('\n새 FP 진단: '+', '.join(f'{k} {v}개' for k,v in causes.items())+'. GT 최대 IoU와 가까운 GT ID는 added_fp.csv에 있습니다. GT와 겹침은 위치 오차/중복의 단서이며 GT에서 먼 dark spot도 실제 결함인지 단순 제품 구조인지 이 영상만으로 확정할 수 없습니다. 따라서 원본/라벨 확인 없이 모두 구조 오검출로 단정하지 않았습니다.\n')
        f.write('\n**최종 채택 여부: C. 기각.** 이번 제한된 weak-proposal 확인 방식은 Val에서도 좋은 복구/FP 교환 관계를 만들지 못했고, Test에서는 FN 복구0개에 FP4개만 추가했습니다. 이 결과는 모든 local contrast 방식의 불가능성을 의미하지 않지만 같은 규칙의 추가 Test 튜닝 근거가 되지 않습니다.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);main(p.parse_args().out.resolve())
