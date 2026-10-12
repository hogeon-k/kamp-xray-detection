"""GT bbox 내부 영상 특성을 이용한 이물질 영역 추정. 입력 파일은 읽기 전용."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'outputs/yolo_512_telea'
OUT = ROOT/'outputs/foreign_object_condition_analysis'
METHODS = ['Otsu','adaptive','local_difference','connected_component','intensity_contrast']
FONT = ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16)

def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def write_csv(path, rows, fields=None):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0])); w.writeheader(); w.writerows(rows)

def save_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def fingerprint():
    paths=list(DATA.rglob('*'))+list((ROOT/'outputs/clean_512_split').rglob('*'))+list((ROOT/'outputs/clean_512').rglob('*'))
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}

def load_records():
    manifest=read_csv(DATA/'manifest.csv'); frozen={r['unique_id']:r for r in read_csv(ROOT/'outputs/clean_512_split/split_manifest.csv')}
    records=[]
    for r in manifest:
        assert all(r[k]==frozen[r['unique_id']][k] for k in ['split','filename','machine','capture_date'])
        image=cv2.imread(str(DATA/'images'/r['split']/r['restored_filename']),cv2.IMREAD_GRAYSCALE)
        h,w=image.shape
        lines=(DATA/'labels'/r['split']/(Path(r['filename']).stem+'.txt')).read_text().splitlines()
        for i,line in enumerate(lines):
            cls,x,y,bw,bh=map(float,line.split()); assert cls==0
            x1,y1=max(0,math.ceil((x-bw/2)*w)),max(0,math.ceil((y-bh/2)*h))
            x2,y2=min(w,math.ceil((x+bw/2)*w)),min(h,math.ceil((y+bh/2)*h))
            crop=image[y1:y2,x1:x2]
            records.append(dict(r,bbox_id=i,bbox_x1=x1,bbox_y1=y1,bbox_x2=x2,bbox_y2=y2,bbox_area=crop.size,image_width=w,image_height=h,proxy_mean=float(crop.mean()),proxy_edge=min(x*w,y*h,(1-x)*w,(1-y)*h)))
    assert len(manifest)==512 and len(records)==1163
    return records

def select_poc(records):
    train=[r for r in records if r['split']=='train']; selected=[]; used=set()
    for key in ['bbox_area','proxy_mean','proxy_edge']:
        ordered=sorted(train,key=lambda r:(r[key],r['unique_id'],r['bbox_id']))
        for q in np.linspace(0,1,9):
            idx=int(round(q*(len(ordered)-1)))
            choices=sorted(range(len(ordered)),key=lambda j:abs(j-idx))
            r=next(ordered[j] for j in choices if (ordered[j]['unique_id'],ordered[j]['bbox_id']) not in used)
            used.add((r['unique_id'],r['bbox_id'])); selected.append(r)
    return selected

def component(mask, strength, interior=False):
    n,lab,stats,_=cv2.connectedComponentsWithStats(mask.astype('uint8'),8)
    if n<=1:return np.zeros_like(mask,bool),0
    scores=[float(np.maximum(strength[lab==i],0).sum()) if not interior or (stats[i,0]>0 and stats[i,1]>0 and stats[i,0]+stats[i,2]<mask.shape[1] and stats[i,1]+stats[i,3]<mask.shape[0]) else 0 for i in range(1,n)]
    if max(scores)==0:return np.zeros_like(mask,bool),1
    order=np.argsort(scores)[::-1]; best=int(order[0])+1
    ambiguity=(scores[int(order[1])]/max(scores[int(order[0])],1e-6)) if len(order)>1 else 0
    return lab==best,float(ambiguity)

def extract_candidates(gray,box):
    x1,y1,x2,y2=box; crop=gray[y1:y2,x1:x2]; h,w=crop.shape
    sigma=max(2,min(h,w)/3)
    bg=cv2.GaussianBlur(gray.astype('float32'),(0,0),sigma)[y1:y2,x1:x2]
    residual=bg-crop.astype('float32')
    _,otsu=cv2.threshold(crop,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    block=max(3,min(31,(min(h,w)//2)*2+1))
    adaptive=cv2.adaptiveThreshold(crop,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,block,3)
    positive=np.maximum(residual,0); scaled=np.uint8(np.clip(positive/max(float(positive.max()),1e-6)*255,0,255))
    t,_=cv2.threshold(scaled,0,255,cv2.THRESH_OTSU)
    raw=(scaled>t)&(residual>3)
    cc,ambiguity=component(raw,residual,interior=True)
    # Intensity+contrast: both darker-than-local-background and below crop median.
    combined,_=component(raw&(crop<np.median(crop)),residual,interior=True)
    return dict(zip(METHODS,[otsu>0,adaptive>0,raw,cc,combined])),residual,ambiguity

def box_of(r):return tuple(r[k] for k in ['bbox_x1','bbox_y1','bbox_x2','bbox_y2'])

def panel_row(r,gray,candidates,size=150):
    x1,y1,x2,y2=box_of(r); crop=gray[y1:y2,x1:x2]; base=cv2.cvtColor(crop,cv2.COLOR_GRAY2RGB)
    canvas=Image.new('RGB',(size*4,5*(size+25)+35),'white'); d=ImageDraw.Draw(canvas)
    d.text((5,2),f"{r['unique_id']} bbox {r['bbox_id']} / {crop.shape[1]}×{crop.shape[0]}",font=FONT,fill='black')
    for j,(name,mask) in enumerate(candidates.items()):
        contour=base.copy(); cs,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(contour,cs,-1,(0,255,0),1)
        overlay=base.copy(); overlay[mask]=(0.55*overlay[mask]+0.45*np.array([255,0,0])).astype('uint8')
        for k,(im,title) in enumerate([(base,'원본 crop'),(mask.astype('uint8')*255,'mask'),(contour,'contour'),(overlay,'object overlay')]):
            canvas.paste(Image.fromarray(im).convert('RGB').resize((size,size),Image.Resampling.NEAREST),(k*size,60+j*(size+25)))
            short=['전역 Otsu','적응 임계값','국소 배경 차이','연결 성분','명암+대비'][j]
            d.text((k*size+2,35+j*(size+25)),short if k==0 else title,font=FONT,fill='black')
    return canvas

def make_poc(records):
    OUT.mkdir(exist_ok=False); (OUT/'poc').mkdir()
    baseline=fingerprint(); save_json(OUT/'input_hashes.json',baseline)
    chosen=select_poc(records); write_csv(OUT/'poc/selected_bbox.csv',chosen)
    sheets=[]
    for i,r in enumerate(chosen):
        gray=cv2.imread(str(DATA/'images'/r['split']/r['restored_filename']),0)
        candidates,_,_=extract_candidates(gray,box_of(r)); im=panel_row(r,gray,candidates)
        im.save(OUT/'poc'/f"{i+1:02d}_{r['unique_id']}_bbox{r['bbox_id']}.png"); sheets.append(im)
        for method,m in candidates.items(): Image.fromarray(m.astype('uint8')*255).save(OUT/'poc'/f"{i+1:02d}_{method}_mask.png")
    for start in range(0,len(sheets),3):
        sheet=Image.new('RGB',(1800,935),'white')
        for j,im in enumerate(sheets[start:start+3]):sheet.paste(im,(600*j,0))
        sheet.save(OUT/'poc'/f'비교_{start//3+1:02d}.png')
    assert fingerprint()==baseline
    print('PoC 27 bbox 생성 완료')

OBJECT_FIELDS=['foreign_object_area','foreign_object_width','foreign_object_height','object_area_ratio','object_center_x','object_center_y','edge_distance','normalized_edge_distance','aspect_ratio','perimeter','circularity','solidity','extent','major_minor_axis_ratio','object_mean_intensity','object_median_intensity','object_std_intensity','background_mean_intensity','background_std_intensity','relative_intensity_difference','contrast_abs','contrast_normalized','michelson_contrast','local_z_score']

def measure(r,gray,mask,residual,ambiguity,all_boxes,restoration):
    x1,y1,x2,y2=box_of(r); crop=gray[y1:y2,x1:x2]; ys,xs=np.where(mask); area=len(xs)
    full=np.zeros_like(gray,np.uint8); full[y1:y2,x1:x2]=mask
    radius=max(3,min(12,round(math.sqrt(max(area,1)))))
    ring=cv2.dilate(full,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))).astype(bool)&~full.astype(bool)
    for b in all_boxes:
        if b!=box_of(r):ring[b[1]:b[3],b[0]:b[2]]=False
    ring[restoration>0]=False
    bg=gray[ring].astype(float); obj=crop[mask].astype(float)
    reason=[]
    if area<4:reason.append('추정 픽셀 부족')
    if area and (abs(xs.mean()/(max(1,crop.shape[1]-1))-.5)>.35 or abs(ys.mean()/(max(1,crop.shape[0]-1))-.5)>.35):reason.append('bbox 외곽 성분: 제품 구조 의심')
    if area>crop.size*0.65:reason.append('제품 구조/넓은 영역 가능성')
    if ambiguity>0.5:reason.append('복수 성분 모호')
    border=int(mask[0].sum()+mask[-1].sum()+mask[:,0].sum()+mask[:,-1].sum())
    if border>max(2,math.sqrt(max(area,1))):reason.append('bbox 경계 접촉')
    if len(bg)<12:reason.append('정상 배경 표본 부족')
    if area and len(bg) and bg.mean()-obj.mean()<5:reason.append('배경 대비 불충분')
    if area and float(residual[mask].mean())<5:reason.append('국소 대비 불충분')
    restored_overlap=int(restoration[y1:y2,x1:x2][mask].astype(bool).sum())
    if restored_overlap>max(1,area*0.1):reason.append('추정 영역 복원 픽셀 영향')
    result={k:'' for k in OBJECT_FIELDS}; result.update(segmentation_reliable=not reason,segmentation_reason='; '.join(reason),estimated_area_diagnostic=area,component_ambiguity=ambiguity,object_restored_pixels=restored_overlap,background_pixel_count=len(bg))
    if reason:return result,full
    width=int(xs.max()-xs.min()+1); height=int(ys.max()-ys.min()+1); cx=float(xs.mean()+x1+.5); cy=float(ys.mean()+y1+.5)
    cs,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); c=max(cs,key=cv2.contourArea); perimeter=float(cv2.arcLength(c,True)); ca=float(cv2.contourArea(c)); hull=float(cv2.contourArea(cv2.convexHull(c)))
    # Shape contour area is kept separate from pixel area: pixel masks on tiny objects are discrete.
    eig=np.linalg.eigvalsh(np.cov(np.array([xs,ys],float)))
    edge=min(cx,cy,r['image_width']-cx,r['image_height']-cy)
    normedge=min(cx/r['image_width'],(r['image_width']-cx)/r['image_width'],cy/r['image_height'],(r['image_height']-cy)/r['image_height'])
    contrast=abs(float(obj.mean()-bg.mean()))
    result.update(dict(zip(OBJECT_FIELDS,[area,width,height,area/gray.size,cx,cy,edge,normedge,width/height,perimeter,4*math.pi*ca/perimeter**2 if perimeter else 0,ca/hull if hull else 0,area/(width*height),math.sqrt(eig[1]/eig[0]) if eig[0]>0 else '',float(obj.mean()),float(np.median(obj)),float(obj.std()),float(bg.mean()),float(bg.std()),float((obj.mean()-bg.mean())/max(bg.mean(),1)),contrast,contrast/255,contrast/max(float(obj.mean()+bg.mean()),1),float((obj.mean()-bg.mean())/max(bg.std(),1))])))
    if ca<=0 or hull<=0:result.update(circularity='',solidity='')
    return result,full

def distribution(features):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Malgun Gothic'; plt.rcParams['axes.unicode_minus']=False
    keys=['foreign_object_area','object_area_ratio','edge_distance','normalized_edge_distance','aspect_ratio','circularity','solidity','extent','object_mean_intensity','contrast_abs','contrast_normalized']
    rows=[]; candidates={}
    fig,axes=plt.subplots(4,3,figsize=(15,16))
    for ax,key in zip(axes.flat,keys):
        for group in ['train','all']:
            v=np.array([float(r[key]) for r in features if r['segmentation_reliable'] and r[key]!='' and (group=='all' or r['split']==group)],float)
            if not len(v):continue
            qs=np.quantile(v,[0,.25,1/3,.5,2/3,.75,1]); rows.append(dict(feature=key,scope=group,n=len(v),minimum=qs[0],q25=qs[1],q33=qs[2],median=qs[3],q67=qs[4],q75=qs[5],maximum=qs[6],mean=v.mean(),std=v.std()))
            ax.hist(v,bins=25,alpha=.45,label=f'{group} n={len(v)}')
            if group=='train':candidates[key]={'q25':float(qs[1]),'q33':float(qs[2]),'q67':float(qs[4])}
        ax.set_title(key); ax.legend()
    axes.flat[-1].axis('off'); fig.suptitle('신뢰도 기준 통과 영역: Train / 전체 분포'); fig.tight_layout(); fig.savefig(OUT/'figures/특성_분포.png',dpi=130); plt.close(fig)
    write_csv(OUT/'train_distribution.csv',rows); save_json(OUT/'train_threshold_candidates.json',candidates)
    return rows,candidates

def full_analysis(records):
    for folder in ['masks','visualizations','figures']:(OUT/folder).mkdir(exist_ok=True)
    assert (OUT/'poc/visual_review.json').exists(),'PoC 육안검토와 방법 선정 기록이 먼저 필요합니다.'
    rejects_path=OUT/'poc/manual_rejections.json'
    rejects=json.loads(rejects_path.read_text(encoding='utf-8')) if rejects_path.exists() else {}
    features=[]; diagnostics=[]; byid={}
    for r in records:byid.setdefault(r['unique_id'],[]).append(r)
    for image_rows in byid.values():
        r0=image_rows[0]; gray=cv2.imread(str(DATA/'images'/r0['split']/r0['restored_filename']),0)
        restoration=cv2.imread(str(DATA/'restoration_masks'/r0['split']/r0['restored_filename']),0)
        if restoration is None:raise FileNotFoundError(r0['restored_filename'])
        for r in image_rows:
            c,residual,ambiguity=extract_candidates(gray,box_of(r)); values,mask=measure(r,gray,c['connected_component'],residual,ambiguity,[box_of(q) for q in image_rows],restoration)
            reject=rejects.get(f"{r['unique_id']}:{r['bbox_id']}")
            if reject:
                values.update({k:'' for k in OBJECT_FIELDS}); values.update(segmentation_reliable=False,segmentation_reason=reject)
            row={k:r[k] for k in ['unique_id','filename','split','bbox_id','bbox_x1','bbox_y1','bbox_x2','bbox_y2','bbox_area','machine','capture_date','image_width','image_height']}; row.update(bbox_aspect_ratio=(r['bbox_x2']-r['bbox_x1'])/(r['bbox_y2']-r['bbox_y1']),**values,prediction_status='未指定モデル',prediction_confidence='',prediction_iou='')
            row['prediction_status']='모델 미지정'; features.append(row)
            Image.fromarray(mask*255).save(OUT/'masks'/f"{r['unique_id']}_bbox{r['bbox_id']}_estimated.png")
            if len(diagnostics)<30 and not values['segmentation_reliable']:diagnostics.append((r,gray.copy(),c))
    write_csv(OUT/'features.csv',features)
    for i,(r,gray,c) in enumerate(diagnostics):panel_row(r,gray,c,120).save(OUT/'visualizations'/f"불안정_{i+1:02d}_{r['unique_id']}_bbox{r['bbox_id']}.png")
    for start in range(0,min(12,len(diagnostics)),3):
        sheet=Image.new('RGB',(1440,760),'white')
        for j,(r,gray,c) in enumerate(diagnostics[start:start+3]):sheet.paste(panel_row(r,gray,c,120),(j*480,0))
        sheet.save(OUT/'visualizations'/f'어려운_사례_{start//3+1:02d}.png')
    rows,candidates=distribution(features)
    runs=[str(p.relative_to(ROOT)) for p in (ROOT/'outputs/yolo_runs').glob('*/weights/best.pt')]
    write_csv(OUT/'condition_performance.csv',[],['model','split','condition','GT_count','TP','FN','recall','mean_confidence','mean_iou'])
    reliable=sum(r['segmentation_reliable'] for r in features)
    summary={'표현':'GT bbox 내부 영상 특성을 이용한 이물질 영역 추정','PoC_bbox수':27,'방법':METHODS,'선택':'local_difference + connected_component','전체_bbox':len(features),'신뢰도_기준_통과':reliable,'비율':reliable/len(features),'split_통과':{s:sum(r['segmentation_reliable'] for r in features if r['split']==s) for s in ['train','val','test']},'분포':rows,'기준_후보':candidates,'성능_분석':'최종 모델 미지정: TP/FN을 계산하지 않았습니다.','모델_후보':runs,'어려운_사례':[{'unique_id':r['unique_id'],'bbox_id':r['bbox_id'],'이유':r['segmentation_reason']} for r in features if not r['segmentation_reliable']][:30]}
    before=json.loads((OUT/'input_hashes.json').read_text(encoding='utf-8')); assert fingerprint()==before
    summary['입력_해시_검증']='전체 보호 입력 동일'; save_json(OUT/'summary.json',summary)
    (OUT/'README.md').write_text('''# 이물질 조건 분석\n\nGT bbox 내부 영상 특성을 이용한 이물질 영역 추정입니다. segmentation 정답은 없으며 segmentation_reliable은 알고리즘 품질 기준 통과 여부입니다. 실제 물체로 검증된 비율이 아닙니다.\n\nTrain 27개 PoC를 먼저 비교하고 local background difference에서 양의 어두운 잔차를 Otsu로 나눈 후 잔차 합이 가장 큰 connected component를 선택했습니다. 제품 구조를 완전히 배제하지 못합니다. 밝은 이물질은 이 방법의 한계입니다.\n\n신뢰도: 4픽셀 미만, bbox 65% 초과, 두 번째 성분 잔차합/첫 번째 >0.5, 경계 픽셀 수 >max(2,sqrt(area)), 배경 12픽셀 미만, 평균 명암차 또는 국소잔차 <5, 복원 픽셀 >max(1,area*0.1)은 실패. 이는 크기/위치 조건 분류 기준이 아니라 추정 안정성 휴리스틱입니다. 실패 행의 object 특성은 공란입니다. masks는 실패 포함 진단용 추정 마스크입니다.\n\n좌표: 원 YOLO 부동소수점 bbox를 ceil로 raster 변환, 상한 exclusive. object centroid는 픽셀 중심(+0.5). edge_distance=min(cx,cy,W-cx,H-cy). normalized_edge_distance=min(cx/W,1-cx/W,cy/H,1-cy/H). 종횡비=object 외접 폭/높이. 원형도=4π contour_area/perimeter²; solidity=contour_area/convex_hull_area; extent=pixel_area/(object width*height); major/minor=픽셀 좌표 공분산 고유값 비의 제곱근. 작은 물체에서 contour와 pixel 면적은 다릅니다.\n\n배경 ring: object dilation 반경 max(3,min(12,round(sqrt(area)))), object 제외, 다른 GT bbox 및 TELEA mask 제외, 이미지 경계 clipping. 같은 GT bbox의 나머지 영역은 ring에 포함될 수 있습니다. 정상 배경이라는 보장은 없으므로 제품 구조 영향이 남습니다. contrast_normalized=abs(object_mean-background_mean)/255; relative_intensity_difference=(object_mean-background_mean)/max(background_mean,1); Michelson=abs difference/max(sum of means,1); z_score=(object_mean-background_mean)/max(background_std,1). 영상상 상대 밀도/방사선 감쇠 특성 proxy이며 물리 밀도(g/cm³)가 아닙니다.\n\n분포는 신뢰도 기준 통과 행에 한정됩니다. train_threshold_candidates.json은 Train 분위수 후보로서 확정 분류가 아닙니다. Val/Test로 조정하지 않았습니다. 최종 모델 지정 전 예측 상태는 미정이며 FN으로 처리하지 않았습니다. condition_performance.csv는 헤더만 있습니다.\n\n재현: .venv\\Scripts\\python.exe -B -X utf8 scripts\\analyze_foreign_object_conditions.py --stage poc (새 출력에만 실행); PoC 검토 기록 후 --stage full. 모듈 함수는 notebook에서 import 가능. 기존 출력/데이터/라벨/split/학습 결과는 수정하지 않습니다. 입력 SHA256 전후 동일 검증.\n''',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['분포','모델_후보','어려운_사례']},ensure_ascii=False))

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--stage',choices=['poc','full','performance','review'],required=True); parser.add_argument('--model',type=Path); parser.add_argument('--reuse-predictions',action='store_true'); args=parser.parse_args()
    if args.stage=='review':make_review_artifacts(); return
    if args.stage=='performance':
        if args.model is None:parser.error('--model 명시 필요')
        link_performance(args.model,args.reuse_predictions); return
    records=load_records()
    if args.stage=='poc':make_poc(records)
    else:full_analysis(records)

def box_iou(a,b):
    a=np.asarray(a,float).reshape(-1,4); b=np.asarray(b,float).reshape(-1,4)
    lo=np.maximum(a[:,None,:2],b[None,:,:2]); hi=np.minimum(a[:,None,2:],b[None,:,2:]); inter=np.maximum(hi-lo,0).prod(2)
    return inter/np.maximum(((a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-inter),1e-9)

def match_predictions(gt,preds,conf=.25,iou_threshold=.5):
    preds=[p for p in preds if p['confidence']>=conf and p['class_id']==0]
    overlaps=box_iou(gt,[p['box'] for p in preds]); matches={}; used=set()
    pairs=sorted([(float(overlaps[i,j]),i,j) for i in range(len(gt)) for j in range(len(preds)) if overlaps[i,j]>=iou_threshold],reverse=True)
    for overlap,i,j in pairs:
        if i not in matches and j not in used:matches[i]=(preds[j]['confidence'],overlap); used.add(j)
    return matches,len(preds)-len(used)

def link_performance(model_path,reuse_predictions=False):
    from ultralytics import YOLO
    assert model_path.is_file()
    features=read_csv(OUT/'features.csv'); manifests=read_csv(DATA/'manifest.csv'); byid={}
    for r in features:byid.setdefault(r['unique_id'],[]).append(r)
    protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in model_path.parent.parent.rglob('*') if p.is_file()}
    cache={}; summaries=[]
    if reuse_predictions:
        meta=json.loads((OUT/'prediction_provenance.json').read_text(encoding='utf-8'))
        assert meta['model_sha256']==hashlib.sha256(model_path.read_bytes()).hexdigest()
        cache=json.loads((OUT/'predictions.json').read_text(encoding='utf-8'))
    else:model=YOLO(str(model_path))
    for split in ([] if reuse_predictions else ['val','test']):
        rows=[r for r in manifests if r['split']==split]; paths=[str(DATA/'images'/split/r['restored_filename']) for r in rows]
        results=model.predict(source=paths,stream=True,imgsz=1280,conf=.001,iou=.7,max_det=300,device=0,batch=4,save=False,verbose=False,augment=False)
        for r,result in zip(rows,results):
            boxes=result.boxes; cache[r['unique_id']]=[dict(box=b,confidence=float(c),class_id=int(k)) for b,c,k in zip(boxes.xyxy.cpu().numpy().tolist(),boxes.conf.cpu().numpy().tolist(),boxes.cls.cpu().numpy().tolist())]
        print(f'{split} 추론 {len(rows)}장 완료',flush=True)
    save_json(OUT/'predictions.json',cache)
    save_json(OUT/'prediction_provenance.json',{'model':str(model_path),'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),'imgsz':1280,'conf':.001,'nms_iou':.7,'max_det':300,'augment':False,'image_count':len(cache)})
    for conf in [.1,.25,.5]:
        for split in ['val','test']:
            tp=fn=fp=0
            for r in [r for r in manifests if r['split']==split]:
                h,w=int(byid[r['unique_id']][0]['image_height']),int(byid[r['unique_id']][0]['image_width'])
                gt=[]
                for line in (DATA/'labels'/split/(Path(r['filename']).stem+'.txt')).read_text().splitlines():
                    _,x,y,bw,bh=map(float,line.split()); gt.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
                matched,extra=match_predictions(gt,cache[r['unique_id']],conf); tp+=len(matched); fn+=len(gt)-len(matched); fp+=extra
                if conf==.25:
                    for row in byid[r['unique_id']]:
                        m=matched.get(int(row['bbox_id'])); row.update(prediction_status='TP' if m else 'FN',prediction_confidence=m[0] if m else '',prediction_iou=m[1] if m else '')
            summaries.append(dict(split=split,confidence_threshold=conf,iou_threshold=.5,GT_count=tp+fn,TP=tp,FN=fn,FP=fp,recall=tp/(tp+fn)))
    write_csv(OUT/'features.csv',features); write_csv(OUT/'confidence_sensitivity.csv',summaries)
    thresholds=json.loads((OUT/'train_threshold_candidates.json').read_text(encoding='utf-8')); performance=[]
    for split in ['val','test']:
        allrows=[r for r in features if r['split']==split]; groups=[('전체','전체',allrows),('추정 안정성','불안정',[r for r in allrows if r['segmentation_reliable']=='False'])]
        for key in ['foreign_object_area','normalized_edge_distance','aspect_ratio','circularity','object_mean_intensity','contrast_abs']:
            cuts=sorted(set([thresholds[key]['q33'],thresholds[key]['q67']]))
            if key=='normalized_edge_distance':cuts=[thresholds[key]['q25']]
            for i in range(len(cuts)+1):
                lo=cuts[i-1] if i else -float('inf'); hi=cuts[i] if i<len(cuts) else float('inf')
                selected=[r for r in allrows if r['segmentation_reliable']=='True' and r[key]!='' and lo<float(r[key])<=hi]
                groups.append((key,f'Train 분위수 후보 구간{i+1}: ({lo}, {hi}]',selected))
        for key,label,selected in groups:
            matched=[r for r in selected if r['prediction_status']=='TP']; n=len(selected)
            performance.append(dict(model=str(model_path),split=split,condition=key,group=label,GT_count=n,TP=len(matched),FN=n-len(matched),recall=len(matched)/n if n else '',mean_confidence=float(np.mean([float(r['prediction_confidence']) for r in matched])) if matched else '',mean_iou=float(np.mean([float(r['prediction_iou']) for r in matched])) if matched else ''))
    write_csv(OUT/'condition_performance.csv',performance)
    summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8')); summary.update(성능_분석='사용자 지정 1280 No-Aug Baseline; Train 분위수 후보별 탐색 분석',모델=str(model_path),모델_SHA256=hashlib.sha256(model_path.read_bytes()).hexdigest(),전체_성능=summaries,조건_성능=performance)
    assert fingerprint()==json.loads((OUT/'input_hashes.json').read_text(encoding='utf-8')); save_json(OUT/'summary.json',summary)
    assert {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in model_path.parent.parent.rglob('*') if p.is_file()}==protected
    with (OUT/'README.md').open('a',encoding='utf-8') as f:f.write('\n\n## 지정 모델 성능 분석 완료\n위의 모델 미지정 설명은 특성 생성 단계 상태입니다. 현재 사용자 지정 1280 No-Aug best.pt로 Val/Test를 추론했습니다. 추론 imgsz1280, NMS IoU0.7, max_det300, augment=False, 저장 confidence0.001; 기본 TP/FN confidence0.25 및 GT IoU≥0.5, class0, IoU 내림차순 일대일 매칭. 매칭에는 ceil raster 대신 원 라벨 부동소수점 좌표를 사용합니다. confidence0.1/0.25/0.5 민감도도 저장. confidence와 IoU 평균은 TP만 대상으로 하며 FN 값은 공란입니다.\n조건 성능은 Train 신뢰도 기준 통과 표본의 q33/q67, Edge는 q25를 고정한 탐색적 후보 구간이며 확정 기준이 아닙니다. Val/Test 성능으로 경계를 조정하지 않았습니다. 불안정 분할은 전체 성능에는 포함되지만 object 조건 분석에서는 제외합니다. 선택 편향과 적은 표본 때문에 조건별 FN 차이를 인과적으로 단정할 수 없습니다. 재현: --stage performance --model 지정한_best.pt\n')
    print(json.dumps(summaries,ensure_ascii=False),flush=True)

def make_review_artifacts():
    features=read_csv(OUT/'features.csv'); records=load_records(); index={(r['unique_id'],str(r['bbox_id'])):r for r in records}
    # Save method-specific final masks as well as complete method panels for all PoC cases.
    chosen=read_csv(OUT/'poc/selected_bbox.csv'); panels=[]
    for i,s in enumerate(chosen):
        r=index[(s['unique_id'],s['bbox_id'])]; gray=cv2.imread(str(DATA/'images'/r['split']/r['restored_filename']),0); methods,_,_=extract_candidates(gray,box_of(r)); im=panel_row(r,gray,methods)
        im.save(OUT/'poc'/f"최종_{i+1:02d}.png"); panels.append(im)
        for method,m in methods.items():Image.fromarray(m.astype('uint8')*255).save(OUT/'poc'/f"{i+1:02d}_{method}_mask.png")
    for start in range(0,len(panels),3):
        canvas=Image.new('RGB',(1800,935),'white')
        for j,p in enumerate(panels[start:start+3]):canvas.paste(p,(600*j,0))
        canvas.save(OUT/'poc'/f'최종_방법비교_{start//3+1:02d}.png')
    good=sorted([r for r in features if r['segmentation_reliable']=='True'],key=lambda r:float(r['foreign_object_area']))
    groups={'신뢰도_통과_대표':[good[int(i)] for i in np.linspace(0,len(good)-1,24)],'FN_전체':[r for r in features if r['prediction_status']=='FN'],'불안정_대표':[r for r in features if r['segmentation_reliable']=='False'][:24]}
    for name,rows in groups.items():
        for start in range(0,len(rows),6):
            sheet=Image.new('RGB',(1200,6*210),'white'); d=ImageDraw.Draw(sheet)
            for j,f in enumerate(rows[start:start+6]):
                r=index[(f['unique_id'],f['bbox_id'])]; box=box_of(r); x1,y1,x2,y2=box
                gray=cv2.imread(str(DATA/'images'/r['split']/r['restored_filename']),0)
                original=np.array(Image.open(ROOT/'outputs/clean_512/images'/r['filename']).convert('RGB'))
                margin=max(5,(x2-x1)//2,(y2-y1)//2); lo=max(0,x1-margin),max(0,y1-margin); hi=min(gray.shape[1],x2+margin),min(gray.shape[0],y2+margin)
                mask=cv2.imread(str(OUT/'masks'/f"{r['unique_id']}_bbox{r['bbox_id']}_estimated.png"),0)>0
                base=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB); overlay=base.copy(); overlay[mask]=(overlay[mask]*.5+np.array([255,0,0])*.5).astype('uint8')
                cv2.rectangle(overlay,(x1,y1),(x2-1,y2-1),(0,255,255),1)
                text=f"{f['unique_id']} bbox{f['bbox_id']} {f['split']} {f['prediction_status']} / {'안정' if f['segmentation_reliable']=='True' else f['segmentation_reason']}"
                d.text((4,j*210),text,font=FONT,fill='black')
                for k,(im,title) in enumerate([(original,'표식 포함 원본'),(base,'TELEA 입력'),(mask.astype('uint8')*255,'추정 mask'),(overlay,'추정 영역 + GT')]):
                    crop=im[lo[1]:hi[1],lo[0]:hi[0]]; resized=Image.fromarray(crop).convert('RGB'); resized.thumbnail((290,165),Image.Resampling.NEAREST)
                    # Upscale tiny crops with nearest-neighbor while preserving aspect ratio.
                    scale=min(290/crop.shape[1],165/crop.shape[0]); resized=Image.fromarray(crop).convert('RGB').resize((round(crop.shape[1]*scale),round(crop.shape[0]*scale)),Image.Resampling.NEAREST)
                    sheet.paste(resized,(k*300,j*210+40)); d.text((k*300+4,j*210+20),title,font=FONT,fill='black')
            sheet.save(OUT/'visualizations'/f'{name}_{start//6+1:02d}.png')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Malgun Gothic'
    perf=read_csv(OUT/'condition_performance.csv'); fig,axes=plt.subplots(2,3,figsize=(15,8))
    keys=['foreign_object_area','normalized_edge_distance','aspect_ratio','circularity','object_mean_intensity','contrast_abs']; labels=['추정 면적','정규화 가장자리 거리','형상 종횡비','원형도','상대 명암','배경 대비']
    for ax,key,label in zip(axes.flat,keys,labels):
        for split in ['val','test']:
            rows=[r for r in perf if r['split']==split and r['condition']==key]; x=np.arange(len(rows)); ax.plot(x,[float(r['recall']) if r['recall'] else np.nan for r in rows],marker='o',label=split)
            for i,r in enumerate(rows):
                if r['recall']:ax.annotate(f"{r['TP']}/{r['GT_count']}",(i,float(r['recall'])),xytext=(0,6 if split=='val' else -14),textcoords='offset points',ha='center',fontsize=9)
        ax.set_title(label); ax.set_ylim(.7,1.05); ax.set_xticks(x,[f'후보{i+1}' for i in x]); ax.set_ylabel('Recall'); ax.legend()
    fig.suptitle('Train 분위수 후보별 탐색 분석 / confidence 0.25, IoU 0.5'); fig.tight_layout(); fig.savefig(OUT/'figures/조건별_Recall.png',dpi=140); plt.close(fig)
    print('최종 비교/FN 시각화 생성 완료')
    write_final_report()

def write_final_report():
    summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8')); rows=read_csv(OUT/'train_distribution.csv'); train={r['feature']:r for r in rows if r['scope']=='train'}
    def dist(key):
        r=train[key]; return f"중앙값 {float(r['median']):.6g}, Q25–Q75 {float(r['q25']):.6g}–{float(r['q75']):.6g}, 범위 {float(r['minimum']):.6g}–{float(r['maximum']):.6g}"
    body=f'''# 이물질 조건 분석 최종 보고\n\n1. PoC: Train 27 bbox, 면적/명암/위치 proxy의 분위수에서 대표 선정.\n2. 방법: Otsu, adaptive, local background difference, connected component, grayscale intensity+contrast.\n3. 선정: 국소 배경 차이 + bbox 경계 제외 연결 성분. 제품 경계 과추출이 줄고 어두운 중심 성분을 더 타당하게 추정. 경계에 붙은 실제 물체는 미추출될 수 있습니다.\n4. 추정 안정성 기준 통과: {summary['신뢰도_기준_통과']}/{summary['전체_bbox']} ({summary['비율']*100:.2f}%). 전수 segmentation 정답 검증 비율이 아닙니다. 나머지 object 특성은 공란.\n5. 어려운 사례: U0385 bbox1 매우 작은 crop, U0223 bbox2 제품 구조와 연결, U0023 bbox0 texture/극소 성분, U0072 bbox1 주변 성분 선택 위험. 추가 육안검토 U0328 bbox1은 제품 구조 혼입으로 수동 불안정 처리. 다른 입력 데이터/라벨을 제외한 것이 아닙니다.\n6. Train 추정 면적(pixel): {dist('foreign_object_area')}. object_area_ratio: {dist('object_area_ratio')}.\n7. Train 위치(px): {dist('edge_distance')}. 정규화 거리: {dist('normalized_edge_distance')}.\n8. Train 형상: 종횡비 {dist('aspect_ratio')}; 원형도 {dist('circularity')}; solidity {dist('solidity')}; extent {dist('extent')}. 퇴화한 contour의 원형도/solidity는 미정의 공란입니다.\n9. Train 상대 intensity(0–255): {dist('object_mean_intensity')}. 영상상 상대 밀도/방사선 감쇠 특성 proxy이며 물리적 밀도가 아닙니다.\n10. Train 대비: {dist('contrast_abs')}; 정규화 대비 {dist('contrast_normalized')}.\n11. 크기 후보: Train q33=4, q67=5 px → ≤4 / (4,5] / >5. 작은 정수 면적에 몰려 균등 3분할이 되지 않습니다. 확정 Small/Medium/Large 기준으로 권장하기 어렵습니다.\n12. Edge 후보: Train 정규화 거리 q25=0.364382 → ≤q25 / >q25. 상대적으로 가장자리에 가까운 하위 분위수라는 의미이며, 절대 가장자리 영역이라는 뜻은 아닙니다.\n13. 대비 후보: Train q33≈27.02, q67≈34.15 → ≤27.02 / (27.02,34.15] / >34.15. 정확한 경계는 train_threshold_candidates.json. 모두 후보이며 Val/Test로 경계를 조정하지 않았습니다.\n14. 사용자 지정 1280 No-Aug Baseline, confidence≥0.25 및 IoU≥0.5. Val TP170/FN5/Recall97.14%; Test TP148/FN12/Recall92.50%. Test 크기 후보별 FN/GT=5/58,2/13,2/26; 단조 크기 효과 확인 안 됨. Edge 후보=7/51 vs2/46(Recall86.27% vs95.65%), 차이는 관찰되나 표본/제품 조건/추정 선택 편향 때문에 일반화 불가. 대비 후보=3/33,3/32,3/32로 저대비 FN 증가가 확인되지 않음. 평균 confidence/IoU는 TP만 대상으로 condition_performance.csv에 기록.\n15. 먼저 볼 파일: poc/최종_방법비교_01.png → visualizations/신뢰도_통과_대표_01.png → visualizations/FN_전체_01.png → figures/조건별_Recall.png.\n\nGT bbox 내부 영상 특성을 이용한 이물질 영역 추정입니다. 대부분 매우 작은 어두운 중심부가 선택되므로 흐릿한 주변부를 포함한 전체 이물질 면적을 과소추정할 수 있습니다. segmentation GT 없이 4픽셀을 실제 크기 정답으로 해석하면 안 됩니다. 전체650 내외 통과 표본에 선택 편향이 있고 저대비/작은 물체가 불안정 표본에 빠질 수 있습니다. 신뢰도 통과 대표24개와 FN17개를 별도 시각화했습니다.\n\n데이터/라벨/split SHA256 전후 동일, 학습/재분할/Fake Restoration 없음. 분석 코드는 scripts/analyze_foreign_object_conditions.py.\n'''
    tc=summary['기준_후보']
    body=body.replace('q25=0.364382',f"q25={tc['normalized_edge_distance']['q25']:.6f}")
    body=body.replace('q67≈34.15',f"q67≈{tc['contrast_abs']['q67']:.2f}")
    body=body.replace('(27.02,34.15] / >34.15',f"(27.02,{tc['contrast_abs']['q67']:.2f}] / >{tc['contrast_abs']['q67']:.2f}")
    body=body.replace('전체650 내외 통과 표본',f"전체 {summary['신뢰도_기준_통과']}개 통과 표본")
    (OUT/'최종_보고.md').write_text(body,encoding='utf-8')
    p=OUT/'README.md'; text=p.read_text(encoding='utf-8')
    text=text.replace('최종 모델 지정 전 예측 상태는 미정이며 FN으로 처리하지 않았습니다. condition_performance.csv는 헤더만 있습니다.','사용자 지정 최종 모델로 Val/Test TP/FN을 계산했으며 condition_performance.csv에 조건별 성능을 저장했습니다.')
    text=text.replace('위의 모델 미지정 설명은 특성 생성 단계 상태입니다. ','')
    text+='\n추가 품질 기준: 추정 성분 중심이 bbox 중심에서 각 축 길이의 35%를 초과해 벗어나면 제품 구조 의심으로 실패 처리합니다. 밝은 성분/실제 bbox 경계 이물질은 미추출 가능. manual_rejections.json은 육안으로 확인한 오추출만 불안정 처리하며 입력 데이터 제외와 무관합니다. 최종_보고.md에 15항목 결과를 정리했습니다. 비교_*.png 및 新版比較_*.png는 방법 개선 이전의 검토 이력이며 최종_방법비교_*.png가 선정 방법 기준입니다. 원본 crop 열은 TELEA 입력 grayscale입니다. 최종 대표/FN 시트에는 표식 포함 clean_512 원본도 함께 표시합니다.\n'
    p.write_text(text,encoding='utf-8')

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8'); main()
