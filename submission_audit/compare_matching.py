from pathlib import Path
import sys,json,csv,math,hashlib
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'submission_audit'
sys.path.insert(0,str(ROOT/'.cache/shortcut_runtime'))
import torch
from ultralytics.utils.metrics import ConfusionMatrix,box_iou

def iou(a,b):
    inter=math.prod(max(0,min(a[k+2],b[k+2])-max(a[k],b[k])) for k in [0,1])
    return inter/(math.prod(a[k+2]-a[k] for k in [0,1])+math.prod(b[k+2]-b[k] for k in [0,1])-inter)

def own(gt,pred,t=.25,cut=.5):
    eligible=[(i,p) for i,p in enumerate(pred) if p['class_id']==0 and p['confidence']>=t]
    edges=sorted([(iou(g,p['box']),i,k) for i,g in enumerate(gt) for k,p in eligible],reverse=True)
    usedg=set();usedp=set(); matches=[]
    for score,g,p in edges:
        if score<cut:break
        if g not in usedg and p not in usedp:
            usedg.add(g);usedp.add(p);matches.append([g,p,score])
    return dict(TP=len(matches),FP=len(eligible)-len(matches),FN=len(gt)-len(matches),matches=matches,
        fp=[i for i,p in eligible if i not in usedp],fn=[i for i in range(len(gt)) if i not in usedg])

def cm(gt,pred,t=.25,cut=.45):
    det=dict(bboxes=torch.tensor([p['box'] for p in pred],dtype=torch.float32).reshape(-1,4),
        conf=torch.tensor([p['confidence'] for p in pred]),cls=torch.tensor([p['class_id'] for p in pred]))
    batch=dict(bboxes=torch.tensor(gt,dtype=torch.float32).reshape(-1,4),cls=torch.zeros(len(gt)))
    matrix=ConfusionMatrix(names={0:'defect'});matrix.process_batch(det,batch,conf=t,iou_thres=cut)
    eligible=[i for i,p in enumerate(pred) if p['confidence']>t]
    overlaps=box_iou(batch['bboxes'],det['bboxes'][det['conf']>t]).numpy()
    g,p=np.where(overlaps>cut)
    matches=np.column_stack([g,p,overlaps[g,p]])
    if len(matches)>1:
        matches=matches[matches[:,2].argsort()[::-1]]
        matches=matches[np.unique(matches[:,1],return_index=True)[1]]
        matches=matches[matches[:,2].argsort()[::-1]]
        matches=matches[np.unique(matches[:,0],return_index=True)[1]]
    ms=[[int(g),eligible[int(p)],float(v)] for g,p,v in matches]
    return dict(TP=int(matrix.matrix[0,0]),FP=int(matrix.matrix[0,1]),FN=int(matrix.matrix[1,0]),matches=ms,
        fp=[p for p in eligible if p not in {m[1] for m in ms}],fn=[i for i in range(len(gt)) if i not in {m[0] for m in ms}])

def get_data(split='test'):
    with (ROOT/'outputs/yolo_512_telea/manifest.csv').open(encoding='utf-8-sig') as f:
        rows=[r for r in csv.DictReader(f) if r['split']==split]
    cache=json.loads((ROOT/'outputs/foreign_object_condition_analysis/predictions.json').read_text(encoding='utf-8'))
    records={}
    for r in rows:
        with Image.open(ROOT/'outputs/yolo_512_telea/images'/split/r['restored_filename']) as im:w,h=im.size
        gt=[]
        for line in (ROOT/'outputs/yolo_512_telea/labels'/split/(Path(r['filename']).stem+'.txt')).read_text().splitlines():
            c,x,y,bw,bh=map(float,line.split());gt.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
        records[r['unique_id']]=dict(row=r,gt=gt,cache=cache[r['unique_id']])
    return records

if __name__=='__main__':
    records=get_data()
    replay=json.loads((OUT/'validator_replay.json').read_text(encoding='utf-8'))
    vals={r['unique_id']:r for r in replay['records']}
    totals={};differences=[];boundary=[];many=[]
    for uid,r in records.items():
        gt=r['gt'];pred=r['cache'];v=vals[uid]
        modes=dict(cache_own_50=own(gt,pred),cache_cm_50=cm(gt,pred,cut=.5),cache_cm_45=cm(gt,pred),
            val_own_50=own(gt,v['predictions']),val_cm_45=cm(v['gt_input'],[dict(p,box=p['input_box']) for p in v['predictions']]))
        for mode,result in modes.items():
            totals.setdefault(mode,dict(TP=0,FP=0,FN=0))
            for k in ['TP','FP','FN']:totals[mode][k]+=result[k]
        if any([modes[m][k]!=modes['cache_own_50'][k] for m in modes for k in ['TP','FP','FN']]):
            differences.append(dict(unique_id=uid,filename=r['row']['restored_filename'],gt=gt,cache_predictions=pred,
                val_predictions=v['predictions'],val_gt_native=v['gt'],val_input_shape=v['input_shape'],modes=modes))
        for pi,p in enumerate(pred):
            if p['confidence'] in [.2,.25] or any(iou(g,p['box']) in [.45,.5] for g in gt):
                boundary.append([uid,pi,p['confidence']])
        pp=[p for p in pred if p['confidence']>=.25]
        graph=[[iou(g,p['box'])>=.5 for p in pp] for g in gt]
        if any(sum(x)>1 for x in graph) or (pp and any(sum(x)>1 for x in zip(*graph))):many.append(uid)
    data=dict(totals=totals,differences=differences,exact_boundaries=boundary,many_to_one_images=many,
        cache_classes=sorted({p['class_id'] for r in records.values() for p in r['cache']}),
        val_matrix=replay['matrix'],gt_max_abs_delta=max(abs(a-b) for uid,r in records.items() for g,v in zip(r['gt'],vals[uid]['gt']) for a,b in zip(g,v)))
    (OUT/'matching_comparison.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in data.items() if k!='differences'},ensure_ascii=False,indent=2))
    for d in differences:
        print(d['unique_id'],d['val_input_shape'],{m: [r[k] for k in ['TP','FP','FN']] for m,r in d['modes'].items()})
        for pi,p in enumerate(d['cache_predictions']):
            if p['confidence']>=.25:print(' CACHE',pi,p['confidence'],p['box'],'IoUs',[round(iou(g,p['box']),9) for g in d['gt']])
        for pi,p in enumerate(d['val_predictions']):
            if p['confidence']>=.25:print(' VAL',pi,p['confidence'],p['box'],'IoUs',[round(iou(g,p['box']),9) for g in d['gt']])
