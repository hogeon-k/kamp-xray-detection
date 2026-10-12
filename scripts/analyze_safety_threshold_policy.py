"""Select confidence and referral policy on Val; evaluate the locked policy once on Test."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from evaluate_edge_reinspection import ROOT,DATA,OLD,MODEL,CONF,MATCH_IOU,csvread,csvwrite,dump,sha,product_mask,label_boxes,matches,iou
from analyze_fn_root_cause import raster
OUT=ROOT/'outputs/safety_threshold_policy'
FLOOR=.001
GRID=np.unique(np.r_[.001,.005,np.arange(.01,.901,.01),CONF])
OPERATIONAL_LOW_GRID=(.05,.10,.15,.20,CONF)
ADDED_FP_BUDGET_PER_IMAGE=.10
REVIEW_IMAGE_BUDGET=.20


def match_at(gt,preds,threshold):
    """Reuse baseline IoU pairing; inclusion flag permits changing only confidence cutoff."""
    eligible=[p for p in preds if p['class_id']==0 and p['confidence']>=threshold]
    flagged=[dict(p,confidence=max(CONF,p['confidence']),source='global') for p in eligible]
    found,fp,_=matches(gt,flagged)
    used={m['prediction_index'] for m in found.values()}
    for m in found.values():m['confidence']=eligible[m['prediction_index']]['confidence']
    return found,fp,eligible,used


def load_records(split,cache):
    records=[]
    for row in [r for r in csvread(DATA/'manifest.csv') if r['split']==split]:
        gray=cv2.imread(str(DATA/'images'/split/row['restored_filename']),0)
        records.append(dict(uid=row['unique_id'],row=row,gray=gray,gt=label_boxes(row,gray.shape),preds=cache[row['unique_id']]))
    return records


def metrics(records,threshold):
    tp=fn=fp=recovered=lost=0
    for rec in records:
        bm,_,_,_=match_at(rec['gt'],rec['preds'],CONF)
        found,extra,_,_=match_at(rec['gt'],rec['preds'],threshold)
        tp+=len(found);fn+=len(rec['gt'])-len(found);fp+=extra
        recovered+=len(set(found)-set(bm));lost+=len(set(bm)-set(found))
    precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn)
    return dict(threshold=float(threshold),TP=tp,FN=fn,FP=fp,precision=precision,recall=recall,
                F1=2*precision*recall/(precision+recall) if precision+recall else 0,
                FN_Recovery=recovered,Lost_TP=lost,FP_per_image=fp/len(records))


def routing(records,low,high):
    auto=review=band_images=band_detections=review_detections=pass_images=0;states={}
    for rec in records:
        strong=[p for p in rec['preds'] if p['class_id']==0 and p['confidence']>=high]
        band=[p for p in rec['preds'] if p['class_id']==0 and low<=p['confidence']<high]
        band_images+=bool(band);band_detections+=len(band)
        if strong:state='automatic_NG';auto+=1
        elif band:state='reinspection';review+=1;review_detections+=len(band)
        else:state='normal_candidate';pass_images+=1
        states[rec['uid']]=state
    return dict(images=len(records),automatic_NG_images=auto,reinspection_images=review,
                reinspection_rate=review/len(records),reinspection_detections=review_detections,
                band_images=band_images,band_detections=band_detections,normal_candidate_images=pass_images),states


def fn_diagnostics(records):
    result=[]
    for rec in records:
        bm,_,_,_=match_at(rec['gt'],rec['preds'],CONF)
        for i,g in enumerate(rec['gt']):
            if i in bm:continue
            raw=[p for p in rec['preds'] if p['class_id']==0 and p['confidence']>=FLOOR]
            overlap=[(iou(g['box'],p['box']),p) for p in raw]
            qualifying=[(score,p) for score,p in overlap if score>=MATCH_IOU]
            if qualifying:
                score,p=max(qualifying,key=lambda z:z[1]['confidence']);kind='Type1_low_confidence'
                reason='confidence_below_baseline' if p['confidence']<CONF else 'one_to_one_matching_competition'
            else:
                score,p=max(overlap,key=lambda z:z[0],default=(0,None));kind='Type2_no_sufficient_overlap'
                reason='no_IoU50_prediction_in_floor_cache'
            box=p['box'] if p else None
            result.append(dict(unique_id=rec['uid'],bbox_id=i,type=kind,best_overlapping_confidence=p['confidence'] if p else None,
                               best_iou=score,prediction_class=p['class_id'] if p else None,
                               prediction_width=box[2]-box[0] if box else None,prediction_height=box[3]-box[1] if box else None,
                               reason=reason,gt_box=g['box'],prediction_box=box))
    return result


def proposal_features(gray,mask,box):
    x1,y1,x2,y2=raster(box,gray.shape);h,w=gray.shape
    if x2<=x1 or y2<=y1:return None
    bw,bh=box[2]-box[0],box[3]-box[1];pad=int(np.clip(round(max(bw,bh)*.5),5,15))
    a,b,c,d=max(0,x1-pad),max(0,y1-pad),min(w,x2+pad),min(h,y2+pad)
    ring=mask[b:d,a:c].copy();ring[y1-b:y2-b,x1-a:x2-a]=0
    vals=gray[b:d,a:c][ring>0]
    if len(vals)<12:return None
    mean=float(gray[y1:y2,x1:x2].mean());sur=float(vals.mean())
    return dict(area=bw*bh,normalized_contrast=abs(mean-sur)/(sur+1e-6))


def train_cutoffs():
    features=[]
    for row in [r for r in csvread(DATA/'manifest.csv') if r['split']=='train']:
        gray=cv2.imread(str(DATA/'images/train'/row['restored_filename']),0);mask,quality=product_mask(gray)
        if not quality['valid']:continue
        for g in label_boxes(row,gray.shape):
            f=proposal_features(gray,mask,g['box'])
            if f:features.append(f)
    return dict(small_area_q25=float(np.quantile([f['area'] for f in features],.25)),
                low_contrast_q25=float(np.quantile([f['normalized_contrast'] for f in features],.25)),train_GT_with_features=len(features))


def vulnerability(records,low,high,cuts):
    predictions=[]
    for rec in records:
        found,_,eligible,used=match_at(rec['gt'],rec['preds'],low)
        mask,quality=product_mask(rec['gray'])
        if not quality['valid']:continue
        for i,p in enumerate(eligible):
            if p['confidence']>=high:continue
            f=proposal_features(rec['gray'],mask,p['box'])
            if not f:continue
            predictions.append(dict(unique_id=rec['uid'],confidence=p['confidence'],status='TP' if i in used else 'FP',
                                    small=f['area']<=cuts['small_area_q25'],low_contrast=f['normalized_contrast']<=cuts['low_contrast_q25'],**f))
    groups=[]
    for name,predicate in [('all_middle',lambda r:True),('small',lambda r:r['small']),('low_contrast',lambda r:r['low_contrast']),
                           ('small_and_low_contrast',lambda r:r['small'] and r['low_contrast'])]:
        selected=[r for r in predictions if predicate(r)];tp=sum(r['status']=='TP' for r in selected)
        groups.append(dict(group=name,detections=len(selected),TP=tp,FP=len(selected)-tp,TP_fraction=tp/len(selected) if selected else None))
    return groups,predictions


def contact(records,diag,low,high,path,policy=False):
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16);byid={r['uid']:r for r in records}
    items=diag[:12] if not policy else [dict(unique_id=r['uid'],bbox_id=None) for r in records if routing([r],low,high)[1][r['uid']]=='reinspection'][:12]
    if policy and not items:
        ordered=sorted(records,key=lambda r:routing([r],low,high)[1][r['uid']]!='normal_candidate')
        items=[dict(unique_id=r['uid'],bbox_id=None) for r in ordered[:6]]
    sheet=Image.new('RGB',(1120,max(1,len(items))*270),'white');draw=ImageDraw.Draw(sheet)
    if not items:draw.text((10,10),'해당 사례 없음',font=font,fill='black')
    for j,item in enumerate(items):
        rec=byid[item['unique_id']];gray=rec['gray'];base=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
        marked=base.copy()
        for g in rec['gt']:
            a,b,c,d=map(round,g['box']);cv2.rectangle(marked,(a,b),(c,d),(255,0,0),1)
        for p in rec['preds']:
            if p['confidence']<FLOOR:continue
            if policy and p['confidence']<low:continue
            if not policy and item.get('gt_box') and iou(item['gt_box'],p['box'])<.1:continue
            a,b,c,d=map(round,p['box']);color=(0,255,0) if p['confidence']>=high else (255,170,0)
            cv2.rectangle(marked,(a,b),(c,d),color,1)
        state=routing([rec],low,high)[1][rec['uid']]
        draw.text((5,j*270+3),f"{rec['uid']} {state}"+(f" bbox{item['bbox_id']} {item['type']}" if not policy else ''),font=font,fill='black')
        panels=[base,marked]
        if not policy:
            x1,y1,x2,y2=item['gt_box'];a,b,c,d=max(0,int(x1)-20),max(0,int(y1)-20),min(gray.shape[1],int(x2)+21),min(gray.shape[0],int(y2)+21)
            panels.append(marked[b:d,a:c])
        else:panels.append(marked)
        for k,panel in enumerate(panels):
            scale=min(290/panel.shape[1],220/panel.shape[0]);im=Image.fromarray(panel).resize((round(panel.shape[1]*scale),round(panel.shape[0]*scale)),Image.Resampling.NEAREST)
            sheet.paste(im,(k*300,j*270+40))
        text=f'T_low={low:.2f}\nT_high={high:.2f}'
        if not policy:text+=f"\nconf={item['best_overlapping_confidence']:.4f}\nIoU={item['best_iou']:.3f}"
        draw.multiline_text((910,j*270+60),text,font=font,fill='black',spacing=6)
    sheet.save(path)


def plots(records,sweep,policies,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(figsize=(10,6));ax.plot([r['threshold'] for r in sweep],[r['recall'] for r in sweep],label='Recall');ax.plot([r['threshold'] for r in sweep],[r['precision'] for r in sweep],label='Precision')
    offsets={'Baseline':(5,-36),'Best_F1':(5,-52),'Recall_oriented':(0,12),'Two_threshold':(0,-24),'High_sensitivity':(10,12)}
    for p in policies:
        ax.scatter(p['T_low'],p['recall']);ax.annotate(p['policy'],(p['T_low'],p['recall']),xytext=offsets[p['policy']],textcoords='offset points',fontsize=8,arrowprops=dict(arrowstyle='-',lw=.5))
    ax.set_xlabel('Confidence threshold');ax.set_ylim(0,1.05);ax.legend();fig.tight_layout();fig.savefig(out/'val_threshold_curve.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6));ax.scatter([r['FP'] for r in sweep],[r['FN'] for r in sweep],s=15)
    for p in policies:ax.scatter(p['FP'],p['FN']);ax.annotate(p['policy'],(p['FP'],p['FN']),fontsize=8)
    ax.set_xlabel('FP');ax.set_ylabel('FN');fig.tight_layout();fig.savefig(out/'val_fn_fp_tradeoff.png',dpi=150);plt.close(fig)
    distributions={'TP':[],'FP':[]}
    for rec in records:
        _,_,eligible,used=match_at(rec['gt'],rec['preds'],FLOOR)
        for i,p in enumerate(eligible):distributions['TP' if i in used else 'FP'].append(p['confidence'])
    fig,axes=plt.subplots(1,3,figsize=(16,5));rng=np.random.default_rng(42)
    for i,(name,values) in enumerate(distributions.items()):
        v=np.array(values);axes[0].hist(v,bins=np.linspace(0,1,31),alpha=.5,label=f'{name} n={len(v)}')
        s=np.sort(v);axes[1].plot(s,np.arange(1,len(s)+1)/len(s),label=name)
        axes[2].scatter(i+rng.uniform(-.15,.15,len(v)),v,s=12,alpha=.4);axes[2].plot([i-.2,i+.2],[np.median(v)]*2,c='black')
    axes[0].set_title('Histogram');axes[0].legend();axes[1].set_title('ECDF');axes[1].legend();axes[2].set_xticks([0,1],['TP','FP']);axes[2].set_title('Strip / median');fig.tight_layout();fig.savefig(out/'val_confidence_distribution.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6))
    for low in OPERATIONAL_LOW_GRID:
        r,_=routing(records,low,CONF);recall=metrics(records,low)['recall']*100
        ax.scatter(r['reinspection_rate']*100,recall,c='purple',marker='x')
        ax.scatter(r['band_images']/len(records)*100,recall,c='teal')
        ax.annotate(f'T_low={low:.2f}',(r['band_images']/len(records)*100,recall),xytext=(4,5),textcoords='offset points',fontsize=8)
    ax.scatter([],[],c='purple',marker='x',label='실제 재검사: 자동 NG 이미지 제외')
    ax.scatter([],[],c='teal',label='Middle band가 있는 전체 이미지')
    ax.set_xlabel('이미지 비율 (%)');ax.set_ylabel('Potential bbox screening Recall (%)');ax.legend();fig.tight_layout();fig.savefig(out/'val_recall_reinspection_tradeoff.png',dpi=150);plt.close(fig)
    return {k:dict(n=len(v),median=float(np.median(v)),q25=float(np.quantile(v,.25)),q75=float(np.quantile(v,.75))) for k,v in distributions.items()}


def main(out):
    if out.exists():raise FileExistsError(f'기존 출력 보호: 새 --out 경로 필요 {out}')
    meta=json.loads((OLD/'prediction_provenance.json').read_text(encoding='utf-8'));assert meta['model_sha256']==sha(MODEL) and meta['imgsz']==1280 and meta['nms_iou']==.7
    inputs=[MODEL,OLD/'predictions.json',DATA/'manifest.csv',DATA/'dataset.yaml']+[p for folder in ('images','labels') for p in (DATA/folder).rglob('*') if p.is_file()]
    hashes={str(p):sha(p) for p in inputs};cache=json.loads((OLD/'predictions.json').read_text(encoding='utf-8'))
    val=load_records('val',cache);base=metrics(val,CONF);sweep=[]
    for t in GRID:
        r=metrics(val,float(round(t,6)));r.update(delta_Recall=r['recall']-base['recall'],delta_Precision=r['precision']-base['precision'],delta_FN=r['FN']-base['FN'],delta_FP=r['FP']-base['FP'])
        added=r['delta_FP'];r['FN_reduction_per_Added_FP']=(base['FN']-r['FN'])/added if added>0 else None;r['Recall_gain_per_Added_FP']=r['delta_Recall']/added if added>0 else None;sweep.append(r)
    f1=max(sweep,key=lambda r:(r['F1'],r['threshold']))
    fpbudget=math.floor(len(val)*ADDED_FP_BUDGET_PER_IMAGE)
    operational=[]
    for low in OPERATIONAL_LOW_GRID:
        m=metrics(val,low);route,_=routing(val,low,CONF)
        if m['FP']-base['FP']<=fpbudget and route['reinspection_rate']<=REVIEW_IMAGE_BUDGET and m['Lost_TP']==0:operational.append((m,route))
    recall,route=max(operational,key=lambda pair:(pair[0]['FN_Recovery'],-pair[0]['FP'],pair[0]['threshold']))
    sensitive=max(sweep,key=lambda r:(r['recall'],-r['FP'],r['threshold']))
    policies=[]
    for name,m,high in [('Baseline',base,CONF),('Best_F1',f1,f1['threshold']),('Recall_oriented',recall,recall['threshold']),('Two_threshold',recall,CONF),('High_sensitivity',sensitive,sensitive['threshold'])]:
        routing_metrics,_=routing(val,m['threshold'],high)
        policies.append(dict(policy=name,T_low=m['threshold'],T_high=high,**{k:v for k,v in m.items() if k!='threshold'},Added_FP=m['FP']-base['FP'],**routing_metrics))
    chosen=next(p for p in policies if p['policy']==('Two_threshold' if recall['FN_Recovery']>0 else 'Baseline'))
    targets=[]
    for target in (.95,.97,.98):
        candidates=[r for r in sweep if r['recall']>=target]
        if candidates:
            r=max(candidates,key=lambda z:z['threshold']);targets.append(dict(target=target,achievable=True,min_grid_threshold=min(q['threshold'] for q in candidates),max_threshold_meeting_target=r['threshold'],precision=r['precision'],recall=r['recall'],FN=r['FN'],FP=r['FP']))
        else:targets.append(dict(target=target,achievable=False))
    cuts=train_cutoffs();groups,proposalrows=vulnerability(val,chosen['T_low'],chosen['T_high'],cuts)
    valdiag=fn_diagnostics(val)
    out.mkdir(parents=True);(out/'visualizations').mkdir()
    csvwrite(out/'val_threshold_sweep.csv',sweep,list(sweep[0]));csvwrite(out/'val_policy_candidates.csv',policies,list(policies[0]));csvwrite(out/'val_recall_targets.csv',targets,list(targets[0]))
    csvwrite(out/'val_fn_low_conf_analysis.csv',valdiag,list(valdiag[0]));csvwrite(out/'val_middle_vulnerability.csv',groups,list(groups[0]));csvwrite(out/'val_middle_prediction_features.csv',proposalrows,list(proposalrows[0]) if proposalrows else ['unique_id','confidence','status','area','normalized_contrast'])
    lock=dict(selected=chosen,selection_split='val',small_low_contrast_train_cutoffs=cuts,review_image_budget=REVIEW_IMAGE_BUDGET,added_FP_budget=fpbudget,operational_low_grid=OPERATIONAL_LOW_GRID,auto_NG_precedence=True)
    dump(out/'locked_policy.json',lock);print(f"Val policy locked: {chosen['policy']} low={chosen['T_low']} high={chosen['T_high']}",flush=True)
    # Final policy only; no Test sweep or alternate policy selection.
    test=load_records('test',cache);testbase=metrics(test,CONF);testm=metrics(test,chosen['T_low']);testroute,states=routing(test,chosen['T_low'],chosen['T_high'])
    assert (testbase['TP'],testbase['FN'],testbase['FP'])==(148,12,8)
    results=[]
    for name,m,r in [('Baseline',testbase,routing(test,CONF,CONF)[0]),('Final_policy',testm,testroute)]:
        results.append(dict(policy=name,**m,FN_reduction=testbase['FN']-m['FN'],FP_increase=m['FP']-testbase['FP'],**r))
    csvwrite(out/'test_policy_result.csv',results,list(results[0]));testdiag=fn_diagnostics(test);csvwrite(out/'test_fn_low_conf_analysis.csv',testdiag,list(testdiag[0]))
    csvwrite(out/'test_image_routing.csv',[dict(unique_id=uid,state=state) for uid,state in states.items()],['unique_id','state'])
    dist=plots(val,sweep,policies,out/'visualizations');contact(test,testdiag,chosen['T_low'],chosen['T_high'],out/'visualizations/test_low_conf_fn_examples.png');contact(test,testdiag,chosen['T_low'],chosen['T_high'],out/'visualizations/test_policy_examples.png',True)
    assert hashes=={str(p):sha(p) for p in inputs}
    summary=dict(provenance=meta,baseline_confidence=CONF,GT_matching_IoU=MATCH_IOU,raw_confidence_floor=FLOOR,lock=lock,val_baseline=base,val_policies=policies,val_targets=targets,
                 val_confidence_distribution=dist,val_vulnerability=groups,train_cutoffs=cuts,
                 val_FN_types={t:sum(r['type']==t for r in valdiag) for t in ('Type1_low_confidence','Type2_no_sufficient_overlap')},
                 test_FN_types={t:sum(r['type']==t for r in testdiag) for t in ('Type1_low_confidence','Type2_no_sufficient_overlap')},test_results=results,test_policy_count=1,input_hashes_verified=True,
                 final_conclusion='Val에서 0.20–0.25 재검사 band는 잠재 FN 1개 감소/FP 1개 증가였으나 실제 추가 재검사 이미지는 없었고, Test에서 FN/FP 변화가 없어 운영 개선은 확인되지 않았습니다.')
    dump(out/'summary.json',summary);dump(out/'input_hashes.json',hashes)
    report(out,summary,sweep,policies,targets,chosen,testm,testbase,testroute,groups)
    print(json.dumps(dict(selected=chosen['policy'],T_low=chosen['T_low'],T_high=chosen['T_high'],test=results[1],FN_types=summary['test_FN_types']),ensure_ascii=False),flush=True)


def report(out,s,sweep,policies,targets,chosen,testm,testbase,testroute,groups):
    text=f'''# Safety confidence threshold policy

## 기준 및 분석 범위
Baseline confidence={CONF}, NMS IoU=0.7, GT matching IoU={MATCH_IOU}, imgsz=1280, augment=False, max_det300, 모델 `{MODEL}`. 원 예측 캐시는 conf floor0.001에서 NMS0.7을 적용한 고정 결과입니다. 모델과 NMS를 바꾸지 않고 confidence 포함 기준만 변경했습니다. 기존 IoU 일대일 매칭 코드를 재사용합니다.

Val 77장/GT175개에서 floor0.001/0.005와 0.01–0.90의 0.01 grid를 비교했습니다. 최종 운영 low 후보는 0.05/0.10/0.15/0.20/0.25로 제한해 설명 가능성을 유지했습니다. FP 증가 예산은 사전 정의한 0.10/image(Val 7개), 재검사 이미지 비율은 최대20%입니다. 두 예산 안에서 FN 복구 최대, FP 최소, 같은 값에서는 가장 높은 low threshold 순으로 선택했습니다. T_high는 기존 0.25를 유지합니다. Best F1은 참고용입니다.

## Val 정책 비교
Recall/Precision/TP/FP는 T_low 이상 bbox를 모두 포함한 **잠재 screening 성능**입니다. 재검사를 통해 확정된 검출 성능이 아닙니다.

|정책|T_low|T_high|Recall|Precision|FN|FP|복구|실제 재검사율|Band 이미지|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
'''
    for p in policies:text+=f"|{p['policy']}|{p['T_low']:.2f}|{p['T_high']:.2f}|{p['recall']*100:.2f}%|{p['precision']*100:.2f}%|{p['FN']}|{p['FP']}|{p['FN_Recovery']}|{p['reinspection_rate']*100:.2f}%|{p['band_images']}|\n"
    text+='\n## Recall 목표\n\n최소 threshold는 모든 목표에서 grid 하한으로 내려가기 때문에 운영상 정보가 적습니다. 요청한 최소값과 함께 목표를 달성하는 **최대 threshold**도 표시했습니다.\n\n|목표|가능|최소 grid threshold|최대 달성 threshold|Precision|FN|FP|\n|---|---|---:|---:|---:|---:|---:|\n'
    for r in targets:text+=f"|{r['target']*100:.0f}%|{r['achievable']}|{r.get('min_grid_threshold','NA')}|{r.get('max_threshold_meeting_target','NA')}|{r.get('precision','NA')}|{r.get('FN','NA')}|{r.get('FP','NA')}|\n"
    text+=f'''
## FN과 confidence
Val baseline FN {s['val_baseline']['FN']}개: Type1 {s['val_FN_types']['Type1_low_confidence']}개, Type2 {s['val_FN_types']['Type2_no_sufficient_overlap']}개. Type1은 IoU≥0.5 예측이 baseline 아래에 존재하는 경우입니다. Type2는 저장된 conf≥0.001 캐시 안에서 IoU≥0.5 예측이 없는 경우로, 모델이 완전히 아무것도 생성하지 않았음을 의미하지는 않습니다. 최고 IoU 후보의 위치 오류가 포함될 수 있습니다.

Val floor 분포: {s['val_confidence_distribution']}. 이 TP/FP 표시는 floor에서 한 번 정한 매칭이며 개별 sweep에서는 매칭이 달라질 수 있습니다. FN 분석의 confidence는 IoU≥0.5 후보 중 최고 confidence, 그런 후보가 없으면 전체 후보 중 최고 IoU의 confidence입니다. prediction class/폭/높이/제거 이유는 별도 CSV에 있습니다.

## 이미지 단위 운영
한 이미지에 T_high 이상 검출이 하나라도 있으면 자동 NG, 없고 middle band 검출이 있으면 재검사, 둘 다 없으면 정상 후보입니다. 자동 NG 이미지 안의 middle detection은 추가 이미지 재검사 부하로 계산하지 않습니다. band_images와 band_detections는 이를 제외하기 전의 전체 수이고 reinspection_images/detections는 실제 referral 대상입니다. 정상 후보는 무결함 보증이 아닙니다. 현재 데이터는 결함 라벨을 가진 이미지 집합이므로 양품 포함 생산라인의 재검사율과 false reject rate를 추정할 수 없습니다.

Two threshold는 자동 NG 기준을 유지하고 낮은 점수 이미지를 검토 대상으로 보낼 수 있다는 설명상의 장점이 있습니다. 자동 NG 이미지 안에서 추가 GT를 발견한 것은 실제 재검사 이미지 복구와 다릅니다. 단일 threshold 대비 사람 재검사의 정확도를 이 데이터로 측정하지 않았습니다.

## Small / low contrast 우선순위
Train GT 하위25% 기준: area≤{s['train_cutoffs']['small_area_q25']:.3f}, normalized contrast≤{s['train_cutoffs']['low_contrast_q25']:.6f}. Prediction bbox ring은 root-cause의 padding 규칙과 제품 mask를 사용하지만 GT 라벨로 주변을 제외하지 않습니다. 운영 때 사용할 수 있는 영상 feature만 계산합니다.

|Middle 조건|Detection|TP|FP|TP 비율|
|---|---:|---:|---:|---:|
'''
    for r in groups:text+=f"|{r['group']}|{r['detections']}|{r['TP']}|{r['FP']}|{r['TP_fraction']}|\n"
    text+=f'''
Middle 표본이 매우 적어 small/low contrast를 재검사 우선순위 규칙으로 추가하지 않습니다. GT FN의 특성과 prediction bbox의 TP/FP 분리력이 같다고 가정하지 않았습니다.

## 잠긴 정책과 Test 단일 적용
선택: **{chosen['policy']}**, T_low={chosen['T_low']:.2f}, T_high={chosen['T_high']:.2f}. locked_policy.json을 Test 평가 전에 저장했습니다. Test에는 최종 정책 1개와 고정 baseline만 비교했고 정책을 수정하지 않았습니다.

|항목|Baseline|Final policy 잠재 screening|
|---|---:|---:|
|TP|{testbase['TP']}|{testm['TP']}|
|FN|{testbase['FN']}|{testm['FN']}|
|FP|{testbase['FP']}|{testm['FP']}|
|Recall|{testbase['recall']*100:.2f}%|{testm['recall']*100:.2f}%|
|Precision|{testbase['precision']*100:.2f}%|{testm['precision']*100:.2f}%|
|FN 감소|0|{testbase['FN']-testm['FN']}|
|FP 증가|0|{testm['FP']-testbase['FP']}|
|실제 재검사 이미지|0|{testroute['reinspection_images']}/{testroute['images']}|
|실제 재검사율|0%|{testroute['reinspection_rate']*100:.2f}%|

Test baseline FN12개의 Type1/Type2: {s['test_FN_types']}. 이 진단은 Test threshold sweep이나 정책 재선택에 사용하지 않았습니다. Final band detection {testroute['band_detections']}개, band image {testroute['band_images']}개, 실제 재검사 detection {testroute['reinspection_detections']}개입니다.

## 재현과 검증
`.venv\\Scripts\\python.exe -B -X utf8 scripts\\analyze_safety_threshold_policy.py`
기존 출력은 덮어쓰지 않습니다. 재현에는 `--out outputs/safety_threshold_policy_run2`처럼 새 경로를 사용하세요. 모델/이미지/라벨/split/기존 prediction SHA256 전후 동일입니다. seed42는 scatter jitter에만 사용했습니다. 그래프와 FN/정책 contact sheet는 visualizations에 있습니다. 빨강GT, 초록 high confidence, 주황 low confidence입니다.

Val에서 low={chosen['T_low']:.2f}는 baseline 대비 FN {s['val_baseline']['FN']-chosen['FN']}개 감소/FP {chosen['FP']-s['val_baseline']['FP']}개 증가였습니다. Test에서는 FN {testbase['FN']-testm['FN']}개 감소/FP {testm['FP']-testbase['FP']}개 증가였으며, 실제 재검사율은 {testroute['reinspection_rate']*100:.2f}%입니다. 이 Test 결과로 임계값을 다시 선택하지 않습니다.

## 최종 판단
Val에서 잠긴 two-threshold 후보를 그대로 유지하되, 현재 데이터에서 실제 운영 개선이 검증됐다고 주장하지 않습니다. Val의 middle band 후보가 있는 2개 이미지는 모두 다른 high-confidence detection으로 이미 자동 NG였고, Test에는 band 후보 자체가 없었습니다. GT가 있는 Test 이미지 중 {testroute['normal_candidate_images']}개가 정상 후보로 남았습니다. 실운영 채택은 양품 포함 독립 데이터와 실제 재검사 결과에서 부하/오판정을 검증한 뒤 판단해야 합니다. 현재 결과만으로 two threshold가 단일 threshold보다 운영적으로 유리하다는 근거는 없습니다.
'''
    (out/'README.md').write_text(text,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);main(p.parse_args().out.resolve())
