from pathlib import Path
import json,csv,re,collections,math,hashlib
import numpy as np
from PIL import Image
from compare_matching import get_data,own

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'submission_audit'
report=(ROOT/'FINAL_REPORT_DRAFT.md').read_text(encoding='utf-8')
def js(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def cs(p):
    with (ROOT/p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
checks=[]
def check(name,value,ok,source):
    checks.append(dict(item=name,observed=value,passed=bool(ok),source=source))

# Exact cells in all main comparison tables.
manifest=js('outputs/model_comparison/comparison_manifest.json')
for i,name in enumerate(manifest['runs'],1):
    base='outputs/yolo_runs/'+name
    c=js(base+'/experiment_config.json');s=js(base+'/summary.json')
    for split in ['val','test']:
        p,r=s[split+'_precision'],s[split+'_recall']
        values=[p,r,2*p*r/(p+r),s[split+'_map50'],s[split+'_map50_95']]
        row='| R'+format(i,'02d')+' | '+' | '.join(format(v,'.6f') for v in values)+' |'
        check('R'+format(i,'02d')+' '+split+' 5 metrics',values,row in report,base+'/summary.json')
    augmentation='무증강' if c['mosaic']==0 else '기본'
    if c.get('gamma_p',0):augmentation+=' + 감마'
    if c.get('noise_p',0):augmentation+=' + 노이즈'
    log=cs(base+'/results.csv')
    settings=[c['model'],c['imgsz'],c['batch'],c['lr0'],augmentation,s['best_epoch'],len(log)]
    row='| R'+format(i,'02d')+' | '+' | '.join(str(v) for v in settings)+' |'
    check('R'+format(i,'02d')+' settings',settings,row in report,base+'/experiment_config.json; results.csv')

for t in [.02,.05,.10,.15,.20,.25,.31]:
    tot=[0,0,0]
    for r in get_data('val').values():
        a=own(r['gt'],r['cache'],t=t)
        tot=[v+a[k] for v,k in zip(tot,['TP','FP','FN'])]
    tp,fp,fn=tot;metrics=[tp/(tp+fp),tp/(tp+fn),2*tp/(2*tp+fp+fn)]
    row='| '+format(t,'.2f')+' | '+' | '.join(str(v) for v in tot)+' | '+' | '.join(format(v,'.6f') for v in metrics)+' |'
    check('Val threshold '+str(t),tot+metrics,row in report,'predictions.json + labels/val (direct matching)')

for r in cs('outputs/low_contrast_aux_poc/val_parameter_search.csv'):
    row='| '+r['config']+' | '+' | '.join(r[k] for k in ['TP','FN','FP','FN_Recovery','Added_FP'])+' | '+('선택' if r['selected']=='True' else '미채택')+' |'
    check('aux '+r['config'],{k:r[k] for k in ['TP','FN','FP','FN_Recovery','Added_FP']},row in report,'outputs/low_contrast_aux_poc/val_parameter_search.csv')

meta=cs('outputs/clean_512/metadata.csv');m=cs('outputs/yolo_512_telea/manifest.csv')
areas=[];aspects=[];counts=collections.Counter();res=collections.Counter()
for row in m:
    with Image.open(ROOT/'outputs/yolo_512_telea/images'/row['split']/row['restored_filename']) as im:w,h=im.size
    res[(w,h)]+=1
    lines=(ROOT/'outputs/yolo_512_telea/labels'/row['split']/(Path(row['filename']).stem+'.txt')).read_text().splitlines()
    counts[len(lines)]+=1
    for line in lines:
        c,x,y,bw,bh=map(float,line.split());areas.append(bw*w*bh*h);aspects.append(bw*w/(bh*h))
check('512 images /1163 bbox',[len(m),len(areas)],len(m)==512 and len(areas)==1163,'512 actual labels + image headers')
check('split image/bbox counts',[(sp,sum(r['split']==sp for r in m),sum(int(r['bbox_count']) for r in m if r['split']==sp)) for sp in ['train','val','test']],
    all((sum(r['split']==sp for r in m),sum(int(r['bbox_count']) for r in m if r['split']==sp))==expected for sp,expected in [('train',(359,828)),('val',(77,175)),('test',(76,160))]),'actual manifest and labels')
check('bbox-per-image distribution',dict(counts),counts=={1:186,2:1,3:325},'actual labels')
check('resolution',str(res),res=={(316,332):218,(352,332):125,(576,444):169},'actual image headers')
check('bbox area statistics',[min(areas),*np.quantile(areas,[.25,.5,.75]),max(areas),np.mean(areas)],
    np.allclose([min(areas),*np.quantile(areas,[.25,.5,.75]),max(areas),np.mean(areas)],[21,81,104,143,420,115.890800],atol=.000001),'actual TXT × width/height')
check('aspect statistics',[min(aspects),np.median(aspects),max(aspects)],np.allclose([min(aspects),np.median(aspects),max(aspects)],[.428571,1,1.833333],atol=.000001),'actual TXT × width/height')
dates=sorted({r['capture_date'] for r in meta})
check('dates',[len(dates),dates[0],dates[-1]],(len(dates),dates[0],dates[-1])==(27,'2020-06-22','2020-09-22'),'clean_512/metadata.csv')
check('machine images/bbox',{mach:[sum(r['machine']==mach for r in meta),sum(int(r['bbox_count']) for r in meta if r['machine']==mach)] for mach in ['1','2','3']},
    all([sum(r['machine']==mach for r in meta),sum(int(r['bbox_count']) for r in meta if r['machine']==mach)]==expected for mach,expected in [('1',[161,411]),('2',[182,383]),('3',[169,369])]),'metadata.csv')

gt=cs('outputs/error_analysis/summary/all_test_gt.csv');cases=cs('outputs/error_analysis/error_cases.csv')
check('FN/FP images',[len({r['unique_id'] for r in cases if r['error_type']==k}) for k in ['FN','FP']],
    [len({r['unique_id'] for r in cases if r['error_type']==k}) for k in ['FN','FP']]==[8,4],'error_cases.csv')
types=collections.Counter(r['cause'] for r in cases if r['error_type']=='FN')
check('FN primary types',dict(types),types=={'localization_failure':8,'low_confidence_localization_success':3,'low_confidence_localization_failure':1},'error_cases.csv')
check('FP type/physical unknown',[sum(r['error_type']=='FP' and r['cause']=='localization_failure' for r in cases),sum(r['physical_cause']=='unknown' for r in cases)],
    sum(r['error_type']=='FP' and r['cause']=='localization_failure' for r in cases)==8 and sum(r['physical_cause']=='unknown' for r in cases)==20,'error_cases.csv')
for field,expected in [('small_object',[(7,41),(5,119)]),('low_contrast',[(7,60),(5,100)]),('product_edge',[(3,73),(9,87)]),('restoration_intersects',[(0,9),(12,151)])]:
    values=[]
    for flag in ['True','False']:
        group=[r for r in gt if r[field]==flag];values.append((sum(r['status']=='FN' for r in group),len(group)))
    check(field+' denominators',values,values==expected,'all_test_gt.csv')
for r in cases:
    if r['case_id'] in ['U0138_FN_gt0','U0352_FN_gt0','U0380_FN_gt0','U0181_FN_gt0','U0488_FN_gt0']:
        values=[format(float(r[k]),'.6f') for k in ['confidence','candidate_gt_iou']]
        check(r['case_id']+' conf/IoU',values,all(v in report for v in values),'error_cases.csv + predictions.json')
for r in cs('outputs/fn_root_cause_analysis/features/tp_fn_statistics.csv'):
    if r['feature'] in ['bbox_area','normalized_contrast','contour_distance','laplacian_variance','local_difference_mean']:
        value=float(r['median'])
        check(r['feature']+' '+r['status']+' median',value,any(format(value,'.'+str(d)+'f').rstrip('0').rstrip('.') in report for d in [4,5,6]),'tp_fn_statistics.csv')

policy=js('outputs/safety_threshold_policy/locked_policy.json')
check('locked threshold/cutoffs',[policy['selected']['T_low'],policy['selected']['T_high'],policy['small_low_contrast_train_cutoffs']],
    policy['selected']['T_low']==.2 and policy['selected']['T_high']==.25 and policy['small_low_contrast_train_cutoffs']['train_GT_with_features']==816 and policy['small_low_contrast_train_cutoffs']['small_area_q25']==81 and abs(policy['small_low_contrast_train_cutoffs']['low_contrast_q25']-.0698918192)<1e-10,'locked_policy.json')
for size,low,expected in [('False','False',(1,78)),('False','True',(4,41)),('True','False',(4,22)),('True','True',(3,19))]:
    rows=[r for r in gt if r['small_object']==size and r['low_contrast']==low]
    a=(sum(r['status']=='FN' for r in rows),len(rows));check('size/contrast '+size+'/'+low,a,a==expected,'all_test_gt.csv')
for machine,expected in [('1',(3,72)),('2',(1,28)),('3',(8,60))]:
    rows=[r for r in gt if r['machine']==machine];a=(sum(r['status']=='FN' for r in rows),len(rows))
    check('machine FN/GT '+machine,a,a==expected,'all_test_gt.csv')
for field,expected in [('image_edge_near',(0,0)),('restoration_near',(12,160))]:
    rows=[r for r in gt if r[field]=='True'];a=(sum(r['status']=='FN' for r in rows),len(rows))
    check(field+' comparison availability',a,a==expected,'all_test_gt.csv')
predrows=cs('outputs/error_analysis/summary/all_test_predictions.csv')
for lo,hi,expected in [(.2,.5,(0,28)),(.5,.7,(7,105)),(.7,1.00001,(1,23))]:
    rows=[r for r in predrows if lo<=float(r['confidence'])<hi];a=(sum(r['status']=='FP' for r in rows),len(rows))
    check('FP score bin '+str(lo),a,a==expected,'all_test_predictions.csv')
check('FP oversize count',sum(float(r['prediction_to_gt_area_ratio'])>1 for r in cases if r['error_type']=='FP'),
    sum(float(r['prediction_to_gt_area_ratio'])>1 for r in cases if r['error_type']=='FP')==7,'error_cases.csv')
edge=cs('outputs/edge_reinspection_poc/val_comparison.csv')
for name,expected in [('narrow',(170,11,5)),('medium',(170,14,5)),('wide',(170,35,5))]:
    row=next(r for r in edge if r['candidate']==name and r['method']=='fusion' and r['group']=='overall')
    observed=tuple(int(row[k]) for k in ['TP','FP','FN']);check('Edge Val '+name,observed,observed==expected,'val_comparison.csv')
aux=js('outputs/low_contrast_aux_poc/summary.json')
check('Aux Test P/R/F1 and counts',aux['test'],(aux['test']['TP'],aux['test']['FP'],aux['test']['FN'])==(148,12,12) and aux['test']['precision']==aux['test']['recall']==.925,'low_contrast_aux_poc/summary.json')
direction=cs('outputs/directional_refinement_poc/method_summary.csv')
check('Directional proxies',[[r['mean_line_score'],r['mean_artificial_edge_excess']] for r in direction],
    all(format(float(r[k]),'.5f') in report for r in direction for k in ['mean_line_score','mean_artificial_edge_excess']),'method_summary.csv')
region=js('outputs/foreign_object_condition_analysis/summary.json')
check('estimated region cohort',[region['전체_bbox'],region['신뢰도_기준_통과'],region['split_통과']],
    region['전체_bbox']==1163 and region['신뢰도_기준_통과']==649 and region['split_통과']=={'train':452,'val':100,'test':97},'foreign_object_condition_analysis/summary.json')

# The ledger extracts every occurrence, including IDs, paths, hashes and ordinal numbers.
# Source mapping is contextual; a repeated lexical number alone is not evidence of its meaning.
maps={
 '1':['outputs/clean_512/metadata.csv','outputs/clean_512/provenance.csv','outputs/clean_512_split/split_summary.json','outputs/unique_512_audit/','outputs/yolo_512_telea/','outputs/yolo_512_telea_qa/','submission_audit/number_checks.json'],
 '2':['outputs/yolo_runs/*/{experiment_config.json,summary.json,results.csv,environment.txt}','outputs/foreign_object_condition_analysis/{predictions.json,prediction_provenance.json}','outputs/safety_threshold_policy/','submission_audit/{matching_comparison.json,validator_replay.json,predict_replay.json}'],
 '3':['outputs/error_analysis/{error_cases.csv,summary/all_test_gt.csv}','outputs/fn_root_cause_analysis/features/tp_fn_statistics.csv','outputs/safety_threshold_policy/locked_policy.json'],
 '4':['outputs/safety_threshold_policy/{locked_policy.json,test_image_routing.csv}','outputs/error_analysis/','scripts/analyze_safety_threshold_policy.py'],
 '5':['outputs/yolo_runs/*/summary.json','outputs/inpainting_poc_20/','outputs/directional_refinement_poc/method_summary.csv','outputs/restoration_after_similarity/classifier_diagnostic.json','outputs/context_matched_restoration/classifier_comparison.csv','outputs/shortcut_diagnostic_gpu/summary.json','outputs/edge_reinspection_poc/','outputs/low_contrast_aux_poc/','outputs/foreign_object_condition_analysis/'],
 '6':['requirements-notebook.txt','requirements.txt','KAMP_Xray_Final_Analysis.ipynb','report_audit/kernel_verification.json','KAMP_Xray_Submission/included_files.json','report_audit/verification_computed.json','submission_audit/model_inventory.json']
}
ledger=[];chapter='1';section='';fenced=False
pattern=re.compile(r'(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?')
for ln,line in enumerate(report.splitlines(),1):
    if line.startswith('## 제'):chapter=re.search(r'제(\d)장',line)[1]
    if line.startswith('### '):section=line[4:]
    if line.startswith('```'):fenced=not fenced
    spans=[]
    for match in re.finditer(r'\b[0-9a-f]{64}\b|(?:outputs|submission_audit)/[^\s\]—]+|\b(?:R|U|E)\d+\b|\b(?:YOLO|yolo)[\w.]+|\bF1\b',line):spans.append((match.start(),match.end()))
    for match in pattern.finditer(line):
        identifier=any(a<=match.start()<b for a,b in spans)
        kind='식별자·경로·해시' if identifier else ('실행 명령' if fenced else ('목차·공식 배점' if line.startswith('#') else '정량 본문·설정'))
        source='공식 평가구조(사용자 제공)' if line.startswith('#') else '; '.join(maps[chapter])
        if ln<=3:source='작성일 및 공통 데이터 검증'
        ledger.append(dict(occurrence=len(ledger)+1,line=ln,section=section,number=match.group(),kind=kind,
            source=source,verification='식별자/문서 표기 확인' if identifier or line.startswith('#') or fenced else '해당 문맥 원본 대조; 재계산은 number_checks.json 참조',context=line))
with (OUT/'all_report_numbers.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
summary=dict(checks=checks,total_checks=len(checks),passed=sum(c['passed'] for c in checks),
    numeric_occurrences=len(ledger),categories=dict(collections.Counter(r['kind'] for r in ledger)),
    note='숫자 등장 전수 목록은 의미별 원본 대조의 색인이다. 동일 숫자의 어휘 일치만으로 검증했다고 취급하지 않는다.')
(OUT/'number_checks.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=lambda o:o.item()),encoding='utf-8')
print('Checks',len(checks),'passed',summary['passed'],'occurrences',len(ledger),'categories',summary['categories'])
for c in checks:
    if not c['passed']:print('FAIL',c['item'],c['observed'])
assert summary['passed']==len(checks)
