from pathlib import Path
import json,csv,hashlib,collections,ast,io,contextlib,os,zipfile,math
from PIL import Image
R=Path(__file__).resolve().parent; O=R/'report_audit'
def js(p):return json.loads((R/p).read_text(encoding='utf-8-sig'))
def cs(p):
    with (R/p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
def check(label,ok,detail=''):
    checks.append(dict(check=label,passed=bool(ok),detail=detail))
    if not ok: print('FAIL',label,detail)
inv=js('report_audit/inventory.json'); project=[x for x in inv if x['path'].split('/')[0] not in ('.git','.venv','.cache','__pycache__') and x['path']!='report_audit.py']
print('PROJECT ORIGINAL',len(project),'extensions',collections.Counter(x['ext'] for x in project))
print('SPECIAL', [x['path'] for x in project if x['ext'] in ('.xlsx','.pdf','.log')])
pyfiles=[R/x['path'] for x in project if x['ext']=='.py']
code=[]
for p in pyfiles:
    src=p.read_text(encoding='utf-8-sig',errors='replace'); tree=ast.parse(src)
    code.append('FILE '+p.relative_to(R).as_posix()+'\nDOC '+str(ast.get_docstring(tree))+'\nFUNCTIONS '+str([(n.name,n.lineno) for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))])+'\n'+src)
(O/'all_project_code.txt').write_text('\n'.join(code),encoding='utf-8');print('PY read',len(pyfiles))
m=cs('outputs/yolo_512_telea/manifest.csv'); meta=cs('outputs/clean_512/metadata.csv'); audits={r['unique_id']:r for r in cs('outputs/submission_label_audit/selected_512_label_check.csv')}
print('AUDIT columns',list(next(iter(audits.values()))));print('META columns',list(meta[0]))
check('512 IDs',len(m)==len({x['unique_id'] for x in m})==512)
counts={};res=collections.Counter(); labels_ok=True;img_ok=True;source_ok=True
for split in ['train','val','test']:
    rows=[x for x in m if x['split']==split]; b=0
    for r in rows:
        p=R/'outputs/yolo_512_telea/labels'/split/(Path(r['filename']).stem+'.txt');lines=[line.split() for line in p.read_text().splitlines() if line.strip()];b+=len(lines)
        labels_ok &= sha(p)==r['label_sha256']==audits[r['unique_id']]['selected_source_sha256'] and len(lines)==int(r['bbox_count'])
        for l in lines:
            c,x,y,w,h=map(float,l);labels_ok &= c==0 and w>0 and h>0 and x-w/2>=-1e-7 and y-h/2>=-1e-7 and x+w/2<=1+1e-7 and y+h/2<=1+1e-7
        im=R/'outputs/yolo_512_telea/images'/split/r['restored_filename']; img_ok &= sha(im)==r['restored_sha256']
        with Image.open(im) as image:res[image.size]+=1
    counts[split]={'images':len(rows),'bbox':b}
check('labels hash/format/count',labels_ok);check('restored image hashes',img_ok)
provenance={r['unique_id']:r for r in cs('outputs/clean_512/provenance.csv')}
for r in meta:
    p=Path(provenance[r['unique_id']]['final_label_original_path'])
    source_ok &= p.is_file() and sha(p)==r['sha256_label']==audits[r['unique_id']]['selected_source_sha256']
check('original selected label source hashes',source_ok)
print('COUNTS',counts,'RES',res)
features=cs('outputs/foreign_object_condition_analysis/features.csv');allgt=cs('outputs/error_analysis/summary/all_test_gt.csv');pred=js('outputs/foreign_object_condition_analysis/predictions.json')
def iou(a,b):
    it=math.prod(max(0,min(a[k+2],b[k+2])-max(a[k],b[k])) for k in [0,1]);return it/(math.prod(a[k+2]-a[k] for k in [0,1])+math.prod(b[k+2]-b[k] for k in [0,1])-it)
def match(boxes,ps,t):
    ps=[p for p in ps if p['confidence']>=t and p['class_id']==0];pairs=sorted([(iou(g,p['box']),i,j) for i,g in enumerate(boxes) for j,p in enumerate(ps)],reverse=True);gi=set();pi=set()
    for sc,i,j in pairs:
        if sc<.5:break
        if i not in gi and j not in pi:gi.add(i);pi.add(j)
    return len(gi),len(ps)-len(pi),len(boxes)-len(gi)
recalcs=[]
for split,t in [('val',.25),('val',.2),('val',.31),('test',.25),('test',.2)]:
    total=[0,0,0]
    for r in m:
        if r['split']!=split:continue
        with Image.open(R/'outputs/yolo_512_telea/images'/split/r['restored_filename']) as im:w,h=im.size
        boxes=[]
        for line in (R/'outputs/yolo_512_telea/labels'/split/(Path(r['filename']).stem+'.txt')).read_text().splitlines():
            c,x,y,bw,bh=map(float,line.split()); boxes.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
        a=match(boxes,pred[r['unique_id']],t);total=[v+n for v,n in zip(total,a)]
    tp,fp,fn=total;record={'split':split,'threshold':t,'TP':tp,'FP':fp,'FN':fn,'P':tp/(tp+fp),'R':tp/(tp+fn),'F1':2*tp/(2*tp+fp+fn)};recalcs.append(record)
print('REMATCH',recalcs)
check('test TP/FP/FN rematch',[(r['TP'],r['FP'],r['FN']) for r in recalcs if r['split']=='test']==[(148,8,12)]*2)
check('val rematch',[(r['TP'],r['FP'],r['FN']) for r in recalcs if r['split']=='val']==[(170,11,5),(171,12,4),(170,9,5)])
conditions=[]
for field in ['small_object','low_contrast','product_edge','image_edge_near','restoration_near','restoration_intersects']:
    if field not in allgt[0]:continue
    for flag in ['True','False']:
        rows=[r for r in allgt if r[field]==flag];conditions.append(dict(condition=field,group=flag,GT=len(rows),FN=sum(r['status']=='FN' for r in rows)))
print('CONDITIONS',conditions)
check('FN/FP image counts',len({r['unique_id'] for r in cs('outputs/error_analysis/error_cases.csv') if r['error_type']=='FN'})==8 and len({r['unique_id'] for r in cs('outputs/error_analysis/error_cases.csv') if r['error_type']=='FP'})==4)
cm=js('outputs/model_comparison/comparison_manifest.json'); runs=[];index={r['run_name']:r for r in cs('outputs/yolo_runs/experiments.csv')}
for i,name in enumerate(cm['runs'],1):
    root=f'outputs/yolo_runs/{name}';s=js(root+'/summary.json');cfg=js(root+'/experiment_config.json');logs=cs(root+'/results.csv');logs=[{k.strip():v for k,v in row.items()} for row in logs];epoch=int(float(max(logs,key=lambda r:float(r['metrics/mAP50-95(B)']))['epoch']))
    check(name+' best epoch',s['best_epoch']==epoch)
    check(name+' index metrics',all(math.isclose(float(index[name][k]),s[k],abs_tol=1e-12) for k in s if k.startswith(('val_','test_')) and k not in ('val_images','test_images')))
    runs.append({'id':f'R{i:02d}','config':cfg,'summary':s,'logged_epochs':len(logs)})
print('RUNS',json.dumps(runs,ensure_ascii=False))
check('weight hash',sha(R/'outputs/yolo_runs'/cm['fixed_downstream_model']/'weights/best.pt')==js('outputs/safety_threshold_policy/summary.json')['provenance']['model_sha256'])
sub=R/'KAMP_Xray_Submission';items=js('KAMP_Xray_Submission/included_files.json'); print('INCLUDED type',type(items),str(items)[:300])
included_bad=[r['path'] for r in items['files'] if not (sub/r['path']).is_file() or (sub/r['path']).stat().st_size!=r['bytes'] or sha(sub/r['path'])!=r['sha256']]
check('submission included SHA list',not included_bad,str(included_bad));print('INCLUDED',len(items['files']))
with zipfile.ZipFile(R/'KAMP_Xray_Submission.zip') as z:
    entries=[n for n in z.namelist() if not n.endswith('/')]
    bad=[n for n in entries if not (R/n).is_file() or hashlib.sha256(z.read(n)).hexdigest()!=sha(R/n)]
    check('ZIP matches folder',not bad,str(bad));print('ZIP entries',len(entries))
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(v) for v in row)+' |' for row in rows])
settings=[];metric_tables={}
for run in runs:
    c,s=run['config'],run['summary'];a='무증강' if c['mosaic']==0 else '기본'
    if c.get('gamma_p',0):a+=' + 감마'
    if c.get('noise_p',0):a+=' + 노이즈'
    settings.append([run['id'],c['model'],c['imgsz'],c['batch'],c['lr0'],a,s['best_epoch'],run['logged_epochs']])
for split in ['val','test']:
    rows=[]
    for run in runs:
        s=run['summary'];p,r=s[split+'_precision'],s[split+'_recall'];rows.append([run['id']]+[f'{v:.6f}' for v in [p,r,2*p*r/(p+r),s[split+'_map50'],s[split+'_map50_95']]])
    metric_tables[split]=table(['실험','Precision','Recall','F1(P/R)','mAP50','mAP50–95'],rows)
(O/'model_tables.md').write_text(table(['실험','모델','imgsz','batch','lr0','증강','best epoch','실행 epoch'],settings)+'\n\nValidation\n\n'+metric_tables['val']+'\n\nTest\n\n'+metric_tables['test'],encoding='utf-8')
nb=js('KAMP_Xray_Final_Analysis.ipynb');nb2=js('KAMP_Xray_Submission/KAMP_Xray_Final_Analysis.ipynb');check('notebook sources match',[c['source'] for c in nb['cells']]==[c['source'] for c in nb2['cells']])
ns={};buf=io.StringIO(); executed=[]
try:
    with contextlib.redirect_stdout(buf):
        for i,c in enumerate(nb['cells'],1):
            if c['cell_type']=='code':
                exec(compile(''.join(c['source']),f'notebook_cell_{i}','exec'),ns)
                ns['display']=lambda *args,**kwargs:None
                executed.append(i)
    check('17 notebook code cells replay',len(executed)==17,str(executed))
except Exception as e: check('17 notebook code cells replay',False,repr(e)+' executed='+str(executed))
(O/'notebook_replay.txt').write_text(buf.getvalue(),encoding='utf-8')
(O/'verification_computed.json').write_text(json.dumps({'counts':counts,'resolution':{str(k):v for k,v in res.items()},'rematch':recalcs,'conditions':conditions,'runs':runs,'checks':checks},ensure_ascii=False,indent=2),encoding='utf-8')
print('CHECKS',json.dumps(checks,ensure_ascii=False))
