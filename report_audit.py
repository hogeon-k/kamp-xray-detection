from pathlib import Path
import os,json,csv,collections,re,hashlib,zipfile
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'report_audit'; OUT.mkdir(exist_ok=True)
files=[]; dirs=[]
for base,ds,fs in os.walk(ROOT):
    for d in ds: dirs.append(str((Path(base)/d).relative_to(ROOT)))
    for f in fs:
        p=Path(base)/f
        if OUT in p.parents: continue
        files.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'ext':p.suffix.lower()})
(OUT/'inventory.json').write_text(json.dumps(files,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'directories.txt').write_text('\n'.join(dirs),encoding='utf-8')
print('TOTAL',len(files),'TOP',dict(collections.Counter(x['path'].split('/')[0] for x in files)),'EXT',dict(collections.Counter(x['ext'] for x in files)))
project=[x for x in files if x['path'].split('/')[0] not in ('.git','.venv','.cache','__pycache__')]
print('PROJECT',len(project))
nbpaths=[ROOT/x['path'] for x in project if x['ext']=='.ipynb']
for p in nbpaths:
    nb=json.loads(p.read_text(encoding='utf-8')); sections=[]
    for i,c in enumerate(nb['cells'],1):
        src=''.join(c.get('source',[])); outs=[]
        for o in c.get('outputs',[]):
            if 'text' in o: outs.append(''.join(o['text']))
            data=o.get('data',{})
            if 'text/plain' in data: outs.append(''.join(data['text/plain']))
            if 'text/html' in data: outs.append(re.sub('<[^>]+>',' ',''.join(data['text/html'])))
            if o.get('output_type')=='error': outs.append(str(o))
        sections.append(f'\n## Cell {i} [{c["cell_type"]}] execution={c.get("execution_count")}\n{src}\nOUTPUT:\n'+ '\n'.join(outs))
    name=p.relative_to(ROOT).as_posix().replace('/','__')+'.txt'
    (OUT/name).write_text('\n'.join(sections),encoding='utf-8')
    print('NOTEBOOK',p.relative_to(ROOT),len(nb['cells']),len(''.join(sections)))
lines=[]
for x in project:
    p=ROOT/x['path']; rel=x['path']
    if rel.startswith('KAMP_Xray_Submission/') or rel.startswith('dataset/OpenLabeling-master/'): continue
    if x['ext']=='.json' and not any(a in p.parts for a in ['labels','annotations']):
        try:
            obj=json.loads(p.read_text(encoding='utf-8-sig'))
            s=json.dumps(obj,ensure_ascii=False,indent=2)
            lines.append('\n### '+rel+'\n'+(s if len(s)<20000 else s[:4000]+'\n[large JSON: inspect separately]'))
        except Exception as e: lines.append(str(e))
    elif x['ext'] in ('.csv','.xlsx','.yaml','.yml','.md'):
        if x['ext']=='.csv':
            with p.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
            lines.append('\n### '+rel+f' rows={len(rows)}\n'+json.dumps(rows[:3],ensure_ascii=False))
        elif x['ext']!='.xlsx': lines.append('\n### '+rel+'\n'+p.read_text(encoding='utf-8-sig',errors='replace')[:20000])
        else: lines.append('\n### XLSX '+rel)
(OUT/'metadata_dump.txt').write_text('\n'.join(lines),encoding='utf-8')
print('METADATA chars',sum(map(len,lines)))
