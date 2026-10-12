import json,csv
from pathlib import Path
from PIL import Image
r=Path.cwd(); out=r/'presentation_build'
figs=[]
for p in (r/'outputs').rglob('*'):
 if p.suffix.lower() not in ('.png','.jpg','.jpeg'): continue
 if any(x in p.parts for x in ('images','masks','restoration_masks','original','modified','clean','fake','crops','labels')): continue
 try:
  im=Image.open(p); figs.append({'path':str(p.relative_to(r)),'width':im.width,'height':im.height})
 except Exception: pass
(out/'figure_inventory.json').write_text(json.dumps(figs,ensure_ascii=False,indent=2),encoding='utf8')
nb=json.loads((r/'KAMP_Xray_Final_Analysis.ipynb').read_text(encoding='utf8'))
cells=[{'cell':i+1,'type':c['cell_type'],'text':''.join(c['source'])} for i,c in enumerate(nb['cells'])]
(out/'notebook_review.json').write_text(json.dumps(cells,ensure_ascii=False,indent=2),encoding='utf8')
print('Figures indexed',len(figs),'Notebook cells',len(cells))
for f in figs:
 if any(x in f['path'] for x in ['model_comparison','inpainting_poc_20/figures','safety_threshold_policy/visualizations','shortcut_diagnostic_gpu/plots']): print(f)
for name in ['outputs/model_comparison','outputs/shortcut_diagnostic_gpu/summary.json','outputs/error_analysis/summary/summary.json','outputs/safety_threshold_policy/locked_policy.json']:
 p=r/name
 if p.is_dir(): print(name,[x.name for x in p.iterdir()])
 else: print(name,p.read_text(encoding='utf8')[:8500])
