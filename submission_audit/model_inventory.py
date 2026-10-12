from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'.cache/shortcut_runtime'))
import torch
torch.set_num_threads(4)
manifest=json.loads((ROOT/'outputs/model_comparison/comparison_manifest.json').read_text(encoding='utf-8'))
rows=[]
for i,name in enumerate(manifest['runs'],1):
    d=ROOT/'outputs/yolo_runs'/name
    p=d/'weights/best.pt';config=json.loads((d/'experiment_config.json').read_text(encoding='utf-8'))
    checkpoint=torch.load(p,map_location='cpu',weights_only=False)
    model=checkpoint.get('ema') or checkpoint['model']
    statehash=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        statehash.update(k.encode());statehash.update(v.cpu().contiguous().numpy().tobytes())
    sub=ROOT/'KAMP_Xray_Submission'/p.relative_to(ROOT)
    row=dict(id=f'R{i:02d}',run_name=name,architecture=config['model'],config=config,
        weight_path=str(p.relative_to(ROOT)),weight_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        state_sha256=statehash.hexdigest(),parameter_count=sum(v.numel() for v in model.parameters()),
        yaml=model.yaml,checkpoint_date=checkpoint.get('date'),train_args=checkpoint.get('train_args'),
        submission_weight_exists=sub.is_file(),submission_summary_exists=(ROOT/'KAMP_Xray_Submission/outputs/yolo_runs'/name/'summary.json').is_file())
    rows.append(row)
data=dict(rows=rows,unique_weight_files=len(set(r['weight_sha256'] for r in rows)),unique_parameter_states=len(set(r['state_sha256'] for r in rows)))
(ROOT/'submission_audit/model_inventory.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
for r in rows:print(r['id'],r['architecture'],r['parameter_count'],r['state_sha256'][:12],r['submission_weight_exists'],r['submission_summary_exists'])
print('unique weights',data['unique_weight_files'],'states',data['unique_parameter_states'])
