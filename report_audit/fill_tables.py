from pathlib import Path
import csv
R=Path(__file__).resolve().parents[1]
m=(R/'report_audit/model_tables.md').read_text(encoding='utf-8')
settings,rest=m.split('\n\nValidation\n\n')
val,test=rest.split('\n\nTest\n\n')
with (R/'outputs/low_contrast_aux_poc/val_parameter_search.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
a=['| Val 설정 | TP | FN | FP | FN복구 | 추가FP | 선택 |','|---|---:|---:|---:|---:|---:|---|']
for r in rows:a.append('| '+' | '.join([r['config'],r['TP'],r['FN'],r['FP'],r['FN_Recovery'],r['Added_FP'],'선택' if r['selected']=='True' else '미채택'])+' |')
p=R/'FINAL_REPORT_DRAFT.md'
s=p.read_text(encoding='utf-8').replace('@MODEL_SETTINGS@',settings).replace('@VAL_METRICS@',val).replace('@TEST_METRICS@',test).replace('@AUX_TABLE@','\n'.join(a))
assert '@MODEL_SETTINGS@' not in s and '@AUX_TABLE@' not in s
p.write_text(s,encoding='utf-8')
print('tables filled')
