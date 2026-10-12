from pathlib import Path
import json,csv,re,hashlib,zipfile,difflib
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'submission_audit'
checks=[]
def check(name,passed):checks.append(dict(check=name,passed=bool(passed)))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
draft=(ROOT/'FINAL_REPORT_DRAFT.md').read_text(encoding='utf-8')
audit=(ROOT/'FINAL_SUBMISSION_AUDIT.md').read_text(encoding='utf-8')
check('six chapters unchanged',len(re.findall(r'^## 제[1-6]장\.',draft,re.M))==6)
check('obsolete final bbox values removed',not any(v in draft for v in ['114.322442','중앙값 100px²','75분위수 140px²','0.428571–2.0']))
check('official confusion referenced','submission_audit/official_test_confusion.png' in draft)
check('no unresolved confusion wording','수치 불일치 확인 필요' not in draft)
check('matching roots explained',all(v in draft for v in ['0.455820','0.449256','149/7/11','148/8/12','150/6/10']))
check('all source number checks passed',json.loads((OUT/'number_checks.json').read_text(encoding='utf-8'))['passed']==108)
for path in set(re.findall(r'(?:outputs|submission_audit)/[^\s\]—]+\.png',draft)):
    check('proposed figure exists '+path,(ROOT/path).is_file())
for filename in ['REPORT_EVIDENCE.md','REPORT_VALIDATION.md']:
    s=(ROOT/filename).read_text(encoding='utf-8')
    check(filename+' updated confusion conclusion','원인 규명 완료' in s or '차이 규명 완료' in s)
check('audit contains required conclusion',all(v in audit for v in ['제출 가능:','CONDITIONAL','가장 큰 감점 위험:','반드시 수정할 사항:','선택적으로 보강할 사항:']))

# Existing delivered bundle integrity: the audit did not overwrite its models or manifests.
included=json.loads((ROOT/'KAMP_Xray_Submission/included_files.json').read_text(encoding='utf-8'))
check('existing submission 614 SHA entries unchanged',all(sha(ROOT/'KAMP_Xray_Submission'/r['path'])==r['sha256'] and (ROOT/'KAMP_Xray_Submission'/r['path']).stat().st_size==r['bytes'] for r in included['files']))
with zipfile.ZipFile(ROOT/'KAMP_Xray_Submission.zip') as z:
    check('existing ZIP matches unchanged submission',all(sha(ROOT/n)==hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if not n.endswith('/')))
supp=OUT/'model_comparison_supplement'
entries=json.loads((supp/'SHA256.json').read_text(encoding='utf-8'))
check('supplement 12 original SHA entries match',len(entries)==12 and all(sha(supp/r['path'])==r['sha256'] for r in entries))
with zipfile.ZipFile(OUT/'model_comparison_supplement.zip') as z:
    check('supplement ZIP matches files',all(hashlib.sha256(z.read(n)).hexdigest()==sha(OUT/n) for n in z.namelist() if not n.endswith('/')))

diffs=[]
for name in ['REPORT_EVIDENCE.md','FINAL_REPORT_DRAFT.md','REPORT_VALIDATION.md']:
    before=(OUT/('before_'+name)).read_text(encoding='utf-8').splitlines()
    after=(ROOT/name).read_text(encoding='utf-8').splitlines()
    diffs.extend(difflib.unified_diff(before,after,fromfile='before/'+name,tofile=name,lineterm=''))
(OUT/'report_changes.diff').write_text('\n'.join(diffs)+'\n',encoding='utf-8')
(OUT/'final_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print('Final audit checks',len(checks),'passed',sum(r['passed'] for r in checks))
for c in checks:
    if not c['passed']:print('FAIL',c['check'])
assert all(r['passed'] for r in checks)
