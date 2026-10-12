from pathlib import Path
from datetime import datetime
import json,csv,hashlib,zipfile
ROOT=Path(__file__).resolve().parent.parent
PACKAGE=ROOT/'KAMP_Xray_Submission'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

kernel=json.loads((ROOT/'report_audit/kernel_verification.json').read_text(encoding='utf-8'))
replay=json.loads((ROOT/'submission_audit/preprocess_package_replay/verification.json').read_text(encoding='utf-8'))
assert replay['images']==512 and replay['pixels_match_included_images']
assert all(r['code_cells']==17 and not r['errors'] for r in kernel)
with (PACKAGE/'outputs/yolo_512_telea/manifest.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
for r in rows:
    assert sha(PACKAGE/'outputs/yolo_512_telea/images'/r['split']/r['restored_filename'])==r['restored_sha256']
    assert sha(PACKAGE/'outputs/yolo_512_telea/labels'/r['split']/(Path(r['filename']).stem+'.txt'))==r['label_sha256']
original=json.loads((PACKAGE/'verification.json').read_text(encoding='utf-8'))
original['package_repair']=dict(verified_at=datetime.now().astimezone().isoformat(),
    training_images=359,validation_images=77,test_images=76,labels=512,GT=1163,
    source_code_included=True,selected_original_images=512,all_comparison_best_weights=11,
    test_predictions_separated=True,test_prediction_images=76,test_prediction_detections=156,
    portable_training_preflight_passed=True,preprocessing_pixels_matched=512,
    notebook_code_cells_passed=17,training_started=False)
write(PACKAGE/'verification.json',original)
p=PACKAGE/'검증결과.md'
p.write_text(p.read_text(encoding='utf-8')+'''

## 소스코드 ZIP 제출 구성 보완 (2026-10-07)

Train359/Val77/Test76의 영상512장과 라벨512개·1163bbox를 포함하였다. 선택 원본512장·전처리/평가 코드·mask·11run의best.pt 및 사전학습weight를 추가하였다. Test76장 전용 예측JSON과156검출CSV, 검출 없는 영상도 포함한76장 목록을 구성하였다. 상대경로 dataset_submission.yaml에서 학습 사전검사를 통과하였다. 포함 원본의 TELEA재생성512장 모두 기존복원영상과 픽셀이 일치하였고, Notebook코드17셀을 오류 없이 실제 실행하였다. 재학습은 실행하지 않았다. 신규추론은기존제출예측과별도출력이다.
''',encoding='utf-8')
inventory=json.loads((PACKAGE/'included_files.json').read_text(encoding='utf-8'))
old={r['path']:r for r in inventory['files']};records=[]
for p in sorted(PACKAGE.rglob('*')):
    if not p.is_file() or p.name=='included_files.json' or '__pycache__' in p.parts:continue
    name=p.relative_to(PACKAGE).as_posix();row=old.get(name,dict(path=name))
    row.update(bytes=p.stat().st_size,sha256=sha(p))
    if name=='train_yolo.py':row['submission_changes']='Portable training dataset and included pretrained weight; final CONFIG preserved.'
    records.append(row)
inventory.update(description='Source-code submission: code, environment requirements, full frozen training/val/test data, README and separate Test predictions.',
    updated_at=datetime.now().astimezone().isoformat(),files=records,file_count_excluding_inventory=len(records))
write(PACKAGE/'included_files.json',inventory)
allfiles=sorted(p for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
staged=ROOT/'KAMP_Xray_Submission.updated.zip'
with zipfile.ZipFile(staged,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in allfiles:z.write(p,p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(staged) as z:
    assert z.testzip() is None
    assert set(z.namelist())=={p.relative_to(ROOT).as_posix() for p in allfiles}
    for p in allfiles:assert hashlib.sha256(z.read(p.relative_to(ROOT).as_posix())).hexdigest()==sha(p)
staged.replace(ROOT/'KAMP_Xray_Submission.zip')
result=dict(status='passed',files=len(allfiles),inventory_entries=len(records),zip_bytes=(ROOT/'KAMP_Xray_Submission.zip').stat().st_size,
    zip_mb=round((ROOT/'KAMP_Xray_Submission.zip').stat().st_size/1024**2,2),zip_sha256=sha(ROOT/'KAMP_Xray_Submission.zip'),data=original['package_repair'],all_zip_file_hashes_verified=True)
write(ROOT/'submission_audit/package_repair_result.json',result)
print(json.dumps(result,ensure_ascii=False,indent=2))
