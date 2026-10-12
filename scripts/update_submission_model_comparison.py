"""Add saved YOLOv8n/11n comparisons and execute both analysis notebooks."""
from pathlib import Path
from datetime import datetime
import csv
import hashlib
import json
import os
import shutil
import sys
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'KAMP_Xray_Submission'
NAME = 'KAMP_Xray_Final_Analysis.ipynb'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def main():
    existing = sorted(p.parent.name for p in (PACKAGE/'outputs/yolo_runs').glob('*/summary.json'))
    extra = sorted(p.parent.name for p in (ROOT/'outputs/yolo_runs').glob('*/summary.json')
                   if 'img640_b4' in p.parent.name and 'telea_512_' in p.parent.name)
    runs = sorted(set(existing + extra))
    assert len(runs) == 11
    prefix = 'telea_512_{}_e200_img640_b4_adamw_lr0.001_wd0.0005_aug_tr0.005_sc0.2_mos0.2'
    pairs = [{'label': label,
              'v8': prefix.format('yolov8n') + suffix,
              'v11': prefix.format('yolo11n') + suffix}
             for label, suffix in [('基本 증강', '_s42'), ('기본 증강 + 감마', '_gam0.9-1.1p0.2_s42')]]
    pairs[0]['label'] = '기본 증강'
    study = [pairs[0]['v11'], pairs[1]['v11'],
             prefix.format('yolo11n') + '_gam0.9-1.1p0.2_noise1-3p0.1_s42']
    final = 'telea_512_yolo11n_e200_img1280_b4_adamw_lr0.001_wd0.0005_noaug_s42'
    manifest = {'created_at': datetime.now().astimezone().isoformat(), 'runs': runs,
                'matched_architecture_pairs': pairs, 'augmentation_study_runs': study,
                'fixed_downstream_model': final, 'selection_status': 'retained_previous_baseline',
                'test_used_for_new_selection': False,
                'scope': {'train_images':359, 'val_images':77, 'test_images':76},
                'dataset_manifest_sha256':sha(ROOT/'outputs/yolo_512_telea/manifest.csv'),
                'scope_limit': 'Saved counts and dataset settings checked; historical per-run dataset hashes were not recorded.'}
    write(ROOT/'outputs/model_comparison/comparison_manifest.json', manifest)
    for run in extra:
        for filename in ('summary.json', 'experiment_config.json', 'results.csv', 'environment.txt'):
            src = ROOT/'outputs/yolo_runs'/run/filename
            assert src.is_file(), src
            dest = PACKAGE/'outputs/yolo_runs'/run/filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
    shutil.copy2(ROOT/'outputs/model_comparison/comparison_manifest.json',
                 PACKAGE/'outputs/model_comparison/comparison_manifest.json'
                 if (PACKAGE/'outputs/model_comparison').exists() else
                 _create_comparison_dir())
    # Keep just completed runs included in the submission's index.
    with (ROOT/'outputs/yolo_runs/experiments.csv').open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = [row for row in reader if row['run_name'] in runs]
    assert set(row['run_name'] for row in rows) == set(runs)
    with (PACKAGE/'outputs/yolo_runs/experiments.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    notebook = nbformat.read(PACKAGE/NAME, as_version=4)
    backup_dir = ROOT/'.cache/notebook_backups'
    backup_dir.mkdir(parents=True, exist_ok=True)
    for folder in (ROOT, PACKAGE):
        shutil.copy2(folder/NAME, backup_dir/f'{folder.name}_before_model_comparison_{datetime.now():%Y%m%d_%H%M%S}.ipynb')
    notebook.cells[14].source = '''## 5. 동일한 512장 분할의 모델 비교와 최종 모델

**목적:** Train 359장 / Val 77장 / Test 76장에서 완료된 YOLO11n·YOLOv8n 실험을 비교한다.  
**공식 평가항목 연결:** 2. AI 예측모델 개발.  

총 11개 완료 실험은 YOLO11n 설정 실험 9개와 YOLOv8n 비교 실험 2개다.  
실험 수와 모델 구조 수를 구분하며, 구조는 YOLO11n·YOLOv8n의 2종이다.  
취소된 실행과 다른 데이터 규모의 PoC는 포함하지 않는다.  

먼저 전체 설정을 보여주고, **동일 조건의 구조 비교**와 **YOLO11n 증강 비교**를 따로 제시한다.  
구조 비교는 두 모델 모두 imgsz640 / batch4 / AdamW / lr0.001 / 최대 200 epoch / patience50 / seed42다.  
기본 증강은 translate0.005 / scale0.2 / mosaic0.2이고 회전·뒤집기·HSV·mixup은 0이다.  
추가 감마는 0.9–1.1, 확률 20%이며 노이즈는 표준편차 1–3(0–255 밝기 기준), 확률 10%다.  
각 구조 비교 쌍의 저장 설정이 모델명과 실행 메타데이터 외에 같은지 코드로 확인한다.  
각 실험의 저장 이미지 수와 데이터 경로는 확인하지만, 과거 실행별 데이터 해시가 없어 당시 입력의 완전 동일성을 소급 증명하지는 않는다.  

Val과 Test를 분리해 보고하며, 새 비교 결과만으로 기존 후속 분석 모델을 교체하지 않는다.  
`F1_from_reported_PR`은 저장 Precision·Recall의 조화평균으로, 6절의 고정 threshold·IoU 매칭 F1과 다르다.'''
    code = notebook.cells[15].source
    start = code.index("run_dirs = sorted(")
    end = code.index("run_rows=[]", start)
    code = code[:start] + '''comparison_manifest = load_json('outputs/model_comparison/comparison_manifest.json')
assert file_sha(source('outputs/yolo_512_telea/manifest.csv')) == comparison_manifest['dataset_manifest_sha256']
run_dirs = [run_root/name for name in comparison_manifest['runs']]
assert len(run_dirs) == len(set(comparison_manifest['runs'])) == 11
''' + code[end:]
    code = code.replace("'imgsz':config.get('imgsz',fallback.get('imgsz'))", "'batch':config.get('batch',fallback.get('batch')),'imgsz':config.get('imgsz',fallback.get('imgsz'))")
    code = code.replace("models[['experiment_id','imgsz','lr0','augmentation','best_epoch','selected_final']]",
                        "models[['experiment_id','model','imgsz','lr0','augmentation','best_epoch','selected_final']]")
    code = code.replace("['실험','입력 크기','학습률','증강','best epoch','최종 모델']",
                        "['실험','모델','입력 크기','학습률','증강','best epoch','최종 모델']")
    code += '''

# Saved CONFIG comparisons: missing gamma/noise probabilities in older runs mean disabled.
pair_setting_keys = ['preprocessing','data','epochs','imgsz','batch','device','workers','optimizer',
    'lr0','lrf','momentum','weight_decay','warmup_epochs','patience','seed','deterministic','pretrained',
    'degrees','translate','scale','shear','perspective','flipud','fliplr','mosaic','mixup',
    'hsv_h','hsv_s','hsv_v','gamma_p','noise_p','plots','save','conf']
paired_comparison_rows = []
for pair in comparison_manifest['matched_architecture_pairs']:
    configs = [load_json(f"outputs/yolo_runs/{pair[key]}/experiment_config.json") for key in ('v8','v11')]
    for key in pair_setting_keys:
        default = 0.0 if key in ('gamma_p','noise_p') else None
        assert configs[0].get(key, default) == configs[1].get(key, default), (pair['label'], key)
    assert [config['model'] for config in configs] == ['yolov8n.pt','yolo11n.pt']
    for probability, bounds in [('gamma_p', ('gamma_min','gamma_max')),
                                ('noise_p', ('noise_std_min','noise_std_max'))]:
        if configs[0].get(probability, 0) > 0:
            assert all(configs[0][key] == configs[1][key] for key in bounds)
    print('동일 조건 구조 비교:', pair['label'], '(640 / batch4 / seed42)')
    paired = models.set_index('run_name').loc[[pair['v8'], pair['v11']]].copy()
    for split, title in [('val','Validation 77장'), ('test','Test 76장')]:
        metrics = paired[['model',f'{split}_precision',f'{split}_recall',f'{split}_map50',f'{split}_map50_95']].copy()
        metrics.columns = ['모델','Precision','Recall','mAP50','mAP50–95']
        print(title)
        display(metrics.reset_index(drop=True))
        delta = metrics.iloc[1,1:].astype(float) - metrics.iloc[0,1:].astype(float)
        paired_comparison_rows.append({'조건':pair['label'],'분할':title,
            **{f'{key} 차이(pp)': float(value)*100 for key,value in delta.items()}})
print('구조 비교 차이: YOLO11n − YOLOv8n, 퍼센트포인트(pp)')
display(pd.DataFrame(paired_comparison_rows))
print('YOLO11n 증강 비교: 모델·640 해상도·batch4 고정')
study = models.set_index('run_name').loc[comparison_manifest['augmentation_study_runs']].copy()
study['조건'] = ['기본 증강','기본 증강 + 감마','기본 증강 + 감마 + 노이즈']
for split, title in [('val','Validation'), ('test','Test')]:
    table = study[['조건',f'{split}_precision',f'{split}_recall',f'{split}_map50',f'{split}_map50_95']].copy()
    table.columns = ['조건','Precision','Recall','mAP50','mAP50–95']
    print(title)
    display(table.reset_index(drop=True))
assert final_run == comparison_manifest['fixed_downstream_model']
assert len(models) == len(comparison_manifest['runs'])
'''
    notebook.cells[15].source = code
    notebook.cells[16].source = notebook.cells[16].source.replace(
        '512장 주 비교표는 모두 YOLO11n이며 동일 512장에서 다른 모델 구조를 비교한 저장 결과는 없다.',
        '''**5.1. YOLOv8n·YOLO11n 구조 비교 — 관찰됨**  
기본 증강·640 조건에서 Val은 YOLOv8n이 네 지표 모두 높았다.  
Test는 YOLO11n이 네 지표 모두 높았으며, mAP50은 0.926064 대 0.899269, mAP50–95는 0.384190 대 0.359172였다.  
따라서 이 조건의 Test에서는 YOLO11n이 우세하지만, 모델 구조가 모든 조건에서 최고라고 단정하지 않는다.  
감마만 추가한 동일 조건에서는 YOLO11n이 Val의 네 지표 모두 높았다.  
Test에서는 YOLO11n의 Precision·Recall·mAP50이 높고, mAP50–95는 YOLOv8n 0.353280이 YOLO11n 0.352791보다 근소하게 높았다.  
각 조건은 seed42의 단일 실행이며 반복 seed 검증이나 통계적 유의성 검증은 수행하지 않았다.  
1280 최종 모델과 640 YOLOv8n의 차이를 모델 구조만의 효과로 해석하지 않는다.  

**5.2. 감마·노이즈 추가 비교 — 관찰됨**  
YOLO11n의 기본 증강 대비 감마+노이즈는 Val의 네 지표가 상승했지만 Test의 네 지표는 하락했다.  
감마만 사용하면 기본 증강 대비 Val Precision·Recall·mAP50은 상승하고 mAP50–95는 하락했다.  
Test Recall은 같고 mAP50은 상승했지만 Precision과 mAP50–95는 하락했다.  
YOLOv8n의 감마 추가는 Val의 네 지표를 모두 낮췄으며 Test에서는 Precision·Recall·mAP50이 상승하고 mAP50–95가 하락했다.  
추가 증강의 일관된 개선은 확인되지 않았고, 노이즈가 하락의 원인이라고 단정하지 않는다.  

**최종 제출 모델 유지**  
기존 1280 YOLO11n / no augmentation 모델의 Test mAP50–95는 0.407136이다.  
이는 이번 640 구조 비교 후보보다 높지만, 최종 모델 유지 근거는 기존 고정 모델과 임계값·오류 분석의 연결이며 사후 Test 최고값 선택으로 표현하지 않는다.  
새 비교의 best.pt에는 기존 FN12·FP8 및 threshold 정책을 그대로 적용했다고 주장하지 않는다.''')
    notebook.cells[31].source = notebook.cells[31].source.replace(
        '모델 주 비교표는 같은 359/77/76 분할의 YOLO11n 설정만 포함한다.',
        '모델 비교는 같은 359/77/76 분할의 완료 실험 11개와 YOLO11n·YOLOv8n 두 구조를 포함한다.  \n   기본 증강·640 조건에서 Val은 YOLOv8n, Test는 YOLO11n이 네 지표 모두 높았다.  \n   감마·노이즈의 일관된 성능 개선은 확인되지 않았다.')
    notebook.cells[32].source = notebook.cells[32].source.replace(
        '수행하지 않은 모델 구조 비교·앙상블·확률보정·불균형 학습을 실적으로 포함하지 않는다.',
        '동일 조건의 YOLOv8n·YOLO11n 구조 비교를 포함한다.  \n앙상블·확률보정·불균형 학습과 반복 seed 검증은 수행하지 않았다.')
    notebook.cells[33].source = notebook.cells[33].source.replace(
        '동일 512장 YOLO11n 설정 비교 / Val77 정책 선택 / Test76 평가',
        '동일 512장 YOLOv8n·YOLO11n 구조 비교 / 증강 비교 / Val77 정책 선택 / Test76 평가').replace(
        '동일512장 다른 구조 모델 비교 없음; F1 정의 분리',
        '구조 비교 조건 일치 확인; 단일 seed; 기존 후속 분석 모델 유지; F1 정의 분리')
    notebook.cells[35].source = notebook.cells[35].source.replace(
        "'동일 분할 6개 설정, 최종 모델 1개, 지표 유효 범위'",
        "f'동일 분할 {len(models)}개 실험, 구조 2종, 조건 일치 비교 2쌍, 최종 모델 1개'").replace(
        'assert len(models) == 6', "assert len(models) == len(comparison_manifest['runs']) == 11\nassert models['model'].nunique() == 2")
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            compile(cell.source, '<notebook>', 'exec')
            cell.outputs = []
            cell.execution_count = None
    nbformat.validate(notebook)
    for folder in (PACKAGE, ROOT):
        nbformat.write(notebook, folder/NAME)
    readme = PACKAGE/'README.md'
    text = readme.read_text(encoding='utf-8').replace(
        '모델 주 비교표에는 동일한 512장 분할의 YOLO11n 실험만 포함하며',
        '모델 비교표에는 동일한 512장 분할의 YOLO11n·YOLOv8n 완료 실험 11개를 포함하며')
    text += '\n## 추가 모델 비교\n\n640 해상도·batch4의 기본 증강 및 감마 조건에서 YOLOv8n·YOLO11n을 비교했습니다. 기존 1280 YOLO11n 최종 모델과 임계값·오류 분석은 유지합니다. 비교 후보의 체크포인트는 분석 표 재생성에 필요하지 않아 추가로 복사하지 않았습니다. 비교 실행 목록과 조건은 `outputs/model_comparison/comparison_manifest.json`에 기록했습니다.\n'
    readme.write_text(text, encoding='utf-8')
    verification_doc = PACKAGE/'검증결과.md'
    text = verification_doc.read_text(encoding='utf-8').replace('R01–R06', 'R01–R11').replace(
        '동일 분할의 모델 6개 설정과 최종 모델 연결', '동일 분할의 실험 11개·모델 구조 2종과 최종 모델 연결')
    # Remove duplicate identical table rows and prose already present in older report.
    lines, seen = [], set()
    for line in text.splitlines():
        if line and (line.startswith('| 원본 선택') or line.startswith('| 과거 XML') or
                     line.startswith('| 제출된 실제') or line.startswith('라벨 출처를 프로젝트')):
            if line in seen:
                continue
            seen.add(line)
        lines.append(line)
    verification_doc.write_text('\n'.join(lines) + '\n\nYOLOv8n·YOLO11n의 기본 증강/감마 비교 2쌍에서 활성 증강 범위와 공통 학습 설정의 일치를 코드로 검증했습니다. 취소된 실행은 제외했습니다.\n', encoding='utf-8')

    os.environ['JUPYTER_RUNTIME_DIR'] = str(ROOT/'.cache/jupyter-runtime')
    os.environ['IPYTHONDIR'] = str(ROOT/'.cache/ipython')
    for folder in (PACKAGE, ROOT):
        started = time.monotonic()
        current = nbformat.read(folder/NAME, as_version=4)
        manager = KernelManager(kernel_name='python3')
        manager._kernel_spec = KernelSpec(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
                                         display_name='Python 3', language='python')
        NotebookClient(current, timeout=180, km=manager,
                       resources={'metadata':{'path':str(folder)}}).execute()
        codes = [cell for cell in current.cells if cell.cell_type == 'code']
        assert len(codes) == 17 and not any(output.output_type == 'error' for cell in codes for output in cell.outputs)
        current.metadata['kamp_analysis'].update(code_cells_executed=17,
            Jupyter_kernel_execution_verified=True, execution_method='Jupyter kernel execution via nbclient',
            compared_model_architectures=['YOLOv8n','YOLO11n'], completed_comparison_runs=11)
        nbformat.write(current, folder/NAME)
        verification = read(folder/'verification.json')
        verification.update(status='passed', verified_at=datetime.now().astimezone().isoformat(),
                            working_directory=str(folder), elapsed_seconds=round(time.monotonic()-started,3),
                            code_cells_passed=17, kernel_execution_verified=True)
        verification['analysis_scope']['matched_split_model_runs'] = 11
        verification['model_architecture_comparison'] = {'architectures':['YOLOv8n','YOLO11n'],
            'matched_pairs':2, 'common_settings_checked':True, 'fixed_downstream_model':final,
            'cancelled_run_excluded':True, 'test_used_for_new_selection':False}
        write(folder/'verification.json', verification)
        print(json.dumps({'notebook':str(folder/NAME),'code_cells_passed':17,'runs':11,'architectures':2}, ensure_ascii=False))

    included = read(PACKAGE/'included_files.json')
    records = {record['path']:record for record in included['files']}
    for file in PACKAGE.rglob('*'):
        if file.is_file() and file.name != 'included_files.json' and '__pycache__' not in file.parts:
            relative = file.relative_to(PACKAGE).as_posix()
            record = records.setdefault(relative, {'path':relative})
            record.update(bytes=file.stat().st_size, sha256=sha(file))
            if relative == NAME:
                record['submission_changes'] = 'Added saved YOLOv8n/YOLO11n and augmentation comparisons; refreshed real Jupyter outputs.'
    included['files'] = sorted(records.values(), key=lambda row:row['path'])
    included['description'] = '512-image analysis package with 11 completed runs, 2 architectures and 2 matched comparison pairs. Original data unchanged.'
    write(PACKAGE/'included_files.json', included)


def _create_comparison_dir():
    folder = PACKAGE/'outputs/model_comparison'
    folder.mkdir(parents=True, exist_ok=True)
    return folder/'comparison_manifest.json'


if __name__ == '__main__':
    main()
