"""Synchronize submission materials, verify the notebook and build its ZIP."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import re
import sys
import time
import zipfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'KAMP_Xray_Submission'
NAME = 'KAMP_Xray_Final_Analysis.ipynb'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    manifest = read(PACKAGE/'outputs/model_comparison/comparison_manifest.json')
    final = manifest['fixed_downstream_model']
    final_config = read(PACKAGE/'outputs/yolo_runs'/final/'experiment_config.json')
    train_source = (ROOT/'train_yolo.py').read_text(encoding='utf-8-sig')
    train_source = train_source.replace(
        '"""Run reproducible YOLO11n Baseline experiments on the 512-image TELEA dataset."""',
        '"""YOLOv8n/YOLO11n training reference. CONFIG defaults to the fixed final model.\n\nThis analysis submission does not include training images. To retrain, provide\nthe full prepared TELEA dataset and set CONFIG["data"] to its dataset.yaml.\nSaved per-run settings are listed in experiment_configs.json.\n"""')
    train_source = train_source.replace('ROOT = Path(r"C:\\workspace\\Kamp_Xray")',
                                        'ROOT = Path(__file__).resolve().parent')
    train_source = train_source.replace('DATA_YAML = r"C:\\workspace\\Kamp_Xray\\outputs\\yolo_512_telea\\dataset.yaml"',
                                        'DATA_YAML = str(ROOT / "outputs" / "yolo_512_telea" / "dataset.yaml")')
    # CONFIG is a reviewable reference to the model used by threshold/error analysis.
    for key in ('model','imgsz','batch','lr0','translate','scale','mosaic'):
        value = json.dumps(final_config[key])
        train_source, count = re.subn(rf'(?m)^(    "{key}": ).*?,\s*$',
                                     lambda match:match[1]+value+',', train_source, count=1)
        assert count == 1, key
    for key in ('gamma_p','noise_p'):
        train_source = re.sub(rf'(?m)^(    "{key}": ).*?,\s*$',
                              lambda match:match[1]+'0.0,', train_source, count=1)
    train_source = train_source.replace(
        '# Shared augmentation settings for the YOLOv8n / YOLO11n 640/batch4 comparison.',
        '# Fixed final YOLO11n / 1280 reference: the listed augmentations are disabled.')
    compile(train_source, 'train_yolo.py', 'exec')
    (PACKAGE/'train_yolo.py').write_text(train_source, encoding='utf-8')
    configs = []
    lifecycle = {'run_name','start_time','end_time','duration_seconds'}
    for run in manifest['runs']:
        saved = read(PACKAGE/'outputs/yolo_runs'/run/'experiment_config.json')
        config = {key:value for key,value in saved.items() if key not in lifecycle}
        config['data'] = 'outputs/yolo_512_telea/dataset.yaml'
        config['project'] = 'outputs/yolo_runs'
        for key in ('gamma_p','noise_p'):
            config.setdefault(key, 0.0)
        configs.append({'run_name':run,'fixed_downstream_model':run==final,
                        'source':f'outputs/yolo_runs/{run}/experiment_config.json',
                        'CONFIG':config})
    write(PACKAGE/'experiment_configs.json', {
        'purpose':'Portable reference settings for the 11 saved experiments. Historical source JSONs are preserved.',
        'retraining_requires_full_images':True,'runs':configs})
    (PACKAGE/'README.md').write_text('''# KAMP X-ray 제출 자료

**제출 방법:** `KAMP_Xray_Submission` 폴더 전체 또는 같은 이름의 ZIP 파일 하나를 제출합니다.
폴더 구조를 유지해야 노트북의 표와 그림을 다시 실행할 수 있습니다.
ZIP을 제출할 경우 먼저 압축을 풀고 안의 노트북을 엽니다.

## 먼저 볼 파일

`KAMP_Xray_Final_Analysis.ipynb`에 설명, 비교표, 그림과 실행 결과가 저장되어 있습니다.
`검증결과.md`에는 실행 검증 결과를 정리했습니다.

## 분석 범위와 비교 결과

고유 이미지 512장·결함 bbox 1,163개를 분석했습니다.
Train 359장/828개, Validation 77장/175개, Test 76장/160개로 구분합니다.
같은 분할의 완료 실험 11개는 YOLO11n 9개와 YOLOv8n 2개이며, 모델 구조는 2종입니다.

640 해상도·batch4에서 기본 증강과 감마 조건별로 두 구조를 비교했습니다.
기본 증강 조건의 Validation은 YOLOv8n, Test는 YOLO11n이 네 지표 모두 높았습니다.
감마·노이즈의 일관된 성능 개선은 확인되지 않았습니다.
기존 1280 YOLO11n / batch4 / lr0.001 / 증강 비활성 / seed42 최종 모델을 유지했습니다.
임계값 정책과 FN 12건·FP 8건 오류 분석은 이 고정 최종 모델의 결과입니다.
새 비교 후보의 결과와 기존 오류 분석을 혼합하지 않습니다.

기존 20장 PoC는 전처리·shortcut 점검의 보조 근거입니다.
512장 모두 결함 라벨이 있어 실제 생산 불량률이나 양품 오경보율을 측정한 자료는 아닙니다.

## 노트북 실행

Python 3.12 환경을 권장합니다.
압축을 푼 제출 폴더에서 다음 명령을 실행합니다.

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements-notebook.txt
.\\.venv\\Scripts\\python.exe -m notebook KAMP_Xray_Final_Analysis.ipynb
```

Python 3 커널을 선택하고 **Restart Kernel and Run All Cells**를 실행합니다.
VS Code에서도 위 가상환경의 Python을 커널로 선택할 수 있습니다.
macOS/Linux에서는 `.venv/bin/python`을 사용합니다.
최초 패키지 설치에는 인터넷 연결이 필요합니다.
노트북 자체는 저장 결과를 읽으며 학습·신규 추론·데이터 다운로드를 수행하지 않습니다.

## 파일 구성

| 파일·폴더 | 용도 |
|---|---|
| `KAMP_Xray_Final_Analysis.ipynb` | 출력이 저장된 분석 노트북 |
| `outputs/` | 표·그림·학습 로그·평가 결과·기존 검증용 모델 |
| `outputs/model_comparison/comparison_manifest.json` | 완료 실험 11개와 동일 조건 비교 쌍 |
| `outputs/submission_label_audit/` | 원본 라벨 출처 감사 기록 |
| `outputs/yolo_512_telea/labels/` | 해시·형식·bbox 개수를 재검증하는 TXT 512개 |
| `experiment_configs.json` | 실험별 설정 11개를 상대경로로 정리한 참고 자료 |
| `train_yolo.py` | 감마·노이즈 구현을 포함한 학습 참고 코드; 기본 CONFIG는 고정 최종 모델 |
| `requirements-notebook.txt` | 노트북 실행에 필요한 설치 목록 |
| `requirements.txt` | 기존 학습 환경의 참고 설치 목록 |
| `verification.json`, `검증결과.md` | 실제 Jupyter 실행 및 코드 검증 기록 |
| `included_files.json` | 포함 파일의 크기·SHA-256 목록 |

## 재현 범위

원본 선택 라벨과 최종 라벨은 512/512 일치했습니다.
과거 XML–YOLO 차이 이력 23장도 최종 선택 canonical과 일치하며, 노트북에서 제출 TXT를 다시 검사합니다.

이 제출 자료는 저장 결과를 이용한 분석 재현용입니다.
원본 영상, 학습용 TELEA 영상 512장, 가상환경과 캐시는 포함하지 않았습니다.
재학습하려면 전체 전처리 데이터셋을 준비하고 `train_yolo.py`의 `CONFIG["data"]`를 해당 dataset.yaml로 설정해야 합니다.
실험별 원본 설정 JSON의 과거 절대경로는 출처 기록으로 유지했으며, 읽기용 노트북은 제출 폴더의 상대경로를 사용합니다.
새 비교 후보의 체크포인트는 표 재생성에 필요하지 않아 추가하지 않았습니다.
원본 영상과 전체 진단 후보 폴더를 별도로 추가할 필요는 없습니다.
''', encoding='utf-8')

    # Run the submitted trainer's configuration helpers only; never invoke training.
    namespace = {'__name__':'submission_config_check','__file__':str(PACKAGE/'train_yolo.py')}
    exec(compile(train_source, str(PACKAGE/'train_yolo.py'), 'exec'), namespace)
    config = namespace['CONFIG']
    namespace['validate_xray_augmentation'](config)
    assert namespace['build_run_name'](config) == final
    assert namespace['ROOT'] == PACKAGE
    assert all(config[key] == final_config[key] for key in ('model','imgsz','batch','lr0','translate','scale','mosaic'))

    os.environ['JUPYTER_RUNTIME_DIR'] = str(ROOT/'.cache/jupyter-runtime')
    os.environ['IPYTHONDIR'] = str(ROOT/'.cache/ipython')
    started = time.monotonic()
    notebook = nbformat.read(PACKAGE/NAME, as_version=4)
    manager = KernelManager(kernel_name='python3')
    manager._kernel_spec = KernelSpec(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
                                     display_name='Python 3',language='python')
    NotebookClient(notebook, timeout=180, km=manager, resources={'metadata':{'path':str(PACKAGE)}}).execute()
    codes = [cell for cell in notebook.cells if cell.cell_type=='code']
    assert len(codes)==17 and not any(output.output_type=='error' for cell in codes for output in cell.outputs)
    nbformat.validate(notebook)
    nbformat.write(notebook, PACKAGE/NAME)
    verification = read(PACKAGE/'verification.json')
    verification.update(verified_at=datetime.now().astimezone().isoformat(),
                        elapsed_seconds=round(time.monotonic()-started,3))
    verification['submission_materials'] = {
        'README_synchronized':True, 'portable_training_reference':True,
        'training_CONFIG_matches_fixed_final_model':True, 'saved_experiment_configs':11,
        'gamma_noise_implementation_included':True, 'training_started':False,
        'notebook_code_cells_passed':17}
    write(PACKAGE/'verification.json',verification)
    report = PACKAGE/'검증결과.md'
    report.write_text(report.read_text(encoding='utf-8') +
        '\n## 함께 제출하는 자료 정리\n\n학습 참고 코드를 최신 감마·노이즈 구현으로 갱신하고 경로를 제출 폴더 기준으로 변경했습니다. 기본 CONFIG가 기존 1280 YOLO11n 최종 모델과 일치하는지 검증했습니다. 실험별 설정 11개를 `experiment_configs.json`에 정리했고, README와 파일 목록을 갱신한 뒤 제출 노트북의 코드 셀 17개를 다시 실행해 모두 통과했습니다. 학습은 시작하지 않았습니다.\n', encoding='utf-8')
    inventory = read(PACKAGE/'included_files.json')
    prior = {row['path']:row for row in inventory['files']}
    files = sorted(p for p in PACKAGE.rglob('*') if p.is_file() and p.name!='included_files.json')
    records = []
    for file in files:
        relative = file.relative_to(PACKAGE).as_posix()
        row = prior.get(relative, {'path':relative})
        row.update(bytes=file.stat().st_size,sha256=digest(file))
        if relative=='train_yolo.py':
            row['submission_changes']='Latest gamma/noise implementation; portable paths; CONFIG defaults to the fixed final 1280 YOLO11n model.'
        records.append(row)
    inventory.update(files=records,updated_at=datetime.now().astimezone().isoformat(),
                     file_count_excluding_inventory=len(records))
    write(PACKAGE/'included_files.json',inventory)
    zip_path = ROOT/'KAMP_Xray_Submission.zip'
    all_files = sorted(p for p in PACKAGE.rglob('*') if p.is_file())
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for file in all_files:
            archive.write(file, file.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {file.relative_to(ROOT).as_posix() for file in all_files}
        for file in all_files:
            with archive.open(file.relative_to(ROOT).as_posix()) as stream:
                assert hashlib.file_digest(stream,'sha256').hexdigest() == digest(file)
    print(json.dumps({'zip':str(zip_path),'files':len(all_files),'size_mb':round(zip_path.stat().st_size/1024**2,2),
                      'notebook_code_cells_passed':17,'archive_all_file_hashes_verified':True},ensure_ascii=False))


if __name__=='__main__':
    main()
