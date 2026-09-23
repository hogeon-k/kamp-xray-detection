# Random Fake Restoration Augmentation

## 확인한 기존 파이프라인

- `outputs/inpainting_poc_20/run_poc.py`의 `main()`은 palette mask를 저장하고
  `cv2.inpaint(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, radius, flag)`로 복원한다.
- `scripts/create_yolo_poc_20.py`는 TELEA / dilation=0 / radius=3 결과만 선택하여
  train 14장, val 3장, test 3장으로 복사한다.
- 기존 복원, 데이터 분할, 학습 스크립트는 변경하지 않았다.

## 실행

프로젝트 루트에서 numpy, Pillow, opencv-python이 설치된 Python을 사용한다.

```powershell
python scripts/create_fake_restoration_dataset.py --seed 42
python -m unittest discover -s scripts -p test_fake_restoration.py -v
```

현재 `.venv`가 참조하는 Python 3.11 실행 파일이 없는 환경에서 실제 검증에 사용한 명령:

```powershell
& 'C:/Users/kang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -c "import numpy,PIL,sys,runpy; sys.path.append(r'C:/workspace/Kamp_Xray/.venv/Lib/site-packages'); sys.path.insert(0,'scripts'); runpy.run_path('scripts/create_fake_restoration_dataset.py',run_name='__main__')"
```

기존 출력 경로가 있으면 덮어쓰지 않고 중단한다. 재실행은 새 경로를 지정한다.

```powershell
python scripts/create_fake_restoration_dataset.py --seed 42 --output outputs/fake_seed42_v2 --debug-dir outputs/fake_debug_v2
python scripts/create_fake_restoration_dataset.py --seed 7 --scale-range 0.8 1.2 --safety-margin 10 --output outputs/fake_seed7 --debug-dir outputs/fake_debug_seed7
```

생성 데이터로 학습하려면 기존 `train_yolo.py`의 CONFIG에서 `data`를
`C:/workspace/Kamp_Xray/outputs/yolo_poc_20_fake_restoration/dataset.yaml`로,
`preprocessing`을 `telea_fake_seed42` 등으로 지정한 뒤 실행한다.
기본 학습 대상은 자동 변경하지 않으며 이 작업에서는 학습을 실행하지 않았다.

## 구현 및 안전 조건

- `mask_templates`: 해당 이미지의 저장된 실제 mask에서 8-connected component를 추출한다.
  mask의 구멍과 선 두께를 보존하며 다른 split의 mask는 사용하지 않는다.
- `augment_fake_restoration`: train에만 0/1/2/3개를 25/35/25/15%로 추첨한다.
  기본 배율은 1.0이다. 옵션으로만 0.8~1.2배 nearest-neighbor resize를 허용한다.
- GT bbox + 기본 10px margin, 실제 mask, 이미 배치한 fake mask와 겹치는
  후보는 거절한다. 구멍 내부에 GT가 들어가지 않도록 전체 사각 footprint까지 검사한다.
- 각 fake 영역마다 최대 500회 후보를 시도한다. 불가능하면 건너뛰고 기록한다.
  따라서 작은 표본의 분포는 지정 확률과 정확히 같지 않으며 배치 실패 시에도 달라질 수 있다.
- `inpaint_rgb`: 실제 복원과 같은 OpenCV 호출을 사용한다. 기존 결과 파일명에서
  TELEA/NS, radius, dilation을 읽고 해당 dilation의 저장 mask를 재사용한다.
  임의로 방법을 바꾸지 않으며 지원하지 않는 방법은 오류 처리한다.
- `generate`: 입력 SHA-256, GT 개수, 실제 복원 재현 결과의 픽셀 일치를 먼저 확인한다.
  GT bbox 안에는 fake mask가 들어가지 않으며 fake label을 만들지 않는다.
  모든 label과 val/test 이미지는 바이트 그대로 복사한다.
- augmentation은 별도 데이터셋에 한 번 생성하는 offline 방식이다.
  동일 입력, seed, 옵션 및 라이브러리 버전에서 동일 결과를 생성한다.
- 이물질 제외는 GT의 완전성을 전제로 한다. 미라벨 이물질을 판별하는 기능은 없다.

## 결과

- `outputs/yolo_poc_20_fake_restoration/`: 새 images, labels, manifest.csv, dataset.yaml
- `outputs/fake_restoration_debug/*_debug.png`: 대표 20장, cyan=GT/red=실제/green=fake
  (val/test 6장에서는 fake=0임을 확인할 수 있다.)
- `outputs/fake_restoration_debug/*_fake_mask.png`: 이미지별 원해상도 fake mask
- `summary.json`: 전체/학습 이미지 수, 적용 이미지 수, 요청/적용 분포, overlap,
  seed, 옵션, 라이브러리 버전, 원본 무변경 해시 검증 수
- `per_image.csv`, `per_image.json`: 이미지별 요청/적용/누락 개수, 거절 횟수,
  실제 적용 overlap 건수 및 픽셀 수, template ID, 위치, 크기, 배율, 방법과 파라미터

overlap 건수는 실제 적용된 fake 영역의 검증 결과이며, 후보 거절 횟수와 구분한다.
원본 또는 기존 출력과 겹치는 경로, 기존 출력 디렉터리에 대한 덮어쓰기는 거부한다.
