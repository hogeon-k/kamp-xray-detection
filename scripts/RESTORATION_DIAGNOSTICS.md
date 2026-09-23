# 복원 전후 진단

기존 코드 및 데이터는 변경하지 않는다. 기존 결과 파일을 읽는 독립 스크립트이다.

```powershell
python scripts/diagnose_restoration.py
python -m unittest discover -s scripts -p test_diagnose_restoration.py -v
```

이미 출력이 있으면 새 폴더를 지정한다. 덮어쓰기를 허용하는 옵션은 없다.

```powershell
python scripts/diagnose_restoration.py --output outputs/restoration_diagnostics_v2 --diff-gain 8 --padding 16
```

현재 프로젝트의 깨진 `.venv`를 수정하지 않고 실행하는 대체 명령:

```powershell
& 'C:/Users/kang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -c "import numpy,PIL,sys,runpy; sys.path.append(r'C:/workspace/Kamp_Xray/.venv/Lib/site-packages'); sys.path.insert(0,'scripts'); runpy.run_path('scripts/diagnose_restoration.py',run_name='__main__')"
```

입력 기본값:

- `outputs/yolo_poc_20_fake_restoration/manifest.csv`
- `outputs/fake_restoration_debug/per_image.json` 및 저장된 real/fake mask
- manifest에 기록된 원본, 실제 복원, fake 적용 이미지

출력 기본값: `outputs/restoration_diagnostics/`

- `real.csv`: 실제 mask 연결 영역별 지표
- `fake.csv`: fake 배치 영역별 지표
- `summary.csv`: 각 지표의 Real/Fake count, mean, std, min, p05/p25/median/p75/p95/max,
  Fake−Real 평균/중앙값 차이. 전체 및 train-only 비교를 별도로 기록한다.
- `samples.csv`: fake 0개 이미지를 포함한 모든 이미지/group 목록
- `images/{real,fake}/<asset>/`: 전체 before/after/abs_diff_amplified/mask PNG
- `regions/{real,fake}/<asset>/<region>/`: 영역 주변 16px를 포함한 같은 네 가지 PNG
- `metadata.json`: 라이브러리 버전, 입력/기존 코드 해시, 해시 보존 검증 수, diff gain
- `README.md`: 계산 정의 및 해석상 제한

RGB MAE/max 및 changed ratio는 정확한 mask 픽셀에서만 계산한다.
SSIM은 원해상도 grayscale 11×11 Gaussian local SSIM map의 mask 내 평균이다.
Laplacian variance와 Sobel gradient magnitude도 이미지 전체에서 계산한 뒤 mask에서 집계한다.
따라서 가느다란 mask도 주변 문맥을 사용하며 crop 경계로 인한 인공 에지를 만들지 않는다.
면적은 bbox 면적이 아닌 nonzero mask 픽셀 수이다.
변화율은 `(after-before)/before * 100`이며 before=0은 CSV 빈 값으로 남긴다.
diff 이미지는 두 그룹 모두 동일한 고정 배율(기본 8배)로 증폭하고 255로 제한한다.

Real before에는 컬러 마킹이 들어 있다. Real−Fake 차이를 복원 흔적의 강도나
YOLO shortcut learning 여부로 직접 해석하지 않는다.
