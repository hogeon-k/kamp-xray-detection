# After-only Restoration Artifact Similarity Diagnostic

기존 복원·fake augmentation 코드, 데이터셋, 이전 출력은 변경하지 않는다.
두 그룹 모두 **fake 적용까지 끝난 동일 최종 이미지**에서 grayscale 특징을 계산한다.
원본 컬러 마킹 이미지는 무변경 확인을 위해 해시만 읽으며 픽셀 분석에 사용하지 않는다.
기존 진단 모듈의 이미지/mask 로딩, 영역 분리, CSV 저장 함수를 재사용한다.

## 실행

검증 환경은 Python 3.12이며 numpy, Pillow, OpenCV, scipy, scikit-learn, matplotlib을 사용한다.
정상 Python 환경에서 추가 통계 패키지를 설치하고 실행한다.

```powershell
python -m pip install -r scripts/requirements-after-diagnostic.txt
python scripts/diagnose_after_restoration_similarity.py --output outputs/restoration_after_similarity --ring-width 5
python -m unittest discover -s scripts -p test_after_restoration_similarity.py -v
```

기존 출력 폴더가 있으면 덮어쓰지 않고 중단한다. 재실행에는 새 `--output`을 지정한다.
`--seed` 생략 시 기존 metadata의 seed를 상속한다(현재 42).
`--skip-classifier`로 분류기만 생략할 수 있다. `--padding`은 시각화 여백이며 지표에는 영향이 없다.
`--manifest`, `--debug-dir`, `--prior-diagnostics`로 다른 일치하는 입력 세트를 지정할 수 있다.

현재 머신에서 기존 가상환경을 수정하지 않고 검증한 실행 명령:

```powershell
$env:MPLCONFIGDIR = 'C:/workspace/Kamp_Xray/.cache/after_similarity_mpl'
& 'C:/Users/kang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -c "import sys,runpy; sys.path.insert(0,r'C:/workspace/Kamp_Xray/.cache/after_similarity_runtime'); sys.path.append(r'C:/workspace/Kamp_Xray/.venv/Lib/site-packages'); sys.path.insert(0,'scripts'); runpy.run_path('scripts/diagnose_after_restoration_similarity.py',run_name='__main__')"
```

통계·시각화 패키지는 `.cache/after_similarity_runtime/`에 별도로 설치했다.

## 결과 해석

- `real_after_features.csv`, `fake_after_features.csv`: 영역별 전체 특징 및 정의 불가 항목.
- `comparison_summary.csv`: all/train/matched_images 세 범위의 그룹별 count, mean,
  std, median, Q1/Q3, min/max, Mann–Whitney U 및 p, BH 보정 p, Cliff's delta, KS 거리.
  `key_metric=True`가 핵심 6개 지표다. delta 양수는 Real이 더 큰 경향이다.
- `matched_images`: Real과 Fake가 모두 존재하는 동일 원본 이미지 집합으로 한정한다.
  이미지 구성 차이를 줄이지만 서로 다른 위치/실제 이물질 주변 문맥의 차이는 남는다.
- ratio 분모가 0이거나 ring이 비어 있으면 빈 CSV 값/JSON null로 기록한다.
  실제 NaN/inf 출력은 금지한다. 공란은 수학적으로 정의되지 않는 지표이며 오류와 구분한다.
- `classifier_diagnostic.json`: 그룹 교차검증 결과. full_features 및 without_shape,
  all 및 matched_images 총 4개 분석이다. fake가 양성 클래스다.
- `classifier_oof_predictions.csv`: 모든 원본 이미지 그룹을 분리해 얻은 out-of-fold 확률.
- `distributions/`: 핵심 6개 특징의 boxplot과 개별 점, all/matched_images를 나란히 표시.
- `regions/`: 모든 34/19개 영역의 복원된 patch, mask/ring overlay, binary mask/ring,
  확대 panel. 분석 이미지에 표시를 그려 저장하거나 덮어쓰지 않는다.
- `metadata.json`: seed/버전, 0분모 등 정의 불가 항목 수, mask/ring 검사 및 입력 파일 해시.
- 출력 `README.md`: intensity/texture/gradient/boundary와 통계의 정확한 계산 정의.

분류기는 고정 Logistic Regression이며 튜닝하지 않는다. 원본 이미지 ID를 group으로
StratifiedGroupKFold를 적용하며 imputation/scaling도 train fold에서만 fit한다.
작은 표본의 성능은 참고용이다. AUC가 낮다고 완전 동일하거나 동등하다고 결론내리지 않는다.
서로 같은 이미지의 영역은 상관되어 있으므로 region 단위 U test의 p-value는 탐색용이다.
BH 보정도 이런 상관 문제를 해결하지 않는다. 효과 크기와 분포를 함께 확인한다.
Cliff's delta가 0에 가까워도 분산/형태는 다를 수 있으므로 KS 거리도 함께 기록한다.

## 검증

6개 테스트: 경계 ring, 경계 불연속의 알려진 값, 빈 ring/0분모, delta 부호 및 동률,
그룹 누수·분류기 재현성, 실제 데이터 두 번 실행의 특징/통계/OOF 결과 바이트 일치.
통합 테스트는 before 이미지가 로드되지 않는지도 검사한다.
실행 시 전체 원본 데이터셋, 기존 복원 결과, 두 학습 데이터셋, 이전 진단 출력 및
기존 코드의 실행 전후 SHA-256을 검사한다(Python bytecode cache 제외).
