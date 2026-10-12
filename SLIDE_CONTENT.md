# 실제 슬라이드용 콘텐츠

10장 구성. 각 대본은 20~30초를 목표로 작성했으며 리허설로 확인한다. 아래 bullet은 전달할 정보 목록이며 PPTX는 표·그림에 포함된 중복 문장을 줄여 배치하였다.

## 1. KAMP X-ray 이물질 검출과 오류분석

**핵심 메시지:** 컬러 주석을 처리한 X-ray 영상에서 검출 성능과 실패 조건을 함께 분석하였다.

**본문 bullet:**

- 제6회 K-인공지능 제조데이터 분석 경진대회
- 검출 성능과 미검출 위험 조건 분석

**실제 이미지:**

- [001_20200814_163105(1).png](C:/workspace/Kamp_Xray/outputs/yolo_512_telea/images/train/001_20200814_163105(1).png)

**배치:** 짙은 남색 표지. 좌측 제목, 우측 실제 TELEA 영상. 팀명·발표자명은 확인 후 추가한다.

**발표 대본(20~30초):**

이번 발표는 KAMP X-ray 영상의 이물질 검출과 오류분석 결과입니다. 영상에 포함된 컬러 주석을 먼저 처리하고, 두 모델 구조를 비교한 뒤 임계값을 Validation에서 고정했습니다. 최종 성능뿐 아니라 어떤 객체를 놓쳤는지, 복원 흔적에 모델이 반응하는지도 분석했습니다.

**원본 근거:**

- [FINAL_REPORT_DRAFT.md](C:/workspace/Kamp_Xray/FINAL_REPORT_DRAFT.md)
- [KAMP_Xray_Final_Analysis.ipynb](C:/workspace/Kamp_Xray/KAMP_Xray_Final_Analysis.ipynb)

## 2. 문제 정의와 데이터

**핵심 메시지:** 동일한 그룹 분할로 이물질의 위치 검출을 평가하였다.

**본문 bullet:**

- 이미지 512장 / bbox 1,163개
- Train / Val / Test = 359 / 77 / 76
- 원본 해시·라벨 출처 연결, 그룹 단위 분할

**슬라이드 표:**

| 분할 | 이미지 | GT bbox |
| --- | --- | --- |
| Train | 359 | 828 |
| Validation | 77 | 175 |
| Test | 76 | 160 |

**실제 이미지:**

- 이미지 없음. 편집 가능한 표 또는 결론 텍스트로 대체한다.

**배치:** 상단에 512와 1,163을 크게 표시. 아래 편집 가능한 분할표. 세부 수치를 문장으로 반복하지 않는다.

**발표 대본(20~30초):**

분석 대상은 이미지 512장과 이물질 bbox 1,163개입니다. 학습 359장, 검증 77장, 테스트 76장으로 나눴고, 테스트 정답은 160개입니다. 원본 해시와 라벨 출처를 연결하고 그룹 단위로 분할했습니다. 정상 제품 판별이 아니라, 주어진 영상에서 이물질 위치를 검출하는 과제입니다.

**원본 근거:**

- [manifest.csv](C:/workspace/Kamp_Xray/outputs/yolo_512_telea/manifest.csv)
- [split_manifest.csv](C:/workspace/Kamp_Xray/outputs/clean_512_split/split_manifest.csv)

## 3. 컬러 bbox와 TELEA 전처리

**핵심 메시지:** 컬러 주석선은 제거하고 이물질의 영상 신호는 보존하도록 복원하였다.

**본문 bullet:**

- 컬러 표시 mask만 복원
- 20장 PoC에서 TELEA·NS 보존성 비교
- 최종 적용: TELEA radius 3 / dilation 0

**실제 이미지:**

- [01_4184354a2a3c_comparison.png](C:/workspace/Kamp_Xray/outputs/inpainting_poc_20/figures/01_4184354a2a3c_comparison.png)

**배치:** 동일 비교 이미지의 원본·mask·TELEA 세 패널을 슬라이드에서 확대 배치. 별도 그림 생성 없이 원본 이미지 crop 사용.

**보이는 주의 문구:** 20장 PoC 예시. 복원 보존성 비교이며 탐지 성능 향상률이 아님

**발표 대본(20~30초):**

원본에는 이물질 주변에 컬러 bbox가 그려져 있어 모델이 표시선에 의존할 가능성이 있었습니다. 표시 색상을 mask로 추출하고 TELEA로 해당 영역을 복원했습니다. 20장 PoC에서 NS와 보존성을 비교해 radius 3, dilation 0을 적용했습니다. 이 비교는 복원 보존성 평가이며 탐지 성능 향상을 직접 증명한 결과는 아닙니다.

**원본 근거:**

- [method_summary.csv](C:/workspace/Kamp_Xray/outputs/inpainting_poc_20/method_summary.csv)
- [run_poc.py](C:/workspace/Kamp_Xray/outputs/inpainting_poc_20/run_poc.py)
- [manifest.csv](C:/workspace/Kamp_Xray/outputs/yolo_512_telea/manifest.csv)

## 4. YOLO11n과 YOLOv8n 모델 비교

**핵심 메시지:** 같은 학습 조건에서 두 모델 구조를 비교하고 기존 R04를 후속 분석 기준으로 유지하였다.

**본문 bullet:**

- 구조 비교: 640 / batch 4 / AdamW / 기본 증강
- 완료 11run, 모델 구조 2종
- 최종 R04: YOLO11n 1280·무증강, Val F1 최고와 구분

**슬라이드 표:**

| 모델·실험 | Val F1(P/R) | Val mAP50 |
| --- | --- | --- |
| YOLO11n R07 | 95.95% | 93.21% |
| YOLOv8n R11 | 97.69% | 97.14% |

**실제 이미지:**

- 이미지 없음. 편집 가능한 표 또는 결론 텍스트로 대체한다.

**배치:** 동일 조건 R07·R11의 2행 비교표. 하단에 최종 R04 설정과 선정 한계를 분리해 표시. 학습곡선은 상세자료로 남긴다.

**발표 대본(20~30초):**

두 구조는 입력 640, batch 4, AdamW와 기본 증강을 맞춰 비교했습니다. Validation에서는 YOLOv8n R11의 F1이 더 높았습니다. 최종 오류분석은 기존 YOLO11n 1280 무증강 R04를 유지했습니다. 따라서 최종 모델을 Validation F1 최고 모델이라고 주장하지 않습니다. 선정 방식은 기존 기준 모델의 유지입니다.

**원본 근거:**

- [comparison_manifest.json](C:/workspace/Kamp_Xray/outputs/model_comparison/comparison_manifest.json)
- [summary.json](C:/workspace/Kamp_Xray/outputs/yolo_runs/telea_512_yolo11n_e200_img640_b4_adamw_lr0.001_wd0.0005_aug_tr0.005_sc0.2_mos0.2_s42/summary.json)
- [summary.json](C:/workspace/Kamp_Xray/outputs/yolo_runs/telea_512_yolov8n_e200_img640_b4_adamw_lr0.001_wd0.0005_aug_tr0.005_sc0.2_mos0.2_s42/summary.json)
- [summary.json](C:/workspace/Kamp_Xray/outputs/yolo_runs/telea_512_yolo11n_e200_img1280_b4_adamw_lr0.001_wd0.0005_noaug_s42/summary.json)

## 5. Validation에서 threshold 고정

**핵심 메시지:** 미검출 감소와 추가 FP 부담을 함께 확인해 0.20 / 0.25 정책을 고정하였다.

**본문 bullet:**

- 운영 하한 0.20 / 고신뢰 기준 0.25
- Val 0.25 대비 FN 1개 감소, FP 1개 증가
- Test로 threshold 재선정하지 않음

**슬라이드 표:**

| Val threshold | FN | FP |
| --- | --- | --- |
| 0.25 | 5 | 11 |
| 0.20 | 4 | 12 |
| 0.02 | 3 | 117 |

**실제 이미지:**

- [val_threshold_curve.png](C:/workspace/Kamp_Xray/outputs/safety_threshold_policy/visualizations/val_threshold_curve.png)

**배치:** 왼쪽 기존 Validation curve, 오른쪽 3행 FN·FP 비교표. 0.20은 F1 최적값이 아니라 recall 중심 하한임을 짧게 표시.

**보이는 주의 문구:** Val 77장·GT 175개, 객체 매칭 IoU ≥ 0.50

**발표 대본(20~30초):**

임계값은 Validation 77장에서 선택했습니다. 하한을 0.25에서 0.20으로 낮추면 FN이 한 개 줄고 FP는 한 개 늘었습니다. 반면 0.02에서는 FP가 117개까지 증가했습니다. 운영 하한 0.20과 고신뢰 기준 0.25를 고정했고, 테스트로 다시 조정하지 않았습니다. 0.20은 F1 최고값이 아니라 재현율 중심의 선택입니다.

**원본 근거:**

- [val_threshold_sweep.csv](C:/workspace/Kamp_Xray/outputs/safety_threshold_policy/val_threshold_sweep.csv)
- [locked_policy.json](C:/workspace/Kamp_Xray/outputs/safety_threshold_policy/locked_policy.json)

## 6. 최종 Test 성능

**핵심 메시지:** 고정 운영 조건에서 F1-score 93.67%를 확인하였다.

**본문 bullet:**

- TP 148 / FP 8 / FN 12
- Precision 94.87% / Recall 92.50% / F1-score 93.67%
- confidence ≥ 0.20 / GT IoU ≥ 0.50

**실제 이미지:**

- [official_test_confusion.png](C:/workspace/Kamp_Xray/submission_audit/official_test_confusion.png)

**배치:** 왼쪽 3개 지표를 크게, 오른쪽 새 공식 confusion. TP·FP·FN은 그림에서 읽고 문장으로 중복하지 않는다.

**보이는 주의 문구:** Test 76장·GT 160개, NMS IoU 0.70. 정상 음성 평가 및 TN 없음

**발표 대본(20~30초):**

고정한 정책을 테스트 76장에 적용했습니다. 정답 160개 중 TP는 148개, FN은 12개였고 FP는 8개였습니다. Precision은 94.87%, Recall은 92.50%, F1은 93.67%입니다. 오른쪽 그림은 같은 예측 캐시와 IoU 기준으로 다시 만든 공식 confusion입니다. 정상 제품의 오경보율이나 TN은 이 자료로 산출하지 않았습니다.

**원본 근거:**

- [official_test_confusion.json](C:/workspace/Kamp_Xray/submission_audit/official_test_confusion.json)
- [summary.json](C:/workspace/Kamp_Xray/outputs/error_analysis/summary/summary.json)

## 7. 소형·저대비 조건과 위치 오차

**핵심 메시지:** 미검출은 점수 부족과 위치 오차로 구분되며 소형·저대비군에서 높은 FN 비율을 관찰하였다.

**본문 bullet:**

- 소형 객체 FN 17.07% (7/41)
- 저대비 객체 FN 11.67% (7/60)
- 위치 오차 8 / 낮은 점수·위치 성공 3 / 낮은 점수·위치 실패 1

**실제 이미지:**

- [U0138_FN_gt0.png](C:/workspace/Kamp_Xray/outputs/error_analysis/fn/U0138_FN_gt0.png)
- [U0371_FP_pred0.png](C:/workspace/Kamp_Xray/outputs/error_analysis/fp/U0371_FP_pred0.png)

**배치:** 좌측에 조건별 FN 비율과 분모를 크게. 우측에 U0138 FN·U0371 FP 원본의 확대 패널만 crop해 나란히 제시.

**보이는 주의 문구:** 소형 bbox ≤ 81px². 저대비 기준은 Train에서 고정. 조건 중복이 있어 인과 해석 불가

**발표 대본(20~30초):**

소형 객체에서는 41개 중 7개를 놓쳐 FN 비율이 17.07%, 저대비 객체에서는 60개 중 7개로 11.67%였습니다. 두 조건은 겹치므로 독립적인 원인으로 해석하지 않았습니다. U0138은 위치가 맞아도 점수가 하한보다 낮았고, U0371은 예측 bbox가 커 IoU가 부족했습니다. 점수와 위치 오차를 구분해 대응해야 합니다.

**원본 근거:**

- [gt_condition_rates.csv](C:/workspace/Kamp_Xray/outputs/error_analysis/summary/gt_condition_rates.csv)
- [cause_frequency.csv](C:/workspace/Kamp_Xray/outputs/error_analysis/summary/cause_frequency.csv)
- [fn_candidate_diagnostics.csv](C:/workspace/Kamp_Xray/outputs/error_analysis/summary/fn_candidate_diagnostics.csv)
- [summary.json](C:/workspace/Kamp_Xray/outputs/error_analysis/summary/summary.json)

## 8. 창의성: fake-mark 역검증

**핵심 메시지:** 의도적인 표시·복원 흔적을 다시 만들어 모델의 반응을 비교하였다.

**본문 bullet:**

- 컬러 표시 제거 → TELEA 복원 → 의도적 fake-mark 재생성 → 모델 반응 검증
- 20장 PoC, 34pair / holdout 10pair
- 두 PoC 모델의 새 fake FP 0, GT loss/gain 0

**실제 이미지:**

- [overlay.png](C:/workspace/Kamp_Xray/outputs/shortcut_diagnostic_gpu/cases/E_largest_delta_pair_0002/overlay.png)
- [pair_0002.png](C:/workspace/Kamp_Xray/outputs/shortcut_diagnostic_dataset_gpu/masks/pair_0002.png)

**배치:** 상단 4단계 편집 가능한 흐름. 중앙에 실제 baseline original/fake 두 패널과 해당 fake mask. 하단에 PoC 결과와 적용 한계.

**보이는 주의 문구:** fake-mark는 GT 밖으로 표시 mask를 재배치·재복원한 흔적. 최종 모델 shortcut 부재 증명은 아님

**발표 대본(20~30초):**

복원 흔적을 모델이 이물질로 오인하는지 역으로 검증했습니다. 컬러 표시를 제거하고 TELEA로 복원한 뒤, 표시 mask를 정답 밖으로 재배치해 의도적인 fake-mark 복원 흔적을 만들고 모델 반응을 비교했습니다. 20장 PoC의 34쌍에서 새 fake FP와 GT 검출 손실·획득은 없었습니다. 다만 최종 512장 모델의 shortcut 부재까지 증명한 결과는 아닙니다.

**원본 근거:**

- [fake_restoration.py](C:/workspace/Kamp_Xray/scripts/fake_restoration.py)
- [diagnose_context_matched_restoration.py](C:/workspace/Kamp_Xray/scripts/diagnose_context_matched_restoration.py)
- [metadata.csv](C:/workspace/Kamp_Xray/outputs/shortcut_diagnostic_dataset_gpu/metadata.csv)
- [summary.json](C:/workspace/Kamp_Xray/outputs/shortcut_diagnostic_gpu/summary.json)

## 9. 현장 활용: 검사자 재검 지원

**핵심 메시지:** AI는 후보 위치와 점수를 제시하고 검사자가 최종 판정을 수행하도록 제안한다.

**본문 bullet:**

- 고신뢰 후보: 우선 확인
- 낮은 점수·무검출: 확대 확인과 표본 재검
- 제안 단계: 정상 제품·무주석 영상 검증 필요

**슬라이드 표:**

| 영상 후보 조건 | 검사자 조치 |
| --- | --- |
| confidence ≥ 0.25 | 후보 위치 우선 확인 |
| 0.20 ≤ confidence < 0.25만 존재 | 재검 후보 확인 |
| confidence ≥ 0.20 후보 없음 | 정상 보증 없이 표본 재검 |

**실제 이미지:**

- 이미지 없음. 편집 가능한 표 또는 결론 텍스트로 대체한다.

**배치:** 3행의 편집 가능한 운영 제안표. 원본·bbox·점수·검사자 판정 기록을 짧게 설명. 실물 생산라인 이미지는 사용하지 않는다.

**보이는 주의 문구:** 고신뢰 후보 우선 routing. 생산 설비 연동·자동 배출·시간 절감 효과 미검증

**발표 대본(20~30초):**

현장에서는 AI가 후보 위치와 점수를 제시하고 검사자가 원본과 확대 영상을 확인하도록 제안합니다. 고신뢰 후보는 우선 확인하고, 낮은 점수 후보와 무검출 영상도 재검 대상에 포함합니다. 실제 생산 설비 연동과 자동 배출은 구현하지 않았으며, 정상 제품과 무주석 영상에서 오경보와 검사 부담을 별도로 검증해야 합니다.

**원본 근거:**

- [FINAL_REPORT_DRAFT.md](C:/workspace/Kamp_Xray/FINAL_REPORT_DRAFT.md)
- [locked_policy.json](C:/workspace/Kamp_Xray/outputs/safety_threshold_policy/locked_policy.json)

## 10. 결론과 후속 검증

**핵심 메시지:** 고정 조건의 검출 성능과 실패 사례를 근거로 검사 지원의 범위를 제시하였다.

**본문 bullet:**

- 공식 운영 Test와 오류분석의 평가 조건 통일
- 효과 없는 보조검사는 운영 채택에서 제외
- 후속 검증: 정상·무주석 영상, 반복 seed, 계획서 대응

**실제 이미지:**

- 이미지 없음. 편집 가능한 표 또는 결론 텍스트로 대체한다.

**배치:** 핵심 결론 1문장과 3개 짧은 항목. 앞 슬라이드 성능 수치를 반복하지 않는다.

**발표 대본(20~30초):**

이번 분석은 컬러 주석을 처리한 데이터에서 운영 성능과 실패 사례를 같은 조건으로 연결했습니다. 효과가 없는 보조검사는 채택하지 않았고, fake-mark 역검증도 PoC 범위로 해석했습니다. 다음 단계는 정상 제품과 무주석 영상의 검증, 반복 seed 실험입니다. 계획서와 최종 모델 선정의 대응도 제출 전에 확인해야 합니다.

**원본 근거:**

- [FINAL_REPORT_DRAFT.md](C:/workspace/Kamp_Xray/FINAL_REPORT_DRAFT.md)
- [FINAL_SUBMISSION_AUDIT.md](C:/workspace/Kamp_Xray/FINAL_SUBMISSION_AUDIT.md)
- [README.md](C:/workspace/Kamp_Xray/outputs/low_contrast_aux_poc/README.md)
