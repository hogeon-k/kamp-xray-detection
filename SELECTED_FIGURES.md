# 선정 그림과 대체 자료

이미지 내용은 프로젝트 원본을 사용한다. 확대는 PPTX의 crop 정보로 저장하며 새로운 사례 이미지를 생성하지 않는다. 표·도식은 기존 수치와 코드 흐름을 편집 가능한 슬라이드 객체로 구성한다.

| 슬라이드 번호 | 사용할 이미지 | 원본 경로 | 사용 이유 |
|---|---|---|---|
| 1 | 001_20200814_163105(1).png | [C:/workspace/Kamp_Xray/outputs/yolo_512_telea/images/train/001_20200814_163105(1).png](C:/workspace/Kamp_Xray/outputs/yolo_512_telea/images/train/001_20200814_163105(1).png) | 실제 TELEA 입력으로 과제 소개 |
| 2 | 이미지 없음: 편집 가능한 표 | [manifest.csv](C:/workspace/Kamp_Xray/outputs/yolo_512_telea/manifest.csv)<br>[split_manifest.csv](C:/workspace/Kamp_Xray/outputs/clean_512_split/split_manifest.csv) | 실제 수치와 구현 범위 전달 |
| 3 | 01_4184354a2a3c_comparison.png | [C:/workspace/Kamp_Xray/outputs/inpainting_poc_20/figures/01_4184354a2a3c_comparison.png](C:/workspace/Kamp_Xray/outputs/inpainting_poc_20/figures/01_4184354a2a3c_comparison.png) | 컬러 표시와 mask·복원 후 차이를 직접 비교 |
| 4 | 이미지 없음: 편집 가능한 표 | [comparison_manifest.json](C:/workspace/Kamp_Xray/outputs/model_comparison/comparison_manifest.json)<br>[summary.json](C:/workspace/Kamp_Xray/outputs/yolo_runs/telea_512_yolo11n_e200_img640_b4_adamw_lr0.001_wd0.0005_aug_tr0.005_sc0.2_mos0.2_s42/summary.json)<br>[summary.json](C:/workspace/Kamp_Xray/outputs/yolo_runs/telea_512_yolov8n_e200_img640_b4_adamw_lr0.001_wd0.0005_aug_tr0.005_sc0.2_mos0.2_s42/summary.json)<br>[summary.json](C:/workspace/Kamp_Xray/outputs/yolo_runs/telea_512_yolo11n_e200_img1280_b4_adamw_lr0.001_wd0.0005_noaug_s42/summary.json) | 실제 수치와 구현 범위 전달 |
| 5 | val_threshold_curve.png | [C:/workspace/Kamp_Xray/outputs/safety_threshold_policy/visualizations/val_threshold_curve.png](C:/workspace/Kamp_Xray/outputs/safety_threshold_policy/visualizations/val_threshold_curve.png) | Validation에서 선택한 운영점과 recall/precision 교환관계 |
| 6 | official_test_confusion.png | [C:/workspace/Kamp_Xray/submission_audit/official_test_confusion.png](C:/workspace/Kamp_Xray/submission_audit/official_test_confusion.png) | 공식 148/8/12 및 평가 조건 일치 |
| 7 | U0138_FN_gt0.png | [C:/workspace/Kamp_Xray/outputs/error_analysis/fn/U0138_FN_gt0.png](C:/workspace/Kamp_Xray/outputs/error_analysis/fn/U0138_FN_gt0.png) | 점수 부족 FN과 위치 오차 FP를 구분 |
| 7 | U0371_FP_pred0.png | [C:/workspace/Kamp_Xray/outputs/error_analysis/fp/U0371_FP_pred0.png](C:/workspace/Kamp_Xray/outputs/error_analysis/fp/U0371_FP_pred0.png) | 점수 부족 FN과 위치 오차 FP를 구분 |
| 8 | overlay.png | [C:/workspace/Kamp_Xray/outputs/shortcut_diagnostic_gpu/cases/E_largest_delta_pair_0002/overlay.png](C:/workspace/Kamp_Xray/outputs/shortcut_diagnostic_gpu/cases/E_largest_delta_pair_0002/overlay.png) | 실제 paired 모델 반응과 의도한 fake 영역 위치 확인 |
| 8 | pair_0002.png | [C:/workspace/Kamp_Xray/outputs/shortcut_diagnostic_dataset_gpu/masks/pair_0002.png](C:/workspace/Kamp_Xray/outputs/shortcut_diagnostic_dataset_gpu/masks/pair_0002.png) | 실제 paired 모델 반응과 의도한 fake 영역 위치 확인 |
| 9 | 이미지 없음: 편집 가능한 표 | [FINAL_REPORT_DRAFT.md](C:/workspace/Kamp_Xray/FINAL_REPORT_DRAFT.md)<br>[locked_policy.json](C:/workspace/Kamp_Xray/outputs/safety_threshold_policy/locked_policy.json) | 실제 수치와 구현 범위 전달 |
| 10 | 이미지 없음: 결론 텍스트 | [FINAL_REPORT_DRAFT.md](C:/workspace/Kamp_Xray/FINAL_REPORT_DRAFT.md)<br>[FINAL_SUBMISSION_AUDIT.md](C:/workspace/Kamp_Xray/FINAL_SUBMISSION_AUDIT.md)<br>[README.md](C:/workspace/Kamp_Xray/outputs/low_contrast_aux_poc/README.md) | 실제 수치와 구현 범위 전달 |

## 패널 확대 위치

- 3장: 원본 비교 PNG의 5개 열 중 원본·mask·TELEA 열을 사용한다. NS·GT 열은 같은 파일에 남아 있으며 발표에서는 상세 비교를 생략한다.
- 7장: U0138 FN·U0371 FP의 두 번째 확대 패널을 사용한다. 원본 전체 사례 파일과 수치는 발표자 노트로 추적할 수 있다. GT와 후보/예측의 색 의미를 슬라이드에 표시한다.
- 8장: overlay 상단의 baseline original/fake 두 패널과 pair_0002 fake mask를 사용한다. 초록 영역은 진단 fake 위치이며 실제 이물질 예측으로 해석하지 않는다. mask는 위치를 보여주는 기존 이진 마스크이다.

## 본문에서 제외한 주요 figure

- [outputs/yolo_runs/telea_512_yolo11n_e200_img1280_b4_adamw_lr0.001_wd0.0005_noaug_s42/results.png](C:/workspace/Kamp_Xray/outputs/yolo_runs/telea_512_yolo11n_e200_img1280_b4_adamw_lr0.001_wd0.0005_noaug_s42/results.png): 57epoch 실제 학습곡선. 10장 흐름에서는 비교 결과표를 우선하고 질의응답 자료로 사용
- [outputs/yolo_runs/telea_512_yolo11n_e200_img1280_b4_adamw_lr0.001_wd0.0005_noaug_s42/validation/BoxPR_curve.png](C:/workspace/Kamp_Xray/outputs/yolo_runs/telea_512_yolo11n_e200_img1280_b4_adamw_lr0.001_wd0.0005_noaug_s42/validation/BoxPR_curve.png): Val AP 상세자료. threshold curve와 역할이 달라 본문에서는 생략
- [outputs/error_analysis/fn/U0488_FN_gt0.png](C:/workspace/Kamp_Xray/outputs/error_analysis/fn/U0488_FN_gt0.png): 위치 경계 사례. U0371 FP와 겹치므로 질문 대응 자료로 보관
- [outputs/error_analysis/summary/fn_condition_rates.png](C:/workspace/Kamp_Xray/outputs/error_analysis/summary/fn_condition_rates.png): 여러 조건이 포함돼 발표에서는 소형·저대비 두 분모만 강조
- [outputs/shortcut_diagnostic_gpu/plots/fr_fpr_comparison.png](C:/workspace/Kamp_Xray/outputs/shortcut_diagnostic_gpu/plots/fr_fpr_comparison.png): 0값 막대는 정보량이 적어 paired 실제 사례와 결과 문구를 우선
