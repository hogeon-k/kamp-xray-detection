# REPORT_EVIDENCE — 프로젝트 조사 및 근거표

조사 기준일: 2026-10-07 (Asia/Seoul). 프로젝트 루트: C:/workspace/Kamp_Xray.

## 1. 전체 조사 범위

보고서 작성 전에 재귀 파일 목록, 최종 Notebook 전체 셀의 Markdown·코드·저장 출력, 원본 선택 라벨, 학습 설정·로그, CSV·JSON·TXT·YAML 및 대표 그림을 조사하였다. 확인하지 않은 제조품목이나 실험을 외부 설명으로 보충하지 않았다.

조사 시작 시 파일은 **99,235개**였다. 환경·캐시·버전관리 자료를 제외한 프로젝트 파일은 **18,411개**였다. `.cache` 53,328개, `.venv` 26,577개, `.git` 918개 및 루트 `__pycache__` 1개를 별도로 구분하였다. 원시 inventory의 99,236개에는 이번에 만든 조사 스크립트 report_audit.py 1개가 포함되어 있어 시작 시 집계에서 차감하였다. 이번 보고서 및 검증 산출물은 시작 시 수에 포함하지 않았다.

목록 조사와 내용 정독을 구분하였다. 대용량 이미지·바이너리를 전부 실행하거나 디코딩하지 않았다. 최종 영상 512장의 헤더·SHA와 TXT 512개 및 실제 선택 출처를 검증하고 대표 PNG를 열람하였다. 프로젝트 Python 50개를 읽어 전체 소스와 AST 함수 정보를 기록하였다. CSV 244개와 JSON 131개의 역할·스키마·요약을 조사하고 큰 예측·해시·라벨 목록은 필요 항목을 선별 확인하였다. 설치된 라이브러리 전체 소스를 정독하지 않았으며 Ultralytics 평가 정의 관련 코드를 추가 확인하였다. 프로젝트 범위에 XLSX/XLS/YML/PDF는 없었으며 설정파일은 YAML 22개였다.

| 조사 기록 | 내용 |
|---|---|
| report_audit/inventory.json | 전체 파일 상대경로·크기·확장자 |
| report_audit/directories.txt | 전체 하위 디렉터리 |
| report_audit/metadata_dump.txt | CSV 스키마·요약, JSON/README/YAML 내용 |
| report_audit/all_project_code.txt | 프로젝트 Python 50개 전체 소스·함수 위치 |
| report_audit/KAMP_Xray_Final_Analysis.ipynb.txt | 최신 Notebook 모든 셀 및 저장 텍스트/HTML 출력 |
| report_audit/notebook_compact.txt | 셀 순서대로 코드·Markdown·text/plain 출력 |
| report_audit/verification_computed.json | 실제 해시·라벨·예측 재매칭·실험 검증 |
| report_audit/kernel_verification.json | root/submission 각 실제 Jupyter 재실행 |

## 2. 디렉터리 구조와 역할

```text
Kamp_Xray/
├─ dataset/                       원본 영상·라벨 사본·OpenLabeling
│  ├─ 라벨링 6종 세트/           canonical TXT·이미지 사본
│  ├─ OpenLabeling-master/        YOLO_darknet/VOC 대체 출처
│  └─ test1/                     X선이물검출기 NgImage·과거 YOLO
├─ scripts/                      감사·복원·분할·평가·진단·패키징
├─ outputs/                      아래 전 계열 목록 참조
├─ runs/                         과거 탐지 시각화 8개 파일
├─ weights/                      사전학습 weight 1개
├─ KAMP_Xray_Submission/          소스코드·데이터·Test 예측 제출 2,726개 파일
├─ KAMP_Xray_Submission.zip       동일 제출파일 2,726개
├─ KAMP_Xray_Final_Analysis.ipynb 최종 36셀: 코드17·Markdown19
├─ train_yolo.py                 현재 작업 기본값 YOLOv8n640/감마
├─ requirements*.txt             학습·Notebook 환경 별도
├─ verification.json             기존 실제 Jupyter 검증 기록
├─ yolo11n.pt / yolov8n.pt        사전학습 모델
├─ .cache/                       환경·캐시·관련 Notebook 백업3개
├─ .venv/                        과거 학습 환경, 현재 Python 연결 불능
└─ .git/                         버전관리
```

## 3. 평가항목별 근거 매핑 — 보고서 작성 전 확정

셀 번호는 Markdown을 포함한 **1부터 시작하는 물리적 셀 번호**이다. execution_count와 구분한다. R04는 최종 고정 run이며 정확한 경로는 아래 모델표에 연결한다.

| 평가항목 | 확인한 내용 | 수치/결과 | 근거 파일 | 근거 위치 |
|---|---|---|---|---|
| 데이터 이해 E01 | 코호트·클래스·split | 512장·1163bbox·0:defect; 359/77/76장·828/175/160bbox | outputs/yolo_512_telea/manifest.csv, dataset.yaml, labels/*/*.txt | 전체512행 / Cell2·6·7 |
| 데이터 이해 E02 | 실제 선택 출처·과거 이력 | canonical500+fallback12; 선택512/512·이력23/23 일치 | outputs/clean_512/provenance.csv, outputs/submission_label_audit/summary.json, selected_512_label_check.csv | 전체512행 / Cell5·7 |
| 데이터 이해 E03 | TELEA 적용·QA | r3/d0·겹침91장141bbox·mask밖 변경0; 유사910/상세107/우선67 | outputs/yolo_512_telea/restoration_summary.json, outputs/yolo_512_telea_qa/QA_요약.json | 관련 key / Cell9–11 |
| 모델 개발 E04 | 실제 완료 비교 | 11실험=YOLO11n9+YOLOv8n2; matched 구조2쌍 | outputs/model_comparison/comparison_manifest.json, outputs/yolo_runs/*/{summary.json,experiment_config.json,results.csv} | R01–R11 / Cell15–18 |
| 모델 개발 E05 | 고정 모델·학습 환경 | R04 1280/b4/AdamW/lr.001/noaug/s42; 최대200/실행57/best7 | R04의 experiment_config.json, args.yaml, environment.txt, results.csv | CONFIG key / 57행 / Cell17·18 |
| 성능평가 E06 | Val 선택·Test 재평가 | Val .25:170/11/5, .20:171/12/4, .31:170/9/5; Test148/8/12 | outputs/safety_threshold_policy/{val_threshold_sweep.csv,locked_policy.json,test_policy_result.csv}, outputs/foreign_object_condition_analysis/predictions.json | threshold행 / Cell19–21 |
| 성능평가 E07 | validator AP | R04 Val .970014/.473756; Test .934643/.407136 | R04/summary.json, validation/BoxPR_curve.png, test_evaluation/*curve.png | val_*/test_* / Cell16·18 |
| 오류분석 E08 | 오류·이미지 수·기전 | FN12/8장·FP8/4장; FN8+3+1·FP localization8 | outputs/error_analysis/{error_cases.csv,error_classification.csv,summary/cause_frequency.csv} | error20행 / Cell22·23 |
| 오류분석 E09 | 조건별 분자·분모 | 소형7/41 vs5/119·저대비7/60 vs5/100·제품경계3/73 vs9/87 | outputs/error_analysis/summary/{all_test_gt.csv,gt_condition_rates.csv} | GT160행 / Cell24·25 |
| 오류분석 E10 | 대표사례·후보IoU | U0138 .191274/.771798; U0488 .590495/.498477; 저점수 위치성공 후보9/12 | outputs/error_analysis/summary/fn_candidate_diagnostics.csv, error_cases.csv, fn/*.png, fp/*.png | case_id / Cell26–29·36 |
| 현장 활용 E11 | 저장 결과 routing | Val band2장·추가재검0; Test 자동NG72·정상후보4·재검0 | outputs/safety_threshold_policy/{test_image_routing.csv,summary.json} | 전체76행 / Cell30·31 |
| 창의성 E12 | 전처리·directional | TELEA5.672499/NS5.774496; directional line개선1/20·edge개선0/20 | outputs/inpainting_poc_20/method_summary.csv, outputs/directional_refinement_poc/{method_summary.csv,per_image_metrics.csv} | rank/방법행 / Cell10·11; directional은 Notebook 밖 |
| 창의성 E13 | 영상처리·Edge ROI | 보조검사 Val복구0/추가FP2·Test복구0/추가FP4; Edge narrow Test148/8/12 | outputs/low_contrast_aux_poc/{val_parameter_search.csv,summary.json}, outputs/edge_reinspection_poc/{val_comparison.csv,test_comparison.csv} | 선택행 / Notebook에 별도 성능표 없음 |
| 창의성 E14 | 20장 paired 보조진단 | pair34·holdout10·test5; FR-FPR0·GT loss/gain0 | outputs/shortcut_diagnostic_gpu/{model_summary.csv,paired_comparison.csv} | model×scope / Cell12–14 |
| 재현성 E15 | 실재실행·해시 | root/submission 각17셀 통과·Notebook 읽기596파일·보완 후 포함2,725파일SHA·ZIP2,726파일 일치·TELEA512장 픽셀 일치 | KAMP_Xray_Submission/{verification.json,included_files.json}, submission_audit/package_repair_result.json, report_audit/{verification_computed.json,kernel_verification.json} | Cell35·36 / 이번 root·submission 기록 |

## 4. 파일 간 차이의 조사와 적용 규칙

1. 초기 outputs/eda_report.md·eda_bbox_metrics.csv는 bbox1134와 XML 무객체14장을 기록하였다. 최종은 canonical TXT500장 및 fallback TXT12장을 선택하여 bbox1163을 확보하였다. 이미지 SHA로 초기와 최종512장을 연결해 라벨 개수 변화를 별도 기록하였다. 역사적 차이23장은 현행 오류23장이 아니며 무객체14장도 현재 양품14장이 아니다.
2. 학습 summary의 P/R은 Ultralytics validator F1곡선 최대 지점의 집계다. 로컬 .venv/Lib/site-packages/ultralytics/utils/metrics.py:885에서 smooth F1 최대 인덱스를 확인하였다. 운영 평가는 .20 또는 .25 및 IoU≥.5 일대일 매칭이다. R04 Test validator P.942611/R.923911/F1.933167과 운영 P.948718/R.925000/F1.936709를 혼용하지 않는다.
3. **Test confusion 차이 규명 완료:** 원래 R04/test_evaluation/confusion_matrix.png는 149/7/11이었다. 같은 weight·Ultralytics 8.4.158의 실제 validator CPU 재실행에서도 149/7/11을 얻었다. predict 캐시는 1280×1280 입력, validator는 rect=True·pad0.5의 1024×1312/1248×1312/1312×1248 입력이었다. 같은 캐시에서 자체/라이브러리 IoU0.50 매칭은 모두 148/8/12, IoU0.45 매칭은 150/6/10이었다. U0303의 cache IoU0.455820가 validator에서0.449256으로 달라 0.45 통과 여부가 바뀌었다. 최종 운영 값은 잠긴 정책의 cache/conf≥0.20/IoU≥0.50 기준148/8/12로 유지하고 동일 조건의 새 confusion 그림을 사용한다. 원본 PNG는 다른 평가 경로의 역사적 산출물로 보존한다. 과거 GPU의 모든 좌표·AP의 bitwise 동일성은 주장하지 않는다. 자세한 이미지별 비교는 submission_audit/matching_comparison.json 및 FINAL_SUBMISSION_AUDIT.md에 있다.
4. error_analysis_preview는 수정 전 자료다. 최신 error_analysis는 선택 candidate와 해당 GT의 직접 IoU를 계산하며 threshold 미만 후보도 진단한다. 최신 주 분류8/3/1을 사용하였다. 다른 후보의 저점수 위치성공 존재9/12는 중복되는 보조분석이므로 합산하지 않았다.
5. Notebook Cell17의 lr.0005 F1.971176은 1280 무증강 모델 간 비교에서 R04 .969879보다 높다. **11실험 전체 최대는 R11 YOLOv8n 기본증강 .976907**, 다음은 R06 .976861이다. 보고서에서는 전체 최대를 명시하고 lr.0005를 전체 최고로 확대하지 않았다. 원본 Notebook 수치는 수정하지 않았다.
6. 루트 train_yolo.py의 현재 CONFIG는 YOLOv8n640/감마, 제출 CONFIG는 고정 YOLO11n1280/무증강이다. 최종 설정 근거는 R04의 저장 experiment_config.json이다.
7. Train 전체828 GT와 cutoff 집계816 GT는 다르다. scripts/analyze_safety_threshold_policy.py의 train_cutoffs/proposal_features는 product_mask 유효 및 주변 ring12픽셀 이상인 경우만 사용한다. 제외12개 각각의 원인별 개수는 기록이 없어 단정하지 않는다. 영역 추정 안정성649/1163과 Test97/160도 별도 분모이다.
8. outputs/notebook_execution_check/summary.json의 kernel false는 과거 순차 exec 검증이다. 최신 verification.json 및 이번 실제 Jupyter kernel 성공과 구분한다.

## 5. 핵심적으로 조사한 코드 흐름

audit_unique_512 → create_label_review·trace_conflict_label_sources → audit_missing_12_labels → create_clean_512 → split_clean_512 → create_telea_512_dataset → qa_telea_512 → train_yolo → analyze_foreign_object_conditions → evaluate_edge_reinspection → analyze_fn_root_cause → evaluate_low_contrast_auxiliary → analyze_safety_threshold_policy → error_analysis → audit_submission_label_consistency → update_submission_model_comparison·build_final_analysis_notebook·finalize_submission_package 순으로 입력과 산출물 의존성을 확인하였다. 이 순서는 실제 모든 코드를 이번에 재실행했다는 뜻이 아니다.

전처리와 분할 스크립트는 기존 폴더 덮어쓰기를 차단한다. 전체 root에는 전처리 입력과 scripts가 있으나 제출 폴더에는 원본 영상·TELEA 영상·전체 scripts가 포함되지 않았다. 평가자는 저장 결과 재생성은 가능하며 완전한 전처리·재학습에는 root의 입력·코드가 추가로 필요하다.

## 6. 추가 실험 해석

TELEA/NS18조합, Directional20장, Edge band24/40/64, Gaussian/Median/Opening×fixed/Otsu/adaptive×2필터18설정, 영역추정27bbox PoC, real/fake after-only 및 context-matched 진단을 확인하였다. 효과가 없는 실험은 최종보고서 제5장에 기각 사유와 함께 기록하였다. 보조검사의 FP예산3개와 threshold정책 FP예산7개를 구분하였다. CPU/GPU shortcut 결과는 섞지 않고 Notebook의 GPU 계열을 사용하였다.

## 7. 확인되지 않은 사항

구체적 제조품목·라인 공정도·이물질 재질/물리적 크기·촬영 노출조건·양품 포함 검증·양품 오경보율·실제 불량률·현장 처리속도/비용·작업자 재검 효과·마킹 전 clean reference·최종512모델 shortcut 부재·과거 각 run 입력해시·저장 confusion PNG와 현행 캐시의 완전 동일성은 확인되지 않았다.

후속 부록의 Notebook 셀표·출력계열표·실험표·코드파일표는 조사 기록에서 자동 생성하였다. 숫자 검증 및 모든 유보사항은 REPORT_VALIDATION.md에 별도로 기록한다.


## 부록 A. 최신 Notebook 36셀 전체 매핑

| Cell | 유형 | execution_count | 역할 | 평가항목 |
| --- | --- | --- | --- | --- |
| 1 | markdown | — | 전체범위/목차 | 공통 |
| 2 | code | 1 | 경로·SHA·512/1163 연결 | 1/2/6 |
| 3 | markdown | — | 제조검사 목적/라벨/용어 | 1/4 |
| 4 | code | 2 | 변수사전/TXT예시 | 1 |
| 5 | markdown | — | 라벨출처/과거이력 | 1 |
| 6 | code | 3 | 촬영조건/분할 | 1/6 |
| 7 | code | 4 | 라벨해시/품질/해상도 | 1/6 |
| 8 | markdown | — | 분포/누출해석 | 1 |
| 9 | code | 5 | 객체크기/유사쌍그림 | 1/3 |
| 10 | markdown | — | 전처리설명 | 1/5 |
| 11 | code | 6 | 512복원/20장방법비교 | 1/5 |
| 12 | markdown | — | shortcut설계 | 5 |
| 13 | code | 7 | paired통계/overlay | 5 |
| 14 | markdown | — | shortcut한계 | 5 |
| 15 | markdown | — | 본모델비교설계 | 2 |
| 16 | code | 8 | 11설정/Val/Test/구조/증강/weight | 2/6 |
| 17 | markdown | — | 최종모델유지해석 | 2 |
| 18 | code | 9 | 학습로그/Val PR | 2 |
| 19 | markdown | — | threshold설계 | 2/4 |
| 20 | code | 10 | Val sweep/정책/Test | 2/4/6 |
| 21 | markdown | — | 운영효과해석 | 2/4 |
| 22 | markdown | — | 오류분석정의 | 3 |
| 23 | code | 11 | FN/FP/분류/후보 | 3/6 |
| 24 | code | 12 | 조건/조합/설비/FP점수 | 3 |
| 25 | markdown | — | 오류해석제한 | 3 |
| 26 | markdown | — | 저점수FN설명 | 3 |
| 27 | code | 13 | 대표FN4개 | 3 |
| 28 | markdown | — | IoU경계설명 | 3 |
| 29 | code | 14 | U0488/U0371/사람검토 | 3 |
| 30 | markdown | — | 현장제안 | 4 |
| 31 | code | 15 | 현장연결표 | 4 |
| 32 | markdown | — | 결론 | 공통 |
| 33 | markdown | — | 공식평가항목설명 | 공통 |
| 34 | code | 16 | 공식매핑표 | 공통 |
| 35 | markdown | — | 재현성범위 | 6 |
| 36 | code | 17 | 환경/IoU재계산/해시 | 6 |

## 부록 B. 관련 Notebook 백업 전체

| 경로 | 전체셀 | 코드셀 | 해석 |
| --- | --- | --- | --- |
| .cache/notebook_backups/Kamp_Xray_before_model_comparison_20261006_112404.ipynb | 36 | 17 | 과거 단계; 최신 숫자와 구분 |
| .cache/notebook_backups/KAMP_Xray_Final_Analysis_before_label_source_correction.ipynb | 36 | 17 | 과거 단계; 최신 숫자와 구분 |
| .cache/notebook_backups/KAMP_Xray_Submission_before_model_comparison_20261006_112404.ipynb | 36 | 17 | 과거 단계; 최신 숫자와 구분 |

## 부록 C. outputs 전 계열

| 폴더 | 파일수 | 역할 |
| --- | --- | --- |
| augmentation_comparison | 1 | 계획; 실제완료는 학습run 확인 |
| clean_512 | 1561 | 최종 원본512·선택TXT·출처 |
| clean_512_candidate | 1565 | canonical누락12 중간후보 |
| clean_512_split | 7 | 45그룹 split 및 보호해시 |
| color_box_reverse_engineering | 68 | 팔레트·주석·bbox 관계 |
| context_matched_restoration | 247 | real/fake34쌍 문맥대응 |
| directional_refinement_poc | 87 | 20장 방향보간 기각 |
| eda_figures | 5 | 초기1134bbox 그림 |
| edge_reinspection_poc | 19 | 경계band24/40/64·fusion |
| error_analysis | 45 | 최신 FN12/FP8·직접IoU |
| error_analysis_preview | 43 | 수정 전 오류분석 |
| fake_restoration_debug | 43 | fake 생성20장 검사 |
| fn_root_cause_analysis | 11 | Test160 크기·대비·경계·blur |
| foreign_object_condition_analysis | 1467 | 1163bbox 영역추정·예측153장 |
| inpainting_poc_20 | 450 | TELEA/NS18조합20장 |
| label_provenance | 5 | 과거23장·XMLzero14·대체TXT |
| label_review | 131 | 초기라벨검토·미검토초기값 |
| low_contrast_aux_poc | 15 | 18설정·Test복구0/추가FP4 |
| missing_12_label_audit | 64 | fallback12 TXT/VOC검증 |
| model_comparison | 1 | 완료11·구조비교2쌍 |
| notebook_execution_check | 2 | 과거순차exec 검증 |
| restoration_after_similarity | 279 | real34/fake19 after진단 |
| restoration_diagnostics | 378 | before/after 흔적특성 |
| rgb_channel_audit | 126 | 512장 RGB·표시 분석 |
| safety_threshold_policy | 19 | Val92행·locked·Test routing |
| selected_512_original_bmp | 513 | 원본BMP512 선택 |
| shortcut_diagnostic | 18 | 20장 CPU paired진단 |
| shortcut_diagnostic_dataset | 103 | paired34 입력 |
| shortcut_diagnostic_dataset_gpu | 103 | GPU paired입력 |
| shortcut_diagnostic_gpu | 18 | Notebook 채택 GPU진단 |
| submission_label_audit | 5 | 최종512·이력23 출처감사 |
| unique_512_audit | 8 | 원본3904후보·고유2532·옛1134bbox |
| unique_512_review | 1551 | 고유512 및 초기대표라벨 |
| unlabeled_rgb_audit | 39 | 미라벨NG2020 탐색 |
| yolo_512_telea | 1582 | 최종학습영상·mask·TXT·manifest |
| yolo_512_telea_qa | 119 | 유사910/107/67·GT겹침141 |
| yolo_poc_20 | 57 | Train14/Val3/Test3·bbox24/5/5 |
| yolo_poc_20_fake_restoration | 45 | 20장 Train-only fake |
| yolo_runs | 1613 | 학습·평가·weights·곡선 |

## 부록 D. 초기 EDA와 최종 선택라벨

이미지SHA로512장을 연결하였다. bbox수 변경 15장, 합계차이 +29개, 1134→1163.

| ID | 파일 | 초기 | 최종 | delta | 초기형식 | 최종출처 |
| --- | --- | --- | --- | --- | --- | --- |
| U0003 | 001_20200623_003520(5).bmp | 0 | 3 | 3 | VOC | CANONICAL_POOL |
| U0009 | 001_20200623_043227(1).bmp | 0 | 3 | 3 | VOC | CANONICAL_POOL |
| U0014 | 001_20200727_204954(7).bmp | 0 | 1 | 1 | VOC | CANONICAL_POOL |
| U0030 | 001_20200728_014351(3).bmp | 0 | 1 | 1 | VOC | CANONICAL_POOL |
| U0047 | 001_20200623_003536(0).bmp | 0 | 3 | 3 | VOC | CANONICAL_POOL |
| U0076 | 002_20200622_203101(3).bmp | 6 | 3 | -3 | VOC | CANONICAL_POOL |
| U0191 | 001_20200623_003532(5).bmp | 0 | 3 | 3 | VOC | CANONICAL_POOL |
| U0198 | 001_20200623_043231(6).bmp | 0 | 3 | 3 | VOC | CANONICAL_POOL |
| U0212 | 001_20200623_043236(1).bmp | 0 | 3 | 3 | VOC | CANONICAL_POOL |
| U0218 | 001_20200728_083116(1).bmp | 0 | 1 | 1 | VOC | CANONICAL_POOL |
| U0227 | 001_20200623_003529(0).bmp | 0 | 3 | 3 | VOC | CANONICAL_POOL |
| U0274 | 001_20200728_123129(2).bmp | 0 | 1 | 1 | VOC | CANONICAL_POOL |
| U0277 | 001_20200623_003524(0).bmp | 0 | 3 | 3 | VOC | CANONICAL_POOL |
| U0364 | 002_20200622_203121(4).bmp | 0 | 3 | 3 | VOC | CANONICAL_POOL |
| U0491 | 001_20200727_204958(2).bmp | 0 | 1 | 1 | VOC | CANONICAL_POOL |

## 부록 E. 본실험 설정 및 Val/Test 전체표

| 실험 | 모델 | imgsz | batch | lr0 | 증강 | best epoch | 실행 epoch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R01 | yolo11n.pt | 1280 | 4 | 0.0005 | 무증강 | 6 | 56 |
| R02 | yolo11n.pt | 1280 | 4 | 0.0005 | 무증강 | 6 | 56 |
| R03 | yolo11n.pt | 1280 | 4 | 0.001 | 기본 | 44 | 94 |
| R04 | yolo11n.pt | 1280 | 4 | 0.001 | 무증강 | 7 | 57 |
| R05 | yolo11n.pt | 640 | 4 | 0.001 | 기본 + 감마 + 노이즈 | 21 | 71 |
| R06 | yolo11n.pt | 640 | 4 | 0.001 | 기본 + 감마 | 61 | 111 |
| R07 | yolo11n.pt | 640 | 4 | 0.001 | 기본 | 43 | 93 |
| R08 | yolo11n.pt | 640 | 8 | 0.001 | 무증강 | 8 | 58 |
| R09 | yolo11n.pt | 960 | 4 | 0.001 | 무증강 | 4 | 54 |
| R10 | yolov8n.pt | 640 | 4 | 0.001 | 기본 + 감마 | 36 | 86 |
| R11 | yolov8n.pt | 640 | 4 | 0.001 | 기본 | 36 | 86 |

Validation

| 실험 | Precision | Recall | F1(P/R) | mAP50 | mAP50–95 |
| --- | --- | --- | --- | --- | --- |
| R01 | 0.971414 | 0.970938 | 0.971176 | 0.977442 | 0.448577 |
| R02 | 0.971414 | 0.970938 | 0.971176 | 0.977442 | 0.448577 |
| R03 | 0.965583 | 0.961907 | 0.963741 | 0.954928 | 0.494868 |
| R04 | 0.965989 | 0.973800 | 0.969879 | 0.970014 | 0.473756 |
| R05 | 0.964053 | 0.965714 | 0.964883 | 0.964420 | 0.473706 |
| R06 | 0.976580 | 0.977143 | 0.976861 | 0.968612 | 0.457394 |
| R07 | 0.959072 | 0.960000 | 0.959536 | 0.932074 | 0.462831 |
| R08 | 0.959987 | 0.959680 | 0.959833 | 0.948866 | 0.436085 |
| R09 | 0.931386 | 0.925714 | 0.928541 | 0.946788 | 0.447237 |
| R10 | 0.958713 | 0.960000 | 0.959356 | 0.950134 | 0.454533 |
| R11 | 0.976672 | 0.977143 | 0.976907 | 0.971408 | 0.464537 |

Test

| 실험 | Precision | Recall | F1(P/R) | mAP50 | mAP50–95 |
| --- | --- | --- | --- | --- | --- |
| R01 | 0.948987 | 0.937500 | 0.943208 | 0.925544 | 0.352720 |
| R02 | 0.948987 | 0.937500 | 0.943208 | 0.925544 | 0.352720 |
| R03 | 0.937482 | 0.937214 | 0.937348 | 0.914896 | 0.362167 |
| R04 | 0.942611 | 0.923911 | 0.933167 | 0.934643 | 0.407136 |
| R05 | 0.931216 | 0.930757 | 0.930987 | 0.884734 | 0.367971 |
| R06 | 0.955614 | 0.956250 | 0.955932 | 0.932648 | 0.352791 |
| R07 | 0.956072 | 0.956250 | 0.956161 | 0.926064 | 0.384190 |
| R08 | 0.906333 | 0.907142 | 0.906737 | 0.887411 | 0.347358 |
| R09 | 0.892086 | 0.887500 | 0.889787 | 0.904466 | 0.378679 |
| R10 | 0.948740 | 0.950000 | 0.949369 | 0.910148 | 0.353280 |
| R11 | 0.943084 | 0.943750 | 0.943417 | 0.899269 | 0.359172 |

| R번호 | outputs/yolo_runs 아래 저장폴더 |
| --- | --- |
| R01 | telea_512_yolo11n_e200_img1280_b4_adamw_lr0.0005_wd0.0005_noaug_s42 |
| R02 | telea_512_yolo11n_e200_img1280_b4_adamw_lr0.0005_wd0.0005_noaug_s42_run02 |
| R03 | telea_512_yolo11n_e200_img1280_b4_adamw_lr0.001_wd0.0005_aug_tr0.005_sc0.2_mos0.2_s42 |
| R04 | telea_512_yolo11n_e200_img1280_b4_adamw_lr0.001_wd0.0005_noaug_s42 |
| R05 | telea_512_yolo11n_e200_img640_b4_adamw_lr0.001_wd0.0005_aug_tr0.005_sc0.2_mos0.2_gam0.9-1.1p0.2_noise1-3p0.1_s42 |
| R06 | telea_512_yolo11n_e200_img640_b4_adamw_lr0.001_wd0.0005_aug_tr0.005_sc0.2_mos0.2_gam0.9-1.1p0.2_s42 |
| R07 | telea_512_yolo11n_e200_img640_b4_adamw_lr0.001_wd0.0005_aug_tr0.005_sc0.2_mos0.2_s42 |
| R08 | telea_512_yolo11n_e200_img640_b8_adamw_lr0.001_wd0.0005_noaug_s42 |
| R09 | telea_512_yolo11n_e200_img960_b4_adamw_lr0.001_wd0.0005_noaug_s42 |
| R10 | telea_512_yolov8n_e200_img640_b4_adamw_lr0.001_wd0.0005_aug_tr0.005_sc0.2_mos0.2_gam0.9-1.1p0.2_s42 |
| R11 | telea_512_yolov8n_e200_img640_b4_adamw_lr0.001_wd0.0005_aug_tr0.005_sc0.2_mos0.2_s42 |

## 부록 F. 20장 PoC 실제 모델결과

experiments.csv 총16행 중5행은20장 PoC이다. YOLO26n도 실제 학습되었으나512장 구조비교2종에 포함하지 않았다. 아래는14/3/3 split이다. 별도 telea_baseline_20은 max50/optimizer auto/증강·로그44epoch 및weight가 있으나summary가 없어최종split별 성능을 만들지 않았다. 512장640/b4/noaug는 cancelled.json·로그7epoch로완료표에서 제외하였다.

| PoC run | Train/Val/Test장 | ValP | ValR | ValAP50 | ValAP50–95 | TestP | TestR | TestAP50 | TestAP50–95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| telea_yolo11n_e200_img640_b8_adamw_lr0.001_wd0.0005_noaug_s42 | 14/3/3 | 0.595510 | 0.600000 | 0.645000 | 0.316536 | 0.794343 | 0.777371 | 0.636000 | 0.230000 |
| telea_yolo11n_e200_img1280_b8_adamw_lr0.001_wd0.0005_noaug_s42 | 14/3/3 | 0.783841 | 0.735364 | 0.636000 | 0.192650 | 0.782586 | 0.730344 | 0.715000 | 0.233937 |
| telea_yolo26n_e200_img640_b8_adamw_lr0.001_wd0.0005_noaug_s42 | 14/3/3 | 0.483092 | 0.800000 | 0.454286 | 0.217214 | 0.591285 | 0.600000 | 0.695000 | 0.327610 |
| telea_fake_restoration_yolo11n_e200_img640_b8_adamw_lr0.001_wd0.0005_noaug_s42 | 14/3/3 | 0.953336 | 0.800000 | 0.795000 | 0.254154 | 0.790597 | 0.800000 | 0.675000 | 0.332316 |
| telea_fake_restoration_yolo26n_e200_img640_b8_adamw_lr0.001_wd0.0005_noaug_s42 | 14/3/3 | 0.428275 | 0.599377 | 0.495000 | 0.193375 | 0.492317 | 0.600000 | 0.595000 | 0.197715 |

## 부록 G. 프로젝트 Python50개 조사 위치

전체소스는 all_project_code.txt의 FILE 구분자로 연결된다. 테스트파일 존재를 테스트실행 성공으로 해석하지 않았다.

| 경로 | 모듈설명 | 대표함수:행 |
| --- | --- | --- |
| train_yolo.py | Run reproducible YOLO11n Baseline experiments on the 512-image TELEA dataset. | UserFacingError:135, _safe_name:139, _compact_number:144, augmentation_label:148, validate_xray_augmentation:170, build_xray_trainer:185, build_run_name:233, get_unique_run_name:254 |
| dataset/OpenLabeling-master/main/dasiamrpn.py | Author : Will Stone Date   : 190407 Desc   : Wrapper class for the DaSiamRPN tracking method. This class has the          methods required to interface with the tracking class impl | dasiamrpn:40 |
| dataset/OpenLabeling-master/main/main.py | docstring없음; 소스확인 | pointInRect:91, dragBBox:97, display_text:190, set_img_index:196, set_class_index:205, draw_edges:212, decrease_index:222, increase_index:229 |
| dataset/OpenLabeling-master/main/main_auto.py | \Description Using object detection and tracker to automatically label data \Brief Algorithm Step 1. Using Object Detection to find objects in the current frame         1.1 If do n | display_text:103, set_img_index:109, set_class_index:118, draw_edges:125, decrease_index:135, increase_index:142, draw_line:151, yolo_format:156 |
| dataset/OpenLabeling-master/object_detection/tf_object_detection.py | docstring없음; 소스확인 | ObjectDetector:5 |
| dataset/OpenLabeling-master/object_detection/utils.py | docstring없음; 소스확인 | format_results:1 |
| KAMP_Xray_Submission/train_yolo.py | YOLOv8n/YOLO11n training reference. CONFIG defaults to the fixed final model.  This analysis submission does not include training images. To retrain, provide the full prepared TELE | UserFacingError:140, _safe_name:144, _compact_number:149, augmentation_label:153, validate_xray_augmentation:175, build_xray_trainer:190, build_run_name:238, get_unique_run_name:259 |
| outputs/color_box_reverse_engineering/analyze.py | docstring없음; 소스확인 | save:15, geom:20, rect:34, assignment:36, table:50, main:53 |
| outputs/color_box_reverse_engineering/edge_palette.py | docstring없음; 소스확인 | main:4 |
| outputs/color_box_reverse_engineering/supplement.py | docstring없음; 소스확인 | main:3 |
| outputs/color_box_reverse_engineering/verify.py | docstring없음; 소스확인 | main:3 |
| outputs/directional_refinement_poc/run_directional.py | Directional refinement of the fixed 20-image TELEA PoC baseline.  Reads the existing selected_20.csv, exact d0 masks, TELEA/d0/r3 outputs, and GT. Writes only below outputs/directi | sha256:34, safe_div:42, save_new:46, write_csv:52, build_gt_masks:63, exact_palette_mask:82, axis_candidates:94, local_orientations:138 |
| outputs/inpainting_poc_20/recompute_metrics.py | Recompute GT-internal contrast and all dependent PoC summaries. |  |
| outputs/inpainting_poc_20/refresh_figures.py | Refresh comparison figures from saved PoC artifacts after report QA. |  |
| outputs/inpainting_poc_20/refresh_palette_metadata.py | Separate palette table entries from actually used overlay indices. |  |
| outputs/inpainting_poc_20/run_poc.py | 20-image palette-mask/OpenCV inpainting proof of concept.  Inputs are read-only. This script writes only inside outputs/inpainting_poc_20. It intentionally never deletes or overwri | refuse_overwrite:40, save_image:45, write_csv:50, sha256:60, safe_div:68, finite:72, raster_box:76, masks_for_boxes:85 |
| scripts/analyze_fn_root_cause.py | Descriptive Test TP/FN analysis. Reuses frozen detections and evaluation definitions. | raster:22, calculate:28, statistics:71, fmt:82, sheets:85, plots:122, write_interpretation:139, main:169 |
| scripts/analyze_foreign_object_conditions.py | GT bbox 내부 영상 특성을 이용한 이물질 영역 추정. 입력 파일은 읽기 전용. | read_csv:15, write_csv:18, save_json:22, fingerprint:25, load_records:29, select_poc:46, component:57, extract_candidates:66 |
| scripts/analyze_safety_threshold_policy.py | Select confidence and referral policy on Val; evaluate the locked policy once on Test. | match_at:18, load_records:28, metrics:36, routing:49, fn_diagnostics:64, proposal_features:87, train_cutoffs:99, vulnerability:111 |
| scripts/audit_missing_12_labels.py | Read-only source audit for the 12 images lacking canonical labeling-set TXT. | hash_file:32, load:40, save:45, box_equivalent:52, main:66 |
| scripts/audit_submission_label_consistency.py | Verify the selected 512 labels against project sources without changing inputs. | read_csv:22, digest:27, main:35 |
| scripts/audit_unique_512.py | Recheck the historical labeled 512-image cohort without touching source data. | write_csv:40, sha256:48, phash:56, source_priority:64, context:78, inventory:91, small_gray:118, compare:123 |
| scripts/build_final_analysis_notebook.py | Refresh the current corrected final analysis notebook in a Jupyter kernel.  No training, inference, package installation, or modification of prior outputs. The current notebook is  | md:23, code:28, execute_and_save:649, execute_current_notebook:712 |
| scripts/create_clean_512.py | Create the final unsplit 512-image BMP/YOLO dataset from verified sources. | sha256:38, load_csv:46, save_csv:51, font:58, visualize:63, make_sheets:77, main:94 |
| scripts/create_clean_512_candidate.py | Copy the selected BMPs and only same-stem canonical TXT labels for review. | digest:26, load:34, save:39, font:46, validate_label:51, render:92, make_sheets:106, main:120 |
| scripts/create_fake_restoration_dataset.py | Create a separate augmented copy of the existing manifest-based YOLO PoC. | load_rgb:21, draw_debug:26, generate:41, main:158 |
| scripts/create_label_review.py | Create read-only human review materials for the existing 512-image audit. | digest:25, read_csv:33, write_csv:38, font:45, parse_xml:52, parse_yolo:64, check_boxes:80, iou:87 |
| scripts/create_telea_512_dataset.py | Create TELEA data from a frozen split manifest; no training or label edits. | load_poc_mask:16, overlap:27, panel:46, audit_relations:60, main:85 |
| scripts/create_yolo_poc_20.py | docstring없음; 소스확인 | read_csv:22, write_csv:27, sha256:34, parse_voc:42, parse_yolo:57, choose_splits:71, main:119 |
| scripts/diagnose_after_restoration_similarity.py | After-only local restoration artifact diagnostic. All inputs are read-only. | read_csv:30, ratio:35, make_ring:39, after_maps:51, boundary_values:59, measure_region:74, cliffs_delta:106, scope_rows:113 |
| scripts/diagnose_context_matched_restoration.py | Same-image, context-matched diagnostic pairs; never changes training augmentation. | context_vector:26, crop_mask:36, ring_offsets:42, ring_coordinates:48, window_sums:54, evaluate_candidates:59, standardized_distances:86, translated_mask:94 |
| scripts/diagnose_restoration.py | Read-only Real/Fake restoration diagnostics; never calls inpainting. | sha256:22, load_rgb:27, load_mask:32, change_pct:40, feature_maps:44, region_metrics:67, real_regions:89, fake_regions:94 |
| scripts/diagnose_yolo_shortcut.py | Paired YOLO shortcut diagnostic on immutable restoration image pairs. | sha256:33, read_csv:38, write_csv:43, load_rgb:49, load_mask:54, box_iou:62, spatial_relation:72, annotate_predictions:86 |
| scripts/error_analysis.py | Frozen Test FN/FP audit. Reuses predictions/features; never trains or infers.  Run with Python 3 + numpy + Pillow: python -B -X utf8 scripts/error_analysis.py Existing outputs are  | read_csv:28, write_csv:33, dump:41, sha:45, iou:49, candidate_diagnosis:55, match:66, boxes:82 |
| scripts/evaluate_edge_reinspection.py | Evaluate product-contour edge crops with the frozen YOLO11n. Inputs are read only. | csvread:26, csvwrite:31, dump:37, sha:41, product_mask:45, edge_distance:77, crop_tiles:81, iou:92 |
| scripts/evaluate_low_contrast_auxiliary.py | Val-selected local-contrast confirmation of uncertain YOLO proposals; Test once. | maps:17, components:24, candidates:57, evaluate:76, prepare:90, contact:107, main:137, post_analysis:246 |
| scripts/fake_restoration.py | Seeded, GT-aware relocation of actual restoration mask components. | RestorationSettings:15, inpaint_rgb:25, bbox_mask:38, mask_templates:51, augment_fake_restoration:61 |
| scripts/finalize_submission_package.py | Synchronize submission materials, verify the notebook and build its ZIP. | read:22, write:26, digest:30, main:35 |
| scripts/qa_telea_512.py | 고정된 512장 데이터셋의 읽기 전용 학습 전 QA 자료 생성. | text:17, load:20, fitted:31, draw_panels:40, sheets:58, prepare:71, finalize:159 |
| scripts/rgb_channel_audit.py | Audit RGB-channel outliers in the 512-image deduplicated X-ray set.  All inputs are read-only. Results are written below outputs/rgb_channel_audit. | sha256_file:28, normalize_rel_path:36, candidate_paths:43, build_basename_index:54, resolve_images:62, clipped_box:126, boundary_mask_for_box:135, connected_components:151 |
| scripts/split_clean_512.py | Read-only validation and reproducible session-group split; never overwrite outputs. | sha256:16, read_csv:23, write_csv:27, write_json:33, snapshot:37, verify_snapshot:40, labels:46, validate_source:60 |
| scripts/test_after_restoration_similarity.py | docstring없음; 소스확인 | AfterOnlyTests:13 |
| scripts/test_context_matched_restoration.py | docstring없음; 소스확인 | ContextMatchedTests:14 |
| scripts/test_diagnose_restoration.py | docstring없음; 소스확인 | DiagnosticTests:14 |
| scripts/test_error_analysis.py | Regression checks for confidence-independent displayed-candidate IoU. | CandidateIoUTest:7 |
| scripts/test_fake_restoration.py | Run: python -m unittest discover -s scripts -p test_fake_restoration.py -v | FixedCount:16, FakeRestorationTests:31, ExistingDatasetIntegrationTests:106 |
| scripts/test_yolo_shortcut.py | docstring없음; 소스확인 | ShortcutLogicTests:8 |
| scripts/trace_conflict_label_sources.py | Trace conflict YOLO TXT bytes back to the supplied labeling-set TXT pool. | sha256:21, load:29, save:34, main:41 |
| scripts/unlabeled_rgb_audit.py | Audit 2,020 unique unlabeled NG X-rays and search for unmarked originals. | safe_ratio:34, sha256_file:38, path_candidates:46, resolve_assets:60, outline_fit:114, component_details:128, dhash:180, parse_filename:190 |
| scripts/update_submission_model_comparison.py | Add saved YOLOv8n/11n comparisons and execute both analysis notebooks. | read:22, write:26, sha:31, main:35, _create_comparison_dir:264 |

## 부록 H. 제출 전 감사에서 추가 확인한 근거

| 항목 | 원본·검증 기록 | 확정 내용 |
|---|---|---|
| 최종 bbox 크기 | 실제 최종 TXT×512 영상 헤더; submission_audit/number_checks.json | 1163개 면적 min21/Q25=81/median104/Q75=143/max420/mean115.890800; 종횡비 min0.428571/median1/max1.833333 |
| 초기 크기 통계 분리 | outputs/eda_bbox_metrics.csv, outputs/class_distribution.csv | 초기 EDA median100/Q75=140/mean114.322442/aspect max2를 최종 통계로 사용하지 않음 |
| validator 재실행 | submission_audit/instrument_validation.py, validator_replay.json | 동일 weight CPU 실제 추론; matrix149/7/11 재현 |
| predict 재실행 | submission_audit/instrument_predict.py, predict_replay.json | 동일 manifest 순서·batch4 CPU 실제 추론; square1280; own50=148/8/12, CM45=150/6/10 재현 |
| 매칭 방식 | compare_matching.py, matching_comparison.json | cache IoU0.50의 자체/Ultralytics 모두148/8/12; class0뿐·정확한 threshold/IoU 동률0·다중 적격 매칭0 |
| 공식 confusion | generate_official_confusion.py, official_test_confusion.json/png, official_test_matching.csv | 잠긴 threshold0.20/IoU0.50,148/8/12; threshold0.25도 동일; TN 미정의 |
| 실제 모델 구조·weight | model_inventory.py/json | 11별도 run/2구조/11파일 SHA/10파라미터 상태; R01/R02 상태 동일; R07/R11은 구조·상태 모두 다름 |
| 제출 비교 weight 보강 | model_comparison_supplement.zip, 내부 SHA256.json | 기존 제출본에 빠진 R07/R11의 weight·config·summary·로그·환경 원본 복사; 기존 ZIP은 변경하지 않음 |
| 숫자 등장 전수 색인 | all_report_numbers.csv, verify_report_numbers.py, number_checks.json | 식별자·경로·배점과 정량값 구분; 단순 숫자 문자열 일치를 의미 검증으로 취급하지 않음 |

새 감사의 공식 결론과 제출 조건은 FINAL_SUBMISSION_AUDIT.md를 따른다. 구체적 계획서 원문은 프로젝트에서 발견하지 못했으므로 ‘계획서에 따라’의 충족 여부는 제출자가 확인해야 한다.
