# REPORT_VALIDATION — 결과보고서 수치 및 재현 검증

검증일: 2026-10-07. 검증 대상: FINAL_REPORT_DRAFT.md의 제1장~제6장. 상대경로는 C:/workspace/Kamp_Xray 기준이다. 상세 파일·셀 근거는 REPORT_EVIDENCE.md의 E01~E15 및 부록에 연결하였다.

## 1. 검증 방법과 범위

파일명이나 Notebook 설명만으로 수치를 확정하지 않았다. 최종 TXT 512개를 직접 파싱하고 영상 헤더·SHA·선택 원본 라벨을 대조하였다. 저장 예측 predictions.json과 실제 Val/Test 라벨·영상 크기로 confidence와 IoU 기준을 다시 적용하여 일대일 매칭과 TP/FP/FN을 재계산하였다. 학습 11개 run의 config·summary·results.csv·실험 인덱스를 대조하였다. 전체 학습이나 GPU 추론을 다시 실행하지 않았으므로 mAP는 기존 validator 결과의 원본 대조이며 새 추론으로 독립 산출한 값은 아니다.

검증 코드는 report_verify.py, 재계산 결과는 report_audit/verification_computed.json에 보존하였다. report_audit/replay_notebook.py는 루트 및 제출본 Notebook을 실제 Jupyter 커널에서 실행하고 원본 불변 여부를 확인하였다. 환경·설치 라이브러리를 포함한 전체 파일 목록과 분석 대상 내용의 범위는 근거 문서에서 구분하였다.

## 2. 데이터 및 전처리 검증

| 항목 | 보고서 값 | 원본 근거·직접 검증 | 판정 |
|---|---|---|---|
| 조사 시작 파일 | 99,235개; 환경·캐시·Git 제외 프로젝트 18,411개 | inventory.json 99,236에서 새 조사 스크립트 1개 차감; 제외 목록은 근거 문서 §1 | 집계 일치 |
| 최종 이미지·bbox | 512장·1163개 | clean_512/metadata.csv, yolo_512_telea/labels의 512 TXT 파싱 | 일치 |
| 클래스 | 1개, 0:defect | dataset.yaml과 모든 TXT 클래스 값 | 일치 |
| Train | 이미지 359·bbox 828 | split_manifest.csv 및 실제 labels/train | 일치 |
| Val | 이미지 77·bbox 175 | 실제 labels/val | 일치 |
| Test | 이미지 76·bbox 160 | 실제 labels/test | 일치 |
| 합계 | 359+77+76=512; 828+175+160=1163 | 직접 합산 | 일치 |
| 그룹 수 | Train/Val/Test 24/12/9=45 | 기계·시리얼·날짜 그룹과 분할 | 일치 |
| cross-split 누수 | ID·파일 SHA·픽셀·그룹·검사 세션 중복 0 | 저장 감사 및 직접 SHA/그룹 대조 | 현재 자료에서 일치 |
| 해상도 | 316×332:218; 352×332:125; 576×444:169 | 512 영상 헤더 직접 판독; 합계 512 | 일치 |
| bbox 분포 | 1개:186장, 2개:1장, 3개:325장 | 186+1+325=512; 186+2+975=1163 | 일치 |
| 원본 라벨 출처 | canonical 500·검증된 fallback 12 | provenance의 final_label_original_path와 실제 최종 TXT 해시 | 일치 |
| 초기/최종 bbox | 1134→1163, +29 | 동일 영상 SHA로 초기 XML과 최종 TXT 연결: 14장 +32, U0076 -3 | 차이 설명 완료 |
| 과거 충돌 라벨 | XML–YOLO 차이 23장; 현재 canonical 23/23 일치 | 초기 감사와 현재 선택 출처 구분 | 현재 23장 오류로 기재하지 않음 |
| TELEA 적용 | 512장, radius 3·dilation 0 | 생성 코드·manifest·QA | 일치 |
| mask–GT 교차 | 91장·141bbox·2498 pixels | 복원 QA; mask 밖 변경 0, 라벨 변경 0 | 일치 |

원본 후보 3904개 중 unique SHA 2532개·중복 사본 1372개는 후보 자료의 수치이며 최종 512장의 중복 수가 아니다. 확장자가 JPG이나 실제 BMP인 1080개 및 XML 객체 0인 초기 14장은 최종 데이터 정의와 구분하였다. near-duplicate 후보 910개·상세 107개·우선 검토 67개는 확정 중복 수로 표현하지 않았다.

## 3. 성능 지표의 정의 및 직접 계산

운영 평가의 IoU 기준은 0.50, confidence는 baseline 0.25 및 잠금 low threshold 0.20이다. NMS IoU 0.70과 검출 평가 IoU 0.50은 서로 다른 설정이다. 저장 예측의 confidence 하한은 0.001이다.

Precision=TP/(TP+FP), Recall=TP/(TP+FN), F1=2TP/(2TP+FP+FN)으로 계산하였다. 최종 Test에서는 TP+FN=160, TP+FP=156이었다.

| 항목 | 원본 값 및 재계산 | 판정 |
|---|---|---|
| Test TP/FP/FN | 148/8/12 | predictions.json·실제 라벨 재매칭 일치 |
| Test Precision | 148/(148+8)=0.948717948718 | 보고서 0.948718 일치 |
| Test Recall | 148/(148+12)=0.925000000000 | 보고서 0.925000 일치 |
| Test F1 | 296/(296+8+12)=0.936708860759 | 보고서 0.936709 일치 |
| Val baseline 0.25 | 170/11/5; P 0.939226519337, R 0.971428571429, F1 0.955056179775 | 직접 재매칭 일치 |
| Val low 0.20 | 171/12/4; P 0.934426229508, R 0.977142857143, F1 0.955307262570 | 직접 재매칭 일치 |
| Val F1 최대 0.31 | 170/9/5; F1 0.960451977401 | 탐색 원본 및 직접 재매칭 일치 |
| Test low 0.20 | baseline과 같은 148/8/12 | 직접 재매칭 일치; 개선 없음 |
| Test validator mAP50 | 0.934643398327 | 최종 run/summary.json의 test_map50 대조 |
| Test validator mAP50-95 | 0.407135716534 | 같은 원본 대조; 새 추론 재산출 아님 |
| Test validator P/R/F1 | 0.942611110746 / 0.923910641947 / 약 0.933167 | validator 최적 F1 curve 지표; 운영 지표와 분리 |
| 최종 run epoch | 최대 200, 실제 로그 57, best 7 | experiment_config.json·results.csv 직접 대조 |

Val의 저임계값 후보에서 추가 FP 예산은 floor(77×0.10)=7개였다. 기존 TP 손실을 금지하고 FN 복구, 추가 FP 최소, 높은 threshold 순으로 고른 정책을 확인하였다. Val에서 정한 0.20/0.25를 Test에 적용하였다. Test 재검 band 0장, 자동 NG 72장, 정상 후보 4장이며, 전체 Test가 GT 양성 영상이므로 정상 후보 4장을 확인된 정상으로 표현하지 않았다.

## 4. 오류 및 영향요인 검증

| 항목 | 분자·분모 및 재계산 | 원본·판정 |
|---|---|---|
| FN 영상·FP 영상 | FN 12개/8장; FP 8개/4장 | 최신 error_analysis CSV; FP 영상은 FN 영상의 부분집합 |
| FN 주 유형 | localization failure 8 + low confidence/localization success 3 + low confidence/localization failure 1 = 12 | 상호 배타 주 분류 합계 일치 |
| unknown | 주 매칭 분류 unknown 0; 물리적 발생 원인은 FN/FP 모두 미확정 | 매칭 유형을 제조 원인으로 단정하지 않음 |
| 낮은 점수의 위치 성공 후보 | 9/12 FN에 score<0.20, IoU≥0.50 후보 존재 | 주 유형과 중복 가능한 후보 통계; 합산 금지 |
| FP 유형 | 8/8 GT 근처 위치 오차 쌍; 7/8 예측 bbox가 GT보다 큼 | 원본 CSV 및 U0488 예외 비율 확인 |
| 소형 | 7/41=17.0732%; 그 외 5/119=4.2017% | 41+119=160, 7+5=12 |
| 저대비 | 7/60=11.6667%; 그 외 5/100=5.0000% | 60+100=160, 7+5=12 |
| 제품 경계 ≤48px | 3/73=4.1096%; 그 외 9/87=10.3448% | 73+87=160, 3+9=12; 경계 FN 집중 주장하지 않음 |
| 영상 경계 5% | 해당 GT 0개 | 비교 분모 없어 영향 판단 불가 |
| 복원 mask 근처 5px | 해당 GT 160개; 비교군 0개 | 근접 여부로 집단 비교 불가 |
| 복원 mask 직접 교차 | FN 0/9; 비교차 12/151=7.9470% | 9+151=160; 표본 수 한계 |
| 기계 1/2/3 | FN 3/72, 1/28, 8/60 | GT 합계 160·FN 합계 12; 독립 인과효과 아님 |
| 크기×대비 | 비소형·일반 1/78, 비소형·저대비 4/41, 소형·일반 4/22, 소형·저대비 3/19 | 분모 합계 160·분자 12; 상승작용 단정하지 않음 |
| FP confidence | 0.20~0.50:0/28; 0.50~0.70:7/105; 0.70~1.00:1/23 | 예측 합계 156·FP 합계 8 |

소형 컷은 Train 유효 특징 816개 중 bbox area Q25=81px², 저대비 컷은 정규화 contrast Q25=0.069891819200이다. Train 전체 828개와 특징 유효 816개를 구분하였다. Train cutoff 코드의 ring은 대상 bbox를 제외하고 Test 영향요인 코드의 ring은 모든 GT를 제외하므로 대비 특징이 완전히 같은 계산이라고 주장하지 않았다. 제외 12개 각각의 사유는 상세 기록이 부족하다.

U0488의 confidence 0.590495·IoU 0.498477은 IoU 0.50 바로 아래여서 FN과 FP를 함께 유발하였다. U0138의 confidence 0.191274·IoU 0.771798은 low threshold 0.20 아래였다. 좌표와 점수는 저장 예측에 대조하고 IoU를 재계산하였다. 이런 개별 사례는 위치·점수 실패를 설명하며 물리적 결함 원인을 증명하지 않는다.

## 5. 비교 모델 및 추가 실험 검증

512장 비교는 YOLO11n·YOLOv8n의 11개 run을 사용하였다. R07/R11과 R06/R10은 저장 설정상 architecture를 제외한 조건이 맞았다. 나머지 입력 크기·batch·증강 변화는 혼합된 조건으로 구분하였다. 취소된 7epoch run 및 20장 PoC는 본 비교표에서 제외하였다. 20장 PoC의 YOLO26n 실제 학습도 별도 부록에 기록하였다.

전체 Val F1 최대는 R11 0.976907이며 R06은 0.976861이었다. 최종 R04는 0.969879였다. R01/R02의 0.971176을 전체 최대로 표현하지 않았다. 최종 선택의 사전 공식 점수함수는 파일에서 확인되지 않아 기존 기준 모델 유지와 Val 지표의 교환관계를 설명하고, Test를 보고 모델을 다시 선정한 것으로 쓰지 않았다.

| 추가 실험 | 확인 결과 | 채택 판단·한계 |
|---|---|---|
| 복원 20장·18조합 | TELEA d0/r3 composite proxy 5.672499, NS 대응 5.774496 | TELEA 선택; detector 성능 또는 clean 영상 오차가 아님 |
| directional 후처리 | line proxy +9.11%, artificial edge +8.97%; 개선 1/20·0/20 | 기각; intensity 개선을 line 개선으로 혼동하지 않음 |
| shortcut 진단 | after-only AUC 0.986068; context matched 약 0.9239; shape excluded 약 0.9014 | 별도 분류기 결과; YOLO shortcut 입증 아님 |
| 20장 fake mark 학습 | 새 fake FP 0, GT 손실·복구 0 | 소규모 진단; 최종 512장 모델 결과 아님 |
| Edge ROI narrow/medium/wide Val | FN 복구 모두 0; 추가 FP 0/3/24 | Val narrow 선택 |
| Edge ROI narrow Test | global/fusion 모두 148/8/12; 복구·추가 FP 0 | 개선 없음 |
| Gaussian/Median/Opening × fixed/Otsu/adaptive × lenient/strict | Val 18개 결과를 모두 보고서 표와 원본 대조 | 추가 FP 예산 3; FN 복구 우선·TP 손실 금지 |
| 선택 Gaussian/Otsu lenient Val | 170/13/5; FN 복구 0·추가 FP 2 | Val 선정과 최종 운영 채택 구분 |
| 선택 설정 Test | 148/12/12; P=R=F1=0.925; FN 복구 0·추가 FP 4 | 기각 |
| 추정영역 안정성 | 649/1163=55.80396%; Train/Val/Test 452/100/97 | 안정성 통과율; segmentation 정확도 아님 |
| 추정영역 Test 분석 | 면적군 FN 5/58·2/13·2/26; 대비군 3/33·3/32·3/32 | 분모 97; 전체 GT 160 기반 분석과 혼합 금지 |

## 6. 불일치 및 확인이 필요한 사항

**제출 전 감사에서 원인 규명 완료 — 서로 다른 추론 전처리 및 IoU 정의.** 원래 저장 PNG의 149/7/11을 같은 weight의 validator 실제 CPU 재실행으로 재현하였다. 동일 캐시에서 자체 매칭과 Ultralytics 매칭을 IoU0.50으로 맞추면 모두148/8/12이며, 캐시를 IoU0.45로 평가하면150/6/10이다. validator는 rect 배치와pad0.5를 사용하여 입력 shape와 예측 좌표가 predict 캐시의 square1280과 다르다. U0303의 IoU가0.455820→0.449256으로 달라 0.45 기준에서도 예측 경로별 집계가 달라졌다. 최종 공식 운영 성능은 잠긴 cache/conf≥0.20/IoU≥0.50의148/8/12이며 기존 PNG 대신 submission_audit/official_test_confusion.png를 사용한다. 과거 GPU 예측 전체의 bitwise 복원은 검증하지 않았다. 상세 분해는 FINAL_SUBMISSION_AUDIT.md와matching_comparison.json에 수록하였다.

| 사항 | 처리 |
|---|---|
| 초기 1134bbox와 최종 1163bbox | 실제 15장 라벨 변화 +29로 설명 완료; label_count_changes.csv 보존 |
| Notebook 일부 모델 선택 서술과 전체 Val F1 순위 | 11개 원본 결과를 기준으로 전체 최대 R11을 명시; 1280 비교 범위와 구분 |
| validator P/R와 고정 threshold P/R | 최적 F1 curve와 운영 고정 threshold 정의 차이로 별도 표기 |
| root train 기본값과 최종 설정 | root는 v8/640/gamma, 최종 저장 설정은 y11/1280/noaug; 재학습 절차에 명시 |
| 현재 .venv 실행 실패 | 과거 Python 설치 경로 부재; 기존 notebook-check-env로 실제 커널 검증 |
| 제조품목·공정·물성·촬영조건 | 파일에 없는 정보는 미확인으로 명시 |
| 정상 음성 영상·실제 정상 판정 성능 | 전체 최종 영상이 GT 양성; specificity·정상 통과율 검증 불가 |
| 생산속도·설비 연동·비용·라인 성과 | 실측 기록 없음; 현장 제안과 구현을 분리 |
| 학습 run별 과거 입력 해시·반복 seed | 현재 입력의 무결성은 검증, 역사적 동일성·분산은 미검증 |
| 완전 재학습 재현 | 제출본에 원본/학습 영상 및 전체 scripts가 부족; 전체 루트 자료 필요 |
| 대비 ring 정의 | Train cutoff와 Test 특징 코드 차이를 보고서에 명시 |

## 7. Notebook·weight·제출 패키지 검증

전체 36셀 중 코드 17셀의 순서는 2, 4, 6, 7, 9, 11, 13, 16, 18, 20, 23, 24, 27, 29, 31, 34, 36이었다. 두 Notebook의 소스가 일치하였다. Python 3.12.14의 실제 Jupyter 커널로 루트·제출본 각각 코드 17셀을 실행하여 오류 0건, 원본 불변을 확인하였다. 실행 증거는 report_audit/kernel_verification.json 및 두 executed Notebook에 있다.

최종 best.pt SHA256:

```text
264f4cdf1b2ef5dd51d7c0fdb827156e1c66fade27348747f91ac55c34073864
```

실제 weight와 threshold/error provenance를 대조하였다. 최초 감사 시 제출 포함 목록 614개의 크기·SHA와 ZIP 615개 파일이 일치하였다. 후속 제출 패키지 보완 후에는 포함 목록 2,725파일과 ZIP 2,726파일이 일치하였으며 모든 ZIP 항목의 SHA·CRC 검증을 통과하였다. Notebook 코드 17셀 재실행과 선택 원본에서 TELEA 512장 픽셀 일치, 학습 입력 사전검사를 통과하였다. 재학습은 실행하지 않았다. Notebook의 실행 전후 입력 596파일 SHA 대조와 FN 후보 12개의 IoU 검증도 통과하였다. 보고서와 report_audit는 소스코드 ZIP에 포함하지 않았다.

## 8. 작성 후 확인 결과

제1장~제6장의 공식 평가 배점 15/40/15/10/10/10과 필수 내용을 확인하였다. 이미지·bbox·분할·TP/FP/FN·P/R/F1·threshold·오류 유형 합계·크기/대비 분모·Notebook 셀 위치를 원본 및 재계산과 대조하였다. 현재 구현은 저장 결과 분석·추론·정책/보조실험 코드이며 생산 설비 연동, 자동 배출 및 독립 정상 판정은 현장 적용 제안으로 구분하였다. 성능 개선이 없는 실험과 추론 경로·IoU 정의가 다른 confusion PNG 차이를 명시하였다.

다음 자동 검증 상세는 직접 재계산 기록에서 생성하였다. 자동 검증 통과는 위의 미확인 사항이 해소되었다는 뜻이 아니다.

문서 자체도 report_audit/report_document_checks.json의 41개 검사로 대조하였다. 공식 6개 장, 미완성 표 자리표시자 부재, Val/Test 모델 성능 22행, 18개 보조 설정, 제안 그림의 실제 경로 및 불일치 명시를 확인하였다. Validation PR curve를 직접 열람하여 표시 mAP50 0.970을 확인하였다. Test F1 curve의 표시값 0.93 at 0.232는 validator 곡선의 설명이며 운영 threshold를 Test에서 0.232로 바꾼 근거로 사용하지 않았다.

## 부록 A. 원본 기반 자동 검사 전체 결과

| 검사 | 결과 | 상세 |
|---|---|---|
| 512 IDs | 통과 |  |
| labels hash/format/count | 통과 |  |
| restored image hashes | 통과 |  |
| original selected label source hashes | 통과 |  |
| test TP/FP/FN rematch | 통과 |  |
| val rematch | 통과 |  |
| FN/FP image counts | 통과 |  |
| telea_512_yolo11n_e200_img1280_b4_adamw_lr0.0005_wd0.0005_noaug_s42 best epoch | 통과 |  |
| telea_512_yolo11n_e200_img1280_b4_adamw_lr0.0005_wd0.0005_noaug_s42 index metrics | 통과 |  |
| telea_512_yolo11n_e200_img1280_b4_adamw_lr0.0005_wd0.0005_noaug_s42_run02 best epoch | 통과 |  |
| telea_512_yolo11n_e200_img1280_b4_adamw_lr0.0005_wd0.0005_noaug_s42_run02 index metrics | 통과 |  |
| telea_512_yolo11n_e200_img1280_b4_adamw_lr0.001_wd0.0005_aug_tr0.005_sc0.2_mos0.2_s42 best epoch | 통과 |  |
| telea_512_yolo11n_e200_img1280_b4_adamw_lr0.001_wd0.0005_aug_tr0.005_sc0.2_mos0.2_s42 index metrics | 통과 |  |
| telea_512_yolo11n_e200_img1280_b4_adamw_lr0.001_wd0.0005_noaug_s42 best epoch | 통과 |  |
| telea_512_yolo11n_e200_img1280_b4_adamw_lr0.001_wd0.0005_noaug_s42 index metrics | 통과 |  |
| telea_512_yolo11n_e200_img640_b4_adamw_lr0.001_wd0.0005_aug_tr0.005_sc0.2_mos0.2_gam0.9-1.1p0.2_noise1-3p0.1_s42 best epoch | 통과 |  |
| telea_512_yolo11n_e200_img640_b4_adamw_lr0.001_wd0.0005_aug_tr0.005_sc0.2_mos0.2_gam0.9-1.1p0.2_noise1-3p0.1_s42 index metrics | 통과 |  |
| telea_512_yolo11n_e200_img640_b4_adamw_lr0.001_wd0.0005_aug_tr0.005_sc0.2_mos0.2_gam0.9-1.1p0.2_s42 best epoch | 통과 |  |
| telea_512_yolo11n_e200_img640_b4_adamw_lr0.001_wd0.0005_aug_tr0.005_sc0.2_mos0.2_gam0.9-1.1p0.2_s42 index metrics | 통과 |  |
| telea_512_yolo11n_e200_img640_b4_adamw_lr0.001_wd0.0005_aug_tr0.005_sc0.2_mos0.2_s42 best epoch | 통과 |  |
| telea_512_yolo11n_e200_img640_b4_adamw_lr0.001_wd0.0005_aug_tr0.005_sc0.2_mos0.2_s42 index metrics | 통과 |  |
| telea_512_yolo11n_e200_img640_b8_adamw_lr0.001_wd0.0005_noaug_s42 best epoch | 통과 |  |
| telea_512_yolo11n_e200_img640_b8_adamw_lr0.001_wd0.0005_noaug_s42 index metrics | 통과 |  |
| telea_512_yolo11n_e200_img960_b4_adamw_lr0.001_wd0.0005_noaug_s42 best epoch | 통과 |  |
| telea_512_yolo11n_e200_img960_b4_adamw_lr0.001_wd0.0005_noaug_s42 index metrics | 통과 |  |
| telea_512_yolov8n_e200_img640_b4_adamw_lr0.001_wd0.0005_aug_tr0.005_sc0.2_mos0.2_gam0.9-1.1p0.2_s42 best epoch | 통과 |  |
| telea_512_yolov8n_e200_img640_b4_adamw_lr0.001_wd0.0005_aug_tr0.005_sc0.2_mos0.2_gam0.9-1.1p0.2_s42 index metrics | 통과 |  |
| telea_512_yolov8n_e200_img640_b4_adamw_lr0.001_wd0.0005_aug_tr0.005_sc0.2_mos0.2_s42 best epoch | 통과 |  |
| telea_512_yolov8n_e200_img640_b4_adamw_lr0.001_wd0.0005_aug_tr0.005_sc0.2_mos0.2_s42 index metrics | 통과 |  |
| weight hash | 통과 |  |
| submission included SHA list | 통과 | [] |
| ZIP matches folder | 통과 | [] |
| notebook sources match | 통과 |  |
| 17 notebook code cells replay | 통과 | [2, 4, 6, 7, 9, 11, 13, 16, 18, 20, 23, 24, 27, 29, 31, 34, 36] |

## 부록 B. 예측·라벨 직접 재매칭

| Split | confidence | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| val | 0.25 | 170 | 11 | 5 | 0.939226519337 | 0.971428571429 | 0.955056179775 |
| val | 0.2 | 171 | 12 | 4 | 0.934426229508 | 0.977142857143 | 0.955307262570 |
| val | 0.31 | 170 | 9 | 5 | 0.949720670391 | 0.971428571429 | 0.960451977401 |
| test | 0.25 | 148 | 8 | 12 | 0.948717948718 | 0.925000000000 | 0.936708860759 |
| test | 0.2 | 148 | 8 | 12 | 0.948717948718 | 0.925000000000 | 0.936708860759 |

## 부록 C. 실제 커널 재실행

| 대상 | 코드 셀 | 오류 | Python | 원본 불변 |
|---|---:|---:|---|---|
| root | 17 | 0 | 3.12.14 | True |
| submission | 17 | 0 | 3.12.14 | True |

## 부록 D. 제출 전 감사에 따른 정정

최종 1163개의 bbox 크기 통계를 실제 TXT와 영상 헤더로 다시 계산하였다. 초기 EDA의 면적 중앙값100/Q75=140/평균114.322442 및 종횡비 최대2.0은 최종 통계가 아니었다. FINAL_REPORT_DRAFT.md §1.3을 중앙값104/Q75=143/평균115.890800 및 종횡비 최대1.833333으로 정정하였다. Q25=81과 소형 FN7/41 등 오류분석의 기준·분모에는 변경이 없었다.

제출 전 감사에서 원본 기반108개 검사를 통과하였다. 추가 기록은 submission_audit/number_checks.json과 all_report_numbers.csv이다. 이전41개 문서 검사 기록은 첫 초안 시점의 검증이며 최종 감사 결과로 대체한다. 이번 감사는 새로운 학습을 수행하지 않았고 별도의 CPU validator/predict 추론을 진단용으로 실행하였다. 원래 저장 AP와 새로운 CPU AP의 미세한 차이를 숨기거나 덮어쓰지 않았다.

실제 모델은 YOLO11n/YOLOv8n의 두 구조이고 11run의 파라미터 상태는10종이었다. 최초 감사에서 누락된 YOLOv8n weight는 후속 보완으로 다른 완료 run과 함께 제출 ZIP에 포함하였다. 학습 영상·라벨·선택 원본·복원 마스크·코드·Test 전용 예측도 포함하고 ZIP·included_files.json을 재생성하였다. 검증 기록은 submission_audit/package_repair_result.json에 있다. 계획서 대응, README 학습 환경 및 최종 한글 문서·제출 파일 구성은 제출 전 사람 확인이 필요하다.
