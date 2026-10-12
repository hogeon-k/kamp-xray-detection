# PPT 및 PDF 생성 기록

생성일: 2026-10-07. 결과: PPTX 10장 및 PPTX에서 변환한 PDF 10페이지 생성 완료.

## 입력과 구성 변경

입력 문서 5개를 읽고 SELECTED_FIGURES.md의 이미지 경로 13개(본문 선정 8개, 제외/질문 대응 5개)를 확인하였다. 모든 경로가 존재했으며 누락 이미지는 없었다. 최신 요청에 맞춰 기존 10장 계획에서 문제 정의와 데이터 구성을 분리하고, 현장 활용과 결론을 마지막 장으로 합쳤다. 원본 입력 문서와 프로젝트 이미지·학습 산출물·제출 ZIP은 변경하지 않았다.

| 슬라이드 | 제목 | 이미지 또는 대체 구성 |
|---|---|---|
| 1 | KAMP X-ray 이물질 검출과 오류분석 | 001_20200814_163105(1).png |
| 2 | 문제 정의 | 텍스트 |
| 3 | 데이터 구성 | 편집 가능한 표 |
| 4 | TELEA 전처리 | 01_4184354a2a3c_comparison.png |
| 5 | 모델 비교 | 편집 가능한 표 |
| 6 | Threshold 선정 | val_threshold_curve.png |
| 7 | 최종 Test 성능 | official_test_confusion.png |
| 8 | 오류분석 | U0138_FN_gt0.png, U0371_FP_pred0.png |
| 9 | 창의성: fake-mark 역검증 | overlay.png, pair_0002.png |
| 10 | 현장 활용 및 결론 | 편집 가능한 표 |

## 사용 이미지

아래 8개 원본 이미지를 11개 이미지 객체에 삽입하였다. 전처리 비교·오류 사례·paired 반응은 PPTX의 native crop으로 확대하였다. 원본 이미지 바이트는 수정하지 않았으며 PPTX에 포함된 이미지 해시가 원본과 일치한다.

| 슬라이드 | 원본 이미지 경로 | 확인 |
|---|---|---|
| 1 | [001_20200814_163105(1).png](C:/workspace/Kamp_Xray/outputs/yolo_512_telea/images/train/001_20200814_163105(1).png) | 존재·SHA 일치 |
| 4 | [01_4184354a2a3c_comparison.png](C:/workspace/Kamp_Xray/outputs/inpainting_poc_20/figures/01_4184354a2a3c_comparison.png) | 존재·SHA 일치 |
| 6 | [val_threshold_curve.png](C:/workspace/Kamp_Xray/outputs/safety_threshold_policy/visualizations/val_threshold_curve.png) | 존재·SHA 일치 |
| 7 | [official_test_confusion.png](C:/workspace/Kamp_Xray/submission_audit/official_test_confusion.png) | 존재·SHA 일치 |
| 8 | [U0138_FN_gt0.png](C:/workspace/Kamp_Xray/outputs/error_analysis/fn/U0138_FN_gt0.png) | 존재·SHA 일치 |
| 8 | [U0371_FP_pred0.png](C:/workspace/Kamp_Xray/outputs/error_analysis/fp/U0371_FP_pred0.png) | 존재·SHA 일치 |
| 9 | [overlay.png](C:/workspace/Kamp_Xray/outputs/shortcut_diagnostic_gpu/cases/E_largest_delta_pair_0002/overlay.png) | 존재·SHA 일치 |
| 9 | [pair_0002.png](C:/workspace/Kamp_Xray/outputs/shortcut_diagnostic_dataset_gpu/masks/pair_0002.png) | 존재·SHA 일치 |

## 누락 이미지 및 대체 구성

누락 이미지: **0개**. 누락에 따른 대체 제안은 필요하지 않았다. 문제 정의는 텍스트, 데이터 구성·모델 비교·threshold·현장 활용은 5행 이하의 편집 가능한 표로 구성하였다. 생산라인·정상 제품 이미지 등 프로젝트에 없는 그림은 생성하지 않았다.

## 숫자 및 표현 검증

| 항목 | 사용 값 | 근거 |
|---|---|---|
| 이미지 / bbox | 512 / 1,163 | outputs/yolo_512_telea/manifest.csv |
| Train / Val / Test | 359 / 77 / 76 | 동일 manifest 및 split_manifest.csv |
| Train / Val / Test GT | 828 / 175 / 160 | 동일 manifest |
| TP / FP / FN | 148 / 8 / 12 | submission_audit/official_test_confusion.json 및 error_analysis summary |
| Precision / Recall / F1 | 94.87% / 92.50% / 93.67% | 148/156, 148/160, 296/316 반올림 |
| 소형 객체 FN | 17.07% (7/41) | gt_condition_rates.csv 및 error_analysis summary |
| 저대비 객체 FN | 11.67% (7/60) | 동일 오류 자료 |
| 모델 비교 Val F1 | R07 95.95% / R11 97.69% | 각 run summary의 P/R 조화평균 |
| 모델 비교 Val mAP50 | R07 93.21% / R11 97.14% | 각 run summary |
| Val threshold 0.25 / 0.20 / 0.02 | FN 5/4/3, FP 11/12/117 | val_threshold_sweep.csv |
| fake-mark | 20장 PoC, 34pair, holdout 10pair, 새 fake FP 0 | shortcut_diagnostic_gpu/summary.json |

공식 그림은 official_test_confusion.png이며 149/7/11 validator confusion을 사용하지 않았다. 최종 모델은 YOLO11n R04 유지로 설명하고 전체 지표 최고 선정이라고 주장하지 않았다. Test에서 threshold를 재조정했다고 표현하지 않았다. fake-mark는 표시 mask를 GT 밖으로 재배치·재복원한 흔적 진단이며 최종 512장 모델의 shortcut 부재 증명으로 해석하지 않았다. 현장 활용은 제안이며 설비 연동·자동 배출·시간 절감 효과를 검증 실적으로 제시하지 않았다.

## 생성과 변환 방법

- PPTX: 제공된 Node.js 런타임의 @oai/artifact-tool로 생성. 16:9(1280×720), 맑은 고딕, 남색·회색·흰색. 제목 32.25pt(표지 36pt), 본문 16.5~21.75pt, 핵심 수치는 48pt와 강조 박스를 사용하였다.
- 발표자 노트: 10장 모두 해당 대본과 원본 근거 경로를 포함하고, 저장된 PPTX XML에서 대본 일치를 확인하였다.
- 이미지 crop: 라이브러리가 직접 지정한 crop을 중앙 맞춤으로 덮어쓰는 현상을 확인해 native a:srcRect 메타데이터를 수정하였다. 이미지 픽셀을 편집하거나 새 그림을 생성한 작업이 아니다. 수정 후 PPTX를 다시 읽어 전체 슬라이드를 검토하였다.
- PDF: 완성된 루트 KAMP_Xray_Presentation.pptx를 다시 import → 각 슬라이드 3200×1800 PNG 렌더링 → 960×540pt PDF로 변환. 240DPI의 이미지 기반 PDF이며 텍스트 선택·검색은 지원하지 않는다. Markdown을 따로 조판해 만든 PDF가 아니다.
- PowerPoint·LibreOffice 변환기는 이 환경에서 확인되지 않아 사용하지 않았다. PowerPoint 프로그램에서 직접 열거나 변환했다고 주장하지 않는다.
- PDF를 Poppler pdftoppm으로 다시 렌더링해 10페이지를 각각 열람하였다. 제목·수치·표·이미지·crop·페이지 순서와 겹침/잘림을 확인하였다.

## 생성 결과 및 검증

PPTX 패키지 무결성·폰트 정책·10장·16:9·편집 가능한 표 4개·bullet/heading geometry 및 재import 검사를 통과하였다. PDF 페이지 수와 16:9 MediaBox를 확인하였다. 원본 입력 문서 5개의 생성 전후 SHA가 일치한다. 모든 필수 이미지가 내장되어 다른 컴퓨터에서 원본 경로를 찾을 필요가 없다.

- [KAMP_Xray_Presentation.pptx](C:/workspace/Kamp_Xray/KAMP_Xray_Presentation.pptx): 1,337,301 bytes, SHA256 `c562004aca0579a229bdbd569ad15b426d0a602c56c015679bb8d6edb91ddf25`
- [KAMP_Xray_Presentation.pdf](C:/workspace/Kamp_Xray/KAMP_Xray_Presentation.pdf): 2,671,357 bytes, SHA256 `c82f27b942c5c2ab2a1c2f5fd98e1b58aa39d1347534e6888881c0c3f9004a4b`

빌드 소스와 기록: presentation_build/build_submission.mjs, fix_native_crops.py, convert_pptx.mjs, pack_pdf.py, submission_inputs.json, submission_validation.json. 중간 렌더링은 presentation_build/submission_previews 및 pdf_review에 저장하였다.

## 제출 전 사람이 확인할 사항

팀명·발표자명은 제공되지 않아 임의 기재하지 않았다. 필요하면 표지에 실제 정보를 추가한다. 계획서 원문과 R04 유지 설명의 대응, 제출 포털의 파일 용량·표지 필수 항목을 확인한다. PDF의 화면 표시와 PPTX 편집 구조는 검증했으나 다른 컴퓨터의 PowerPoint 글꼴 렌더링까지 검증한 것은 아니다.
