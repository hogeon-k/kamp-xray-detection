# 제출 전 최종 감사

감사일: 2026-10-07. 대상: REPORT_EVIDENCE.md, FINAL_REPORT_DRAFT.md, REPORT_VALIDATION.md. 보고서를 재작성하지 않고 모순·근거 연결·평가기준 미충족 위험을 조사하여 필요한 문단만 정정하였다. 아래 판단은 실제 심사 점수의 예측이 아니라 제공된 평가 문구에 대한 충족도 판단이다.

## 1. 발견된 문제와 처리 결과

| 문제 | 감사 전 | 감사 결과·처리 |
|---|---|---|
| Test 집계 모순 | 자체148/8/12와 PNG149/7/11의 잔여 차이 미해명 | 동일 weight의 validator를 실제 실행하여149/7/11 재현; predict/val 전처리와 IoU 정의 차이를 이미지별로 확인 |
| 최종 bbox 크기 통계 오류 | 초기 XML EDA의 median100/Q75=140/mean114.322442/aspect max2를 최종1163개 통계로 사용 | 실제 TXT에서 median104/Q75=143/mean115.890800/aspect max1.833333을 계산하고 제1장 수정 |
| 본실험11개 표현 | 모델 구조·설정 변경·동일 seed 반복이 혼동될 가능성 | 2구조의11학습run, 파라미터 상태10종으로 구분; threshold/후처리 실험 제외 명시 |
| 구조 비교의 제출 근거 누락 | 기존 제출 패키지에 YOLOv8n 비교 weight 없음 | 실제 R07/R11의 weight·config·summary·results·args·환경과 SHA를 보강 ZIP으로 준비 |
| 계획서 대응 미확인 | 실제 계획서 파일 없음 | ‘계획서에 따른 선정’이라고 단정하지 않음; 사람 확인 필수 |
| 최종 모델 선정 논리 | Val F1 최고 R11/R06과 최종 R04가 다름 | 사후 유지 설명과 사전 선택 규칙을 분리; 실제 manifest retained_previous_baseline 근거 명시 |
| AP 근거 경로 오류 | 검증 문서가 존재하지 않는 test_evaluation/summary.json을 지칭 | 실제 run/summary.json의 test_map50/test_map50_95로 정정 |
| 전체 재현 범위 부족 | 제출본에 학습 영상·전체 scripts 누락 | 저장 결과 재현과 완전 재학습을 구분하고 미충족 위험 명시 |

원본 모델·예측·라벨·Notebook·기존 제출 ZIP을 변경하지 않았다. 새 진단 추론·confusion 그림·보강 ZIP은 submission_audit/에 별도 저장하였다. 감사 전 세 문서는 submission_audit/before_*.md로 보존하였다.

## 2. TP/FP/FN 차이의 원인 규명

### 2.1 입력과 평가 설정 비교

| 항목 | 공식 운영 평가 | 저장 validator confusion |
|---|---|---|
| 모델 | R04 best.pt | 동일 R04 best.pt |
| 모델 SHA | 264f4cdf1b2ef5dd51d7c0fdb827156e1c66fade27348747f91ac55c34073864 | 실제 재실행에서 같은 SHA 확인 |
| 예측 경로 | analyze_foreign_object_conditions.py의 model.predict() 저장 캐시 | train_yolo.py의 model.val(split=test) |
| 원본 prediction 파일 | outputs/foreign_object_condition_analysis/predictions.json | 당시 전체 좌표 JSON 미저장; 이번 instrument_validation.py가 새 기록 |
| GT | 최종 labels/test 76 TXT·160bbox | 같은 TXT의 복사본; float32 좌표 변환만 다름 |
| 실제 입력 크기 | predict 재실행76장 모두1280×1280 | rect=True; 1024×1312,1248×1312,1312×1248 |
| resize/padding | predict의 square letterbox | validator의 aspect ratio 정렬·사각 배치·pad0.5 |
| raw confidence 하한 | 0.001 | val 기본0.001 |
| 평가 confidence | 공식≥0.20; baseline≥0.25도 같은 개수 | confusion>.25 |
| NMS IoU / max_det | .70 /300 | .70 /300 |
| augmentation | False | False |
| NMS class 옵션 | class-aware, agnostic_nms=False | class-aware, single_cls=False·agnostic_nms=False |
| NMS multi_label | 기본False | True; 실제1클래스이므로 다중 클래스 출력 차이 없음 |
| class filtering | class_id=0만 포함 | 클래스별 matrix; 실제 데이터·예측 모두class0 |
| GT 매칭 IoU | ≥.50 | >.45 |
| 매칭 | IoU 내림차순 greedy; GT·prediction 사용 여부를 동시 검사 | IoU 내림차순→prediction 중복 제거→재정렬→GT 중복 제거 |
| confidence/IoU 동률 | ≥는 포함 | >는 제외; 실제 cache에 .20/.25 confidence 및 .45/.50 IoU 정확한 동률0 |

근거: train_yolo.py의 _evaluation_args/run_test, scripts/analyze_foreign_object_conditions.py:221–248, scripts/error_analysis.py:66–80, Ultralytics 8.4.158의 engine/model.py의 val defaults, data/build.py의 validation pad, engine/predictor.py의 pre_transform 및 utils/metrics.py의 ConfusionMatrix.process_batch. 이번 실행의 args·입력 shape·GT·전체 prediction은 submission_audit/validator_replay.json과 predict_replay.json에 보존하였다.

직접 TXT 좌표와 validator가 원좌표로 역변환한 GT의 최대 차이는 약0.000092px였다. GT 클래스나 개수는 달라지지 않았다. cache와 validator의 confidence>.25 및 IoU>.45/.50 적격 그래프를 모두 검사하였고, 하나의 GT에 여러 prediction 또는 하나의 prediction에 여러 GT가 동시에 적격인 영상은 각각0개였다. 따라서 두 중복 제거 순서의 일반적인 차이는 존재하지만 이번148/8/12 차이의 원인은 아니었다. 동일 cache에서 실제 Ultralytics ConfusionMatrix를 사용해도 IoU.50은148/8/12였다.

다중 적격 쌍이 있다면 자체 방식은 가장 높은 IoU 쌍부터 두 대상을 함께 사용 처리하고 남은 적격 쌍으로 진행한다. Ultralytics confusion 방식은 prediction별 최고 겹침을 먼저 남긴 뒤 GT별 최고 겹침을 남겨, 일부 대체 매칭으로 다시 돌아가지 않는다. 단일 클래스라서 오분류를 FP/FN으로 나누는 class 처리도 이번 차이에 영향을 주지 않았다.

### 2.2 조건을 하나씩 맞춘 분해

| 예측 입력 | 매칭 | TP | FP | FN | 해석 |
|---|---|---:|---:|---:|---|
| 기존 predict cache | 자체 confidence≥.25·IoU≥.50 | 148 | 8 | 12 | 기존 운영 baseline |
| 같은 cache | Ultralytics confidence>.25·IoU>.50 | 148 | 8 | 12 | 알고리즘·경계값 조건 차이는 이번 집계에 영향 없음 |
| 같은 cache | Ultralytics confidence>.25·IoU>.45 | 150 | 6 | 10 | U0303·U0488의 추가 매칭 |
| validator 재실행 prediction | Ultralytics confidence>.25·IoU>.45 | 149 | 7 | 11 | 원래 PNG의 집계 재현 |
| validator 재실행 prediction | 자체 confidence≥.25·IoU≥.50 | 147 | 9 | 13 | 별도 전처리 경로의 진단값; 공식 성능으로 교체하지 않음 |
| predict 재실행 prediction | 자체 confidence≥.25·IoU≥.50 | 148 | 8 | 12 | cache 경로 자체도 CPU 실제 추론으로 재현 |

validator CPU 재실행의 AP50/95는 약0.934377/0.407386이며 과거 GPU 저장값0.934643/0.407136과 미세하게 달랐다. 원래 결과를 덮어쓰거나 새 값을 과거 성능으로 바꾸지 않았다. 재실행에서 같은 confusion 개수와 원인 경로를 재현한 것이며 과거 GPU 좌표 전체의 bitwise 복원은 아니다.

### 2.3 이미지별 차이

표의 prediction 번호는 각 파일 안의 index이다. IoU는 실제 TXT의 동일 GT와 원좌표 prediction으로 계산하였다. 모든 후보의 클래스는0이다.

| 이미지 | GT (x1,y1,x2,y2) | Prediction·score·IoU | 자체 계산 / cache IoU.50 | validator confusion / IoU.45 | 차이 원인 |
|---|---|---|---|---|---|
| U0488 | gt0=(249,173,260,184) | cache p0 score.590495, IoU.498477; val p0 score.609888, IoU.500334 | gt0 FN+p0 FP; 영상0/1/1 | gt0 TP; 영상1/0/0 | IoU 기준.50→.45와 사각배치 예측 좌표 차이; 공식 두 결과의 개수 차이에 해당 |
| U0303 | gt2=(370,276,378,287) | cache p2 score.512530, IoU.455820; val p1 score.504317, IoU.449256 | 영상1/2/2 | 영상1/2/2 | 두 공식 집계에서는 동일 실패. cache를.45로만 바꾸면 TP가 늘지만 val은.45 아래여서 늘지 않음. 이전150/6/10 대149/7/11의 잔여1개를 설명 |
| U0207 | gt0=(363,228,372,237) | cache p2 score.366193, IoU.505926; val p2 score.375091, IoU.484472 | 영상3/0/0 | 영상3/0/0 | val도.45 통과하여 공식 집계 동일; val prediction을.50에 맞추면 추가 FN/FP 발생 |
| U0299 | gt2=(211,173,217,182) | cache p1 score.513702, IoU.501667; val p1 score.591904, IoU.472852 | 영상3/0/0 | 영상3/0/0 | val도.45 통과하여 공식 집계 동일; val prediction을.50에 맞추면 추가 FN/FP 발생 |

전체 차이4개 영상의 모든 bbox·score·매칭 index는 matching_comparison.json에 있다. U0488에서 운영 기준의 FN/FP 하나가 validator 기준의 TP로 바뀌어148/8/12→149/7/11을 설명한다. U0303은 같은 cache에서 IoU만 낮춘150/6/10이 원래 PNG와 달랐던 이유이다. NMS IoU·클래스·GT 변경이나 threshold 동률로 차이를 설명할 근거는 없었다.

### 2.4 최종 공식 값과 confusion 그림

최종 공식 **운영 객체 단위 Test 성능은148/8/12**로 결정하였다. 이는 이미 Val에서 잠근 정책, 저장 캐시, FN/FP 사례 및 조건별 분모가 연결된 동일 평가 경로이다. P=148/156=94.8718%, R=148/160=92.5000%, F1=296/316=93.6709%이다. 저장 validator AP는 별도의 원본 지표로 유지한다.

기존149/7/11 PNG는 공식148/8/12 성능의 그림으로 사용할 수 없다. 다른 조건의 비교자료로 쓸 경우 ‘validator rect 입력·IoU>.45’라는 별도 제목이 필요하지만 제출 본문에서는 혼동을 줄이기 위해 제외를 권고한다.

같은 조건의 새 그림은 submission_audit/official_test_confusion.png, 수치 JSON은 official_test_confusion.json, 이미지별 표는 official_test_matching.csv이다. 재생성 방법:

```powershell
.\.cache\notebook-check-env\Scripts\python.exe submission_audit\generate_official_confusion.py
```

코드는 frozen cache, 최종 TXT, confidence≥.20, class0, IoU≥.50, IoU 내림차순 일대일 매칭으로 생성한다. 새 추론·학습 없이 재생성한다. 그림의 background/background는 TN=0이 아니라 N/A이다. 정상 음성 영상이 없는 객체 탐지 평가에서 TN이나 specificity를 산출하지 않았다.

## 3. 제2장 요건 감사 — 11개 실험의 정확한 분류

각 학습 run에 실제 config·results.csv·summary·best.pt가 존재하였다. 모델 checkpoint를 직접 로드하여 구조와 파라미터 수·state SHA를 확인하였다. YOLO11n은 fusion 전2,590,035개 파라미터, YOLOv8n은3,011,043개였고 두 구조의 YAML 및 state도 달랐다. 11개 파일의 SHA는 모두 달랐지만 R01/R02의 모델 state SHA는 동일하여 파라미터 상태는10종이었다. 이는 새로운11종 모델이 있다는 뜻이 아니다.

| 실험 | 모델 구조 | 학습 조건 (입력/batch/lr/증강) | threshold만 변경? | 별도 학습 모델? | 베이스라인 비교 인정 가능성 |
|---|---|---|---|---|---|
| R01 | YOLO11n | 1280/4/.0005/없음 | 아니오 | 별도run·weight | 학습률 비교 가능; R04와 구조 동일 |
| R02 | YOLO11n | R01과 동일 | 아니오 | 별도run; R01과 state 동일 | 반복성 확인용; 독립 개선·새 구조로 인정하지 않음 |
| R03 | YOLO11n | 1280/4/.001/기본 | 아니오 | 예 | R04와 증강만 다른 재학습 비교 |
| R04 | YOLO11n | 1280/4/.001/없음 | 아니오 | 예 | 기존 후속 분석 기준 모델; 모든 Val 지표 최고 아님 |
| R05 | YOLO11n | 640/4/.001/기본+감마+노이즈 | 아니오 | 예 | R06/R07과 증강 비교; 구조 동일 |
| R06 | YOLO11n | 640/4/.001/기본+감마 | 아니오 | 예 | R10과 같은 조건의 구조 비교 가능 |
| R07 | YOLO11n | 640/4/.001/기본 | 아니오 | 예 | R11과 같은 조건의 구조 비교 가능 |
| R08 | YOLO11n | 640/8/.001/없음 | 아니오 | 예 | R04/R09와 입력·batch 혼합 차이; 순수 해상도 효과 아님 |
| R09 | YOLO11n | 960/4/.001/없음 | 아니오 | 예 | R04와 입력 크기 비교; 구조 동일 |
| R10 | YOLOv8n | 640/4/.001/기본+감마 | 아니오 | 예 | R06과 구조 비교 가능; 기존 제출본에 weight 없음 |
| R11 | YOLOv8n | 640/4/.001/기본 | 아니오 | 예 | R07과 구조 비교 가능; 기존 제출본에 weight 없음 |

공통 조건은 AdamW, 최대200epoch, patience50, seed42, weight_decay.0005 및 동일359/77/76 분할이다. 실제 best·실행epoch는 보고서 실험 설정표와 원본 results.csv에 있다. R07/R11, R06/R10의 architecture 외 config 차이는 run명·실행시간 등 식별 정보이며 핵심 학습·평가 설정은 맞았다. 현재 자료의 동일성은 확인하였지만 과거 run별 입력 해시는 미기록이다.

threshold 탐색·Two_threshold 정책은 같은 R04 weight의 평가 정책 변경이다. Edge ROI는 같은 R04의 crop 신규 추론·fusion 후처리이고, Gaussian/Median/Opening은 weak proposal과 영상처리 후처리이다. 이 실험들은 별도 학습 모델이나 새로운 architecture로 세지 않았다. 감마·노이즈는 학습 입력 증강이며 실제 재학습·별도 weight가 있어 학습 조건 비교로 인정할 근거가 있다.

**판정:** 프로젝트 전체에서는 두 실제 모델 구조의 동일 조건 비교가 확인되어 기술적 요건을 뒷받침한다. 제출 패키지 보완으로 완료 11run의 best.pt가 포함되어 비교 weight 누락은 해결하였다. 그러나 ‘계획서에 따라’의 계획서 원문은 확인되지 않았다. 최종 선정은 ‘Val F1 최대 선정’이 아니라 기존 R04 유지이다. 그러므로 공식 기준을 무조건 충족한다고 보증할 수 없다.

최소 보강은 추가 학습이 아니다. R07/R11을 구조 비교의 기준 쌍으로 명시하고 기존 학습 원본을 제출하는 것이다. 이를 위한 model_comparison_supplement.zip을 준비하였다. 내부에는 두 weight·experiment_config.json·summary.json·results.csv·args.yaml·environment.txt와 SHA 목록이 있다. 원래 제출 ZIP에는 자동 합치지 않았다. 계획서의 베이스라인·모델 선정 문구와 이 쌍의 대응은 사람이 확인해야 한다. 계획서가 반드시 최고 Val F1 모델의 최종 채택을 요구한다면 글만 고쳐 충족시킬 수 없으며 그 모델의 정책·오류분석 재실험이 필요하다.

## 4. 제1~6장 평가항목별 감점 위험

| 장 | 배점 | 현재 충족도 | 감점 위험 | 수정 필요·판정 |
|---|---:|---|---|---|
| 제1장 데이터 이해 및 진단 | 15 | 변수·출처·분할·전처리·품질 진단 충분; 크기 통계 정정 완료 | 구체 제조품목·공정/촬영 정보 미제공; 초기 EDA 혼용 시 모순 | **보강 권장** — 품목·공정의 확인 가능한 정보가 있으면 추가; 없으면 현 한계 유지 |
| 제2장 AI 모델 및 성능 | 40 | 두 구조·동일 조건2쌍·Val/Test·정책·선정 차이 제시 | 계획서 대응 미확인, 비교 weight 제출 누락, 최종 유지의 사전 기준 부족 | **반드시 수정** — 준비한 비교 근거를 제출에 포함하고 계획서 대응 확인; 본문 모순은 수정 완료 |
| 제3장 영향요인·오류 | 15 | FN/FP·사례·크기/대비/위치 분모와 기전 구분 충분 | 작은 표본·중복 조건·Train/Test ring 정의 차이; 인과·물리 원인 미확정 | **충분함** — 원인 단정 금지 및 분모 유지; 정의를 통일한 재분석은 선택 보강 |
| 제4장 현장 활용 | 10 | 현재 구현/제안 구분, routing·FN/FP 관리·역할 분담 충분 | 정상 음성 자료·현장 실증 없음; 정상 보증·자동 배출 주장 불가 | **충분함** — 실제 구현 성과와 제안 구분 유지 |
| 제5장 창의성·차별성 | 10 | threshold·후보IoU·조건 분석·18설정 및 기각 근거 제시 | Test FN 개선0, 높은 AUC를 shortcut 입증으로 확대하면 감점 | **충분함** — 효과 없는 실험을 기각한 분석 기여로 제시 |
| 제6장 재현성 | 10 | Notebook 17셀 실행·전처리 512장 픽셀 일치·학습 입력 및 코드 포함·ZIP 해시 통과 | 재학습 실행 및 최초 원본 후보 전체 감사는 미검증 | **보강 권장** — README 환경·실행 방법을 확인하고 검증 범위를 정확히 표시 |

공식 제2장 문구의 ‘2개 이상 모델’은 두 구조·실제 weight로 뒷받침하였으나 ‘계획서에 따라’는 자료가 없어 미확정이다. 제6장 문구의 ‘전처리부터 학습·추론·결과생성까지’는 기존 제출본만으로는 충족하지 못한다. 이 두 사항은 문장 길이를 늘려 해결할 수 없다.

## 5. 과장·근거 없는 표현 감사 및 수정안

전체 본문을 문장 단위로 검토하였다. 실제 라인 배포·생산 효과·촬영 전압·특정 제조품목·자동 배출을 구현한 것으로 단정한 문장은 없었다. 다음은 수치 오류 또는 오해 위험이 있어 수정·유지 판단을 기록한 문장 전체 목록이다. 근거 없는 ‘향상하였다/개선하였다’의 단정 문장은 발견하지 못했으며 조건별 증강 비교의 상승/하락은 원본 표로 확인하였다.

| 원문/대상 문장 | 위험·근거 | 수정 또는 판단 |
|---|---|---|
| ‘1163개…중앙값100…Q75=140…평균114.322442…범위…2.0’ | 초기 EDA 통계를 최종 TXT 통계로 잘못 사용 | 최종 TXT 직접 계산104/143/115.890800/1.833333으로 수정 |
| ‘당시 validator와 캐시의 예측 차이는 추가 확인이 필요’ | 미해결로 남겨 평가 일관성 손상 | 같은 weight 재실행·image별 IoU·입력 shape로 규명하고 공식 값/그림 결정 |
| ‘완료 실험…총11개’ | 독자가11개 서로 다른 구조로 이해할 가능성 | 2구조의11run, 반복 state 동일 및 threshold/후처리 제외를 추가 |
| ‘기존에 잠긴 모델…연결…사후 판단’ | 사전 객관적 선정 기준처럼 읽힐 가능성 | manifest의 retained_previous_baseline/false를 인용하고 사후 문서화임을 명시 |
| ‘저장 결과와 라벨·weight의 분석 재현은 가능’ | 모든 비교 weight가 제출되어 있다는 오해 | ‘포함된’ weight로 한정하고 v8 누락·보강 ZIP·전체 입력 부족을 명시 |
| ‘X-ray 검사 영상은 제품의 내부 구조와 국소적인 명암 차이를 보여 주며…’ | 일반적인 투과영상 설명; 특정 실제 품목/재질 정보로 확대될 위험 | 개념 설명으로 유지; 다음 문단의 품목·촬영조건 미확인 제한과 함께 읽도록 구성 |
| ‘소형 및 저대비…높은 FN 비율’ | 인과 주장으로 확대될 위험 | 실제7/41 대5/119·7/60 대5/100의 관찰만 제시하고 독립·인과 효과 아님을 명시한 문장 유지 |
| ‘기대효과는…위치정보 제공…점검 우선순위’ | 실측 생산 효과로 오해될 위험 | 제4장 현장 적용 제안의 기대 항목으로 유지; 시간/인력/비용 수치는 없음 |
| ‘신뢰성/차별점…효과 없는 후처리를 배제’ | 실제 Test 향상으로 오해될 위험 | 제5장 Test 개선 부재·기각 결과를 유지; 신규 알고리즘이나 운영 성능 향상 주장 없음 |

NgImage를 전문가 확정 불량으로, confidence를 보정 확률로, 정상 후보4장을 정상 정답으로, bbox 면적을 실제 이물질 물리적 크기로, AUC를 YOLO shortcut 입증으로 표현하지 않았음을 재확인하였다. 양품 오경보율·specificity·생산 불량률·처리속도·금전효과는 산출하지 않았다.

## 6. 숫자 전수검사

수정 후 본문에 등장하는 숫자는 submission_audit/all_report_numbers.csv에 **등장 위치·문맥·역할·원본 근거**와 함께 전수 추출하였다. 날짜·목차·공식 배점·모델명·파일경로·해시 숫자도 제외하지 않고 식별자와 정량값을 구분하였다. 같은 숫자가 원본 어딘가에 있다는 이유만으로 의미가 검증된 것으로 처리하지 않았다.

원본 기반 검사는108개를 통과하였다(number_checks.json). 여기에는 모델별 Val/Test22행의110개 지표,11개 학습설정 행,18개 보조설정 행,7개 Val threshold의 직접 재매칭, 최종512라벨의 개수·면적·종횡비·영상헤더,날짜·설비,오류 유형·조건 분모·대표 사례·FP bins·Edge 결과 등이 포함된다. 감사에서 발견한 수치 오류는 최종 bbox 크기 통계였으며 수정하였다. 원래 의미가 다른 초기1134·20장PoC·149/7/11은 삭제하거나 동일 값으로 바꾸지 않고 역할을 구분하였다.

| 핵심 검사 | 확정 값·직접 계산 | 판정 |
|---|---|---|
| 이미지/bbox | 512/1163 | 실제 라벨·manifest 일치 |
| Train/Val/Test | 이미지359/77/76; bbox828/175/160 | 합계·실파일 일치 |
| 공식 Test | TP148 FP8 FN12 | 잠긴 캐시 직접 재매칭 |
| P/R/F1 | .948717948718/.925/.936708860759 | 148/156,148/160,296/316 |
| FN/FP 발생 이미지 | 8/4 | error_cases unique_id 재집계 |
| FN 주 기전 | 8/3/1=12 | 상호 배타 주 분류 |
| FP 기전·unknown | localization8/8; physical unknown20/20 | 기전 unknown0과 물리 원인 구분 |
| 소형/비소형 | 7/41=17.07%;5/119=4.20% | 분모160·FN12 |
| 저대비/일반 | 7/60=11.67%;5/100=5.00% | 분모160·FN12 |
| 제품 경계/비경계 | 3/73=4.11%;9/87=10.34% | 48px 정의, 경계 집중 없음 |
| 영상 경계/복원 근접 | 영상 경계 GT0;복원 근접 GT160·비근접0 | 비교 불가를0% 효과로 해석하지 않음 |
| 임계값 | Val 선정.20/.25;Test 모두148/8/12 | Test 재튜닝 없음 |
| 보조검사 | Val170/13/5;Test148/12/12;FN복구0·FP+4 | 기각 유지 |
| Edge ROI | narrow Test148/8/12;복구0·추가FP0 | 개선 없음 |
| 최종 bbox 크기 | area21/81/104/143/420/115.890800;aspect max1.833333 | 최종 TXT 직접 재계산으로 정정 |

조건별 비율은 탐색적 표본 관찰이며 신뢰구간·독립 효과 검정을 수행한 것으로 쓰지 않았다. 문서 숫자 전수 색인은 계산 감사의 위치를 찾는 자료이고 모든 문장의 의미를 자동 증명하는 프로그램은 아니다.

## 7. 그림·표 감사 및 최종 선정

기존 제안의11개 실제 PNG와 새 공식 confusion1개를 모두 열어 확인하였다. 경로가 모두 존재하였다. 비교용 contact sheet는 submission_audit/figure_review_1.png와 figure_review_2.png에 보존하였다. 표는 원본 CSV/JSON 또는 직접 재계산에 대조하였다.

| 자료 | 파일/수치·본문 일치 | 중복·심사 기여도 | 최종 권고 |
|---|---|---|---|
| 데이터 분할·해상도 표 | 512/1163 및3해상도 합계 일치 | 데이터 범위·누수 설명 핵심 | 필수 표 |
| 11run 설정·Val 성능 표 | 원본 config/log/summary 일치 | 같은 조건2구조 쌍을 강조하면 모델 요건 증거 | 필수 표; R01/R02 반복을 새 모델로 세지 않음 |
| Test validator11행 표 | summary와 일치 | Val 표와 함께 길어질 수 있음 | 선택 표/부록; 본문에는 같은 조건 R07/R11과 최종 R04 중심 |
| 고정 운영 Test TP/FP/FN 표 | cache 직접 계산 일치 | 최종 성능·분모·정의 핵심 | 필수 표 |
| Val7threshold 표 | 직접 재매칭 일치 | Test 선택 누출 방지·운영 tradeoff | 필수 표 |
| 오류 유형·조건별 FN/GT 표 | 8/3/1 및7/41·7/60 등 일치 | 오류분석 배점 핵심 | 필수 표 |
| 크기×대비 조합 표 | 분모160·FN12 일치 | 독립·상승작용 오해 방지 | 선택 표 |
| Edge Val4행·보조18설정 표 | 원본 결과 일치 | 실패 기각의 근거;18행은 본문 과밀 가능 | 선택 상세표/부록; 채택 설정의 Val/Test 요약은 필수 표 |
| TELEA/NS20장 그림1 | 원본·mask·복원·GT 패널 확인 | 최종512의 전처리 이유 설명;20장 범위 표시 필요 | 필수 그림 |
| Validation PR 그림2 | 표시AP50 .970, 원본.970014 반올림 일치 | AP 정의와 실제 곡선 증거 | 선택 그림 |
| 학습 results 그림3 | 57행곡선;최대200/best7과 구분 | val loss 증가도 보임; 안정적인 수렴 주장 불가 | 선택 그림 |
| Val threshold 그림4 | 0.20/0.25/0.31 정책점과 표 일치 | threshold 선택 과정 핵심 | 필수 그림 |
| FN/FP 빈도 그림5 | 8/3/1 및8/8 일치 | 동일 표와 중복;FP 한 막대는 정보 적음 | 선택 그림;FP 빈도 그림 제거 권고 |
| 조건별 FN rate 그림6 | 분자·분모 일치;분모0는산출불가 표시 | 조건표와 중복, 일부 대비군 없는 항목 많음 | 선택 그림;소형/대비 표 우선 |
| U0138 FN 그림7 | score.191274·IoU.771798 일치 | threshold 실패와 실물 확대 사례 | 필수 그림 |
| U0488 FN/FP 그림8 | 동일cache score.590495·IoU.498477 일치 | 위치 경계 및 이번 집계차이 설명 핵심;두 그림 거의중복 | 필수 그림은FN1장, FP패널은선택 |
| U0371 FP 그림9 | 넓은 예측·IoU미달 사례 및score 확인 | 같은기전 반복이지만작은GT 설명 가능 | 선택 그림 |
| 새 공식 confusion | official_test_confusion.png의148/8/12·TN N/A | 공식 성능표와조건 일치 | 필수 그림 |
| 기존149/7/11 confusion | 다른전처리/IoU.45·재실행 개수재현 | 공식148/8/12와혼용하면모순 | 본문 제외;감사 부록에서만 사용 가능 |

**[필수 표]** 데이터 분할·해상도, 같은 조건 모델쌍의 설정/Val 결과, 최종 운영 Test, Val threshold, FN/FP 기전 및 크기·대비·경계 분모, 보조검사 선택설정의 Val/Test 결과.

**[필수 그림]** 주석 복원 예시(20장 PoC 표시), Val threshold 곡선, U0138 낮은점수 사례, U0488 위치경계 사례1장, 같은조건으로 재생성한 공식 confusion.

**[선택 표]** 11run 전체 Test,18개 보조설정,Edge ROI4행,크기×대비 조합,이번매칭차이 진단표.

**[선택 그림]** Val PR,57epoch 학습곡선,FN 기전 막대,조건 rate,U0371 FP,U0488의 중복 FP패널. 본문 길이 제한이 있으면 중복 막대·상세표부터 부록으로 이동한다.

## 8. 반영한 수정과 아직 해결되지 않은 문제

FINAL_REPORT_DRAFT.md의 제1장 크기통계, 제2장 실험 정의·집계차이·공식 confusion·최종 선정 표현, 제6장 비교weight와재현 범위를 수정하였다. 나머지 근거가 충분한 표·문단은 재작성하지 않았다. REPORT_EVIDENCE.md와 REPORT_VALIDATION.md의 미해결 confusion 설명 및 잘못된 AP 근거 경로도 새 감사 결과에 맞추었다.

남은 문제는 계획서 원문 부재, 양품/무주석 현장 검증 부재, 사전 모델 선택 규칙 부재이다. 없는 실험·규칙을 만들어 해결하지 않았다. 제출 패키지의 영상·코드·비교 weight 누락은 후속 보완으로 해결하였다. 재학습 실행, 역사적 run별 데이터 해시 및 GPU 예측의 bitwise 복원은 확인되지 않았으나 운영 집계의 혼용 원인은 설명하였다.

2026-10-07 제출 패키지 보완: 소스코드·requirements.txt·README, Train 359/Val 77/Test 76장의 영상 및 라벨, 선택 원본 512장, 복원 마스크, 완료 11run의 best.pt, Test 전용 예측 JSON·CSV와 공식 confusion을 포함하였다. Notebook 코드 17셀, 전처리 512장 픽셀 비교, 학습 입력 사전검사 및 포함 2,725파일 SHA를 통과하였다. 새 ZIP은 목록 자체를 포함해 2,726파일이며 모든 항목의 SHA와 CRC를 검증하였다. 재학습은 실행하지 않았다. 기록은 submission_audit/package_repair_result.json 및 제출 verification.json에 보존하였다.

## 9. 제출 전 사람이 직접 확인할 사항

1. 실제 계획서의 베이스라인·모델 비교·최종 선정 기준과 R07/R11 및 R04 유지 설명을 대조한다. 계획서와 맞지 않는다면 사후 유지 설명만으로 충족한 것으로 제출하지 않는다.
2. 보완된 KAMP_Xray_Submission.zip을 제출한다. 비교 weight와 새 공식 confusion은 이 ZIP에 포함되어 별도 보강 ZIP은 중복 제출할 필요가 없다.
3. 데이터 반출 규정·제출 용량 제한과 README의 학습 환경·실행 방법을 확인한다. 선택 원본·학습 영상·전처리 코드가 포함되었으나 최초 후보 전체 감사와 동일 성능의 재학습까지 검증한 것은 아니다.
4. HWP에서는 공식148/8/12 표와 새 confusion을 같은 조건으로 배치한다.149/7/11 그림을 공식 성능 그림으로 복사하지 않는다. 초기 EDA 크기 통계나그림은 ‘초기’로 표시한다.
5. 제조품목·촬영조건·공정 정보가 별도 제공되어 있으면 확인된 내용만 보강한다. 없는 내용을 채우지 않는다. 정상 판정 성능·현장 효과를 검증된 실적으로 표현하지 않는다.
6. HWP 표의 단위·소수자리·캡션·페이지 넘김을 직접 확인한다. 이번 산출물은 Markdown이므로 실제 HWP 레이아웃을 검증한 결과는 아니다.

## 10. 최종 제출 판단

- 제출 가능: **CONDITIONAL**
- 가장 큰 감점 위험: **40점 모델 항목의 계획서 대응·최종 선정 근거. 재학습 실행 및 현장 검증은 미실시이다.**
- 반드시 수정할 사항: **계획서 대응을 확인하고 보완된 ZIP과 공식148/8/12 confusion을 최종 제출에 반영할 것. 크기 통계·집계 혼용 및 제출 패키지의 학습 영상·코드·비교 weight 누락은 수정 완료하였다.**
- 선택적으로 보강할 사항: **확인 가능한 제조공정 정보,정상 음성/무주석 현장 표본 검증,반복seed·Train/Test 대비 정의 통일. 중복 그림과 긴 상세표는 부록으로 이동할 것.**
