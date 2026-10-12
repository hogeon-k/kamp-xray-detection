from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
p=ROOT/'REPORT_EVIDENCE.md';s=p.read_text(encoding='utf-8')
lines=s.splitlines()
for i,line in enumerate(lines):
    if line.startswith('3. **수치 불일치 확인 필요'):
        lines[i]='3. **Test confusion 차이 규명 완료:** 원래 R04/test_evaluation/confusion_matrix.png는 149/7/11이었다. 같은 weight·Ultralytics 8.4.158의 실제 validator CPU 재실행에서도 149/7/11을 얻었다. predict 캐시는 1280×1280 입력, validator는 rect=True·pad0.5의 1024×1312/1248×1312/1312×1248 입력이었다. 같은 캐시에서 자체/라이브러리 IoU0.50 매칭은 모두 148/8/12, IoU0.45 매칭은 150/6/10이었다. U0303의 cache IoU0.455820가 validator에서0.449256으로 달라 0.45 통과 여부가 바뀌었다. 최종 운영 값은 잠긴 정책의 cache/conf≥0.20/IoU≥0.50 기준148/8/12로 유지하고 동일 조건의 새 confusion 그림을 사용한다. 원본 PNG는 다른 평가 경로의 역사적 산출물로 보존한다. 과거 GPU의 모든 좌표·AP의 bitwise 동일성은 주장하지 않는다. 자세한 이미지별 비교는 submission_audit/matching_comparison.json 및 FINAL_SUBMISSION_AUDIT.md에 있다.'
s='\n'.join(lines)+'\n'
s+='''
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
'''
p.write_text(s,encoding='utf-8')

p=ROOT/'REPORT_VALIDATION.md';s=p.read_text(encoding='utf-8')
s=s.replace('최종 run/test_evaluation/summary.json 대조','최종 run/summary.json의 test_map50 대조')
lines=s.splitlines()
for i,line in enumerate(lines):
    if line.startswith('**수치 불일치 확인 필요'):
        lines[i]='**제출 전 감사에서 원인 규명 완료 — 서로 다른 추론 전처리 및 IoU 정의.** 원래 저장 PNG의 149/7/11을 같은 weight의 validator 실제 CPU 재실행으로 재현하였다. 동일 캐시에서 자체 매칭과 Ultralytics 매칭을 IoU0.50으로 맞추면 모두148/8/12이며, 캐시를 IoU0.45로 평가하면150/6/10이다. validator는 rect 배치와pad0.5를 사용하여 입력 shape와 예측 좌표가 predict 캐시의 square1280과 다르다. U0303의 IoU가0.455820→0.449256으로 달라 0.45 기준에서도 예측 경로별 집계가 달라졌다. 최종 공식 운영 성능은 잠긴 cache/conf≥0.20/IoU≥0.50의148/8/12이며 기존 PNG 대신 submission_audit/official_test_confusion.png를 사용한다. 과거 GPU 예측 전체의 bitwise 복원은 검증하지 않았다. 상세 분해는 FINAL_SUBMISSION_AUDIT.md와matching_comparison.json에 수록하였다.'
s='\n'.join(lines)+'\n'
s=s.replace('성능 개선이 없는 실험과 아직 해소되지 않은 confusion PNG 차이를 숨기지 않았다.','성능 개선이 없는 실험과 추론 경로·IoU 정의가 다른 confusion PNG 차이를 명시하였다.')
s+='''
## 부록 D. 제출 전 감사에 따른 정정

최종 1163개의 bbox 크기 통계를 실제 TXT와 영상 헤더로 다시 계산하였다. 초기 EDA의 면적 중앙값100/Q75=140/평균114.322442 및 종횡비 최대2.0은 최종 통계가 아니었다. FINAL_REPORT_DRAFT.md §1.3을 중앙값104/Q75=143/평균115.890800 및 종횡비 최대1.833333으로 정정하였다. Q25=81과 소형 FN7/41 등 오류분석의 기준·분모에는 변경이 없었다.

제출 전 감사에서 원본 기반108개 검사를 통과하였다. 추가 기록은 submission_audit/number_checks.json과 all_report_numbers.csv이다. 이전41개 문서 검사 기록은 첫 초안 시점의 검증이며 최종 감사 결과로 대체한다. 이번 감사는 새로운 학습을 수행하지 않았고 별도의 CPU validator/predict 추론을 진단용으로 실행하였다. 원래 저장 AP와 새로운 CPU AP의 미세한 차이를 숨기거나 덮어쓰지 않았다.

실제 모델은 YOLO11n/YOLOv8n의 두 구조이고 11run의 파라미터 상태는10종이었다. 제출 패키지에서 YOLOv8n weight가 누락된 점을 확인하여 기존 R07/R11 원본을 model_comparison_supplement.zip으로 준비하였다. 원래 제출 ZIP·included_files.json은 변경하지 않았다. 계획서 대응, 완전 재학습 입력 제공 및 최종 한글 문서·제출 파일 구성은 제출 전 사람 확인이 필요하다.
'''
p.write_text(s,encoding='utf-8')
print('Supporting reports updated')
