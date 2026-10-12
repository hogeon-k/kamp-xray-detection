from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
p=ROOT/'FINAL_REPORT_DRAFT.md'
changes={
 '최종 TELEA 데이터의 동일': '''최종 TELEA 데이터의 동일 359/77/76 분할에서 YOLO11n 및 YOLOv8n을 비교하였다. 완료 실험은 YOLO11n 설정 실험 9개와 YOLOv8n 실험 2개, 총 11개였다. 이는 서로 다른 모델 구조 2종의 학습 run이며 11종의 architecture를 뜻하지 않았다. 각 run에 별도 학습 로그와 best.pt가 존재하였고 threshold 또는 후처리만 바꾼 실험은 이 11개에 포함하지 않았다. R02는 R01의 동일 seed 반복 실행이며 저장 파라미터도 동일하여 독립적인 모델 개선으로 세지 않았다. 취소된 640/batch 4/무증강 로그 7epoch는 완료 비교에서 제외하였다.''',
 '기본 증강의 동일 640 조건에서는': '''기본 증강의 동일 640 조건에서는 YOLOv8n R11이 YOLO11n R07보다 P/R/mAP50/mAP50–95가 모두 높았다. 감마 조건에서는 YOLO11n R06이 YOLOv8n R10보다 네 지표 모두 높았다. 구조의 우열은 조건에 따라 달랐다. 구조 비교의 기준은 R11과 R07의 같은 학습 설정 쌍이며, R04는 기존에 유지한 후속 분석 기준 모델이었다. 두 역할을 동일한 단일 통제 실험으로 해석하지 않았다.''',
 '저장 Test confusion matrix': '''저장 Test confusion matrix의 TP 149/FP 7/FN 11은 본 보고서와 다른 평가 경로의 결과였다. 동일 best.pt의 validator를 다시 실행하여 해당 개수를 재현하였다. validator는 사각 배치 전처리와 GT IoU>0.45를 사용하였고, 본 보고서는 1280×1280 predict 캐시와 GT IoU≥0.50을 사용하였다. 같은 캐시에 두 매칭 방식을 IoU 0.50으로 적용하면 모두 148/8/12였으며, IoU 0.45에서는 150/6/10이었다. U0303의 후보 IoU는 캐시에서 0.455820, validator 재실행에서 0.449256으로 달라 0.45 통과 여부가 바뀌었다. U0488도 캐시 IoU 0.498477로 0.50 미달이었으나 validator에서는 0.500334였다. 따라서 149/7/11은 단순 집계 오류가 아니라 입력 전처리·예측 좌표 및 IoU 정의가 다른 결과였다. CPU 재실행은 개수 차이를 재현한 진단이며 과거 GPU의 모든 좌표·AP가 bitwise 동일하다는 증명은 아니었다.

최종 공식 운영 Test 값은 잠긴 정책·오류분석과 같은 predict 캐시, confidence≥0.20, GT IoU≥0.50의 148/8/12로 유지하였다. 기존 confusion PNG를 이 공식 성능의 근거 그림으로 사용하지 않았다. 같은 조건으로 다시 생성한 객체 단위 confusion 그림과 이미지별 매칭표를 사용하였고, 정상 음성 영상이 없는 자료이므로 TN은 정의하지 않았다.

[필수 그림 삽입 권장: submission_audit/official_test_confusion.png — 고정 운영 평가의 TP 148/FP 8/FN 12. 생성 코드: submission_audit/generate_official_confusion.py]''',
 '추가 640 비교에서는': '''추가 640 비교에서는 R11/R06이 더 높은 Val F1과 Recall을 기록하였다. 따라서 R04가 모든 후보보다 검출률이 높거나 모든 지표에서 최적이었다고 주장하지 않았다. 비교 manifest의 selection_status는 retained_previous_baseline이었고 test_used_for_new_selection은 false였다. 최종 유지 설명은 기존 모델·threshold·오류분석의 연결과 Val 지표 간 교환관계를 고려한 사후 문서화였다. 사전에 정한 성능 최대화 규칙이나 계획서에 따른 선정 규칙은 프로젝트에서 확인되지 않았다. Test 최고값을 보고 모델을 바꾼 것으로 설명하지 않았으며, 더 높은 Val F1 후보를 교체하려면 그 후보로 동일한 정책 선정과 오류분석을 새로 수행해야 한다. mAP50–95를 반복 촬영 간 bbox 안정성의 직접 검증으로 표현하지 않았다. [근거 E04·E05]''',
 '제출 폴더에는': '''제출 폴더에는 원본 영상과 학습용 TELEA 영상 512장 및 전체 scripts가 포함되어 있지 않았다. 따라서 제출 패키지만으로 전처리부터 재학습까지 완전 재현할 수 있다고 주장하지 않았다. 저장 결과와 포함된 라벨·weight의 분석 재현은 가능하며, 완전한 학습 재현에는 전체 루트의 준비된 입력과 코드가 추가로 필요하였다. YOLOv8n 구조 비교 weight도 기존 제출 패키지에 포함되지 않았다. 동일 조건 비교의 R07/R11 weight·설정·로그·summary 및 SHA 목록은 submission_audit/model_comparison_supplement.zip에 보강 자료로 구성하였다. 제출 시 이 자료와 전체 전처리·학습 입력의 제공 범위를 함께 확인해야 한다.''',
 '제출 시 초기': '''제출 시 초기 1134bbox·XML 무객체 14장·preview 오류 자료를 최종 수치와 혼합하지 않아야 한다. validator 지표·운영 고정 threshold 지표·confusion PNG는 추론 전처리와 평가 정의가 다르다. 최종 운영 지표와 같은 조건으로 생성한 confusion 그림을 사용하고 기존 validator 그림을 혼용하지 않아야 한다. 최종 모델이 F1 최적이라고 주장하지 않았으며, 임계값 정책·Edge ROI·영상처리 보조 검사의 Test 개선 부재를 명시하였다. 구체적인 제조품목과 생산라인 효과는 추가 자료 없이 단정하지 않았다. 계획서와의 대응 및 제출본만으로 가능한 재현 범위는 제출자가 별도로 확인해야 한다.'''
}
s=p.read_text(encoding='utf-8');lines=s.splitlines();applied=[]
for i,line in enumerate(lines):
 for prefix,new in changes.items():
  if line.startswith(prefix):lines[i]=new;applied.append(prefix);break
assert len(applied)==len(changes),(applied,changes.keys())
p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Changed paragraphs',len(applied))
