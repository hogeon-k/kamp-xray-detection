from pathlib import Path
import re
import json

root = Path(__file__).resolve().parent.parent
p = root / 'FINAL_REPORT_DRAFT.md'
s = p.read_text(encoding='utf-8')
replacements = {
    '원본·라벨출처': '원본·라벨 출처', '출처감사': '출처 감사',
    '컬러주석처리': '컬러 주석 처리', '같은조건': '같은 조건',
    '구조·증강비교': '구조·증강 비교', '정책잠금': '정책 잠금',
    '직접후보IoU': '직접 후보 IoU', '실패기전': '실패 기전',
    '운영부담': '운영 부담', '검사전략': '검사 전략', '적용범위': '적용 범위',
    '주변대비': '주변 대비', '복원mask': '복원 mask', '제품mask': '제품 mask',
    '주변ring': '주변 ring', '객체탐지': '객체 탐지', '검사효과': '검사 효과',
    '재검효과': '재검 효과', '저장추론': '저장 추론', '정책선정': '정책 선정',
    '이미지routing': '이미지 routing', '저장routing': '저장 routing',
    '사례그림': '사례 그림', '최종판정': '최종 판정', '검사이력': '검사 이력',
    '모델버전': '모델 버전', '검사자판정': '검사자 판정', '재검결과': '재검 결과',
    '후보탐지': '후보 탐지', '영상품질': '영상 품질', '확대영상': '확대 영상',
    '품질기준': '품질 기준', '공정점검': '공정 점검', '주석복원': '주석 복원',
    '실센서': '실제 센서', '입력처리': '입력 처리', '우선확인': '우선 확인',
    '추가검토': '추가 검토', '위험조건': '위험 조건', '위험분석': '위험 분석',
    '수동검사': '수동 검사', '센서상태': '센서 상태', '운영연동': '운영 연동',
    '사례분석': '사례 분석', '자동보정': '자동 보정', '판정기준': '판정 기준',
    '위치오차': '위치 오차', '크기·위치오차': '크기·위치 오차',
    '후보관리': '후보 관리', '자동후처리': '자동 후처리', '정상제품': '정상 제품',
    '작업부담': '작업 부담', '중복경보': '중복 경보', '위치정보': '위치 정보',
    '검사기록': '검사 기록', '오류사례': '오류 사례', '검사시간': '검사 시간',
    '인력절감률': '인력 절감률', '생산영상': '생산 영상', '무주석입력': '무주석 입력',
    '촬영조건': '촬영 조건', '재학습시': '재학습 시', 'FN관리는': 'FN 관리는',
    'FP관리는': 'FP 관리는', '좋은보존성': '좋은 보존성', 'GT통계': 'GT 통계',
    'core침범': 'core 침범', '잔존표시': '잔존 표시', '인공경계': '인공 경계',
    '탐지성능': '탐지 성능', 'FN복구': 'FN 복구', '추가FP': '추가 FP',
    '복원흔적': '복원 흔적', '주변평균': '주변 평균', '제품contour': '제품 contour',
    '작은bbox': '작은 bbox', '작은객체': '작은 객체', '정답에서': '정답에서',
    '예측bbox': '예측 bbox', '작은GT': '작은 GT', '소형GT': '소형 GT',
    '대응GT': '대응 GT', '예측면적': '예측 면적', '위치성공': '위치 성공',
    '동시발생': '동시 발생', '기본640': '기본 640', '동일640': '동일 640',
    '1280 무증강': '1280 무증강', '기하/HSV/mosaic/mixup': '기하/HSV/mosaic/mixup',
    '초기학습률': '초기 학습률', 'FP예산': 'FP 예산', 'F1곡선': 'F1 곡선',
    '검출/점수': '검출·점수', 'FN집중': 'FN 집중', '최종512': '최종 512',
}
for a,b in replacements.items():
    s = s.replace(a,b)

# Apply spacing only to prose, preserving code, paths, and identifiers.
out=[]
fenced=False
for line in s.splitlines():
    if line.startswith('```'):
        fenced=not fenced
    if not fenced and not line.startswith(('|','[그림','[표','#')):
        line=re.sub(r'(Val|Test|Train|GT|TP|FP|FN|Recall|Precision|confidence|threshold|batch|imgsz|epoch|seed|Python|PyTorch|Torchvision|CUDA|Ultralytics|RTX|TELEA|dilation|radius|IoU|mAP50–95|mAP50|T_low|T_high)(?=\d)',r'\1 ',line)
        line=re.sub(r'(?<=[가-힣])(?=\d)', ' ',line)
        line=re.sub(r'(?<=[가-힣])(?=(?:FN|FP|GT|TP|Test|Val|Train|bbox|confidence|IoU|mAP50|Recall|TELEA|NS)\b)', ' ',line)
        line=re.sub(r'(?<=[0-9]) (?=(?:에서|으로|보다|은|이|와|과))','',line)
    out.append(line)
s='\n'.join(out)+'\n'
run='outputs/yolo_runs/telea_512_yolo11n_e200_img1280_b4_adamw_lr0.001_wd0.0005_noaug_s42'
s=s.replace('R04/validation/BoxPR_curve.png',run+'/validation/BoxPR_curve.png')
s=s.replace('R04/results.png',run+'/results.png')
s=s.replace('및 fp_cause_frequency.png','및 outputs/error_analysis/summary/fp_cause_frequency.png')
s=s.replace('및 fp/U0488_FP_pred0.png','및 outputs/error_analysis/fp/U0488_FP_pred0.png')
s=s.replace('translate0.005','translate 0.005').replace('scale0.2','scale 0.2').replace('mosaic0.2','mosaic 0.2')
s=s.replace('TELEA와NS','TELEA와 NS').replace('및radius','및 radius').replace('TELEA최선','TELEA 최선').replace('NS최선','NS 최선')
s=s.replace('자동NG','자동 NG').replace('과Recall','과 Recall').replace('이며원본','이며 원본')
s=s.replace('high후보','high 후보').replace('영상mask','영상 mask').replace('AI판정','AI 판정').replace('GT기반','GT 기반')
s=s.replace('Test에서IoU','Test에서 IoU').replace('작은GT','작은 GT').replace('최대200','최대 200').replace('실제57epoch','실제 57epoch')
s=s.replace('좋은 보존성 proxy','보존성이 양호하다고 평가하는 proxy')
p.write_text(s,encoding='utf-8')

# Verify final document structure and exact model rows against computed originals.
checks=[]
def check(name,ok):
    checks.append({'check':name,'passed':bool(ok)})
check('six official chapters',len(re.findall(r'^## 제[1-6]장\.',s,re.M))==6)
check('no unfilled table placeholders',not re.search(r'@[A-Z_]+@',s))
x=json.loads((root/'report_audit/verification_computed.json').read_text(encoding='utf-8'))
for r in x['runs']:
    for split in ['val','test']:
        t=r['summary']; pr=t[split+'_precision']; rc=t[split+'_recall']
        values=[pr,rc,2*pr*rc/(pr+rc),t[split+'_map50'],t[split+'_map50_95']]
        row='| '+r['id']+' | '+' | '.join(format(v,'.6f') for v in values)+' |'
        check(r['id']+' '+split+' table',row in s)
figures=re.findall(r'outputs/[^\s\]—]+\.png',s)
for path in figures:
    check('figure '+path,(root/path).is_file())
check('all 34 source validations passed',all(r['passed'] for r in x['checks']))
check('confusion discrepancy explicit','수치 불일치 확인 필요' in s)
check('auxiliary 18 configurations',len(re.findall(r'^\| (?:gaussian|median|opening)_(?:fixed|otsu|adaptive)_(?:lenient|strict) \|',s,re.M))==18)
for name in ['REPORT_EVIDENCE.md','FINAL_REPORT_DRAFT.md','REPORT_VALIDATION.md']:
    check(name+' exists and nonempty',(root/name).stat().st_size>10000)
(root/'report_audit/report_document_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print('Document checks:',len(checks),'passed:',sum(r['passed'] for r in checks))
for r in checks:
    if not r['passed']: print('FAILED:',r['check'])
assert all(r['passed'] for r in checks)
