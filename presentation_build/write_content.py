import csv, json, hashlib
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'presentation_build'
def load(p): return json.loads((ROOT/p).read_text(encoding='utf-8'))
def rows(p):
    with (ROOT/p).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def absolute(p): return (ROOT/p).as_posix()

manifest = rows('outputs/yolo_512_telea/manifest.csv')
assert len(manifest)==512 and sum(int(x['bbox_count']) for x in manifest)==1163
assert {s:sum(x['split']==s for x in manifest) for s in ('train','val','test')} == {'train':359,'val':77,'test':76}
official=load('submission_audit/official_test_confusion.json')
assert official['totals']=={'TP':148,'FP':8,'FN':12}
err=load('outputs/error_analysis/summary/summary.json')
assert any(x['condition']=='small_object' and x['group']=='True' and x['GT']==41 and x['FN']==7 for x in err['conditions'])
assert any(x['condition']=='low_contrast' and x['group']=='True' and x['GT']==60 and x['FN']==7 for x in err['conditions'])
runmanifest=load('outputs/model_comparison/comparison_manifest.json')
runs=runmanifest['runs']; final=runs[3]
model_table=[['모델·실험','Val F1(P/R)','Val mAP50']]
for i in (6,10):
    s=load(f'outputs/yolo_runs/{runs[i]}/summary.json')
    p,r=s['val_precision'],s['val_recall']
    model_table.append([('YOLO11n R07' if i==6 else 'YOLOv8n R11'),f'{2*p*r/(p+r)*100:.2f}%',f"{s['val_map50']*100:.2f}%"])
policy=load('outputs/safety_threshold_policy/locked_policy.json')
assert policy['selected']['T_low']==.2 and policy['selected']['T_high']==.25
threshold_table=[['Val threshold','FN','FP']]
for t in (.25,.20,.02):
    x=next(x for x in rows('outputs/safety_threshold_policy/val_threshold_sweep.csv') if float(x['threshold'])==t)
    threshold_table.append([f'{t:.2f}',x['FN'],x['FP']])
shortcut=load('outputs/shortcut_diagnostic_gpu/summary.json')
assert shortcut['pairs']==34 and shortcut['heldout_pairs']==10
for x in shortcut['model_summary']:
    if x['scope'] in ('all','holdout'): assert x['fr_fpr']==0 and x['new_fake_fp_count']==0
sample=next(x for x in manifest if x['unique_id']=='U0001')
cover=f"outputs/yolo_512_telea/images/{sample['split']}/{sample['restored_filename']}"
comparison='outputs/inpainting_poc_20/figures/01_4184354a2a3c_comparison.png'
threshold='outputs/safety_threshold_policy/visualizations/val_threshold_curve.png'
confusion='submission_audit/official_test_confusion.png'
fn='outputs/error_analysis/fn/U0138_FN_gt0.png'
fp='outputs/error_analysis/fp/U0371_FP_pred0.png'
overlay='outputs/shortcut_diagnostic_gpu/cases/E_largest_delta_pair_0002/overlay.png'
fake_mask='outputs/shortcut_diagnostic_dataset_gpu/masks/pair_0002.png'

slides=[
 dict(n=1,title='KAMP X-ray 이물질 검출과 오류분석',message='컬러 주석을 처리한 X-ray 영상에서 검출 성능과 실패 조건을 함께 분석하였다.',
      bullets=['제6회 K-인공지능 제조데이터 분석 경진대회','검출 성능과 미검출 위험 조건 분석'],
      figures=[cover],layout='짙은 남색 표지. 좌측 제목, 우측 실제 TELEA 영상. 팀명·발표자명은 확인 후 추가한다.',
      notes='이번 발표는 KAMP X-ray 영상의 이물질 검출과 오류분석 결과입니다. 영상에 포함된 컬러 주석을 먼저 처리하고, 두 모델 구조를 비교한 뒤 임계값을 Validation에서 고정했습니다. 최종 성능뿐 아니라 어떤 객체를 놓쳤는지, 복원 흔적에 모델이 반응하는지도 분석했습니다.',
      sources=['FINAL_REPORT_DRAFT.md','KAMP_Xray_Final_Analysis.ipynb']),
 dict(n=2,title='문제 정의와 데이터',message='동일한 그룹 분할로 이물질의 위치 검출을 평가하였다.',
      bullets=['이미지 512장 / bbox 1,163개','Train / Val / Test = 359 / 77 / 76','원본 해시·라벨 출처 연결, 그룹 단위 분할'],figures=[],
      table=[['분할','이미지','GT bbox'],['Train','359','828'],['Validation','77','175'],['Test','76','160']],
      layout='상단에 512와 1,163을 크게 표시. 아래 편집 가능한 분할표. 세부 수치를 문장으로 반복하지 않는다.',
      notes='분석 대상은 이미지 512장과 이물질 bbox 1,163개입니다. 학습 359장, 검증 77장, 테스트 76장으로 나눴고, 테스트 정답은 160개입니다. 원본 해시와 라벨 출처를 연결하고 그룹 단위로 분할했습니다. 정상 제품 판별이 아니라, 주어진 영상에서 이물질 위치를 검출하는 과제입니다.',
      sources=['outputs/yolo_512_telea/manifest.csv','outputs/clean_512_split/split_manifest.csv']),
 dict(n=3,title='컬러 bbox와 TELEA 전처리',message='컬러 주석선은 제거하고 이물질의 영상 신호는 보존하도록 복원하였다.',
      bullets=['컬러 표시 mask만 복원','20장 PoC에서 TELEA·NS 보존성 비교','최종 적용: TELEA radius 3 / dilation 0'],
      figures=[comparison],layout='동일 비교 이미지의 원본·mask·TELEA 세 패널을 슬라이드에서 확대 배치. 별도 그림 생성 없이 원본 이미지 crop 사용.',
      caption='20장 PoC 예시. 복원 보존성 비교이며 탐지 성능 향상률이 아님',
      notes='원본에는 이물질 주변에 컬러 bbox가 그려져 있어 모델이 표시선에 의존할 가능성이 있었습니다. 표시 색상을 mask로 추출하고 TELEA로 해당 영역을 복원했습니다. 20장 PoC에서 NS와 보존성을 비교해 radius 3, dilation 0을 적용했습니다. 이 비교는 복원 보존성 평가이며 탐지 성능 향상을 직접 증명한 결과는 아닙니다.',
      sources=['outputs/inpainting_poc_20/method_summary.csv','outputs/inpainting_poc_20/run_poc.py','outputs/yolo_512_telea/manifest.csv']),
 dict(n=4,title='YOLO11n과 YOLOv8n 모델 비교',message='같은 학습 조건에서 두 모델 구조를 비교하고 기존 R04를 후속 분석 기준으로 유지하였다.',
      bullets=['구조 비교: 640 / batch 4 / AdamW / 기본 증강','완료 11run, 모델 구조 2종','최종 R04: YOLO11n 1280·무증강, Val F1 최고와 구분'],figures=[],table=model_table,
      layout='동일 조건 R07·R11의 2행 비교표. 하단에 최종 R04 설정과 선정 한계를 분리해 표시. 학습곡선은 상세자료로 남긴다.',
      notes='두 구조는 입력 640, batch 4, AdamW와 기본 증강을 맞춰 비교했습니다. Validation에서는 YOLOv8n R11의 F1이 더 높았습니다. 최종 오류분석은 기존 YOLO11n 1280 무증강 R04를 유지했습니다. 따라서 최종 모델을 Validation F1 최고 모델이라고 주장하지 않습니다. 선정 방식은 기존 기준 모델의 유지입니다.',
      sources=['outputs/model_comparison/comparison_manifest.json',f'outputs/yolo_runs/{runs[6]}/summary.json',f'outputs/yolo_runs/{runs[10]}/summary.json',f'outputs/yolo_runs/{final}/summary.json']),
 dict(n=5,title='Validation에서 threshold 고정',message='미검출 감소와 추가 FP 부담을 함께 확인해 0.20 / 0.25 정책을 고정하였다.',
      bullets=['운영 하한 0.20 / 고신뢰 기준 0.25','Val 0.25 대비 FN 1개 감소, FP 1개 증가','Test로 threshold 재선정하지 않음'],figures=[threshold],table=threshold_table,
      layout='왼쪽 기존 Validation curve, 오른쪽 3행 FN·FP 비교표. 0.20은 F1 최적값이 아니라 recall 중심 하한임을 짧게 표시.',
      caption='Val 77장·GT 175개, 객체 매칭 IoU ≥ 0.50',
      notes='임계값은 Validation 77장에서 선택했습니다. 하한을 0.25에서 0.20으로 낮추면 FN이 한 개 줄고 FP는 한 개 늘었습니다. 반면 0.02에서는 FP가 117개까지 증가했습니다. 운영 하한 0.20과 고신뢰 기준 0.25를 고정했고, 테스트로 다시 조정하지 않았습니다. 0.20은 F1 최고값이 아니라 재현율 중심의 선택입니다.',
      sources=['outputs/safety_threshold_policy/val_threshold_sweep.csv','outputs/safety_threshold_policy/locked_policy.json']),
 dict(n=6,title='최종 Test 성능',message='고정 운영 조건에서 F1-score 93.67%를 확인하였다.',
      bullets=['TP 148 / FP 8 / FN 12','Precision 94.87% / Recall 92.50% / F1-score 93.67%','confidence ≥ 0.20 / GT IoU ≥ 0.50'],figures=[confusion],
      layout='왼쪽 3개 지표를 크게, 오른쪽 새 공식 confusion. TP·FP·FN은 그림에서 읽고 문장으로 중복하지 않는다.',
      caption='Test 76장·GT 160개, NMS IoU 0.70. 정상 음성 평가 및 TN 없음',
      notes='고정한 정책을 테스트 76장에 적용했습니다. 정답 160개 중 TP는 148개, FN은 12개였고 FP는 8개였습니다. Precision은 94.87%, Recall은 92.50%, F1은 93.67%입니다. 오른쪽 그림은 같은 예측 캐시와 IoU 기준으로 다시 만든 공식 confusion입니다. 정상 제품의 오경보율이나 TN은 이 자료로 산출하지 않았습니다.',
      sources=['submission_audit/official_test_confusion.json','outputs/error_analysis/summary/summary.json']),
 dict(n=7,title='소형·저대비 조건과 위치 오차',message='미검출은 점수 부족과 위치 오차로 구분되며 소형·저대비군에서 높은 FN 비율을 관찰하였다.',
      bullets=['소형 객체 FN 17.07% (7/41)','저대비 객체 FN 11.67% (7/60)','위치 오차 8 / 낮은 점수·위치 성공 3 / 낮은 점수·위치 실패 1'],figures=[fn,fp],
      layout='좌측에 조건별 FN 비율과 분모를 크게. 우측에 U0138 FN·U0371 FP 원본의 확대 패널만 crop해 나란히 제시.',
      caption='소형 bbox ≤ 81px². 저대비 기준은 Train에서 고정. 조건 중복이 있어 인과 해석 불가',
      notes='소형 객체에서는 41개 중 7개를 놓쳐 FN 비율이 17.07%, 저대비 객체에서는 60개 중 7개로 11.67%였습니다. 두 조건은 겹치므로 독립적인 원인으로 해석하지 않았습니다. U0138은 위치가 맞아도 점수가 하한보다 낮았고, U0371은 예측 bbox가 커 IoU가 부족했습니다. 점수와 위치 오차를 구분해 대응해야 합니다.',
      sources=['outputs/error_analysis/summary/gt_condition_rates.csv','outputs/error_analysis/summary/cause_frequency.csv','outputs/error_analysis/summary/fn_candidate_diagnostics.csv','outputs/error_analysis/summary/summary.json']),
 dict(n=8,title='창의성: fake-mark 역검증',message='의도적인 표시·복원 흔적을 다시 만들어 모델의 반응을 비교하였다.',
      bullets=['컬러 표시 제거 → TELEA 복원 → 의도적 fake-mark 재생성 → 모델 반응 검증','20장 PoC, 34pair / holdout 10pair','두 PoC 모델의 새 fake FP 0, GT loss/gain 0'],figures=[overlay,fake_mask],
      flow=['컬러 표시 제거','TELEA 복원','의도적 fake-mark\n재생성','모델 반응 검증'],
      layout='상단 4단계 편집 가능한 흐름. 중앙에 실제 baseline original/fake 두 패널과 해당 fake mask. 하단에 PoC 결과와 적용 한계.',
      caption='fake-mark는 GT 밖으로 표시 mask를 재배치·재복원한 흔적. 최종 모델 shortcut 부재 증명은 아님',
      notes='복원 흔적을 모델이 이물질로 오인하는지 역으로 검증했습니다. 컬러 표시를 제거하고 TELEA로 복원한 뒤, 표시 mask를 정답 밖으로 재배치해 의도적인 fake-mark 복원 흔적을 만들고 모델 반응을 비교했습니다. 20장 PoC의 34쌍에서 새 fake FP와 GT 검출 손실·획득은 없었습니다. 다만 최종 512장 모델의 shortcut 부재까지 증명한 결과는 아닙니다.',
      sources=['scripts/fake_restoration.py','scripts/diagnose_context_matched_restoration.py','outputs/shortcut_diagnostic_dataset_gpu/metadata.csv','outputs/shortcut_diagnostic_gpu/summary.json']),
 dict(n=9,title='현장 활용: 검사자 재검 지원',message='AI는 후보 위치와 점수를 제시하고 검사자가 최종 판정을 수행하도록 제안한다.',
      bullets=['고신뢰 후보: 우선 확인','낮은 점수·무검출: 확대 확인과 표본 재검','제안 단계: 정상 제품·무주석 영상 검증 필요'],figures=[],
      table=[['영상 후보 조건','검사자 조치'],['confidence ≥ 0.25','후보 위치 우선 확인'],['0.20 ≤ confidence < 0.25만 존재','재검 후보 확인'],['confidence ≥ 0.20 후보 없음','정상 보증 없이 표본 재검']],
      layout='3행의 편집 가능한 운영 제안표. 원본·bbox·점수·검사자 판정 기록을 짧게 설명. 실물 생산라인 이미지는 사용하지 않는다.',
      caption='고신뢰 후보 우선 routing. 생산 설비 연동·자동 배출·시간 절감 효과 미검증',
      notes='현장에서는 AI가 후보 위치와 점수를 제시하고 검사자가 원본과 확대 영상을 확인하도록 제안합니다. 고신뢰 후보는 우선 확인하고, 낮은 점수 후보와 무검출 영상도 재검 대상에 포함합니다. 실제 생산 설비 연동과 자동 배출은 구현하지 않았으며, 정상 제품과 무주석 영상에서 오경보와 검사 부담을 별도로 검증해야 합니다.',
      sources=['FINAL_REPORT_DRAFT.md','outputs/safety_threshold_policy/locked_policy.json']),
 dict(n=10,title='결론과 후속 검증',message='고정 조건의 검출 성능과 실패 사례를 근거로 검사 지원의 범위를 제시하였다.',
      bullets=['공식 운영 Test와 오류분석의 평가 조건 통일','효과 없는 보조검사는 운영 채택에서 제외','후속 검증: 정상·무주석 영상, 반복 seed, 계획서 대응'],figures=[],
      layout='핵심 결론 1문장과 3개 짧은 항목. 앞 슬라이드 성능 수치를 반복하지 않는다.',
      notes='이번 분석은 컬러 주석을 처리한 데이터에서 운영 성능과 실패 사례를 같은 조건으로 연결했습니다. 효과가 없는 보조검사는 채택하지 않았고, fake-mark 역검증도 PoC 범위로 해석했습니다. 다음 단계는 정상 제품과 무주석 영상의 검증, 반복 seed 실험입니다. 계획서와 최종 모델 선정의 대응도 제출 전에 확인해야 합니다.',
      sources=['FINAL_REPORT_DRAFT.md','FINAL_SUBMISSION_AUDIT.md','outputs/low_contrast_aux_poc/README.md']),
]

for s in slides:
    for p in s['sources']+s['figures']: assert (ROOT/p).is_file(),p
    s['sources']=[absolute(p) for p in s['sources']]
    s['figures']=[absolute(p) for p in s['figures']]
    s['duration_seconds']=25
(BUILD/'slides.json').write_text(json.dumps(slides,ensure_ascii=False,indent=2),encoding='utf8')

plan='''# 발표자료 구성안

10장(표지 포함), 각 20~30초를 기준으로 약 4분 10초를 계획하였다. 실제 소요시간은 발표 리허설에서 조정한다. 대상은 제6회 K-인공지능 제조데이터 분석 경진대회 심사위원이며, 데이터 처리와 모델 비교, 고정 운영 성능, 오류분석 및 활용 범위를 설명한다.

## 구성

| 슬라이드 | 제목 | 역할 | 시간 |
|---|---|---|---|
'''
for s in slides: plan+=f"| {s['n']} | {s['title']} | {s['message']} | 20~30초 |\n"
plan+='''
## 디자인 및 수치 사용 원칙

- 16:9, 남색·청록과 밝은 바탕, 맑은 고딕. 제목 약 32pt, 본문 약 18~21pt, 핵심 수치는 더 크게 표시한다.
- 슬라이드당 핵심 메시지 하나, 본문 2~4항목. 실제 슬라이드에서는 표·수치·그림에 이미 표시한 내용을 문장으로 반복하지 않는다.
- 분할표 3행, 모델 비교표 2행, threshold 표 3행, 운영표 3행으로 제한한다. 표와 흐름은 편집 가능한 슬라이드 객체로 만든다.
- 프로젝트 이미지의 비율을 유지한다. 복원·오류 사례는 기존 패널을 슬라이드 crop으로 확대하며 이미지 내용·수치를 새로 생성하지 않는다.
- 발표 대본과 상세 출처는 PPTX 발표자 노트에 넣는다. 팀명과 발표자명은 제공되지 않아 임의로 작성하지 않았다.

## 심사에서 혼동하기 쉬운 부분

1. 공식 Test는 저장 predict 캐시, confidence ≥ 0.20, GT IoU ≥ 0.50의 148/8/12이다. 기존 149/7/11 validator 그림을 사용하지 않는다. NMS IoU 0.70과 GT matching IoU 0.50을 구분한다.
2. 같은 조건의 구조 비교는 R07/R11(640, batch 4, 기본 증강)이다. 최종 R04(1280, 무증강)는 기존 후속 분석 기준을 유지한 것이며 Val F1 최고 선정이 아니다. 모델 비교표의 F1은 validator P/R의 조화평균으로 운영 Test F1과 정의가 다르다.
3. 소형 7/41, 저대비 7/60은 객체 분모이다. 기준은 Train에서 정했고 두 군이 중복되므로 인과 또는 결합 효과를 주장하지 않는다.
4. fake-mark는 실제 컬러 표시 mask 형태를 GT 밖에 재배치하고 inpaint한 흔적의 paired 반응 진단이다. 새 컬러 선을 실제 영상에 다시 그린 구현으로 설명하지 않는다. 20장 PoC의 34pair, holdout 10pair는 최종 Test 76장과 다르며 34개의 독립 영상도 아니다.
5. 현장 routing은 구현된 후보 분류에 기반한 운영 제안이다. 실제 출하 판정, 정상 제품 성능, 생산라인 연동, 시간·비용 절감 실적은 제시하지 않는다.
6. 계획서 원문과 사전 최종 모델 선정 규칙은 확인되지 않았다. 발표 전 계획서 대응과 팀명·발표자명을 사람이 확인한다.

## 조사 및 근거

FINAL_REPORT_DRAFT.md, FINAL_SUBMISSION_AUDIT.md를 읽고 Notebook 36셀의 소스와 그림 참조를 조사하였다. outputs의 주요 비교·복원·정책·오류·shortcut·학습 figure를 목록화하고 선정 자료를 직접 열람하였다. manifest, 모델 summary, threshold sweep, 잠금 정책, 오류 summary, 공식 confusion JSON 및 PoC GPU summary를 대조하였다. 전체 원본 이미지 모두를 그림 후보로 정독한 것은 아니다.

비공개 빌드 기록: presentation_build/figure_inventory.json, notebook_review.json, source_validation.json. 추가 학습·추론으로 새로운 성능을 만들지 않았다. 원본 보고서와 분석 산출물 및 제출 ZIP은 이번 발표자료 제작에서 변경하지 않았다.
'''
(ROOT/'PRESENTATION_PLAN.md').write_text(plan,encoding='utf8')

content='# 실제 슬라이드용 콘텐츠\n\n10장 구성. 각 대본은 20~30초를 목표로 작성했으며 리허설로 확인한다. 아래 bullet은 전달할 정보 목록이며 PPTX는 표·그림에 포함된 중복 문장을 줄여 배치하였다.\n'
for s in slides:
    content+=f"\n## {s['n']}. {s['title']}\n\n**핵심 메시지:** {s['message']}\n\n**본문 bullet:**\n\n"+''.join('- '+x+'\n' for x in s['bullets'])
    if s.get('table'):
        t=s['table']; content+='\n**슬라이드 표:**\n\n| '+' | '.join(t[0])+' |\n| '+' | '.join(['---']*len(t[0]))+' |\n'
        content+=''.join('| '+' | '.join(row)+' |\n' for row in t[1:])
    content+='\n**실제 이미지:**\n\n'+(''.join(f'- [{Path(p).name}]({p})\n' for p in s['figures']) or '- 이미지 없음. 편집 가능한 표 또는 결론 텍스트로 대체한다.\n')
    content+=f"\n**배치:** {s['layout']}\n"
    if s.get('caption'): content+=f"\n**보이는 주의 문구:** {s['caption']}\n"
    content+=f"\n**발표 대본(20~30초):**\n\n{s['notes']}\n\n**원본 근거:**\n\n"+''.join(f'- [{Path(p).name}]({p})\n' for p in s['sources'])
(ROOT/'SLIDE_CONTENT.md').write_text(content,encoding='utf8')

figmd='# 선정 그림과 대체 자료\n\n이미지 내용은 프로젝트 원본을 사용한다. 확대는 PPTX의 crop 정보로 저장하며 새로운 사례 이미지를 생성하지 않는다. 표·도식은 기존 수치와 코드 흐름을 편집 가능한 슬라이드 객체로 구성한다.\n\n| 슬라이드 번호 | 사용할 이미지 | 원본 경로 | 사용 이유 |\n|---|---|---|---|\n'
reason={1:'실제 TELEA 입력으로 과제 소개',3:'컬러 표시와 mask·복원 후 차이를 직접 비교',5:'Validation에서 선택한 운영점과 recall/precision 교환관계',6:'공식 148/8/12 및 평가 조건 일치',7:'점수 부족 FN과 위치 오차 FP를 구분',8:'실제 paired 모델 반응과 의도한 fake 영역 위치 확인'}
for s in slides:
    for p in s['figures']: figmd+=f"| {s['n']} | {Path(p).name} | [{p}]({p}) | {reason[s['n']]} |\n"
    if not s['figures']: figmd+=f"| {s['n']} | 이미지 없음: {'편집 가능한 표' if s.get('table') else '결론 텍스트'} | "+'<br>'.join(f'[{Path(p).name}]({p})' for p in s['sources'])+' | 실제 수치와 구현 범위 전달 |\n'
figmd+='''
## 패널 확대 위치

- 3장: 원본 비교 PNG의 5개 열 중 원본·mask·TELEA 열을 사용한다. NS·GT 열은 같은 파일에 남아 있으며 발표에서는 상세 비교를 생략한다.
- 7장: U0138 FN·U0371 FP의 두 번째 확대 패널을 사용한다. 원본 전체 사례 파일과 수치는 발표자 노트로 추적할 수 있다. GT와 후보/예측의 색 의미를 슬라이드에 표시한다.
- 8장: overlay 상단의 baseline original/fake 두 패널과 pair_0002 fake mask를 사용한다. 초록 영역은 진단 fake 위치이며 실제 이물질 예측으로 해석하지 않는다. mask는 위치를 보여주는 기존 이진 마스크이다.

## 본문에서 제외한 주요 figure

'''
excluded=[(f'outputs/yolo_runs/{final}/results.png','57epoch 실제 학습곡선. 10장 흐름에서는 비교 결과표를 우선하고 질의응답 자료로 사용'),(f'outputs/yolo_runs/{final}/validation/BoxPR_curve.png','Val AP 상세자료. threshold curve와 역할이 달라 본문에서는 생략'),('outputs/error_analysis/fn/U0488_FN_gt0.png','위치 경계 사례. U0371 FP와 겹치므로 질문 대응 자료로 보관'),(f'outputs/yolo_runs/{final}/test/confusion_matrix.png','149/7/11의 validator 조건. 공식 148/8/12 그림과 혼용 금지'),('outputs/error_analysis/summary/fn_condition_rates.png','여러 조건이 포함돼 발표에서는 소형·저대비 두 분모만 강조'),('outputs/shortcut_diagnostic_gpu/plots/fr_fpr_comparison.png','0값 막대는 정보량이 적어 paired 실제 사례와 결과 문구를 우선')]
for p,why in excluded:
    if (ROOT/p).exists(): figmd+=f'- [{p}]({absolute(p)}): {why}\n'
(ROOT/'SELECTED_FIGURES.md').write_text(figmd,encoding='utf8')

assets=[]
for s in slides:
    for p in s['figures']:
        im=Image.open(p); assets.append({'slide':s['n'],'path':p,'dimensions':im.size,'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()})
proof={'slide_count':10,'images':512,'bbox':1163,'split':[359,77,76],'test':official['totals'],'metrics':{'precision':round(148/156*100,2),'recall':round(148/160*100,2),'F1':round(296/316*100,2)},'small_FN_rate':round(7/41*100,2),'low_contrast_FN_rate':round(7/60*100,2),'matched_model_table':model_table,'threshold_table':threshold_table,'selected_assets':assets,'new_inference':False}
(BUILD/'source_validation.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf8')
print('Created three requested Markdown files and verified 10-slide content against project sources.')
