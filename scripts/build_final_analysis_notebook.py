"""Refresh the current corrected final analysis notebook in a Jupyter kernel.

No training, inference, package installation, or modification of prior outputs.
The current notebook is the authoritative template. Historical cell definitions
are retained below for provenance but are not the regeneration entry point.
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import textwrap
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'KAMP_Xray_Final_Analysis.ipynb'
cells = []


def md(source):
    cells.append(dict(cell_type='markdown', id=uuid.uuid4().hex[:8], metadata={},
                      source=textwrap.dedent(source).strip()+'\n'))


def code(source):
    cells.append(dict(cell_type='code', id=uuid.uuid4().hex[:8], metadata={},
                      source=textwrap.dedent(source).strip()+'\n', execution_count=None, outputs=[]))


md('''
# KAMP X-ray 최종 분석

**서면평가용 결과 모음 · 기존 저장 결과만 읽는 노트북**

평가항목: **1. 데이터 이해 및 진단 / 2. AI 예측모델 개발 / 3. 영향요인 및 오류분석 / 4. 현장 활용방안 / 5. 창의성·차별성 / 6. 코드 및 재현성**

노트북은 프로젝트 root에 둡니다. Python 3, pandas, numpy, Pillow가 필요하고 Jupyter에서는 IPython의 표·이미지 표시를 사용합니다.
위에서 아래로 실행하면 기존 CSV·JSON·TXT·PNG와 학습 로그를 읽어 표와 시각화를 표시합니다.
모델 학습·신규 추론·재복원·데이터 변경은 실행하지 않습니다. PNG는 저장된 파일을 그대로 사용합니다.
각 표의 수치는 저장 결과 또는 그 결과에서 계산한 비율이며, 생성할 때 실행한 출력도 노트북에 포함했습니다.

목차: 1 프로젝트 개요 → 2 데이터 이해 및 진단 → 3 사각형 표시 제거 → 4 Shortcut 검증 → 5 모델 비교 → 6 Threshold → 7 FN/FP → 8 현장 활용 → 9 최종 결론 → KAMP 공식 평가항목1~6 대응표 → 코드 및 재현성(공식 항목6).

**상태 표기:** 검증됨=기존 파일·계산의 일치 확인, 관찰됨=현재 표본의 결과, 제안됨=운영 아이디어, 미구현=자동 기능·현장 효과 검증이 없는 상태. 기술 통계를 인과관계나 운영 효과로 해석하지 않는다.
''')
code(r'''
from pathlib import Path, PureWindowsPath
import hashlib
import json
import math
import sys
import numpy as np
import pandas as pd
from PIL import Image as PILImage
try:
    from IPython.display import display
except ImportError:
    def display(value):
        print(value.to_string(index=False) if isinstance(value, pd.DataFrame) else value)

def find_project_root():
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate/'outputs/yolo_512_telea/manifest.csv').exists() and (candidate/'train_yolo.py').exists():
            return candidate
    raise FileNotFoundError('프로젝트 root 또는 하위 폴더에서 노트북을 실행하세요.')

ROOT = find_project_root()
pd.set_option('display.max_columns', 24)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.float_format', lambda v: f'{v:.6f}')
SOURCE_LEDGER = {}

def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()

def source(relative):
    path = ROOT/relative
    if not path.is_file():
        raise FileNotFoundError(f'필수 저장 결과 누락: {relative}')
    SOURCE_LEDGER.setdefault(path.relative_to(ROOT).as_posix(), file_sha(path))
    return path

def load_json(relative):
    return json.loads(source(relative).read_text(encoding='utf-8-sig'))

def load_csv(relative):
    return pd.read_csv(source(relative), encoding='utf-8-sig')

def show_png(relative, caption=''):
    if caption:
        print(caption)
    path = source(relative)
    with PILImage.open(path) as image:
        display(image.copy())
    print(f'출처: {relative}')

def harmonized_f1(precision, recall):
    return 2*precision*recall/(precision+recall) if precision+recall else 0.0

manifest = load_csv('outputs/yolo_512_telea/manifest.csv')
clean_meta = load_csv('outputs/clean_512/metadata.csv')
split_summary = load_json('outputs/clean_512_split/split_summary.json')
restoration_summary = load_json('outputs/yolo_512_telea/restoration_summary.json')
threshold_summary = load_json('outputs/safety_threshold_policy/summary.json')
locked_policy = load_json('outputs/safety_threshold_policy/locked_policy.json')
error_summary = load_json('outputs/error_analysis/summary/summary.json')
print('준비 완료: 저장 결과 읽기 전용 / 학습·추론 없음 / 모든 데이터 경로는 ROOT 기준')
display(pd.DataFrame([{'Python': sys.version.split()[0], 'pandas': pd.__version__,
                       'numpy': np.__version__, 'Notebook': 'KAMP_Xray_Final_Analysis.ipynb'}]))
''')

md('''
## 1. 프로젝트 개요

**목적:** X-ray 영상에서 이물질 후보를 탐지하고 bbox와 confidence를 제공하여 작업자 품질검사를 보조한다.
**공식 평가항목 연결:** 1. 데이터 이해 및 진단, 2. AI 예측모델 개발, 4. 현장 활용방안.

생산 관측 단위는 촬영 이미지이며 객체 평가 단위는 이미지 안의 이물질 bbox다. 하나의 이미지에 여러 bbox가 있을 수 있다.
장비(`machine`), 제품/일련 식별자(`serial`), 촬영일(`capture_date`)은 manifest에서 연결하며, 같은 촬영 조건이 split 간 섞이는 영향을 줄이기 위해 그룹 분할했다.
TXT 라벨은 YOLO 형식 `class_id center_x center_y width height`이고 좌표는 이미지 크기로 정규화된 0–1 값이다. 최종 클래스는 `0: defect` 하나다.
목표는 객체 위치와 점수를 함께 평가하고, FN·FP 집중 조건을 도출해 재검 우선순위에 연결하는 것이다.
현재 최종 데이터는 모두 결함 bbox가 있으므로 양품 생산라인의 false reject rate나 실제 불량률을 측정한 데이터로 해석하지 않는다.
''')
code('''
variable_dictionary = pd.DataFrame([
    ['unique_id', '고유 이미지 식별자', '중복·라벨·예측 결과 연결'],
    ['filename / restored_filename', '원본 / TELEA 입력 파일', '전처리 전후 추적'],
    ['machine', '촬영 설비 식별값', '설비별 분포와 오류 점검'],
    ['serial', '제품/일련 식별값', '촬영일·설비와 함께 그룹 구성'],
    ['capture_date', '촬영일', '시간적 유사성과 split 누출 점검'],
    ['group_id / session_id', 'machine + serial + capture_date 그룹', '그룹 전체를 한 split에 배치'],
    ['bbox_count', '이미지의 GT 객체 개수', '객체 기준 분모'],
    ['TXT bbox', 'class_id, cx, cy, width, height (정규화)', '실제 크기와 위치 평가'],
    ['confidence', '모델의 객체 점수', '불량 확률로 보정된 값은 아님'],
], columns=['변수', '의미', '분석 역할'])
display(variable_dictionary)
sample = manifest.iloc[0]
label_relative = f"outputs/yolo_512_telea/labels/{sample['split']}/{Path(sample['filename']).stem}.txt"
sample_txt = source(label_relative).read_text().strip()
print('TXT 실제 예시:', sample_txt)
print('클래스 설명 출처: outputs/yolo_512_telea/dataset.yaml')
source('outputs/yolo_512_telea/dataset.yaml')
''')

md('''
## 2. 데이터 이해 및 진단

**목적:** 최종 데이터의 생산단위·변수·시간/설비/제품 관계, 결측·중복·라벨 문제와 분할 검증 전략을 확인한다.
**공식 평가항목 연결:** 1. 데이터 이해 및 진단.

최종 기준은 `clean_512`와 `yolo_512_telea`다. 초기 EDA/라벨 충돌 점검에서 기록한 1,134개 bbox와 초기 후보 1,147개 bbox를 최종 1,163개 bbox와 혼합하지 않는다.
500장은 canonical TXT, 12장은 다른 YOLO TXT 간 바이트/SHA 일치와 VOC bbox/class 일치가 확인된 fallback TXT를 사용했다.
형식·연결 검증 통과는 전문가의 모든 라벨 승인과 다르며, 기존 충돌 23장과 검토 상태를 보존한다.
''')
code('''
split_order = ['train', 'val', 'test']
split_table = manifest.groupby('split').agg(images=('unique_id','size'), GT=('bbox_count','sum'),
                                          groups=('group_id','nunique')).reindex(split_order)
assert len(manifest)==512 and manifest['bbox_count'].sum()==1163
assert split_table['images'].tolist()==[359,77,76]
assert split_table['GT'].tolist()==[828,175,160]
display(split_table.reset_index())
display(pd.crosstab(manifest['split'], manifest['machine']).reindex(split_order).rename_axis(columns='machine'))
display(manifest.groupby(['split','serial']).agg(images=('unique_id','size'),
        dates=('capture_date','nunique')).reset_index())
display(manifest.groupby('split').agg(first_date=('capture_date','min'), last_date=('capture_date','max'),
                                    unique_dates=('capture_date','nunique')).reindex(split_order).reset_index())
display(pd.DataFrame(split_summary['leakage'].items(), columns=['split 누출 점검','건수']))
''')
code('''
audit = load_json('outputs/unique_512_audit/statistics.json')
clean_validation = load_json('outputs/clean_512/validation_report.json')
review_summary = load_json('outputs/label_review/summary.json')
qa_summary = load_json('outputs/yolo_512_telea_qa/QA_요약.json')
diagnostics = pd.DataFrame([
    ['전체 진단 후보 파일',audit['candidate_image_files'],'최종 학습 이미지 수와 다른 모수'],
    ['전체 후보 SHA 고유 수',audit['all_dataset_unique_sha'],'최종 라벨 연결 고유 후보는512장'],
    ['전체 후보 exact duplicate 파일',audit['exact_duplicate_files'],'파일별 중복을 점검한 역사적 집계'],
    ['최종 중복 이미지',clean_validation['duplicate_final_images'],'최종 clean_512 기준'],
    ['최종 라벨 누락',clean_validation['missing_labels'],'최종 기준'],
    ['최종 YOLO 형식 오류',clean_validation['invalid_yolo_labels'],'최종 기준'],
    ['보존된 기존 라벨 충돌 이미지',clean_validation['previous_conflict_images'],'전문가 검토 완료 주장 아님'],
    ['검증 fallback TXT 이미지',clean_validation['fallback_verified_images'],'canonical 없는12장의 출처 검증'],
    ['최종 bbox',clean_validation['total_bbox'],'최종 모수'],
    ['cross-split 유사쌍 후보',restoration_summary['near_cross_split_review_candidates'],'exact duplicate 확정과 다름'],
    ['유사쌍 상세 육안 검토',qa_summary['상세_육안검토_쌍'],'전체910쌍 중107쌍 상세 검토'],
    ['유사쌍 우선 위험 후보',qa_summary['위험_cross_split_유사쌍_수'],'67쌍은 확정 중복 아님'],
    ['복원 추가 검토 이미지',qa_summary['GT_mask_육안검토_필요_이미지_수'],'8장은 실제 결함 소실 확정 아님'],
], columns=['진단 항목','개수','해석'])
display(diagnostics)
display(manifest[['machine','serial','capture_date','bbox_count','label_source_type']].isna().sum()
        .rename('결측수').reset_index().rename(columns={'index':'변수'}))
display(clean_meta['review_status'].value_counts().rename('images').reset_index())
print('라벨 불일치 역사적 상세: outputs/label_review/conflict_review.csv')
print('최종 이미지당 bbox 개수:')
display(manifest['bbox_count'].value_counts().sort_index().rename('images').reset_index())
print('최종 해상도 분포:')
display(clean_meta.groupby(['width','height']).size().rename('images').reset_index())
''')
md('''
**분포와 검증 전략:** 결측은 핵심 메타데이터의 null 값을 대상으로 확인한다. 형식 오류·누락·중복은 위 저장 진단 결과로 확인하고,
극단적으로 작은 bbox는 자동 삭제할 이상치가 아니라 실제 검출 위험으로 본다. 클래스는 단일 클래스이며 최종512장 모두 결함 라벨이 있어 양품/불량 불균형의 생산 분포를 추정할 수 없다.
설비·제품·촬영일의 45개 그룹을 보존하는 분할(seed42)을 사용했으며 exact/pixel 중복의 split 누출은0건이다.
외관상 유사쌍910개와 위험67개는 남아 있어 완전한 독립성 보장으로 표현하지 않는다.
아래 크기 분포는 최종 저장 feature CSV의 bbox 기준이다. 이물질 segmentation 추정 면적과 구분한다.
''')
code('''
object_features = load_csv('outputs/foreign_object_condition_analysis/features.csv')
assert len(object_features)==1163
display(object_features[['bbox_area','bbox_aspect_ratio','image_width','image_height']].describe())
display(object_features.groupby('split')['bbox_area'].agg(['count','min','median','max']).reindex(split_order))
show_png('outputs/yolo_512_telea_qa/위험_cross_split/위험_유사쌍_01.png',
         '기존 상세 비교판: 유사쌍 검토 자료이며 동일 제품 확정 판정이 아님')
''')

md('''
## 3. 사각형 표시 제거 전처리

**목적:** GT와 관련된 컬러 사각형을 탐지 특징으로 사용하는 지름길을 줄이고, 실제 X-ray 신호 기반 학습 입력을 구성한다.
**공식 평가항목 연결:** 1. 데이터 이해 및 진단, 5. 창의성·차별성.

사각형 표시는 원래 센서 신호가 아니라 후처리 주석이므로 실제 생산 입력에 같은 표시가 없다면 모델 일반화가 깨질 수 있다.
정확한 palette/RGB 마스크로 컬러 픽셀을 식별하고 TELEA/NS, dilation0/1/2, radius1/2/3의18조합을20장 PoC에서 비교했다.
composite score는 GT 신호 변화·core 침범·잔존 표시·경계 변화를 종합한 보존성 proxy이며 낮을수록 좋다.
TELEA, dilation0, radius3이 기존 composite 최선(약5.672499)이고 NS 최선은약5.774496이다.
차이는 작으며 clean reference 없이 마킹으로 가려진 실제 신호 복원을 증명할 수 없다. PoC high-risk7/20과 전체 적용 후141개 GT-mask 겹침 bbox를 별도 검토했다.
''')
code('''
method_table = load_csv('outputs/inpainting_poc_20/method_summary.csv')
method_comparison = method_table.groupby('method').agg(
    best_composite_score=('composite_score','min'), avg_composite_score=('composite_score','mean'),
    avg_contrast_change=('mean_abs_gt_contrast_change','mean'),
    avg_edge_excess=('mean_artificial_edge_excess','mean')).reset_index()
display(method_comparison)
display(method_table.sort_values('rank')[['method','dilation_px','radius','images','composite_score',
        'mean_abs_gt_contrast_change','mean_mask_fraction_of_gt_core','residual_palette_pixels']].head(6))
display(pd.DataFrame([{'method':restoration_summary['method'],'radius':restoration_summary['radius'],
        'dilation':restoration_summary['dilation'],'images':restoration_summary['images'],
        'GT_overlap_images':restoration_summary['gt_mask_overlap_images'],
        'GT_overlap_bbox':restoration_summary['gt_mask_overlap_bbox'],
        'outside_mask_changed_pixels':restoration_summary['outside_mask_changed_pixels']}]))
for path in sorted((ROOT/'outputs/inpainting_poc_20/figures').glob('*_comparison.png'))[:2]:
    show_png(path.relative_to(ROOT).as_posix(), '기존 비교판: ORIGINAL / MASK / TELEA / NS / GT')
''')

md('''
## 4. Shortcut 검증

**목적:** 이물질이 없는 영역에 fake restoration을 추가해 복원 흔적 자체가 모델의 탐지 단서인지 점검한다.
**공식 평가항목 연결:** 5. 창의성·차별성, 2. AI 예측모델 개발.

기존20장 PoC의 baseline과 fake augmentation 모델을 동일 설정으로 비교했다. 전체34개 paired 영역 중 holdout10개(원본 이미지6장), test5개 pair만 별도로 해석한다.
FR-FPR의 분모는 fake 영역 수이며, 영상 전체 일반 FP rate와 다르다. Train pair는 이미 본 이미지이므로 holdout/test를 우선한다.
이 진단은 최종512장 모델의 새로운 테스트가 아니다. 정상 성능 P/R/mAP 열은 기존20장 실험의 저장 test 지표다.
''')
code('''
shortcut = load_csv('outputs/shortcut_diagnostic_gpu/model_summary.csv')
shortcut_cols = ['model','scope','diagnostic_pairs','source_images','fake_regions','fr_fpr',
                 'fake_fp_total','mean_fake_region_max_conf','mean_delta_confidence',
                 'new_fake_fp_count','gt_detection_loss_count','gt_detection_gain_count']
display(shortcut[shortcut['scope'].isin(['all','holdout','test'])][shortcut_cols])
normal_comparison = shortcut[shortcut['scope']=='all'][['model','normal_precision','normal_recall',
                                                     'normal_map50','normal_map50_95']].copy()
normal_comparison['F1_from_reported_PR'] = [harmonized_f1(p,r) for p,r in
                                          zip(normal_comparison.normal_precision,normal_comparison.normal_recall)]
display(normal_comparison)
display(load_csv('outputs/shortcut_diagnostic_gpu/paired_comparison.csv')[['scope','metric','test','n',
        'paired_mean_difference','p_value']])
show_png('outputs/shortcut_diagnostic_gpu/cases/E_largest_delta_pair_0002/overlay.png',
         '최대 confidence 변화 사례의 기존 overlay: fake FP 확정 사례가 아님')
''')
md('''
**저장 결과에 따른 결론:** baseline과 fake augmentation 모두 전체/holdout/test FR-FPR=0, 새 fake FP=0, GT loss/gain=0이었다.
현재 PoC에서 복원 흔적 shortcut을 지지하는 강한 증거는 확인되지 않았다. 표본 규모·conf floor0.001·fake 생성 방식의 한계가 있어 shortcut 부재를 증명한 것은 아니다.
확률보정, 앙상블, 불확실성 추정 또는 불균형 학습을 수행했다고 주장하지 않는다. 차별성은 주석 제거 보존성 비교와 paired negative control 검증에 있다.
''')

md('''
## 5. 모델 비교와 최종 모델

**목적:** 실제 저장된 학습 실행을 자동 탐색해 baseline 포함 모델·설정별 Precision/Recall/F1/mAP를 비교한다.
**공식 평가항목 연결:** 2. AI 예측모델 개발.

`outputs/yolo_runs`의 results/summary와 experiments 인덱스를 합쳐 탐색한다. weights/학습 로그 존재 여부와 평가 지표 저장 상태를 함께 표시한다.
20장 PoC의 YOLO11n/YOLO26n 구조 비교와512장 본 실험은 데이터 규모가 달라 같은 순위표로 섞지 않는다.
512장 본 실험은 YOLO11n의 해상도·학습률·augmentation 설정 비교다. F1_from_reported_PR은 저장된 P/R의 조화평균이며 validator의 최고 F1 또는 고정 threshold 집계 F1과 구분한다.
''')
code('''
experiment_index = load_csv('outputs/yolo_runs/experiments.csv')
indexed = experiment_index.set_index('run_name').to_dict('index')
run_root = ROOT/'outputs/yolo_runs'
run_dirs = sorted(p for p in run_root.iterdir() if p.is_dir() and
                  ((p/'results.csv').exists() or (p/'summary.json').exists() or p.name in indexed))
run_rows=[]
for run_dir in run_dirs:
    relative=run_dir.relative_to(ROOT).as_posix()
    saved=load_json(relative+'/summary.json') if (run_dir/'summary.json').exists() else {}
    config=load_json(relative+'/experiment_config.json') if (run_dir/'experiment_config.json').exists() else {}
    fallback=indexed.get(run_dir.name,{})
    row={'run_name':run_dir.name,'model':config.get('model',fallback.get('model')),
         'imgsz':config.get('imgsz',fallback.get('imgsz')),'lr0':config.get('lr0',fallback.get('lr0')),
         'augmentation':fallback.get('augmentation','미저장'),'best_epoch':saved.get('best_epoch',fallback.get('best_epoch')),
         'train_images':saved.get('train_images'),'val_images':saved.get('val_images'),'test_images':saved.get('test_images'),
         'weights_present':(run_dir/'weights/best.pt').exists(),
         'metrics_source':relative+'/summary.json' if saved else ('outputs/yolo_runs/experiments.csv' if fallback else '미저장')}
    for split in ('val','test'):
        for metric in ('precision','recall','map50','map50_95'):
            row[f'{split}_{metric}']=saved.get(f'{split}_{metric}',fallback.get(f'{split}_{metric}',np.nan))
        p,r=row[f'{split}_precision'],row[f'{split}_recall']
        row[f'{split}_F1_from_reported_PR']=harmonized_f1(p,r) if pd.notna(p) and pd.notna(r) else np.nan
    row['comparison_group']='512장 본 실험' if saved.get('train_images')==359 else ('20장 PoC' if saved.get('train_images')==14 else '평가 미완료/모수 미확인')
    if (run_dir/'results.csv').exists():
        logs=load_csv(relative+'/results.csv')
        row['logged_epochs']=len(logs)
    else:
        row['logged_epochs']=0
    row['metrics_status']='평가 저장됨' if pd.notna(row['test_map50']) else '평가 지표 없음'
    run_rows.append(row)
models=pd.DataFrame(run_rows)
final_run=PureWindowsPath(error_summary['provenance']['model']).parents[1].name
models['selected_final']=models['run_name']==final_run
display(models[['run_name','model','comparison_group','logged_epochs','weights_present','metrics_status','selected_final']])
metric_cols=['run_name','model','imgsz','lr0','augmentation','val_precision','val_recall',
             'val_F1_from_reported_PR','val_map50','val_map50_95','test_precision','test_recall',
             'test_F1_from_reported_PR','test_map50','test_map50_95','selected_final']
for group in ['20장 PoC','512장 본 실험','평가 미완료/모수 미확인']:
    print(group)
    display(models[models.comparison_group==group][metric_cols])
final_model = models[models.selected_final].iloc[0]
assert final_model['weights_present']
final_weight=f'outputs/yolo_runs/{final_run}/weights/best.pt'
assert file_sha(source(final_weight))==error_summary['provenance']['model_sha256']
print('기존 후속 분석에서 고정된 최종 모델:',final_weight)
''')
md('''
**최종 모델 선정 근거와 범위:** 기존 후속 평가에서 고정한 모델은 TELEA512 / YOLO11n / imgsz1280 / batch4 / lr0.001 / no augmentation / seed42의 best.pt(best epoch7)이다.
저장 Val Recall≈0.973800, mAP50≈0.970014, mAP50–95≈0.473756으로640/960 no-aug baseline보다 높은 위치·검출 성능을 보였다.
lr0.0005는 Val mAP50≈0.977442로 더 높지만 mAP50–95≈0.448577이며, augmentation 실행은 mAP50–95≈0.494868이 더 높지만 Recall≈0.961907이다.
**관찰됨:** 최종 모델 Val F1_from_reported_PR=**0.969879**이고 lr0.0005 모델은 **0.971176**으로, **F1 최대 모델은 별도로 존재했다**(본 실험 비교 기준). 최종 모델이 F1 최적이라고 주장하지 않는다.
**사후 판단:** 최종 모델 유지는 Recall, bbox 안정성 검토의 지표인 Val mAP50–95, 기존 고정 baseline과 후속 분석의 일관성을 함께 고려한 설명이다. lr0.0005보다 Recall과 mAP50–95는 높지만 F1과 mAP50은 낮다는 교환관계를 숨기지 않는다. mAP50–95는 반복 촬영 간 bbox 안정성을 직접 측정한 지표가 아니므로 bbox 안정성 우위를 독립적으로 검증했다고 표현하지 않는다.
기록에 없는 사전 선정 기준을 추가하지 않았으며, 현재 설명은 기존 결과에 대한 종합적인 사후 판단이다. 현재 노트북은 기존 고정 baseline과 후속 검증의 연결을 유지하며 Test 결과로 모델을 재선택하지 않는다.
lr0.0005의 run02는 반복 실행이므로 별도 신규 모델 구조로 세지 않는다. 구조가 다른 두 모델의 비교는20장 PoC 수준에서만 수행됐다는 한계를 명시한다.
''')
code('''
final_logs=load_csv(f'outputs/yolo_runs/{final_run}/results.csv')
final_logs.columns=final_logs.columns.str.strip()
print('기존 학습 로그: 첫3 epoch / 마지막3 epoch')
display(pd.concat([final_logs.head(3),final_logs.tail(3)]).drop_duplicates())
validation_curve=f'outputs/yolo_runs/{final_run}/validation/BoxPR_curve.png'
if (ROOT/validation_curve).exists():
    show_png(validation_curve,'기존 최종 모델 validation PR curve')
else:
    print('해당 PR PNG는 저장되어 있지 않음; 기존 성능표와 로그를 사용')
''')

md('''
## 6. Threshold 분석

**목적:** Val에서 선택한 threshold 정책의 Precision/Recall/F1과 FN/FP 비용을 검토하고 잠긴 정책을 Test 결과에 연결한다.
**공식 평가항목 연결:** 2. AI 예측모델 개발, 4. 현장 활용방안.

최종 정책은 **T_low=0.20, T_high=0.25**로 유지한다. **0.20은 F1 optimum이 아니라 screening lower bound**이며 자동 NG 후보 상한은0.25다. confidence는 보정된 불량 확률이 아니다.
Val에서 T_low 후보0.05/0.10/0.15/0.20/0.25와 추가 FP 예산7개(0.10/image), 재검 이미지 예산20% 안에서 FN 복구 최대·FP 최소·동률이면 높은 threshold를 선택했다.
Best-F1 단일 threshold0.31은 비교용이고,0.20은 재현율 중심 정책으로 잠겼다. Test로 threshold를 재선택하지 않았다.
''')
code('''
val_sweep=load_csv('outputs/safety_threshold_policy/val_threshold_sweep.csv')
val_policies=load_csv('outputs/safety_threshold_policy/val_policy_candidates.csv')
test_policies=load_csv('outputs/safety_threshold_policy/test_policy_result.csv')
display(val_sweep[val_sweep['threshold'].isin([.02,.05,.10,.15,.20,.25,.31])]
        [['threshold','TP','FN','FP','precision','recall','F1','FP_per_image']])
display(val_policies[['policy','T_low','T_high','TP','FN','FP','precision','recall','F1',
        'FN_Recovery','Added_FP','band_images','reinspection_images']])
assert locked_policy['selection_split']=='val'
assert locked_policy['selected']['T_low']==.20 and locked_policy['selected']['T_high']==.25
display(pd.DataFrame([{'selection_split':locked_policy['selection_split'],
        'T_low':locked_policy['selected']['T_low'],'T_high':locked_policy['selected']['T_high'],
        'added_FP_budget':locked_policy['added_FP_budget'],'review_image_budget':locked_policy['review_image_budget']}]))
show_png('outputs/safety_threshold_policy/visualizations/val_threshold_curve.png',
         '기존 Val threshold 곡선')
display(test_policies[['policy','threshold','TP','FN','FP','precision','recall','F1',
        'band_detections','reinspection_images','normal_candidate_images']])
''')
md('''
**결과 해석:** Val에서0.25→0.20은 TP170→171, FN5→4, FP11→12로 FN1개 복구/FP1개 증가였다.
그러나 Val middle band가 있는2개 이미지는 이미 다른 high-confidence bbox로 자동 NG여서 추가 재검 이미지는0장이었다.
Test에는0.20–0.25 band 자체가 없고 baseline/final 정책 모두 TP148/FN12/FP8, Recall92.50%, Precision94.87%, F1≈93.67%였다.
Test 재검 이미지도0/76으로 운영 개선을 입증하지 못했다. ‘정상 후보’4장도 GT가 있는 이미지이므로 정상 보증이 아니다.
학습 summary의 Test P/R은 validator 집계이고, 이 표는 고정 threshold와 IoU0.5 일대일 매칭 집계이므로 값이 조금 다른 것이 오류는 아니다.
''')

md('''
## 7. FN/FP 오류분석

**목적:** 저장된 Test FN12건·FP8건의 bbox·confidence·IoU·크기·대비·위치·복원 조건을 비교하여 오류 집중 조건을 도출한다.
**공식 평가항목 연결:** 3. 영향요인 및 오류분석, 5. 창의성·차별성.

수정된 `outputs/error_analysis`를 사용한다. FN candidate IoU는 표시 candidate와 해당 GT의 직접 IoU이며 threshold 미만 후보도 계산한다.
평가 매칭 대상은 threshold 이상 후보로 유지하므로 `low_confidence_localization_success`라도 Test FN이다.
주 분류는 기존 시각화 candidate 선택 기준이며, 다른 FN에 존재하는 저점수 IoU≥0.5 후보는 별도 열에 보존한다.
localization_failure는 관찰된 위치/크기 실패 기전이며 정상 구조 오인이나 복원 흔적의 물리적 원인을 확정한 라벨이 아니다.

**선택 candidate 기준 주 분류:** localization_failure8/12, low_confidence_localization_success3/12, low_confidence_localization_failure1/12를 유지한다. **선택 candidate 기준 주 분류에서는66.7%가 localization failure**이며, FN 원인의66.7%가 유일하게 위치 실패라고 단정하지 않는다. FN12건은 **8개 이미지**에서 발생해 독립적인12개 촬영 표본으로 볼 수 없다.

**별도 보조 분석 — 관찰됨:** 기존 진단 CSV의 `low_confidence_gt_iou`와 `low_confidence_confidence`를 보면 FN **9/12**에 score<T_low=0.20이면서 IoU≥0.5인 후보가 존재한다. 주 분류3건은 선택된 candidate 자체가 저점수 위치 성공인 경우이며, 보조9건은 다른 후보까지 포함한 존재 여부다. 두 분석은 분모가 같은 중복 진단으로 서로 합산하거나 주 분류를 바꾸지 않는다.
''')
code('''
error_cases=load_csv('outputs/error_analysis/error_cases.csv')
classification=load_csv('outputs/error_analysis/error_classification.csv')
cause_frequency=load_csv('outputs/error_analysis/summary/cause_frequency.csv')
fn_diagnostics=load_csv('outputs/error_analysis/summary/fn_candidate_diagnostics.csv')
gt_conditions=load_csv('outputs/error_analysis/summary/gt_condition_rates.csv')
all_gt=load_csv('outputs/error_analysis/summary/all_test_gt.csv')
fp_confidence=load_csv('outputs/error_analysis/summary/fp_confidence_bins.csv')
verification=load_json('outputs/error_analysis/summary/verification.json')
fn=error_cases[error_cases.error_type=='FN'].copy()
fp=error_cases[error_cases.error_type=='FP'].copy()
expected_causes={'localization_failure':8,'low_confidence_localization_success':3,'low_confidence_localization_failure':1}
assert len(fn)==12 and len(fp)==8 and fn.cause.value_counts().to_dict()==expected_causes
assert verification['candidate_iou_verified']
display(cause_frequency)
display(fn_diagnostics[['case_id','prediction_index','confidence','candidate_confidence_threshold',
        'candidate_gt_iou','best_eligible_iou','cause']])
display(fp[['case_id','confidence','candidate_gt_iou','prediction_to_gt_area_ratio','cause','review_required']])
show_png('outputs/error_analysis/summary/fn_cause_frequency.png','기존 수정 완료 FN 빈도 그래프')
show_png('outputs/error_analysis/summary/fp_cause_frequency.png','기존 FP 빈도 그래프')
auxiliary_low_score = fn[(fn['low_confidence_gt_iou'] >= .5) &
                         (fn['low_confidence_confidence'] < locked_policy['selected']['T_low'])]
assert len(auxiliary_low_score)==9 and fn['unique_id'].nunique()==8
print('별도 보조 분석: FN9/12에 저점수 IoU≥0.5 후보 존재; 주 분류는8/3/1 유지')
display(auxiliary_low_score[['case_id','cause','low_confidence_confidence','low_confidence_gt_iou']])
''')
code('''
display(gt_conditions[gt_conditions.condition.isin(['small_object','low_contrast','product_edge',
        'image_edge_near','restoration_near','restoration_intersects'])])
show_png('outputs/error_analysis/summary/fn_condition_rates.png','기존 조건별 FN/GT 비교 그래프')
print('크기 × 대비의 조건 조합 (재학습 또는 정책 선택에 사용하지 않은 기술 통계)')
interaction=all_gt.groupby(['small_object','low_contrast'],dropna=False).agg(
    GT=('status','size'),FN=('status',lambda x:(x=='FN').sum())).reset_index()
interaction['FN_rate']=interaction['FN']/interaction['GT']
display(interaction)
print('설비별 FN/GT (인과 원인 해석 불가)')
machine_condition=all_gt.groupby('machine').agg(GT=('status','size'),FN=('status',lambda x:(x=='FN').sum())).reset_index()
machine_condition['FN_rate']=machine_condition['FN']/machine_condition['GT']
display(machine_condition)
display(fp_confidence)
show_png('outputs/error_analysis/summary/fp_confidence_bins.png','기존 confidence 구간별 FP/전체 예측')
''')
md('''
**확인된 집중 조건과 한계:** Train 하위25% 기준 소형 bbox≤81px²는 FN7/41(17.07%), 그 외5/119(4.20%)다.
Train 기준 낮은 bbox 주변 정규화 대비≤0.0698918은 FN7/60(11.67%), 그 외5/100(5.00%)다.
**관찰됨:** 소형 객체 전체와 저대비 객체 전체는 각각 비교군보다 높은 FN 비율을 보였다. 그러나 소형+저대비는 **3/19=15.79%**, 소형+비저대비는 **4/22=18.18%**로, **소형군 내부에서는 저대비의 추가 위험 증가가 확인되지 않았다**. 두 조건이 결합되면 위험이 더 증가한다는 결론은 내리지 않는다.
이 두 조건은 중복되며 표본·설비·제품 구조가 얽혀 있으므로 두 조건의 인과관계나 독립적인 효과를 단정하지 않는다. 기존 조합표는 기술 통계로만 해석한다.
제품 경계군 FN3/73(4.11%)은 비경계9/87(10.34%)보다 낮아 이번 Test에서 경계 집중을 주장하지 않는다.
모든 GT가 복원 영역5px 이내라 비근접 비교군이 없으며, 복원 원인 집중을 주장할 수 없다.
FP8개는 모두 FN이 있는4개 이미지에서 미검출 GT와 겹치지만 IoU0.5를 통과하지 못한다.7개는 예측 면적이 GT보다 크고, U0488은 중심 이동이 관찰된다.
소형/저대비는 측정 조건이고 주 오류 기전·물리적 원인과 구분한다. 전 사례는 전문가 bbox/라벨 검토가 필요하다.
''')
md('''
### 7.1 위치는 맞지만 점수가 낮은 FN

**목적:** U0138/U0352/U0380의 candidate IoU≥0.5와 confidence<0.20을 함께 확인한다.
**공식 평가항목 연결:** 3. 영향요인 및 오류분석.

각각 IoU≈0.771798/0.690004/0.746874, confidence≈0.191274/0.023290/0.097960이며
`low_confidence_localization_success`다. 위치 통과는 low-confidence 후보의 진단 결과이고 평가 TP로 전환하지 않는다.
U0181의 기존 최고 IoU candidate는 IoU≈0.365997, confidence≈0.001634로 점수와 위치 기준 모두 미달이다.
''')
code('''
selected_ids=['U0138_FN_gt0','U0352_FN_gt0','U0380_FN_gt0','U0181_FN_gt0']
display(fn_diagnostics[fn_diagnostics.case_id.isin(selected_ids)]
        [['case_id','confidence','candidate_gt_iou','best_eligible_iou','cause']])
for case_id in selected_ids:
    show_png(f'outputs/error_analysis/fn/{case_id}.png',case_id+' / 기존 개별 분석 이미지')
''')
md('''
### 7.2 위치/크기 오차와 IoU 경계 사례

**목적:** FN과 FP가 같은 물체의 bbox 오차로 동시에 발생할 수 있음을 확인한다.
**공식 평가항목 연결:** 3. 영향요인 및 오류분석.

U0488의 confidence≈0.590495는 threshold 이상이지만 IoU **0.498477**로0.5에 근소하게 미달한다.
따라서 GT는 FN, 대응 accepted prediction은 FP로 기록된다. 이를 ‘탐지가 전혀 없음’ 또는 ‘정상 구조 오인’으로 단정하지 않는다.
U0371의3개 작은 GT도 대응 예측 bbox가 더 넓어 IoU 기준을 놓친다. Test를 보고 IoU 기준을 낮추거나 threshold를 재조정하지 않는다.
''')
code('''
display(error_cases[error_cases.unique_id.isin(['U0488','U0371'])][
        ['case_id','error_type','confidence','candidate_gt_iou','cause','bbox_area','gt_area_ratio']])
show_png('outputs/error_analysis/fn/U0488_FN_gt0.png','U0488 FN / IoU0.498477')
show_png('outputs/error_analysis/fp/U0488_FP_pred0.png','U0488 FP / 같은 IoU 기준 미달')
show_png('outputs/error_analysis/fp/U0371_FP_pred0.png','U0371의 소형 GT 대응 FP 대표')
print('사람 검토 대상:')
display(error_cases[['case_id','image_name','cause','review_required','physical_cause']])
''')

md('''
## 8. 현장 활용방안

**목적:** bbox·confidence와 오류분석을 작업자 재검, AI 보조 판정, 품질 이력관리 흐름으로 연결한다.
**공식 평가항목 연결:** 4. 현장 활용방안, 3. 영향요인 및 오류분석, 5. 창의성·차별성.

**제안됨: 현장 활용 흐름 (실제 운영 효과 검증과 구분)**

X-ray 입력 → 전처리 → AI 이물질 탐지 → bbox + confidence 출력 → 고신뢰 후보 우선 확인 → 저신뢰 후보 재검 → 소형/저대비 위험조건 추가 확인 → 작업자 최종판정 → 결과 저장

전처리는 기존 컬러 주석이 있을 때 적용하며, 실센서 영상에 표시가 없다면 불필요한 복원을 추가하지 않는다.
이미지 단위 잠긴 정책은 하나라도 confidence≥0.25면 자동 NG 후보, high가 없고0.20≤confidence<0.25면 재검 후보, 둘 다 없으면 정상 후보다.
이는 AI 경보/검사 우선순위이며 최종 출하 판정은 작업자가 원본·확대 영상과 제품 기준으로 결정한다.
하나의 이미지에 high 후보가 있더라도 저점수 bbox와 소형/저대비 위치를 추가 확인할 수 있다.
U0352/U0380처럼0.20보다 낮은 후보는 현재 band 정책으로 복구되지 않는다. 모든 낮은 점수를 자동 통과/자동 NG 처리하는 것은 별도 검증 없이 제안하지 않는다.
소형/저대비 진단은 GT 기반으로 계산한 조건이므로 실운영 후보 bbox feature의 TP/FP 구분력과 같다고 가정하지 않는다.

**검증됨/관찰됨:** 저장 결과에서 잠긴 T_low=0.20/T_high=0.25의 screening·이미지 routing 정책과 Test 집계를 확인했다. 이는 저장 결과 분석이며 현장 시스템 배포나 실제 작업자 재검 효과의 검증이 아니다.
**제안됨·미구현:** score<0.20 후보 재검은 현재 구현된 정책이 아닌 운영 제안이다. 저점수 후보를 현장 화면으로 제공하고 재검을 연결하는 추가 기능과 효과는 미구현/미검증 상태다.
**작업자 참고 조건·자동판정 미구현:** GT 기반 소형/저대비 조건은 오류분석에서 도출한 작업자 참고 조건이다. GT 없는 실제 운영 자동판정 기능으로 구현되거나 검증되지 않았다. 작업자 최종판정과 품질 이력관리도 현장 적용 제안으로 구분한다.
''')
code('''
field_use = pd.DataFrame([
    ['입력/전처리','원본, 표시 유무, 기존 mask, 처리 버전','주석 shortcut 예방·원본 보존','전처리 보존성 비교와 연결','기존 처리 검증됨; 현장 적용 제안됨'],
    ['고신뢰 후보 우선 확인','bbox와 점수≥0.25','자동 NG 후보 경보 후 작업자 확인','U0488의 고점수 위치 오차도 확인','저장 routing 검증됨; 작업자 흐름 제안됨'],
    ['저신뢰 후보 재검','0.20≤점수<0.25 band 및 확대 영상','고신뢰가 없을 때 이미지 재검 후보','Test band0이므로 개선 미검증','기존 정책 분석됨; 현장 재검 제안됨'],
    ['위치 일치/저점수 점검','점수<0.20의 진단 후보·제품 검사 기준','결함 가능성 추가 검토','U0138/U0352/U0380; 자동 복구 주장 아님','운영 제안됨; 정책 확장 미구현'],
    ['소형/저대비 추가 확인','확대·주변 대비·제품 구조','작업자 참고 조건','소형17.07%, 저대비11.67%; GT 기반','오류조건 관찰됨; 자동판정 미구현'],
    ['bbox 위치/크기 검토','GT 검증 자료, 원본/표시 영상','과대 bbox·IoU 경계 원인 점검','FP8/8와 선택 candidate FN8/12','IoU 검증됨; 현장 확인 제안됨'],
    ['최종판정/품질 이력','작업자 판정, 설비/serial/시간, 모델/threshold 버전','이력 조회·설비/제품별 오류 모니터링','자동 NG 후보와 실제 판정·재검 결과 분리','현장 기능 제안됨·미구현'],
], columns=['단계','확인 정보','활용','오류분석 연결','상태'])
display(field_use)
display(pd.DataFrame([
    ['경보 품질','양품/불량 모두 포함한 신규 독립 생산 데이터','false reject, 미검출 이미지율, 작업자 재검 정확도'],
    ['운영 부하','실제 검사시간·작업자 referral 기록','재검률, cycle time, 검사 우선순위 효율'],
    ['공정점검','설비·제품·촬영조건별 품질 이력','오류 발생 조건 변화 모니터링; 원인 단정 금지'],
],columns=['확인 대상','추가 검증 자료','지표']))
''')

md('''
## 9. 최종 결론

**목적:** 전처리·모델·threshold·오류분석·현장 활용의 확인된 결과와 한계를 서면평가 문장으로 정리한다.
**공식 평가항목 연결:** 1–6 전체.

1. 최종 라벨 연결 고유512장과1,163개 bbox를 기준으로 장비·제품·촬영일 그룹을 보존하여 Train359/Val77/Test76장으로 분할했으며, 형식 오류·라벨 누락·exact split 누출은0건이다.
2. 실제 X-ray 신호 대신 컬러 사각형 표시를 학습하는 위험을 줄이기 위해 palette/RGB mask 기반 TELEA/NS를 비교했고 기존 composite 기준 TELEA/dilation0/radius3을 선택했으나 clean reference 없는 복원 보존성에는 한계가 있다.
3. Fake Restoration paired 진단에서 두 모델 모두 fake FP와 GT loss/gain이0으로 강한 shortcut 증거가 없었지만,20장 PoC가 shortcut 부재나 최종512장 모델의 일반화를 입증하지는 않는다.
4. 최종 모델 Val F1=0.969879보다 lr0.0005 모델의0.971176이 높았지만, Recall·Val mAP50–95의 bbox 정밀도·기존 baseline 일관성을 종합한 사후 판단으로 기존 모델을 유지하며 기록에 없는 사전 기준은 만들지 않았다.
5. T_low=0.20/T_high=0.25를 유지하고0.20은 F1 optimum이 아닌 screening lower bound로 해석하며, Val의 잠재 FN1개 복구·FP1개 증가와 달리 Test는 TP148/FN12/FP8, Precision94.87%, Recall92.50%, F1약93.67%로 baseline 대비 개선이 없었다.
6. 8개 이미지의 FN12건에서 선택 candidate 기준 주 분류는 localization_failure8/12(66.7%), low_confidence_localization_success3/12, low_confidence_localization_failure1/12이며, 다른 후보를 포함하면9/12에 저점수 IoU≥0.5 후보가 존재한다는 별도 보조 결과와 섞지 않는다.
7. 소형 전체와 저대비 전체의 FN 비율은 각각17.07%와11.67%로 비교군보다 높았지만, 소형군 내 저대비3/19(15.79%)는 비저대비4/22(18.18%)보다 낮아 추가 위험 증가와 인과관계를 단정하지 않는다.
8. U0138/U0352/U0380은 위치가 맞아도 낮은 점수 때문에 FN으로 남고, U0488은 IoU0.498477의 경계 사례여서 작업자에게 점수·위치·확대 영상을 함께 제공해야 한다.
9. 현장 검사·판정·이력 흐름은 제안이며 score<0.20 후보 재검과 GT 기반 소형/저대비 자동판정은 미구현이고, 현재 검증 범위는 기존 bbox 재계산과 저장 결과의 일치 및 입력 불변성 확인에 한정한다.
''')

md('''
## KAMP 공식 평가항목 대응표

**목적:** 사용자가 제공한 공식 평가항목1–6을 이 노트북의 검증 결과와 직접 연결한다.
평가항목5에서는 실제 수행한 표시 제거·보존성 비교·negative control·IoU/점수 분리 진단을 차별화 접근으로 제시하며, 수행하지 않은 앙상블·확률보정·불균형 학습을 실적으로 포함하지 않는다.
''')
code('''
kamp_mapping = pd.DataFrame([
    ['1. 데이터 이해 및 진단','1, 2, 3','생산단위/변수, 시간·설비·제품 관계, 결측·중복·라벨·분포, 그룹 분할, 표시 제거',
     'clean_512 검증 / split_summary / 910 유사쌍 QA / TELEA-NS 비교','유사쌍과 미승인 라벨 검토 잔여'],
    ['2. AI 예측모델 개발','1, 4, 5, 6','baseline 포함 모델 비교, Precision/Recall/F1/mAP, 고정 최종 모델, Val threshold',
     '20장11n/26n / 512장11n 설정 비교 / 잠긴0.20–0.25 정책','동일512장 다른 구조 모델 비교 없음; F1 정의 분리'],
    ['3. 영향요인 및 오류분석','7, 7.1, 7.2','FN/FP 사례, 크기×대비 조합, 설비별 FN/GT, IoU·confidence 분리',
     'FN12 / FP8 / candidate 직접 IoU / U0488 / 소형·저대비 발생률','소표본·조건 중복; 물리적 원인 미확정'],
    ['4. 현장 활용방안','1, 6, 8','사전경보, 검사 우선순위, 저점수 재검, 작업자 보조판정, 공정점검·품질 이력',
     'bbox+점수+확대 원본 / 잠긴 이미지 routing / 최종판정 저장 흐름','실제 생산라인 개선은 추가 검증 필요'],
    ['5. 창의성·차별성','3, 4, 7, 8','palette 주석 제거, TELEA/NS 보존성 비교, Fake Restoration negative control, 제조 조건 결합',
     '마킹 shortcut 방지·paired 검증·크기/대비 위험과 작업자 확인 연결','앙상블·확률보정·불균형 학습은 미수행'],
    ['6. 코드 및 재현성','실행 준비(셀2), 5, 7, 코드 및 재현성(셀35–36)','실제 환경·requirements 차이, 필수 상대경로 파일, 위→아래 실행, 기존 bbox 재계산',
     '저장된 셀 출력 / 출처 SHA 목록 / candidate IoU 재계산 / 순차 Python 실행','Jupyter 커널 실행 미검증; SHA는 현재 입력 불변성 확인'],
],columns=['공식 평가항목','Notebook section','대응 내용','실제 근거','평가 시 구분할 한계'])
display(kamp_mapping)
''')
md('''
## 코드 및 재현성 — KAMP 공식 평가항목6

**목적:** 표·이미지·모델 연결에 사용한 기존 결과의 상대경로와 SHA256을 제공하고 실행 중 변경이 없었는지 확인한다.
**공식 평가항목 연결:** 6. 코드 및 재현성(1·2·5의 근거 추적 포함).

**실행 환경·순서 — 검증됨:** 생성 환경은 Python3.12.14/pandas3.0.1/numpy2.3.5이고 Pillow를 사용했다. 실제 현재 버전과 requirements.txt의 차이는 아래 표로 확인한다. 모델 런타임을 호출하지 않으며 패키지를 설치하거나 requirements를 변경하지 않는다. 프로젝트 root에서 셀1→36의 코드 셀17개를 순차 Python 실행하고 rich output을 저장했다. Jupyter 커널 자체의 실행은 미검증이며 Jupyter 사용자는 필요한 환경에서 위→아래로 실행한다.

**필수 파일·상대경로:** 이 ipynb와 root 탐색용 train_yolo.py, requirements.txt, outputs/yolo_512_telea/manifest.csv·dataset.yaml·샘플 TXT, clean_512 메타데이터/검증, clean_512_split 요약, 학습 run의 CSV·JSON·최종 best.pt, 전처리/shortcut/threshold/error_analysis 결과 및 참조 PNG가 필요하다. 기존 predictions.json은 candidate 동일성 검증에 사용한다. 전체 사용 파일은 아래 상대경로 출처 목록으로 확인하고, 노트북을 프로젝트 root 또는 하위 폴더에서 실행한다. 외부 데이터 다운로드나 신규 추론은 없다.

**SHA256 — 검증 범위 제한:** source()에서 읽을 때 저장한 해시와 이 실행 종료 시 해시를 비교하여 **현재 입력 불변성**을 확인한다. 이 확인은 과거 기준파일과의 동일성 증명이 아니다. 최종 best.pt와 저장 provenance의 해시 비교는 별도 확인이며 전체 입력의 역사적 동일성으로 확대하지 않는다.

**candidate_IoU_verified — 기존 bbox 재계산 검증:** 기존 verification.json은 과거 재계산 결과를 기록한 것이며, 아래에서는 기존 CSV의 GT/candidate bbox로 IoU를 직접 재계산하고 기존 predictions.json의 bbox/index/confidence와도 대조한다. 신규 후보 생성·추론이 아니며, 라벨 의미나 물리적 원인의 검증은 아니다. 저장된 검증 기록과 이번 실행의 재계산 결과를 별도 열로 표시한다.
생성 시 셀을 순서대로 실행한 표와 기존 PNG가 노트북에 저장되어 있어 실행 전에도 결과를 열람할 수 있다.
''')
code('''
from importlib import metadata as package_metadata
requirements_lines = source('requirements.txt').read_text(encoding='utf-8-sig').splitlines()
requirements_map = {}
for line in requirements_lines:
    entry = line.strip()
    if entry and not entry.startswith('#'):
        name, separator, version = entry.partition('==')
        requirements_map[name.lower()] = version if separator else '버전 미고정'
def installed_version(package):
    try:
        return package_metadata.version(package)
    except package_metadata.PackageNotFoundError:
        return '미설치 (현재 순차 Python 생성 환경)'
display(pd.DataFrame([
    {'항목':'Python','requirements.txt':'미지정','기존 저장 실행환경':'3.12.14','현재 실행환경':sys.version.split()[0]},
    {'항목':'numpy','requirements.txt':requirements_map.get('numpy','미지정'),'기존 저장 실행환경':'2.3.5','현재 실행환경':np.__version__},
    {'항목':'pandas','requirements.txt':requirements_map.get('pandas','미지정'),'기존 저장 실행환경':'3.0.1','현재 실행환경':pd.__version__},
    {'항목':'Pillow','requirements.txt':requirements_map.get('pillow','미지정'),'기존 저장 실행환경':'셀 출력에 버전 미기록','현재 실행환경':installed_version('Pillow')},
    {'항목':'IPython','requirements.txt':requirements_map.get('ipython','미지정'),'기존 저장 실행환경':'Jupyter 커널 검증 없음','현재 실행환경':installed_version('IPython')},
]))
print('requirements는 학습 의존성도 포함하며, 이 읽기 전용 분석 환경과 동일한 lockfile이 아님')

cached_predictions = load_json('outputs/foreign_object_condition_analysis/predictions.json')
candidate_checks = []
for _, case in fn.iterrows():
    gt_box = json.loads(case['gt_bbox'])
    candidate_box = json.loads(case['prediction_bbox'])
    cached = cached_predictions[case['unique_id']][int(case['prediction_index'])]
    assert candidate_box == cached['box']
    assert math.isclose(float(case['confidence']),float(cached['confidence']),rel_tol=1e-12)
    intersection = math.prod(max(0.,min(gt_box[j+2],candidate_box[j+2])-max(gt_box[j],candidate_box[j])) for j in (0,1))
    gt_area = math.prod(gt_box[j+2]-gt_box[j] for j in (0,1))
    candidate_area = math.prod(candidate_box[j+2]-candidate_box[j] for j in (0,1))
    recalculated = intersection/(gt_area+candidate_area-intersection)
    candidate_checks.append(math.isclose(recalculated,float(case['candidate_gt_iou']),rel_tol=1e-12,abs_tol=1e-12))
candidate_iou_recalculated_verified = len(candidate_checks)==12 and all(candidate_checks)
assert candidate_iou_recalculated_verified
assert all(file_sha(ROOT/relative)==initial for relative,initial in SOURCE_LEDGER.items())
source_table=pd.DataFrame([{'relative_path':path,'sha256':digest} for path,digest in SOURCE_LEDGER.items()])
display(source_table)
display(pd.DataFrame([{'used_files':len(source_table),'input_files_unchanged':True,
                       'training_executed':False,'inference_executed':False,
                       'FN':len(fn),'FN_images':fn['unique_id'].nunique(),'FP':len(fp),
                       'candidate_IoU_verified_saved_record':verification['candidate_iou_verified'],
                       'candidate_IoU_verified_current_recalculation':candidate_iou_recalculated_verified,
                       'historical_baseline_identity_proven_by_current_input_hashes':False,
                       'Jupyter_kernel_execution_verified':False}]))
''')


def execute_and_save():
    """Execute actual code cells, capture pandas HTML and original PNG bytes."""
    import pandas as pd
    from PIL import Image

    if TARGET.exists():
        previous=json.loads(TARGET.read_text(encoding='utf-8'))
        assert len(previous['cells'])==len(cells), 'Preserve the existing cell numbering'
        for new,old in zip(cells,previous['cells']):
            new['id']=old['id']
    ns={'__name__':'__notebook__'}
    executed=0
    current_outputs=[]

    def capture_display(obj):
        if isinstance(obj,pd.DataFrame):
            data={'text/html':obj.to_html(index=True,border=0), 'text/plain':obj.to_string()}
        elif isinstance(obj,Image.Image):
            buffer=io.BytesIO();obj.save(buffer,format='PNG')
            data={'image/png':base64.b64encode(buffer.getvalue()).decode(),
                  'text/plain':f'<Existing PNG result {obj.width}×{obj.height}>'}
        else:
            data={'text/plain':str(obj)}
        current_outputs.append(dict(output_type='display_data',metadata={},data=data))

    old_cwd=Path.cwd()
    os.chdir(ROOT)
    try:
        for cell in cells:
            if cell['cell_type']!='code':
                continue
            executed+=1;current_outputs=[];stream=io.StringIO()
            if executed>1:
                ns['display']=capture_display
            with contextlib.redirect_stdout(stream):
                exec(compile(cell['source'],f'Notebook cell {executed}','exec'),ns)
            cell['execution_count']=executed
            text=stream.getvalue()
            if text:
                current_outputs.insert(0,dict(output_type='stream',name='stdout',text=text))
            cell['outputs']=current_outputs
        notebook=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata={
            'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
            'language_info':{'name':'python','version':ns['sys'].version.split()[0]},
            'kamp_analysis':{'execution_method':'Sequential Python execution with captured rich outputs',
                             'code_cells_executed':executed,'no_training':True,'no_inference':True,
                             'input_files_unchanged':True,
                             'official_rubric_source':'User-provided KAMP evaluation criteria 1–6',
                             'candidate_iou_recalculation_verified':ns['candidate_iou_recalculated_verified'],
                             'Jupyter_kernel_execution_verified':False}})
        TARGET.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
        # Structural and sequential-execution checks without optional nbformat.
        saved=json.loads(TARGET.read_text(encoding='utf-8'))
        assert saved['nbformat']==4 and len({c['id'] for c in saved['cells']})==len(saved['cells'])
        assert [c['execution_count'] for c in saved['cells'] if c['cell_type']=='code']==list(range(1,executed+1))
        assert not any(o['output_type']=='error' for c in saved['cells'] if c['cell_type']=='code' for o in c['outputs'])
        print(json.dumps({'notebook':str(TARGET),'cells':len(cells),'executed_code_cells':executed,
                          'used_source_files':len(ns['SOURCE_LEDGER']),'model_runs':len(ns['models']),
                          'FN':len(ns['fn']),'FP':len(ns['fp']),'bytes':TARGET.stat().st_size},ensure_ascii=False,indent=2))
    finally:
        os.chdir(old_cwd)


def execute_current_notebook():
    """Refresh the current corrected notebook with a real Jupyter kernel.

    The legacy cell definitions above describe an earlier analysis stage.
    Use the current notebook as the template so historical label conflicts
    cannot silently replace the subsequently verified source status.
    """
    import sys
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpec

    template = TARGET if TARGET.exists() else ROOT / 'KAMP_Xray_Submission' / TARGET.name
    if not template.is_file():
        raise FileNotFoundError('Current notebook template is missing; refusing to recreate obsolete analysis.')
    notebook = nbformat.read(template, as_version=4)
    nbformat.validate(notebook)
    manager = KernelManager(kernel_name='python3')
    manager._kernel_spec = KernelSpec(
        argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        display_name='Python 3', language='python',
    )
    NotebookClient(notebook, timeout=180, kernel_name='python3', km=manager,
                   resources={'metadata': {'path': str(ROOT)}}).execute()
    code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
    assert len(code_cells) == 17
    assert not any(output.output_type == 'error' for cell in code_cells for output in cell.outputs)
    notebook.metadata['kamp_analysis'].update(
        execution_method='Jupyter kernel execution via nbclient',
        code_cells_executed=len(code_cells), Jupyter_kernel_execution_verified=True,
    )
    nbformat.write(notebook, TARGET)
    print(json.dumps({'notebook': str(TARGET), 'executed_code_cells': len(code_cells),
                      'template': str(template), 'Jupyter_kernel_execution_verified': True}, indent=2))


# Keep the original public entry point on the corrected kernel-based executor.
execute_and_save = execute_current_notebook

if __name__=='__main__':
    execute_current_notebook()
