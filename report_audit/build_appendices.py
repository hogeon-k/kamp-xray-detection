from pathlib import Path
import json,csv,ast,collections
R=Path(__file__).resolve().parents[1];O=R/'report_audit'
def js(p):return json.loads((R/p).read_text(encoding='utf-8-sig'))
def cs(p):
 with (R/p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def table(h,rows):return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |']+['| '+' | '.join(str(v).replace('\n',' ').replace('|',' / ') for v in r)+' |' for r in rows])
roles=['전체범위/목차','경로·SHA·512/1163 연결','제조검사 목적/라벨/용어','변수사전/TXT예시','라벨출처/과거이력','촬영조건/분할','라벨해시/품질/해상도','분포/누출해석','객체크기/유사쌍그림','전처리설명','512복원/20장방법비교','shortcut설계','paired통계/overlay','shortcut한계','본모델비교설계','11설정/Val/Test/구조/증강/weight','최종모델유지해석','학습로그/Val PR','threshold설계','Val sweep/정책/Test','운영효과해석','오류분석정의','FN/FP/분류/후보','조건/조합/설비/FP점수','오류해석제한','저점수FN설명','대표FN4개','IoU경계설명','U0488/U0371/사람검토','현장제안','현장연결표','결론','공식평가항목설명','공식매핑표','재현성범위','환경/IoU재계산/해시']
cats=['공통','1/2/6','1/4','1','1','1/6','1/6','1','1/3','1/5','1/5','5','5','5','2','2/6','2','2','2/4','2/4/6','2/4','3','3/6','3','3','3','3','3','3','4','4','공통','공통','공통','6','6']
nb=js('KAMP_Xray_Final_Analysis.ipynb');inv=js('report_audit/inventory.json');a='\n\n## 부록 A. 최신 Notebook 36셀 전체 매핑\n\n'+table(['Cell','유형','execution_count','역할','평가항목'],[(i,c['cell_type'],c.get('execution_count') or '—',roles[i-1],cats[i-1]) for i,c in enumerate(nb['cells'],1)])+'\n'
hist=[]
for x in inv:
 if x['ext']=='.ipynb' and x['path'].startswith('.cache/'):
  p=R/x['path'];b=json.loads(p.read_text(encoding='utf-8'));s='\n'.join('CELL '+str(i)+'\n'+''.join(c.get('source',[]))+'\nOUTPUT\n'+'\n'.join(''.join(o.get('text',o.get('data',{}).get('text/plain',[]))) for o in c.get('outputs',[])) for i,c in enumerate(b['cells'],1));(O/(p.stem+'_historical.txt')).write_text(s,encoding='utf-8');hist.append([x['path'],len(b['cells']),sum(c['cell_type']=='code' for c in b['cells']),'과거 단계; 최신 숫자와 구분'])
a+='\n## 부록 B. 관련 Notebook 백업 전체\n\n'+table(['경로','전체셀','코드셀','해석'],hist)+'\n'
desc={
'augmentation_comparison':'계획; 실제완료는 학습run 확인','clean_512':'최종 원본512·선택TXT·출처','clean_512_candidate':'canonical누락12 중간후보','clean_512_split':'45그룹 split 및 보호해시','color_box_reverse_engineering':'팔레트·주석·bbox 관계','context_matched_restoration':'real/fake34쌍 문맥대응','directional_refinement_poc':'20장 방향보간 기각','eda_figures':'초기1134bbox 그림','edge_reinspection_poc':'경계band24/40/64·fusion','error_analysis':'최신 FN12/FP8·직접IoU','error_analysis_preview':'수정 전 오류분석','fake_restoration_debug':'fake 생성20장 검사','fn_root_cause_analysis':'Test160 크기·대비·경계·blur','foreign_object_condition_analysis':'1163bbox 영역추정·예측153장','inpainting_poc_20':'TELEA/NS18조합20장','label_provenance':'과거23장·XMLzero14·대체TXT','label_review':'초기라벨검토·미검토초기값','low_contrast_aux_poc':'18설정·Test복구0/추가FP4','missing_12_label_audit':'fallback12 TXT/VOC검증','model_comparison':'완료11·구조비교2쌍','notebook_execution_check':'과거순차exec 검증','restoration_after_similarity':'real34/fake19 after진단','restoration_diagnostics':'before/after 흔적특성','rgb_channel_audit':'512장 RGB·표시 분석','safety_threshold_policy':'Val92행·locked·Test routing','selected_512_original_bmp':'원본BMP512 선택','shortcut_diagnostic':'20장 CPU paired진단','shortcut_diagnostic_dataset':'paired34 입력','shortcut_diagnostic_dataset_gpu':'GPU paired입력','shortcut_diagnostic_gpu':'Notebook 채택 GPU진단','submission_label_audit':'최종512·이력23 출처감사','unique_512_audit':'원본3904후보·고유2532·옛1134bbox','unique_512_review':'고유512 및 초기대표라벨','unlabeled_rgb_audit':'미라벨NG2020 탐색','yolo_512_telea':'최종학습영상·mask·TXT·manifest','yolo_512_telea_qa':'유사910/107/67·GT겹침141','yolo_poc_20':'Train14/Val3/Test3·bbox24/5/5','yolo_poc_20_fake_restoration':'20장 Train-only fake','yolo_runs':'학습·평가·weights·곡선'}
a+='\n## 부록 C. outputs 전 계열\n\n'+table(['폴더','파일수','역할'],[(k,sum(x['path'].startswith('outputs/'+k+'/') for x in inv),d) for k,d in desc.items()])+'\n'
old={r['asset_id']:r for r in cs('outputs/eda_image_metrics.csv')};changes=[]
for r in cs('outputs/clean_512/metadata.csv'):
 before=old[r['sha256_image']];n=int(r['bbox_count']);o=int(before['object_count'])
 if n!=o:changes.append([r['unique_id'],r['filename'],o,n,n-o,before['label_format'],r['label_source_type']])
with (O/'label_count_changes.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['unique_id','filename','old_bbox','final_bbox','delta','old_format','final_source']);w.writerows(changes)
a+=f'\n## 부록 D. 초기 EDA와 최종 선택라벨\n\n이미지SHA로512장을 연결하였다. bbox수 변경 {len(changes)}장, 합계차이 {sum(r[4] for r in changes):+d}개, 1134→1163.\n\n'+table(['ID','파일','초기','최종','delta','초기형식','최종출처'],changes)+'\n'
v=js('report_audit/verification_computed.json');a+='\n## 부록 E. 본실험 설정 및 Val/Test 전체표\n\n'+(O/'model_tables.md').read_text(encoding='utf-8')+'\n\n'+table(['R번호','outputs/yolo_runs 아래 저장폴더'],[(r['id'],r['summary']['run_name']) for r in v['runs']])+'\n'
poc=[]
for r in cs('outputs/yolo_runs/experiments.csv'):
 if not r['run_name'].startswith('telea_512_'):
  s=js('outputs/yolo_runs/'+r['run_name']+'/summary.json');poc.append([r['run_name'],f'{s["train_images"]}/{s["val_images"]}/{s["test_images"]}']+[f'{s[k]:.6f}' for k in ['val_precision','val_recall','val_map50','val_map50_95','test_precision','test_recall','test_map50','test_map50_95']])
a+='\n## 부록 F. 20장 PoC 실제 모델결과\n\nexperiments.csv 총16행 중5행은20장 PoC이다. YOLO26n도 실제 학습되었으나512장 구조비교2종에 포함하지 않았다. 아래는14/3/3 split이다. 별도 telea_baseline_20은 max50/optimizer auto/증강·로그44epoch 및weight가 있으나summary가 없어최종split별 성능을 만들지 않았다. 512장640/b4/noaug는 cancelled.json·로그7epoch로완료표에서 제외하였다.\n\n'+table(['PoC run','Train/Val/Test장','ValP','ValR','ValAP50','ValAP50–95','TestP','TestR','TestAP50','TestAP50–95'],poc)+'\n'
py=[]
for x in inv:
 if x['ext']=='.py' and x['path'].split('/')[0] not in ['.venv','.cache','.git'] and x['path']!='report_audit.py':
  tree=ast.parse((R/x['path']).read_text(encoding='utf-8-sig',errors='replace'));doc=(ast.get_docstring(tree) or 'docstring없음; 소스확인')[:180];fns=[f'{n.name}:{n.lineno}' for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))];py.append([x['path'],doc,', '.join(fns[:8])])
a+='\n## 부록 G. 프로젝트 Python50개 조사 위치\n\n전체소스는 all_project_code.txt의 FILE 구분자로 연결된다. 테스트파일 존재를 테스트실행 성공으로 해석하지 않았다.\n\n'+table(['경로','모듈설명','대표함수:행'],py)+'\n'
with (R/'REPORT_EVIDENCE.md').open('a',encoding='utf-8') as f:f.write(a)
print('evidence appendices',len(changes),sum(x[4] for x in changes),len(py))
