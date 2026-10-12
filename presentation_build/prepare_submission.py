from pathlib import Path
import json,re,hashlib
r=Path.cwd();b=r/'presentation_build'; s=(b/'build_deck.mjs').read_text(encoding='utf8')
s=s.replace("const specs=JSON.parse(await fs.readFile(path.join(build,'slides.json'),'utf8'));", """const oldSpecs=JSON.parse(await fs.readFile(path.join(build,'slides.json'),'utf8'));
const problem={n:0,title:'문제 정의',message:'컬러 주석과 복원 흔적의 영향을 확인하며 이물질 위치 검출을 평가한다.',notes:'이 과제는 X-ray 영상에서 이물질의 위치를 검출하는 것입니다. 원본에 그려진 컬러 bbox는 모델이 정답 표시를 학습하는 단서가 될 수 있어 먼저 처리했습니다. 이후 미검출과 위치 오차를 구분하고 복원 흔적에 대한 모델 반응을 확인했습니다. 정상 제품의 판별 성능과 실제 현장 효과는 별도 검증이 필요합니다.',sources:[root+'/FINAL_REPORT_DRAFT.md',root+'/FINAL_SUBMISSION_AUDIT.md'],figures:[]};
const specs=[oldSpecs[0],problem,...oldSpecs.slice(1,9)].map((s,i)=>({...s,number:i+1}));
specs[2].title='데이터 구성';
specs[3].title='TELEA 전처리';
specs[4].title='모델 비교';
specs[5].title='Threshold 선정';
specs[7].title='오류분석';
specs[9].title='현장 활용 및 결론';
specs[9].message='고정 성능과 실패 사례를 근거로 검사자의 재검을 지원하도록 제안한다.';
specs[9].notes='이번 분석은 컬러 주석 처리부터 고정 Test 성능과 오류 사례까지 연결했습니다. 현장에서는 AI가 위치와 점수를 제시하고 검사자가 원본과 확대 영상을 확인하도록 제안합니다. 무검출 영상도 정상으로 보증하지 않고 표본 재검이 필요합니다. 실제 설비 연동과 시간 절감은 미검증이며, 정상 제품과 무주석 영상에서 후속 검증이 필요합니다.';""")
s=s.replace("teal='#087E8B'", "teal='#233E55'")
s=s.replace(" const shape=s.shapes.add({geometry:'textbox'", " if(size<22)size=22;else if(size>29&&size<42)size=29;\n const shape=s.shapes.add({geometry:'textbox'")
s=s.replace("function metric(s,label,value,x,y,w=350){text", "function metric(s,label,value,x,y,w=350){s.shapes.add({geometry:'rect',name:label+' 강조 박스',position:{left:x-7,top:y-7,width:w,height:137},fill:'#EDF1F4',line:{fill:'#C4CED6',width:1}});text")
s=s.replace("String(spec.n).padStart", "String(spec.number).padStart")
s=s.replace("switch(spec.n){", "switch(spec.n){\n case 0:\n  text(s,'이물질 위치 검출',76,240,1100,74,43,navy,true);\n  bullets(s,['컬러 정답 표시와 복원 흔적의 영향 확인','미검출과 위치 오차를 구분해 검사 위험 분석','정상 제품 판별·현장 효과는 별도 검증'],77,361,1110,194,29);\n  text(s,'표시선 처리, 모델 비교, 고정 평가와 오류분석으로 연결',77,612,1110,46,27,gray);\n  break;")
s=s.replace("185,57,white,true", "185,48,white,true")
s=s.replace("['원본·bbox·점수·검사자 판정 기록','정상 제품·무주석 영상 검증 필요']", "['결론: AI 후보 제시와 검사자 최종 판정','후속 검증: 정상 제품·무주석 영상']")
s=s.replace(" const candidate=", " const candidate=")
s=s.replace("path.join(build,'candidate.pptx')", "path.join(build,'submission_candidate.pptx')")
s=s.replace("[path.join(build,'fix_native_crops.py'),candidate]", "[path.join(build,'fix_native_crops.py'),candidate,'submission']")
s=s.replace("path.join(build,'previews')", "path.join(build,'submission_previews')")
s=s.replace("path.join(build,'previews',", "path.join(build,'submission_previews',")
s=s.replace("'KAMP_Xray_Presentation_Final.pptx'", "'KAMP_Xray_Presentation.pptx'")
s=s.replace("[2,4,5,9]", "[3,5,6,10]")
s=s.replace("'validation_final.json'", "'submission_validation.json'")
s += "\nawait fs.copyFile(finalPath,path.join(root,'KAMP_Xray_Presentation.pptx'));\nawait fs.writeFile(path.join(build,'submission_specs.json'),JSON.stringify(specs,null,2));\n"
(b/'build_submission.mjs').write_text(s,encoding='utf8')
inputs=['PRESENTATION_PLAN.md','SLIDE_CONTENT.md','SELECTED_FIGURES.md','FINAL_REPORT_DRAFT.md','FINAL_SUBMISSION_AUDIT.md']
images=set(re.findall(r'C:/[^\r\n<>|]+?\.(?:png|jpg|jpeg)(?=[)\]])',(r/'SELECTED_FIGURES.md').read_text(encoding='utf8')))
assert all(Path(p).is_file() for p in images)
check={'input_files':[{ 'path':str(r/p),'sha256':hashlib.sha256((r/p).read_bytes()).hexdigest()} for p in inputs], 'selected_figure_image_paths':[{'path':p,'exists':Path(p).is_file()} for p in sorted(images)],'missing_images':[]}
(b/'submission_inputs.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf8')
print('All selected figure image paths exist:',len(images))
