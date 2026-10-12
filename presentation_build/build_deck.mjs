import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {Presentation,PresentationFile} from '@oai/artifact-tool';

const root='C:/workspace/Kamp_Xray';
const build=path.join(root,'presentation_build');
const skill='C:/Users/kang/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations';
const python='C:/Users/kang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const {resolvePresentationFont,makeNativeBulletParagraphs,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const font=resolvePresentationFont({fontFamily:'Malgun Gothic'});
const specs=JSON.parse(await fs.readFile(path.join(build,'slides.json'),'utf8'));
const deck=Presentation.create({slideSize:{width:1280,height:720}});
const navy='#142E43',teal='#087E8B',gray='#536574',white='#FFFFFF';

function text(s,t,x,y,w,h,size=26,color=navy,bold=false){
 const shape=s.shapes.add({geometry:'textbox',name:t.slice(0,35),position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 shape.text=t; shape.text.style={typeface:font,fontSize:size,color,bold,autoFit:'none'};
 return shape;
}
function bullets(s,items,x,y,w,h,size=25){
 const shape=text(s,'',x,y,w,h,size);
 shape.text=makeNativeBulletParagraphs(items,{marginLeftPoints:16,hangingPoints:8,spaceAfterPoints:12});
 shape.text.style={typeface:font,fontSize:size,color:navy,autoFit:'none'};
 return shape;
}
async function image(s,p,x,y,w,h,crop){
 const im=s.images.add({blob:new Uint8Array(await fs.readFile(p)),contentType:'image/png',alt:path.basename(p),fit:crop?'cover':'contain',position:{left:x,top:y,width:w,height:h}});
 if(crop) im.crop=crop;
 return im;
}
function table(s,values,x,y,w,h,colWidths,size=25){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:colWidths});
 t.borders.assign({style:'solid',fill:'#D9E2E7',width:1});
 t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({textStyle:{typeface:font,fontSize:size,color:navy},fill:white,margins:{left:12,right:8,top:9,bottom:9}});
 t.cells.block({row:0,column:0,rowCount:1,columnCount:values[0].length}).assign({fill:navy,textStyle:{typeface:font,fontSize:size,color:white,bold:true}});
 return t;
}
function metric(s,label,value,x,y,w=350){text(s,label,x,y,w,36,24,gray);text(s,value,x,y+42,w,90,64,teal,true);}

for (const spec of specs){
 const s=deck.slides.add();s.background.fill=white;
 if(spec.n!==1){
  text(s,spec.title,58,38,1160,65,43,navy,true);
  text(s,spec.message,60,119,1160,70,26,teal);
  text(s,String(spec.n).padStart(2,'0'),1190,667,50,30,19,gray);
 }
 switch(spec.n){
 case 1:
  s.background.fill=navy;
  text(s,'제6회 K-인공지능 제조데이터 분석 경진대회',64,60,1110,50,25,'#AEC7D8');
  text(s,'KAMP X-ray\n이물질 검출과 오류분석',64,217,720,185,57,white,true);
  text(s,'컬러 주석 처리와 고정 운영 평가',67,470,700,50,29,'#B7DADE');
  await image(s,spec.figures[0],822,197,382,342);
  break;
 case 2:
  metric(s,'분석 이미지','512장',70,211,400);
  metric(s,'정답 bbox','1,163개',622,211,440);
  table(s,spec.table,74,375,1112,229,[480,300,332],27);
  text(s,'원본 해시·라벨 출처 연결, 그룹 단위 분할',75,627,1100,39,24,gray);
  break;
 case 3:
  for(let i=0;i<3;i++){
   text(s,['원본 컬러 표시','표시선 mask','TELEA 복원'][i],64+i*396,211,370,42,26,navy,true);
   await image(s,spec.figures[0],64+i*396,264,366,287,{left:i*.2,right:1-(i+1)*.2,top:.04,bottom:0});
  }
  text(s,'표시선 mask만 복원',64,566,430,44,27,teal,true);
  text(s,'radius 3 / dilation 0',657,566,520,44,28,navy,true);
  text(s,spec.caption,64,635,1120,40,21,gray);
  break;
 case 4:
  text(s,'동일 조건: 640 / batch 4 / AdamW / 기본 증강',70,214,1120,43,27,navy,true);
  table(s,spec.table,70,282,1115,201,[455,330,330],28);
  text(s,'최종 분석 기준 R04',72,513,540,40,27,teal,true);
  text(s,'YOLO11n 1280 · 무증강',72,557,560,50,32,navy,true);
  bullets(s,['기존 기준 모델 유지','Validation F1 최고와 구분'],711,520,475,110,25);
  text(s,'완료 11run · 구조 2종. 비교 F1은 validator P/R의 조화평균',72,644,1110,30,20,gray);
  break;
 case 5:
  await image(s,spec.figures[0],45,230,750,409);
  text(s,'운영 하한 0.20\n고신뢰 기준 0.25',842,213,350,91,32,teal,true);
  table(s,spec.table,833,326,371,202,[201,85,85],23);
  text(s,'Test 재선정 없음\n0.20은 recall 중심 하한',842,562,355,77,25,navy);
  text(s,spec.caption,60,660,1080,31,20,gray);
  break;
 case 6:
  metric(s,'Precision','94.87%',68,206,430);
  metric(s,'Recall','92.50%',68,356,430);
  metric(s,'F1-score','93.67%',68,506,430);
  await image(s,spec.figures[0],550,197,655,458);
  text(s,'Test 76장 · GT 160개   /   confidence ≥ 0.20 · GT IoU ≥ 0.50',65,667,1120,34,20,gray);
  break;
 case 7:
  metric(s,'소형 객체 FN (7/41)','17.07%',65,216,400);
  metric(s,'저대비 객체 FN (7/60)','11.67%',65,377,400);
  text(s,'FN 유형 8 / 3 / 1',65,554,430,40,25,navy,true);
  text(s,'위치 실패 / 저점수·위치 성공\n/ 저점수·위치 실패',65,596,450,56,21,gray);
  text(s,'U0138: 낮은 점수 FN',544,212,300,38,25,navy,true);
  text(s,'U0371: 과대 bbox FP',886,212,330,38,25,navy,true);
  await image(s,spec.figures[0],536,260,302,298,{left:360/1400,right:700/1400,top:95/720,bottom:290/720});
  await image(s,spec.figures[1],878,260,302,303,{left:360/1400,right:700/1400,top:95/720,bottom:284/720});
  text(s,'GT 빨강 · 저점수 후보 주황',543,567,307,60,22,gray);
  text(s,'GT 청록 · 오류 예측 빨강',884,567,319,60,22,gray);
  text(s,'기준은 Train에서 고정. 조건군 중복이 있어 인과 해석 불가',544,635,648,52,20,gray);
  break;
 case 8:
  for(let i=0;i<4;i++){
   text(s,spec.flow[i],61+i*303,215,258,76,25,navy,true);
   if(i<3) text(s,'→',319+i*303,232,42,50,35,teal,true);
  }
  await image(s,spec.figures[0],65,300,356,296,{left:0,right:.5,top:0,bottom:.5});
  await image(s,spec.figures[0],486,300,356,296,{left:.5,right:0,top:0,bottom:.5});
  text(s,'fake mask',1036,330,175,35,23,gray);
  await image(s,spec.figures[1],1045,382,143,159,{left:380/576,right:(576-434)/576,top:135/444,bottom:(444-195)/444});
  text(s,'20장 PoC · 34pair (holdout 10pair)',65,601,735,41,28,teal,true);
  text(s,'두 PoC 모델: 새 fake FP 0 · GT loss/gain 0',65,646,753,38,24,navy);
  text(s,'최종 512장 모델의\nshortcut 부재 증명 아님',887,612,317,70,23,gray);
  break;
 case 9:
  table(s,spec.table,72,222,1110,286,[560,550],27);
  bullets(s,['원본·bbox·점수·검사자 판정 기록','정상 제품·무주석 영상 검증 필요'],74,545,1090,96,26);
  text(s,'운영 제안: 생산 설비 연동·자동 배출·시간 절감 효과 미검증',74,651,1110,35,21,gray);
  break;
 case 10:
  text(s,'고정 성능과 실패 사례를\n검사 지원의 근거로 연결',72,242,1110,144,49,teal,true);
  bullets(s,['공식 Test와 오류분석의 평가 조건 통일','효과 없는 보조검사는 운영 채택에서 제외','정상·무주석 영상, 반복 seed, 계획서 대응 확인'],74,437,1090,175,29);
  break;
 }
 s.speakerNotes.textFrame.setText(spec.notes+'\n\n핵심 메시지: '+spec.message+'\n\n원본 근거:\n'+spec.sources.concat(spec.figures).join('\n')+(spec.n===8?'\nfake-mark 구현: 실제 표시 mask를 GT 밖으로 재배치해 TELEA inpaint한 흔적. 새 컬러선을 그렸다는 뜻이 아님.':'') );
}
await fs.mkdir(path.join(build,'previews'),{recursive:true});
const candidate=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidate);
execFileSync(python,[path.join(build,'fix_native_crops.py'),candidate]);
console.log('Draft exported',candidate);
const {FileBlob}=await import('@oai/artifact-tool');
const renderedDeck=await PresentationFile.importPptx(await FileBlob.load(candidate));
for(let i=0;i<renderedDeck.slides.items.length;i++){
 const s=renderedDeck.slides.items[i];
 const b=await renderedDeck.export({slide:s,format:'png',scale:1});
 await fs.writeFile(path.join(build,'previews',`slide-${i+1}.png`),new Uint8Array(await b.arrayBuffer()));
 const layout=await s.export({format:'layout'});
 await fs.writeFile(path.join(build,'previews',`slide-${i+1}.layout.json`),await layout.text());
 console.log('Rendered',i+1);
}
const finalPath=path.join(root,'presentation','KAMP_Xray_Presentation_Final.pptx');
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath,pythonExecutable:python,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[2,4,5,9].flatMap(n=>['--require-native-table-slide',String(n)])],explicitTotalSlideCount:10,requiredNativeTableOwnerSlides:[2,4,5,9],fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:path.join(build,'validation_final.json')});
console.log(JSON.stringify(result));
