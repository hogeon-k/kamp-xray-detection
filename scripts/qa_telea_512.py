"""고정된 512장 데이터셋의 읽기 전용 학습 전 QA 자료 생성."""
from pathlib import Path
import argparse
import json
import sys
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from split_clean_512 import ROOT, read_csv, write_csv, write_json, snapshot, verify_snapshot, labels
from create_telea_512_dataset import overlap

DATA = ROOT/'outputs/yolo_512_telea'
SOURCE = ROOT/'outputs/clean_512'
OUT = ROOT/'outputs/yolo_512_telea_qa'
FONT = ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',13)

def text(im,xy,message):
    ImageDraw.Draw(im).text(xy,message,font=FONT,fill='black')

def load(r):
    with Image.open(SOURCE/'images'/r['filename']) as im:
        original=np.asarray(im.convert('RGB')).copy()
    with Image.open(DATA/'images'/r['split']/r['restored_filename']) as im:
        restored=np.asarray(im.convert('RGB')).copy()
    with Image.open(DATA/'restoration_masks'/r['split']/(Path(r['filename']).stem+'.png')) as im:
        mask=np.asarray(im)>0
    boxes=labels(DATA/'labels'/r['split']/(Path(r['filename']).stem+'.txt'))
    _,raster,_=overlap(boxes,mask)
    return original,restored,mask,raster

def fitted(im,size):
    im=im.copy()
    scale=min(size[0]/im.width,size[1]/im.height)
    method=Image.Resampling.NEAREST if scale>1 else Image.Resampling.LANCZOS
    im=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),method)
    canvas=Image.new('RGB',size,(238,238,238))
    canvas.paste(im,((size[0]-im.width)//2,(size[1]-im.height)//2))
    return canvas

def draw_panels(r,crop=None,width=200,title=''):
    original,restored,mask,boxes=load(r)
    orig_gt=Image.fromarray(original.copy()); restored_gt=Image.fromarray(restored.copy())
    for im in (orig_gt,restored_gt):
        draw=ImageDraw.Draw(im)
        for x0,y0,x1,y1 in boxes:
            draw.rectangle((x0,y0,max(x0,x1-1),max(y0,y1-1)),outline='cyan',width=1)
    panels=[Image.fromarray(original),Image.fromarray(restored),orig_gt,restored_gt,Image.fromarray(mask.astype(np.uint8)*255).convert('RGB')]
    if crop:
        panels=[im.crop(crop) for im in panels]
    height=150 if crop else 180
    canvas=Image.new('RGB',(width*5,height+45),'white')
    text(canvas,(4,0),title or f"{r['unique_id']} {r['split']}")
    for i,(im,name) in enumerate(zip(panels,['원본','TELEA 복원본','원본 + GT bbox','복원본 + GT bbox','복원 마스크'])):
        text(canvas,(i*width+4,23),name)
        canvas.paste(fitted(im,(width,height)),(i*width,45))
    return canvas

def sheets(items,directory,prefix,per_page=10):
    directory.mkdir(parents=True,exist_ok=True)
    paths=[]
    for start in range(0,len(items),per_page):
        batch=items[start:start+per_page]
        sheet=Image.new('RGB',(max(im.width for im in batch),sum(im.height for im in batch)),'white')
        y=0
        for im in batch:
            sheet.paste(im,(0,y)); y+=im.height
        path=directory/f'{prefix}_{start//per_page+1:02d}.png'
        sheet.save(path); paths.append(str(path))
    return paths

def prepare():
    if OUT.exists():
        raise FileExistsError(OUT)
    baseline=snapshot([ROOT/'dataset',ROOT/'outputs'])
    OUT.mkdir()
    write_json(OUT/'보호_해시.json',baseline)
    records=read_csv(DATA/'manifest.csv'); lookup={r['unique_id']:r for r in records}
    pairs=read_csv(DATA/'near_cross_split_review.csv')
    cache={uid:load(r) for uid,r in lookup.items()}
    for r in pairs:
        a,b=cache[r['left_id']],cache[r['right_id']]
        same=a[0].shape==b[0].shape
        r['동일_해상도']=same
        if same:
            # 기존 저장 마스크만 제외. 복원 픽셀을 유사도 근거로 사용하지 않는다.
            valid=~(a[2]|b[2])
            ag=a[0].mean(axis=2)[valid]; bg=b[0].mean(axis=2)[valid]
            r['마스크_제외_MAE']=float(np.abs(ag-bg).mean())
            r['마스크_제외_상관계수']=float(np.corrcoef(ag,bg)[0,1])
        else:
            r['마스크_제외_MAE']=''; r['마스크_제외_상관계수']=''
    pairs.sort(key=lambda r:(int(r['phash_distance']),float(r['마스크_제외_MAE']) if r['마스크_제외_MAE']!='' else float('inf'),r['left_id'],r['right_id']))
    for i,r in enumerate(pairs,1):
        r['유사도_순위']=i
    write_csv(OUT/'cross_split_전체_유사도순.csv',pairs)
    # 저거리 전수와 비마킹 영역 MAE 최상위 60쌍의 합집합을 상세 육안 심사.
    candidates=[r for r in pairs if int(r['phash_distance'])<=1]
    mae_top=sorted([r for r in pairs if r['동일_해상도']],key=lambda r:float(r['마스크_제외_MAE']))[:60]
    keys={(r['left_id'],r['right_id']) for r in candidates+mae_top}
    candidates=[r for r in pairs if (r['left_id'],r['right_id']) in keys]
    write_csv(OUT/'cross_split_상세검토_후보.csv',candidates)
    pair_panels=[]
    for r in candidates:
        canvas=Image.new('RGB',(1000,450),'white')
        for i,key in enumerate(('left_id','right_id')):
            sample=lookup[r[key]]
            im=draw_panels(sample,title=f"순위 {r['유사도_순위']} / pHash {r['phash_distance']} / MAE {float(r['마스크_제외_MAE']):.3f} / {sample['unique_id']} {sample['split']}")
            canvas.paste(im,(0,i*225))
        pair_panels.append(canvas)
    cross_paths=sheets(pair_panels,OUT/'검토용_cross_split','후보',4)
    # 전체 910쌍 축약 비교판도 제공. 상세 후보 밖의 쌍을 검토할 수 있다.
    overview=[]
    for start in range(0,len(pairs),48):
        batch=pairs[start:start+48]
        canvas=Image.new('RGB',(1200,12*125),'white')
        for j,r in enumerate(batch):
            x,y=(j%4)*300,(j//4)*125
            text(canvas,(x+2,y),f"{r['유사도_순위']} d={r['phash_distance']} {r['left_id']}/{r['right_id']}")
            for k,key in enumerate(('left_id','right_id')):
                canvas.paste(fitted(Image.fromarray(cache[r[key]][0]),(150,100)),(x+k*150,y+23))
        overview.append(canvas)
    sheets(overview,OUT/'전체_cross_split','전체',1)
    bbox=[r for r in read_csv(DATA/'gt_mask_bbox_overlap.csv') if float(r['overlap_pixels'])>0]
    bbox.sort(key=lambda r:(-float(r['overlap_ratio_of_bbox']),-int(r['overlap_pixels']),r['unique_id'],int(r['bbox_index'])))
    write_csv(OUT/'GT_mask_겹침_bbox_비율순.csv',bbox)
    grouped=defaultdict(list)
    for b in bbox:
        grouped[b['unique_id']].append(b)
    image_rows=[]
    for uid,bs in grouped.items():
        r=lookup[uid]
        image_rows.append({'unique_id':uid,'split':r['split'],'최대_bbox_겹침비율':max(float(b['overlap_ratio_of_bbox']) for b in bs),
                           'GT_union_겹침비율':float(r['gt_mask_overlap_ratio']),'겹친_bbox_수':len(bs),
                           '겹친_픽셀_GT_union':int(r['gt_mask_overlap_pixels'])})
    image_rows.sort(key=lambda r:(-r['최대_bbox_겹침비율'],-r['GT_union_겹침비율'],r['unique_id']))
    write_csv(OUT/'GT_mask_겹침_이미지_비율순.csv',image_rows)
    crops=[]
    for i,r in enumerate(image_rows,1):
        b=grouped[r['unique_id']][0]
        sample=lookup[r['unique_id']]; o,t,m,boxes=cache[r['unique_id']]
        x0,y0,x1,y1=boxes[int(b['bbox_index'])]
        margin=12
        crop=(max(0,x0-margin),max(0,y0-margin),min(o.shape[1],x1+margin),min(o.shape[0],y1+margin))
        crops.append(draw_panels(sample,crop,title=f"{i} {r['unique_id']} {sample['split']} / bbox {b['bbox_index']} / 겹침 {100*r['최대_bbox_겹침비율']:.1f}%"))
    overlap_paths=sheets(crops,OUT/'검토용_GT_mask','겹침_확대',10)
    all_crops=[]
    for i,b in enumerate(bbox,1):
        r=lookup[b['unique_id']]; o,t,m,boxes=cache[r['unique_id']]
        x0,y0,x1,y1=boxes[int(b['bbox_index'])]
        crop=(max(0,x0-12),max(0,y0-12),min(o.shape[1],x1+12),min(o.shape[0],y1+12))
        all_crops.append(draw_panels(r,crop,title=f"{i} {r['unique_id']} bbox {b['bbox_index']} / 겹침 {float(b['overlap_ratio_of_bbox'])*100:.1f}% / 최근접 확대"))
    sheets(all_crops,OUT/'검토용_GT_mask_141_bbox','전체_bbox_확대',10)
    write_json(OUT/'검토_진행.json',{'전체_cross_split_쌍':len(pairs),'상세_cross_split_후보':len(candidates),'겹침_이미지':len(image_rows),'겹침_bbox':len(bbox),
                                 'cross_split_비교판':cross_paths,'GT_mask_비교판':overlap_paths,
                                 '설명':'pHash 오름차순, 동률은 기존 마스크를 제외한 원본 RGB 평균 밝기 MAE 오름차순. 겹침 이미지는 최대 개별 bbox 겹침 비율 내림차순. 수치 선별은 육안 검토 우선순위이며 중복/복원 실패 판정이 아님.'})
    print(json.dumps({'전체_쌍':len(pairs),'상세_후보':len(candidates),'겹침_이미지':len(image_rows),'겹침_bbox':len(bbox)},ensure_ascii=False),flush=True)
    verify_snapshot(baseline)

def finalize():
    selection=json.loads((OUT/'육안_선별.json').read_text(encoding='utf-8'))
    lookup={r['unique_id']:r for r in read_csv(DATA/'manifest.csv')}
    ranked=read_csv(OUT/'cross_split_전체_유사도순.csv')
    selected_pairs=[]; panels=[]
    for item in selection['cross_split']:
        r=next(r for r in ranked if {r['left_id'],r['right_id']}==set(item['ids']))
        selected_pairs.append(dict(r,육안_검토_이유=item['이유']))
        canvas=Image.new('RGB',(1000,474),'white')
        text(canvas,(4,0),item['이유'])
        for i,uid in enumerate(item['ids']):
            im=draw_panels(lookup[uid],title=f"{uid} {lookup[uid]['split']} / pHash {r['phash_distance']} / 유사도 순위 {r['유사도_순위']}")
            canvas.paste(im,(0,24+i*225))
        panels.append(canvas)
    if selected_pairs:
        write_csv(OUT/'위험_cross_split_육안선별.csv',selected_pairs)
    cross=sheets(panels,OUT/'위험_cross_split','위험_유사쌍',3)
    selected_keys={frozenset(item['ids']) for item in selection['cross_split']}
    review_rows=[]
    for r in read_csv(OUT/'cross_split_상세검토_후보.csv'):
        review_rows.append(dict(r,육안_검토_완료=True,우선_위험_선별=frozenset((r['left_id'],r['right_id'])) in selected_keys,
                               설명='우선 위험 목록 미선별은 안전 확정 또는 중복 부정을 뜻하지 않음'))
    write_csv(OUT/'cross_split_107쌍_육안검토_기록.csv',review_rows)
    bbox=read_csv(OUT/'GT_mask_겹침_bbox_비율순.csv')
    panels=[]; selected_gt=[]
    for item in selection['gt_mask']:
        uid=item['id']; r=lookup[uid]
        selected_gt.append(dict(unique_id=uid,split=r['split'],육안_검토_이유=item['이유']))
        original,restored,mask,boxes=load(r)
        target=[b for b in bbox if b['unique_id']==uid]
        images=[draw_panels(r,title=f"{uid} / {item['이유']}")]
        for b in target:
            x0,y0,x1,y1=boxes[int(b['bbox_index'])]
            crop=(max(0,x0-12),max(0,y0-12),min(original.shape[1],x1+12),min(original.shape[0],y1+12))
            images.append(draw_panels(r,crop,title=f"{uid} bbox {b['bbox_index']} / 겹침 {float(b['overlap_ratio_of_bbox'])*100:.1f}%"))
        canvas=Image.new('RGB',(1000,sum(im.height for im in images)),'white'); y=0
        for im in images:
            canvas.paste(im,(0,y)); y+=im.height
        canvas.save(OUT/f'GT_mask_위험_{uid}.png'); panels.append(canvas)
    if selected_gt:
        write_csv(OUT/'GT_mask_육안검토_필요.csv',selected_gt)
    gt=sheets(panels,OUT/'위험_GT_mask','이물질_변형_검토',2)
    baseline=json.loads((OUT/'보호_해시.json').read_text(encoding='utf-8'))
    checked=verify_snapshot(baseline)
    summary={'위험_cross_split_유사쌍_수':len(selected_pairs),'GT_mask_육안검토_필요_이미지_수':len(selected_gt),
             '육안상_명백한_이물질_중심_소실_또는_심한변형_확인':0,
             '원본_분할_라벨_변경':0,'보호_파일_해시_검증_수':checked,'자동_제외':0,
             '육안_판정_범위':selection['검토범위'],'위험_cross_split_비교판':cross,'위험_GT_mask_비교판':gt,
             '해석':'위험 후보는 검토 요청이며 동일 샘플 또는 이물질 소실의 확정 판정이 아니다. 원본은 이미 컬러 마킹으로 가려져 있어 마킹 전 신호의 실제 복구 여부를 확정할 수 없다.'}
    write_json(OUT/'QA_요약.json',summary)
    (OUT/'README.md').write_text('# 512장 TELEA 학습 전 QA\n\n'+selection['검토범위']+'\n\n전체 pHash 후보는 거리 오름차순으로 정렬했고, 동률은 기존 마스크를 제외한 원본 밝기 MAE로 정렬했습니다. 이미지의 이동 정렬이나 새 복원은 하지 않았습니다. GT 겹침은 개별 bbox 면적 대비 마스크 픽셀 비율을 기준으로 정렬했습니다. 이미지 순위는 이미지 내 최대 bbox 비율이며 GT union 비율도 함께 기록했습니다.\n\n비교판은 원본 / 복원본 / 원본+GT / 복원본+GT / 기존 마스크 순서입니다. 청록색은 검토용 bbox입니다. 확대판은 원래 bbox에 주변 12픽셀을 포함한 crop이며 보기용 크기 조절만 수행했습니다. 데이터 파일은 수정하지 않았습니다.\n\n위험 유사쌍은 유사한 구조·자세·배치가 관측되어 전문가 확인이 필요한 후보입니다. 같은 물체인지 확정하지 않았습니다. 복원 위험 사례는 bbox 안/주변 신호가 약해지거나 연결·경계가 매끈해진 것으로 보이는 사례를 추렸습니다. 마킹 자체 제거와 실제 이물질 신호 소실을 원본만으로 확정할 수 없습니다. 자동 제외는 없습니다.\n\nQA_요약.json과 육안 선별 CSV에 최종 후보 및 이유가 있습니다. 전체 후보 비교판과 겹침 91장 비교판도 보존해 추가 검토할 수 있습니다.\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    finalize() if args.finalize else prepare()
