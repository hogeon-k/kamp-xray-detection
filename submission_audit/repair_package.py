from pathlib import Path
from datetime import datetime
import csv,json,shutil,hashlib,zipfile,ast

ROOT=Path(__file__).resolve().parent.parent
PACKAGE=ROOT/'KAMP_Xray_Submission'
AUDIT=ROOT/'submission_audit'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def copy(src,dst):
    assert dst.resolve().is_relative_to(PACKAGE.resolve())
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)

def main():
    if not (AUDIT/'before_package_repair.zip').exists():shutil.copy2(ROOT/'KAMP_Xray_Submission.zip',AUDIT/'before_package_repair.zip')
    for folder in ['outputs/yolo_512_telea/images','outputs/yolo_512_telea/restoration_masks','outputs/clean_512/images','outputs/clean_512/labels']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file():copy(p,PACKAGE/p.relative_to(ROOT))
    for p in (ROOT/'scripts').rglob('*'):
        if p.is_file() and p.suffix in ['.py','.md','.json'] and '__pycache__' not in p.parts:
            # A historical packager would restore the obsolete analysis-only README.
            if p.name=='finalize_submission_package.py':continue
            copy(p,PACKAGE/p.relative_to(ROOT))
    for p in (ROOT/'outputs').rglob('*.py'):copy(p,PACKAGE/p.relative_to(ROOT))
    copy(ROOT/'yolo11n.pt',PACKAGE/'weights/yolo11n.pt')
    models=read(PACKAGE/'outputs/model_comparison/comparison_manifest.json')
    for name in models['runs']:
        copy(ROOT/'outputs/yolo_runs'/name/'weights/best.pt',PACKAGE/'outputs/yolo_runs'/name/'weights/best.pt')
    (PACKAGE/'outputs/yolo_512_telea/dataset_submission.yaml').write_text('train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: defect\n',encoding='utf-8')
    p=PACKAGE/'train_yolo.py';s=p.read_text(encoding='utf-8')
    s=s.replace('This analysis submission does not include training images. To retrain, provide\nthe full prepared TELEA dataset and set CONFIG["data"] to its dataset.yaml.',
                'This submission includes all prepared images and labels. CONFIG uses the\nportable dataset_submission.yaml and the included pretrained YOLO11n weight.')
    s=s.replace('DATA_YAML = str(ROOT / "outputs" / "yolo_512_telea" / "dataset.yaml")','DATA_YAML = str(ROOT / "outputs" / "yolo_512_telea" / "dataset_submission.yaml")')
    s=s.replace('"model": "yolo11n.pt",','"model": str(ROOT / "weights" / "yolo11n.pt"),',1)
    compile(s,str(p),'exec');p.write_text(s,encoding='utf-8')
    configs=read(PACKAGE/'experiment_configs.json');configs['retraining_requires_full_images']=False
    configs['training_images_included']=True
    for r in configs['runs']:r['CONFIG']['data']='outputs/yolo_512_telea/dataset_submission.yaml'
    write(PACKAGE/'experiment_configs.json',configs)

    with (PACKAGE/'outputs/yolo_512_telea/manifest.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    raw=read(PACKAGE/'outputs/foreign_object_condition_analysis/predictions.json')
    copy(ROOT/'outputs/foreign_object_condition_analysis/prediction_provenance.json',PACKAGE/'outputs/foreign_object_condition_analysis/prediction_provenance.json')
    images={};detections=[];index=[]
    for r in rows:
        if r['split']!='test':continue
        accepted=[(i,p) for i,p in enumerate(raw[r['unique_id']]) if p['class_id']==0 and p['confidence']>=.20]
        images[r['unique_id']]=dict(image_name=r['restored_filename'],predictions=[dict(p,cache_index=i) for i,p in accepted])
        index.append(dict(unique_id=r['unique_id'],image_name=r['restored_filename'],prediction_count=len(accepted)))
        for i,p in accepted:detections.append(dict(unique_id=r['unique_id'],image_name=r['restored_filename'],prediction_index=i,class_id=0,confidence=p['confidence'],**dict(zip(['x1','y1','x2','y2'],p['box']))))
    provenance=read(PACKAGE/'outputs/foreign_object_condition_analysis/prediction_provenance.json')
    metadata=dict(split='test',image_count=len(images),detections=len(detections),confidence=.20,nms_iou=.7,imgsz=1280,
        coordinates='original image pixels xyxy',class_names={'0':'defect'},model='outputs/yolo_runs/'+models['fixed_downstream_model']+'/weights/best.pt',
        model_sha256=provenance['model_sha256'],source_cache='outputs/foreign_object_condition_analysis/predictions.json',source_cache_sha256=sha(PACKAGE/'outputs/foreign_object_condition_analysis/predictions.json'),raw_confidence_floor=.001)
    write(PACKAGE/'test_predictions/predictions.json',dict(metadata=metadata,images=images))
    write(PACKAGE/'test_predictions/raw_predictions.json',dict(metadata=dict(metadata,confidence=.001),images={r['unique_id']:dict(image_name=r['restored_filename'],predictions=raw[r['unique_id']]) for r in rows if r['split']=='test'}))
    for name,data,fields in [('predictions.csv',detections,['unique_id','image_name','prediction_index','class_id','confidence','x1','y1','x2','y2']),('images.csv',index,['unique_id','image_name','prediction_count'])]:
        path=PACKAGE/'test_predictions'/name
        with path.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
    copy(AUDIT/'official_test_confusion.png',PACKAGE/'test_predictions/confusion_matrix.png')
    copy(AUDIT/'official_test_confusion.json',PACKAGE/'test_predictions/evaluation.json')
    (PACKAGE/'test_predictions/README.md').write_text('''# Test 전용 예측결과

predictions.json은 Test 76장 전체를 포함한다. 검출이 없는 영상도 빈 predictions 배열로 남긴다. predictions.csv는 class0·confidence≥0.20의156개 검출 bbox이며 좌표는 원영상 픽셀 xyxy이다. images.csv는76장의 파일명·검출 개수 목록이다. raw_predictions.json은 Test만 분리한 confidence floor0.001 캐시이다. 원본 Val/Test 통합 캐시는 outputs/foreign_object_condition_analysis/predictions.json에 보존한다.

기존 고정 weight의 저장 캐시를 분리했으며 신규 추론이나 모델 변경을 하지 않았다. NMS IoU0.7, GT 평가 IoU0.5를 구분한다. 저장 운영 평가 TP148/FP8/FN12, P94.8718%/R92.5%/F1 93.6709%이다. GT·평가결과는 예측 bbox 자체와 구분된다. background/background TN은 정의하지 않는다.

신규 추론: python scripts/predict_test.py --device 0 --out test_predictions_new. 원본 결과를 덮어쓰지 않으며 환경에 따라 미세한 결과 차이가 발생할 수 있다.
''',encoding='utf-8')
    for p in PACKAGE.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
    print('Added data, code and separated Test predictions:',len(images),len(detections))

if __name__=='__main__':main()
