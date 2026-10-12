from pathlib import Path
import sys,json,csv,shutil,hashlib

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'submission_audit'
sys.path.insert(0,str(ROOT/'.cache/shortcut_runtime'))
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import ops

torch.set_num_threads(4)
model=ROOT/'outputs/yolo_runs/telea_512_yolo11n_e200_img1280_b4_adamw_lr0.001_wd0.0005_noaug_s42/weights/best.pt'
with (ROOT/'outputs/yolo_512_telea/manifest.csv').open(encoding='utf-8-sig') as f:
    rows=[r for r in csv.DictReader(f) if r['split']=='test']
ids={r['restored_filename']:r['unique_id'] for r in rows}
dataset=OUT/'validation_dataset'
for kind in ['images','labels']:
    (dataset/kind/'test').mkdir(parents=True,exist_ok=True)
for r in rows:
    shutil.copy2(ROOT/'outputs/yolo_512_telea/images/test'/r['restored_filename'],dataset/'images/test'/r['restored_filename'])
    label=Path(r['filename']).stem+'.txt'
    shutil.copy2(ROOT/'outputs/yolo_512_telea/labels/test'/label,dataset/'labels/test'/label)
yaml=dataset/'dataset.yaml'
yaml.write_text('path: '+dataset.as_posix()+'\ntrain: images/test\nval: images/test\ntest: images/test\nnames:\n  0: defect\n',encoding='utf-8')

class AuditValidator(DetectionValidator):
    def init_metrics(self,model):
        AuditValidator.last=self
        super().init_metrics(model)

    def update_metrics(self,preds,batch):
        if not hasattr(self,'records'): self.records=[]
        for si,pred in enumerate(preds):
            b=self._prepare_batch(si,batch)
            native=ops.scale_boxes(b['imgsz'],pred['bboxes'].clone(),b['ori_shape'],b['ratio_pad'])
            gt=ops.scale_boxes(b['imgsz'],b['bboxes'].clone(),b['ori_shape'],b['ratio_pad'])
            self.records.append(dict(unique_id=ids[Path(b['im_file']).name],filename=Path(b['im_file']).name,
                input_shape=list(b['imgsz']),ori_shape=list(b['ori_shape']),ratio_pad=b['ratio_pad'],
                gt=gt.cpu().tolist(),gt_input=b['bboxes'].cpu().tolist(),gt_class=b['cls'].cpu().tolist(),
                predictions=[dict(box=box,confidence=float(c),class_id=int(k),input_box=ib)
                    for box,c,k,ib in zip(native.cpu().tolist(),pred['conf'].cpu().tolist(),pred['cls'].cpu().tolist(),pred['bboxes'].cpu().tolist())]))
        super().update_metrics(preds,batch)

if __name__=='__main__':
    y=YOLO(str(model))
    y.val(validator=AuditValidator,data=str(yaml),split='test',imgsz=1280,batch=4,device='cpu',workers=0,
        project=str(OUT),name='validator_replay',exist_ok=True,plots=True,verbose=False,
        conf=None,iou=.7,max_det=300,augment=False)
    v=AuditValidator.last
    data=dict(runtime=dict(torch=torch.__version__,ultralytics=__import__('ultralytics').__version__,device='cpu'),
        args=vars(v.args),cm_conf=v.confusion_matrix_conf,matrix=v.confusion_matrix.matrix.tolist(),
        metrics=v.metrics.results_dict,records=v.records,model_sha256=hashlib.sha256(model.read_bytes()).hexdigest())
    (OUT/'validator_replay.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print('CM',data['matrix'],'confidence',data['cm_conf'])
