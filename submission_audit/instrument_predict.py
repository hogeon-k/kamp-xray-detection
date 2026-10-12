from pathlib import Path
import sys,json,csv
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'submission_audit'
sys.path.insert(0,str(ROOT/'.cache/shortcut_runtime'))
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from compare_matching import get_data,own,cm
torch.set_num_threads(4)

class AuditPredictor(DetectionPredictor):
    def preprocess(self,im):
        tensor=super().preprocess(im)
        self.shapes=getattr(self,'shapes',{})
        for path in self.batch[0]:self.shapes[Path(path).name]=list(tensor.shape[2:])
        return tensor

if __name__=='__main__':
    records=get_data(); rows=[r['row'] for r in records.values()]
    paths=[str(ROOT/'outputs/yolo_512_telea/images/test'/r['restored_filename']) for r in rows]
    model=ROOT/'outputs/yolo_runs/telea_512_yolo11n_e200_img1280_b4_adamw_lr0.001_wd0.0005_noaug_s42/weights/best.pt'
    y=YOLO(str(model))
    results=y.predict(predictor=AuditPredictor,source=paths,stream=True,imgsz=1280,conf=.001,iou=.7,
        max_det=300,device='cpu',batch=4,save=False,verbose=False,augment=False)
    capture=[];totals={'own50':[0,0,0],'cm45':[0,0,0]}
    for r,result in zip(rows,results):
        b=result.boxes
        pred=[dict(box=box,confidence=float(c),class_id=int(k)) for box,c,k in zip(b.xyxy.tolist(),b.conf.tolist(),b.cls.tolist())]
        capture.append(dict(unique_id=r['unique_id'],input_shape=y.predictor.shapes[r['restored_filename']],predictions=pred))
        for mode,res in [('own50',own(records[r['unique_id']]['gt'],pred)),('cm45',cm(records[r['unique_id']]['gt'],pred))]:
            totals[mode]=[a+res[k] for a,k in zip(totals[mode],['TP','FP','FN'])]
    (OUT/'predict_replay.json').write_text(json.dumps(dict(totals=totals,records=capture),ensure_ascii=False,indent=2),encoding='utf-8')
    print('predict replay totals',totals,'shapes',set(tuple(r['input_shape']) for r in capture))
