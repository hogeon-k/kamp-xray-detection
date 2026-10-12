"""Generate object-detection confusion matrix from locked predictions and actual GT.

Run: .cache/notebook-check-env/Scripts/python.exe submission_audit/generate_official_confusion.py
No retraining or inference. Raw cache was made with imgsz1280/conf floor.001/NMS IoU.7.
"""
from pathlib import Path
import sys,csv,json,hashlib
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'.cache/shortcut_runtime'))
from compare_matching import get_data,own
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    records=get_data(); totals={'TP':0,'FP':0,'FN':0}; table=[]
    for uid,r in records.items():
        match=own(r['gt'],r['cache'],t=.20,cut=.50)
        for k in totals:totals[k]+=match[k]
        table.append(dict(unique_id=uid,filename=r['row']['restored_filename'],**{k:match[k] for k in totals},
            matching_pairs=json.dumps(match['matches']),FP_prediction_indices=json.dumps(match['fp']),FN_GT_indices=json.dumps(match['fn'])))
    assert totals==dict(TP=148,FP=8,FN=12)
    out=ROOT/'submission_audit'
    with (out/'official_test_matching.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    src=ROOT/'outputs/foreign_object_condition_analysis/predictions.json'
    meta=dict(totals=totals,confidence=.20,baseline_confidence_same_counts=.25,matching_iou=.50,
        confidence_inclusion='>=',iou_inclusion='>=',class_id=0,matching='global IoU descending greedy one-to-one',
        GT=160,images=76,predictions_accepted=156,matrix_predicted_rows_true_columns=[[148,8],[12,None]],
        TN='undefined for object detection; no normal-image negatives in this cohort',
        predictions_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),source=str(src.relative_to(ROOT)))
    (out/'official_test_confusion.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    a=np.array([[148,8],[12,np.nan]])
    fig,ax=plt.subplots(figsize=(7.8,6.1));ax.imshow(a,cmap='Blues',vmin=0,vmax=160)
    for (row,col),v in np.ndenumerate(a):
        label='N/A\n(no TN)' if np.isnan(v) else f'{int(v)}\n'+[['TP','FP'],['FN','']][row][col]
        ax.text(col,row,label,ha='center',va='center',fontsize=22,color='white' if v>80 else '#163047')
    ax.set_xticks([0,1],['defect GT','unmatched prediction'])
    ax.set_yticks([0,1],['defect prediction','missed GT'])
    ax.set_xlabel('Ground truth / unmatched detections');ax.set_ylabel('Prediction / missed targets')
    ax.set_title('Test object matching: TP 148 / FP 8 / FN 12\nconfidence >= 0.20; GT IoU >= 0.50',pad=18)
    fig.text(.5,.02,'76 images, 160 GT; baseline confidence 0.25 gives the same counts.\nSource: frozen predict() cache; NMS IoU 0.70. No true-negative estimate.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(out/'official_test_confusion.png',dpi=180);plt.close(fig)
    print('Official confusion generated',totals)

if __name__=='__main__':main()
