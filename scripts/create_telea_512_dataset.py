"""Create TELEA data from a frozen split manifest; no training or label edits."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from split_clean_512 import ROOT, SPLITS, sha256, read_csv, write_csv, write_json, labels, validate_source, leakage, summarize, snapshot, verify_snapshot

def load_poc_mask(path):
    # Execute the exact existing function AST without importing unrelated PoC pandas code.
    tree = ast.parse(path.read_text(encoding='utf-8'))
    selected = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in ('safe_div', 'inspect_palette')) or
                (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PALETTE_RGB' for t in n.targets))]
    if len(selected) != 3:
        raise ValueError('PoC mask source structure changed')
    ns = {'Path': Path, 'Image': Image, 'np': np, 'cv2': cv2}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), ns)
    return ns['inspect_palette']

def overlap(boxes, mask):
    height, width = mask.shape
    union = np.zeros(mask.shape, bool)
    records, raster = [], []
    for i, (x,y,w,h) in enumerate(boxes):
        x0, y0 = max(0,min(width,math.ceil((x-w/2)*width))), max(0,min(height,math.ceil((y-h/2)*height)))
        x1, y1 = max(0,min(width,math.ceil((x+w/2)*width))), max(0,min(height,math.ceil((y+h/2)*height)))
        area = (x1-x0)*(y1-y0)
        pixels = int(mask[y0:y1,x0:x1].sum())
        union[y0:y1,x0:x1] = True
        raster.append((x0,y0,x1,y1))
        records.append({'bbox_index': i, 'bbox_area_pixels': area, 'overlap_pixels': pixels,
                        'overlap_ratio_of_bbox': pixels / area if area else 0, 'intersects': bool(pixels)})
    gt_pixels = int(union.sum())
    hit = int((union & mask).sum())
    return records, raster, {'gt_mask_overlap_pixels':hit, 'gt_union_pixels':gt_pixels,
                            'gt_mask_overlap_ratio':hit/gt_pixels if gt_pixels else 0,
                            'overlap_bbox_count':sum(r['intersects'] for r in records)}

def panel(original, mask, restored, boxes, title, panel_width=360):
    overlay = Image.fromarray(restored.copy())
    draw = ImageDraw.Draw(overlay)
    for x0,y0,x1,y1 in boxes:
        draw.rectangle((x0,y0,max(x0,x1-1),max(y0,y1-1)), outline='cyan', width=2)
    panels = [Image.fromarray(original), Image.fromarray(mask.astype(np.uint8)*255).convert('RGB'), Image.fromarray(restored), overlay]
    height = round(original.shape[0]*panel_width/original.shape[1])
    canvas = Image.new('RGB', (panel_width*4, height+42), 'white')
    for i,(im,name) in enumerate(zip(panels, ['ORIGINAL','MASK','TELEA RESTORED','RESTORED + GT'])):
        canvas.paste(im.resize((panel_width,height)), (i*panel_width,42))
        ImageDraw.Draw(canvas).text((i*panel_width+5,23),name,fill='black')
    ImageDraw.Draw(canvas).text((5,5),title,fill='black')
    return canvas

def audit_relations(rows):
    # Map historical aliases by file SHA to the retained canonical record.
    inventory = read_csv(ROOT / 'outputs/unique_512_audit/all_images_inventory.csv')
    by_sha = {r['image_sha256']:r for r in rows}
    aliases, historical_pixels = {}, defaultdict(set)
    for r in inventory:
        if r['sha256'] in by_sha:
            canonical = by_sha[r['sha256']]
            aliases[r['full_path']] = canonical
            historical_pixels[r['pixel_hash']].add(canonical['split'])
    cross_pixel = sum(len(v)>1 for v in historical_pixels.values())
    exact_cross, near = [], []
    for r in read_csv(ROOT / 'outputs/unique_512_audit/duplicate_candidates.csv'):
        left, right = aliases.get(r['anchor_path']), aliases.get(r['candidate_path'])
        if left and right and left['unique_id'] != right['unique_id'] and left['split'] != right['split']:
            if r['reason'] == 'PERCEPTUAL_NEAR_ONLY':
                near.append({'left_id':left['unique_id'],'right_id':right['unique_id'],'left_split':left['split'],
                             'right_split':right['split'],'phash_distance':r['phash_distance'],'decision':'REVIEW_ONLY_NOT_AUTOMATIC_DUPLICATE'})
            elif r['decision'] in ('EXACT','VERIFIED_CONVERSION'):
                exact_cross.append(r)
    if cross_pixel or exact_cross:
        raise ValueError('Historical exact duplicate leakage found')
    unique = {tuple(sorted((r['left_id'],r['right_id']))):r for r in near}
    return list(unique.values()), {'historical_pixel_cross_split':cross_pixel,'historical_exact_relation_cross_split':len(exact_cross)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'outputs/clean_512')
    parser.add_argument('--split-dir',type=Path,default=ROOT/'outputs/clean_512_split')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/yolo_512_telea')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_rows = validate_source(args.source)
    rows = read_csv(args.split_dir/'split_manifest.csv')
    if len(rows)!=512 or {r['unique_id'] for r in rows} != {r['unique_id'] for r in source_rows}:
        raise ValueError('Manifest/source ID mismatch')
    source_by_id = {r['unique_id']:r for r in source_rows}
    for r in rows:
        if r['split'] not in SPLITS or any(r[k] != source_by_id[r['unique_id']][k] for k in ('filename','image_sha256','label_sha256','bbox_count','machine','serial','capture_date','pixel_sha256')):
            raise ValueError('Manifest/source mismatch')
    checks = leakage(rows)
    near, historical = audit_relations(rows)
    checks.update(historical)
    protected = json.loads((args.split_dir/'protected_baseline.json').read_text(encoding='utf-8'))
    # Protect the finalized split and reproducibility scripts as well.
    protected.update(snapshot([args.split_dir]))
    verify_snapshot(protected)
    poc = ROOT/'outputs/inpainting_poc_20/run_poc.py'
    inspect = load_poc_mask(poc)
    args.output.mkdir(parents=True)
    for kind in ('images','labels','restoration_masks'):
        for split in SPLITS:
            (args.output/kind/split).mkdir(parents=True)
    debug = args.output/'debug'
    debug.mkdir()
    results, bbox_records = [], []
    for index, r in enumerate(rows):
        split, stem = r['split'], Path(r['filename']).stem
        source_image = args.source/'images'/r['filename']
        source_label = args.source/'labels'/(stem+'.txt')
        info = inspect(source_image)
        if not info['mask_equal']:
            raise ValueError('Palette/RGB mask mismatch: '+stem)
        mask, original = info['palette_mask'], info['rgb']
        mask_pixels = int(mask.sum())
        if mask_pixels:
            restored = cv2.cvtColor(cv2.inpaint(cv2.cvtColor(original,cv2.COLOR_RGB2BGR),mask.astype(np.uint8)*255,3,cv2.INPAINT_TELEA),cv2.COLOR_BGR2RGB)
            image_name = stem+'.png'
            Image.fromarray(restored).save(args.output/'images'/split/image_name)
        else:
            restored, image_name = original.copy(), r['filename']
            shutil.copy2(source_image,args.output/'images'/split/image_name)
        shutil.copy2(source_label,args.output/'labels'/split/(stem+'.txt'))
        Image.fromarray(mask.astype(np.uint8)*255).save(args.output/'restoration_masks'/split/(stem+'.png'))
        per_box, raster, metrics = overlap(labels(source_label),mask)
        bbox_records.extend(dict(unique_id=r['unique_id'],filename=r['filename'],split=split,**b) for b in per_box)
        delta = np.abs(restored.astype(np.int16)-original.astype(np.int16))
        outside_changed = int(np.any(delta,axis=2)[~mask].sum())
        if outside_changed:
            raise ValueError('Inpainting changed pixels outside mask')
        results.append(dict(r,restored_filename=image_name,mask_pixel_count=mask_pixels,mask_ratio=mask_pixels/mask.size,
                            restoration_applied=bool(mask_pixels),method='TELEA',radius=3,dilation=0,input_sha256=sha256(source_image),
                            restored_sha256=sha256(args.output/'images'/split/image_name),label_sha256=sha256(args.output/'labels'/split/(stem+'.txt')),
                            mean_abs_rgb_change=float(delta.mean()),changed_pixel_count=int(np.any(delta,axis=2).sum()),
                            outside_mask_changed_pixels=outside_changed,palette_rgb_equal=True,**metrics))
        if (index+1)%64==0:
            print(f'Restored/validated {index+1}/512',flush=True)
    # Full on-disk checks rather than trusting generation counters.
    total_images = total_labels = total_bbox = 0
    for split in SPLITS:
        part = [r for r in results if r['split']==split]
        ims = {p.stem:p for p in (args.output/'images'/split).iterdir()}
        lbs = {p.stem:p for p in (args.output/'labels'/split).iterdir()}
        masks = {p.stem:p for p in (args.output/'restoration_masks'/split).iterdir()}
        if ims.keys()!=lbs.keys() or ims.keys()!=masks.keys() or len(ims)!=len(part):
            raise ValueError('Output stems/count mismatch')
        total_images += len(ims)
        total_labels += len(lbs)
        for r in part:
            stem = Path(r['filename']).stem
            if sha256(lbs[stem])!=r['label_sha256'] or r['label_sha256']!=source_by_id[r['unique_id']]['label_sha256'] or sha256(ims[stem])!=r['restored_sha256']:
                raise ValueError('Output SHA/label mismatch')
            total_bbox += len(labels(lbs[stem]))
            with Image.open(ims[stem]) as im:
                if im.size!=(int(source_by_id[r['unique_id']]['width']),int(source_by_id[r['unique_id']]['height'])):
                    raise ValueError('Output dimension mismatch')
                r['restored_pixel_sha256']=hashlib.sha256(str(im.size).encode()+im.convert('RGB').tobytes()).hexdigest()
    restored_checks = leakage([dict(r,pixel_sha256=r['restored_pixel_sha256']) for r in results], 'restored_sha256')
    if (total_images,total_labels,total_bbox)!=(512,512,1163):
        raise ValueError('Final totals failed')
    # Representatives plus ranked review candidates, selected only after split is frozen.
    qa = []
    for split in SPLITS:
        part = [r for r in results if r['split']==split]
        selectors = [('representative',sorted(part,key=lambda r:(abs(r['mask_ratio']-float(np.median([v['mask_ratio'] for v in part]))),r['unique_id']))[:2]),
                     ('largest_mask',sorted(part,key=lambda r:r['mask_pixel_count'],reverse=True)[:3]),
                     ('gt_overlap',sorted([r for r in part if r['gt_mask_overlap_pixels']],key=lambda r:r['gt_mask_overlap_pixels'],reverse=True)[:3]),
                     ('largest_change',sorted(part,key=lambda r:r['mean_abs_rgb_change'],reverse=True)[:3])]
        thumbnails = []
        for category, selected in selectors:
            for r in selected:
                stem = Path(r['filename']).stem
                original = inspect(args.source/'images'/r['filename'])['rgb']
                with Image.open(args.output/'images'/split/r['restored_filename']) as im:
                    restored=np.asarray(im.convert('RGB')).copy()
                with Image.open(args.output/'restoration_masks'/split/(stem+'.png')) as im:
                    mask=np.asarray(im)>0
                _, boxes, _ = overlap(labels(args.source/'labels'/(stem+'.txt')),mask)
                title=f"{split} {r['unique_id']} {category} mask={r['mask_pixel_count']} overlap={r['gt_mask_overlap_pixels']}"
                name=f"{split}_{category}_{r['unique_id']}.png"
                panel(original,mask,restored,boxes,title).save(debug/name)
                thumbnails.append(panel(original,mask,restored,boxes,title,200))
                qa.append({'split':split,'unique_id':r['unique_id'],'category':category,'debug_filename':name})
        sheet=Image.new('RGB',(800,sum(im.height for im in thumbnails)),'white')
        y=0
        for im in thumbnails:
            sheet.paste(im,(0,y)); y+=im.height
        sheet.save(debug/(split+'_contact_sheet.png'))
    write_csv(debug/'qa_selection.csv',qa)
    write_csv(args.output/'gt_mask_bbox_overlap.csv',bbox_records)
    write_csv(args.output/'near_cross_split_review.csv',near,['left_id','right_id','left_split','right_split','phash_distance','decision'])
    write_csv(args.output/'manifest.csv',results)
    (args.output/'dataset.yaml').write_text(f'path: {args.output.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: defect\n',encoding='utf-8')
    protected_count=verify_snapshot(protected)
    # Check that no files were added to the original dataset tree.
    expected_dataset={p for p in protected if Path(p).is_relative_to(ROOT/'dataset')}
    actual_dataset={str(p.resolve()) for p in (ROOT/'dataset').rglob('*') if p.is_file()}
    if expected_dataset!=actual_dataset:
        raise ValueError('Original dataset file inventory changed')
    summary={'images':total_images,'labels':total_labels,'bbox':total_bbox,'class_ids':[0],'splits':summarize(results),
             'processed_images':512,'restoration_applied':sum(r['restoration_applied'] for r in results),
             'no_mask_images':sum(not r['restoration_applied'] for r in results),
             'gt_mask_overlap_images':sum(r['gt_mask_overlap_pixels']>0 for r in results),
             'gt_mask_overlap_bbox':sum(r['overlap_bbox_count'] for r in results),
             'gt_mask_overlap_union_pixels':sum(r['gt_mask_overlap_pixels'] for r in results),
             'method':'TELEA','radius':3,'dilation':0,'poc_mask_source':str(poc),'poc_code_sha256':sha256(poc),
             'mask_logic':'Unmodified AST of PoC inspect_palette + safe_div + PALETTE_RGB; exact palette/RGB equality verified',
             'split_manifest_sha256':sha256(args.split_dir/'split_manifest.csv'),
             'source_leakage':checks,'restored_leakage':restored_checks,'near_cross_split_review_candidates':len(near),
             'protected_files_verified':protected_count,'protected_file_changes':0,'dataset_inventory_changes':0,
             'label_content_changes':0,'integrity_errors':0,'missing_images':0,'missing_labels':0,'orphan_images':0,'orphan_labels':0,
             'palette_rgb_mismatches':0,'outside_mask_changed_pixels':0,'all_checks_passed':True,
             'versions':{'opencv':cv2.__version__,'numpy':np.__version__,'pillow':Image.__version__}}
    write_json(args.output/'restoration_summary.json',summary)
    (args.output/'README.md').write_text('# 512-image TELEA YOLO preparation\n\nLineage: clean_512 → clean_512_split → yolo_512_telea\n\nSplit is frozen before restoration. TELEA radius=3, dilation=0 uses the exact existing PoC palette/RGB mask function. Marked images are lossless RGB PNG; zero-mask images are byte-identical copies preserving their original extension. Labels are byte-identical copies, with dimensions and coordinates unchanged. No training, augmentation, fake restoration or threshold tuning.\n\nSee restoration_summary.json, manifest.csv and gt_mask_bbox_overlap.csv for full verification and per-image/per-box overlap. Bbox rasterization uses clipped ceil edges and exclusive upper bounds, matching PoC convention; image overlap uses GT union to avoid double counting. Mean RGB change is over the whole original-sized image.\n\nReview debug/train_contact_sheet.png first, then val/test sheets. QA images alone are resized for viewing. Near cross-split candidates are historical pHash review candidates only, never automatic duplicates. Restoration quality remains a visual review task; overlap statistics do not establish recovery of hidden defect signal.\n\nReproduce with the project .venv Python: scripts/split_clean_512.py then scripts/create_telea_512_dataset.py. Both refuse existing output directories.\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
