import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

import diagnose_context_matched_restoration as d


class ContextMatchedTests(unittest.TestCase):
    def test_ring_offsets_match_full_image_including_boundary(self):
        template=np.zeros((6,8),bool)
        template[[0,-1],:]=True; template[:,[0,-1]]=True
        for x,y in ((0,0),(12,10),(22,24)):
            mask=d.translated_mask(template,x,y,(30,30))
            yy,xx=d.ring_coordinates(x,y,d.ring_offsets(template,5),mask.shape)
            translated=np.zeros(mask.shape,bool); translated[yy,xx]=True
            np.testing.assert_array_equal(translated,d.make_ring(mask,5))

    def test_candidates_reproducible_safe_unique_and_background_filtered(self):
        rgb=np.random.default_rng(9).integers(20,220,(70,80,3),dtype=np.uint8)
        rgb[:15,:]=255
        template=np.ones((4,5),bool)
        forbidden=np.zeros((70,80),bool); forbidden[25:40,30:40]=True
        selected=np.zeros_like(forbidden); selected[45:55,50:60]=True
        args=(d.after_maps(rgb),template,forbidden,selected)
        a=d.evaluate_candidates(*args,np.random.default_rng(42),candidates=256)
        b=d.evaluate_candidates(*args,np.random.default_rng(42),candidates=256)
        np.testing.assert_array_equal(a[0],b[0]); np.testing.assert_array_equal(a[1],b[1])
        self.assertEqual(len(a[0]),256)
        self.assertEqual(len(set(map(tuple,a[0]))),256)
        for x,y in a[0]:
            self.assertFalse((forbidden|selected)[y:y+4,x:x+5].any())
            self.assertGreaterEqual(y,15)

    def test_no_safe_candidate_and_shape_validation(self):
        rgb=np.full((20,20,3),100,np.uint8)
        occupied=np.ones((20,20),bool)
        pos,vectors,_=d.evaluate_candidates(d.after_maps(rgb),np.ones((3,3),bool),occupied,occupied,np.random.default_rng(1))
        self.assertEqual(len(pos),0)
        blank=np.zeros((20,20),bool); template=np.eye(4,dtype=bool)
        mask=d.translated_mask(template,3,5,blank.shape)
        d.validate_fake(mask,template,3,5,blank,blank,blank,blank)
        with self.assertRaises(RuntimeError):
            d.validate_fake(mask,template,3,5,mask,blank,blank,blank)
        with self.assertRaises(ValueError):
            d.translated_mask(template,19,19,blank.shape)

    def test_standardization_known_nearest_and_constant_columns(self):
        candidates=np.array([[1,100,9],[2,200,9],[3,300,9]],float)
        distances,center,scale=d.standardized_distances(candidates[1],candidates)
        self.assertEqual(np.argmin(distances),1)
        self.assertEqual(distances[1],0)
        self.assertEqual(scale[2],1)
        self.assertTrue(np.isfinite(distances).all())

    def test_paired_effect_direction_zeros_and_missing(self):
        identical={f:1. for f in d.FEATURES}
        results=d.paired_statistics([dict(real=identical,fake=identical)])
        self.assertTrue(all(r['p_value']==1 for r in results))
        self.assertTrue(all(r['paired_rank_biserial']==0 for r in results))
        raised={f:2. for f in d.FEATURES}
        results=d.paired_statistics([dict(real=identical,fake=raised)]*8)
        self.assertTrue(all(r['paired_rank_biserial']==1 for r in results))
        self.assertTrue(all(r['mean_paired_difference']==1 for r in results))
        self.assertTrue(all(r['count']==0 for r in d.paired_statistics([])))

    def test_real_dataset_repeatability_and_protection(self):
        root=d.ROOT; manifest=root/'outputs/yolo_poc_20_fake_restoration/manifest.csv'
        if not manifest.exists():
            self.skipTest('Prepared diagnostic inputs are unavailable')
        with tempfile.TemporaryDirectory(prefix='context_test_',dir=root/'.cache') as tmp:
            outputs=[]
            for name in ('first','second'):
                dest=Path(tmp)/name
                with contextlib.redirect_stdout(io.StringIO()):
                    outputs.append(d.run_diagnostic(manifest,root/'outputs/fake_restoration_debug',
                        root/'outputs/restoration_after_similarity',dest))
            self.assertEqual(outputs[0],outputs[1])
            pairs,unmatched,stats,classifiers=outputs[0]
            self.assertEqual(len(pairs)+len(unmatched),34)
            self.assertTrue(all(p['evaluated_candidates']>=200 for p in pairs))
            self.assertTrue(all(p['context_distance']<=2.5 for p in pairs))
            self.assertTrue(all(p['context_distance']<=p['random_control_distance'] for p in pairs))
            for path in (Path(tmp)/'first').glob('*.csv'):
                self.assertEqual(path.read_bytes(),(Path(tmp)/'second'/path.name).read_bytes())
            for first in (Path(tmp)/'first/paired_images').glob('*.png'):
                self.assertEqual(d.sha256(first),d.sha256(Path(tmp)/'second/paired_images'/first.name))
            for result in classifiers.values():
                for fold in result.get('folds',[]):
                    self.assertFalse(set(fold['train_groups']) & set(fold['test_groups']))
            validation=json.loads((Path(tmp)/'first/metadata.json').read_text())
            self.assertEqual(validation['fake_gt_overlap'],0)
            self.assertEqual(validation['mask_shape_mismatches'],0)
            self.assertTrue(validation['git_diff_unchanged'])
            with self.assertRaises(FileExistsError):
                d.run_diagnostic(manifest,root/'outputs/fake_restoration_debug',root/'outputs/restoration_after_similarity',Path(tmp)/'first')


if __name__=='__main__':
    unittest.main()
