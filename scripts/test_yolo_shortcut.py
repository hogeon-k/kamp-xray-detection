import unittest

import numpy as np

import diagnose_yolo_shortcut as d


class ShortcutLogicTests(unittest.TestCase):
    def test_fake_region_matching_uses_mask_and_excludes_gt(self):
        mask=np.zeros((40,40),bool)
        mask[10:20,10]=mask[10:20,19]=True
        mask[10,10:20]=mask[19,10:20]=True
        box=(10,10,20,20)
        prediction=dict(box=[9,9,21,21],conf=.7,class_id=0)
        result=d.annotate_predictions([prediction],[],mask,box,conf=.25,raw_conf=.001,gt_iou=.5,coverage_threshold=.3)[0]
        self.assertTrue(result['fake_fp_final'])
        self.assertEqual(result['fake_mask_coverage'],1.)
        self.assertLess(result['iou_fake_bbox'],1.)
        gt_result=d.annotate_predictions([prediction],[(9,9,21,21)],mask,box,conf=.25,
                                         raw_conf=.001,gt_iou=.5,coverage_threshold=.3)[0]
        self.assertTrue(gt_result['gt_matched'])
        self.assertFalse(gt_result['fake_fp_candidate'])

    def test_fp_count_and_delta_zero_when_original_empty(self):
        mask=np.zeros((20,20),bool);mask[5:8,5:8]=True
        preds=d.annotate_predictions([dict(box=[4,4,9,9],conf=.4,class_id=0)],[],mask,(5,5,8,8),
                                     conf=.25,raw_conf=.001,gt_iou=.5,coverage_threshold=.3)
        before=d.region_summary([]);after=d.region_summary(preds)
        self.assertEqual(before['max_confidence'],0)
        self.assertEqual(after['final_count'],1)
        self.assertAlmostEqual(after['max_confidence']-before['max_confidence'],.4)
        self.assertEqual(d.gt_confidences([],[(1,1,3,3)],.5),[0.])

    def test_paired_tests_and_effect_direction(self):
        w=d.paired_wilcoxon([.1,.2,.3,.4],[.2,.3,.4,.5])
        self.assertAlmostEqual(w['paired_mean_difference'],.1)
        self.assertIsNotNone(w['p_value'])
        same=d.paired_wilcoxon([0,0],[0,0])
        self.assertEqual(same['p_value'],1)
        # With four equal positive differences, only all-positive or all-negative
        # sign assignments are at least as extreme: exact two-sided p=2/16.
        self.assertEqual(d.exact_signed_rank([1,1,1,1]),(0.,.125))
        m=d.mcnemar_exact([1,1,0,0],[0,0,0,0])
        self.assertEqual(m['baseline_only'],2)
        self.assertEqual(m['fake_aug_only'],0)
        self.assertAlmostEqual(m['risk_difference_fake_aug_minus_baseline'],-.5)

    def test_existing_pairs_config_and_hashes(self):
        configs=d.validate_training_configs()
        self.assertEqual(configs[0]['imgsz'],configs[1]['imgsz'])
        pairs=d.prepare_pairs(d.read_csv(d.CONTEXT/'matched_pairs.csv'))
        self.assertEqual(len(pairs),34)
        self.assertGreater(sum(p['source_split'] in ('val','test') for p in pairs),0)
        self.assertEqual(len({p['pair_id'] for p in pairs}),34)
        for p in pairs:
            self.assertEqual(p['mask_area'],int(p['mask_area']))


if __name__=='__main__':
    unittest.main()
