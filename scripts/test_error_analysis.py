"""Regression checks for confidence-independent displayed-candidate IoU."""
import unittest

from error_analysis import candidate_diagnosis, match


class CandidateIoUTest(unittest.TestCase):
    def test_below_threshold_overlap_is_diagnosed_but_not_a_tp(self):
        gt = [0, 0, 10, 10]
        candidate = dict(box=[0, 0, 10, 10], confidence=.02, class_id=0)
        score, cause = candidate_diagnosis(gt, candidate, .2)
        self.assertEqual(score, 1.)
        self.assertEqual(cause, 'low_confidence_localization_success')
        self.assertEqual(match([gt], [candidate], .2), ({}, [], []))

    def test_below_threshold_localization_failure(self):
        score, cause = candidate_diagnosis([0, 0, 10, 10], dict(box=[0, 0, 6, 6], confidence=.02), .2)
        self.assertAlmostEqual(score, .36)
        self.assertEqual(cause, 'low_confidence_localization_failure')

    def test_boundary_iou_half_is_success(self):
        score, cause = candidate_diagnosis([0, 0, 10, 10], dict(box=[0, 0, 5, 10], confidence=.1999), .2)
        self.assertEqual(score, .5)
        self.assertEqual(cause, 'low_confidence_localization_success')

    def test_threshold_equality_is_not_low_confidence(self):
        score, cause = candidate_diagnosis([0, 0, 10, 10], dict(box=[0, 0, 6, 6], confidence=.2), .2)
        self.assertAlmostEqual(score, .36)
        self.assertEqual(cause, 'localization_failure')

    def test_absent_candidate_has_no_numeric_iou(self):
        self.assertEqual(candidate_diagnosis([0, 0, 10, 10], None, .2), (None, 'unknown'))


if __name__ == '__main__':
    unittest.main()
