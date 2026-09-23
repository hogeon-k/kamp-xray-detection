import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from diagnose_restoration import (ROOT, change_pct, diagnose, distribution_summary,
                                   fake_regions, region_metrics, save_views)


class DiagnosticTests(unittest.TestCase):
    def test_identical_images_and_zero_baseline(self):
        image = np.full((20, 20, 3), 70, dtype=np.uint8)
        mask = np.ones((20, 20), bool)
        row = region_metrics(image, image, mask)
        self.assertAlmostEqual(row['ssim'], 1., places=12)
        self.assertEqual(row['mean_absolute_difference'], 0)
        self.assertEqual(row['changed_pixel_ratio'], 0)
        self.assertEqual(row['laplacian_variance_before'], 0)
        self.assertIsNone(row['laplacian_variance_change_pct'])
        self.assertIsNone(row['gradient_magnitude_change_pct'])
        self.assertEqual(change_pct(4, 6), 50.)

    def test_known_rgb_difference_and_mask_geometry(self):
        before = np.zeros((20, 20, 3), np.uint8)
        after = before.copy()
        mask = np.zeros((20, 20), bool)
        mask[3, 4] = mask[5, 7] = True
        after[3, 4] = [30, 60, 90]
        after[0, 0] = 255  # Must not be counted in RGB differences.
        row = region_metrics(before, after, mask)
        self.assertEqual(row['mean_absolute_difference'], 30.)
        self.assertEqual(row['max_absolute_difference'], 90)
        self.assertEqual(row['changed_pixel_ratio'], .5)
        self.assertEqual((row['mask_width'], row['mask_height'], row['mask_area']), (4, 3, 2))

    def test_ssim_constant_luminance_formula_and_thin_mask(self):
        before = np.full((20, 20, 3), 50, np.uint8)
        after = np.full((20, 20, 3), 100, np.uint8)
        mask = np.zeros((20, 20), bool)
        mask[0, 0] = True
        row = region_metrics(before, after, mask)
        expected = (2*50*100+2.55**2)/(50**2+100**2+2.55**2)
        self.assertAlmostEqual(row['ssim'], expected, places=10)

    def test_fake_regions_reject_missing_or_ambiguous_metadata(self):
        mask = np.zeros((10, 10), bool)
        mask[2, 2] = True
        placement = dict(x=1, y=1, width=3, height=3)
        self.assertEqual(len(fake_regions(mask, [placement])), 1)
        for placements in ([], [placement, placement]):
            with self.assertRaises(ValueError):
                fake_regions(mask, placements)

    def test_empty_group_summary_and_diff_gain(self):
        rows = distribution_summary({'real':[], 'fake':[]})
        self.assertEqual(rows[0]['fake_valid_count'], 0)
        self.assertIsNone(rows[0]['fake_mean'])
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as tmp:
            before = np.full((2, 2, 3), 100, np.uint8)
            after = np.full_like(before, 90)
            save_views(Path(tmp)/'views', before, after, np.ones((2,2), bool), 8)
            with Image.open(Path(tmp)/'views/abs_diff_amplified.png') as image:
                self.assertTrue((np.array(image) == 80).all())

    def test_existing_dataset_integration_and_overwrite_protection(self):
        manifest = ROOT/'outputs/yolo_poc_20_fake_restoration/manifest.csv'
        if not manifest.exists():
            self.skipTest('Generated dataset not present')
        with tempfile.TemporaryDirectory(prefix='diagnostic_test_', dir=ROOT/'outputs') as tmp:
            output = Path(tmp)/'diagnostics'
            with contextlib.redirect_stdout(io.StringIO()):
                result = diagnose(manifest, ROOT/'outputs/fake_restoration_debug', output)
            self.assertEqual(result['images'], 20)
            self.assertEqual(result['fake_regions'], 19)
            self.assertEqual(len(list((output/'images').glob('*/*/before.png'))), 40)
            self.assertEqual(len(list((output/'regions/fake').glob('*/*/before.png'))), 19)
            for filename in ('real.csv', 'fake.csv', 'summary.csv', 'samples.csv', 'metadata.json'):
                self.assertTrue((output/filename).is_file())
            with self.assertRaises(FileExistsError):
                diagnose(manifest, ROOT/'outputs/fake_restoration_debug', output)


if __name__ == '__main__':
    unittest.main()
