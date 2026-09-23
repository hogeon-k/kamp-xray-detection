"""Run: python -m unittest discover -s scripts -p test_fake_restoration.py -v"""
import unittest
import contextlib
import io
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np

from fake_restoration import (RestorationSettings, augment_fake_restoration,
                              bbox_mask, inpaint_rgb, mask_templates)


class FixedCount:
    def __init__(self, count, seed=42):
        self.count = count
        self.rng = np.random.default_rng(seed)

    def choice(self, *args, **kwargs):
        return self.count

    def integers(self, *args, **kwargs):
        return self.rng.integers(*args, **kwargs)

    def uniform(self, *args, **kwargs):
        return self.rng.uniform(*args, **kwargs)


class FakeRestorationTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.random.default_rng(2).integers(0, 256, (120, 150, 3), dtype=np.uint8)
        self.real = np.zeros((120, 150), np.uint8)
        cv2.rectangle(self.real, (15, 20), (30, 40), 255, 1)
        self.settings = RestorationSettings('TELEA', 3, 0)
        self.boxes = [(60., 50., 80., 70.)]

    def run_aug(self, **kwargs):
        args = dict(split='train', rng=FixedCount(3), settings=self.settings)
        args.update(kwargs)
        return augment_fake_restoration(self.rgb, self.real, self.boxes, **args)

    def test_placement_preserves_shape_and_protected_pixels(self):
        before = self.rgb.copy()
        result, fake, audit = self.run_aug()
        self.assertEqual(audit['applied_count'], 3)
        protected = bbox_mask(self.real.shape, self.boxes, 10) | (self.real > 0)
        self.assertFalse(np.any((fake > 0) & protected))
        np.testing.assert_array_equal(result[protected], before[protected])
        np.testing.assert_array_equal(result[fake == 0], before[fake == 0])
        np.testing.assert_array_equal(self.rgb, before)
        patch = mask_templates(self.real)[0]
        for p in audit['placements']:
            np.testing.assert_array_equal(fake[p['y']:p['y']+p['height'], p['x']:p['x']+p['width']], patch)

    def test_val_test_and_zero_are_identity(self):
        for split in ('val', 'test'):
            result, fake, audit = self.run_aug(split=split)
            np.testing.assert_array_equal(result, self.rgb)
            self.assertFalse(fake.any())
            self.assertEqual(audit['requested_count'], 0)
        result, fake, audit = self.run_aug(rng=FixedCount(0))
        np.testing.assert_array_equal(result, self.rgb)
        self.assertFalse(fake.any())

    def test_seed_reproducibility_and_scaling(self):
        for scales in ((1., 1.), (.8, 1.2)):
            a = self.run_aug(rng=FixedCount(3, 77), scale_range=scales)
            b = self.run_aug(rng=FixedCount(3, 77), scale_range=scales)
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])
            self.assertEqual(a[2], b[2])

    def test_full_gt_and_missing_mask_skip_safely(self):
        self.boxes = [(0, 0, 150, 120)]
        result, fake, audit = self.run_aug(max_attempts=5)
        self.assertEqual(audit['skipped_count'], 3)
        self.assertFalse(fake.any())
        self.real[:] = 0
        result, fake, audit = self.run_aug()
        self.assertEqual(audit['skipped_count'], 3)

    def test_same_inpaint_method_for_telea_and_ns(self):
        for method, flag in (('TELEA', cv2.INPAINT_TELEA), ('NS', cv2.INPAINT_NS)):
            expected = cv2.cvtColor(cv2.inpaint(cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR), self.real, 3, flag), cv2.COLOR_BGR2RGB)
            np.testing.assert_array_equal(inpaint_rgb(self.rgb, self.real, RestorationSettings(method, 3, 0)), expected)

    def test_sampling_probabilities(self):
        # Tiny empty mask isolates count sampling from rejection retries.
        self.rgb = self.rgb[:1, :1]
        self.real = np.zeros((1, 1), np.uint8)
        self.boxes = []
        rng = np.random.default_rng(123)
        counts = np.zeros(4)
        for _ in range(10000):
            counts[self.run_aug(rng=rng)[2]['requested_count']] += 1
        np.testing.assert_allclose(counts/10000, [.25, .35, .25, .15], atol=.015)

    def test_invalid_configuration_fails(self):
        for kwargs in ({'split':'other'}, {'safety_margin':-1}, {'scale_range':(.5, 1)}, {'max_attempts':0}):
            with self.assertRaises(ValueError):
                self.run_aug(**kwargs)


class ExistingDatasetIntegrationTests(unittest.TestCase):
    def test_repeated_generation_and_read_only_inputs(self):
        from create_fake_restoration_dataset import ROOT, generate
        from create_yolo_poc_20 import sha256
        source = ROOT/'outputs/yolo_poc_20'
        poc = ROOT/'outputs/inpainting_poc_20'
        if not (source/'manifest.csv').exists():
            self.skipTest('Existing PoC dataset is not available')
        with tempfile.TemporaryDirectory(prefix='fake_restoration_test_', dir=ROOT/'outputs') as tmp:
            base = Path(tmp)
            summaries = []
            for name in ('a', 'b'):
                with contextlib.redirect_stdout(io.StringIO()):
                    summaries.append(generate(source, poc, base/name, base/(name+'_debug')))
            self.assertEqual(summaries[0], summaries[1])
            for path in (base/'a').glob('images/*/*.png'):
                relative = path.relative_to(base/'a')
                self.assertEqual(sha256(path), sha256(base/'b'/relative))
                if relative.parts[1] in ('val', 'test'):
                    self.assertEqual(sha256(path), sha256(source/relative))
            for path in (base/'a').glob('labels/*/*.txt'):
                relative = path.relative_to(base/'a')
                self.assertEqual(sha256(path), sha256(source/relative))
            a = json.loads((base/'a_debug/per_image.json').read_text())
            b = json.loads((base/'b_debug/per_image.json').read_text())
            self.assertEqual([r['placements'] for r in a], [r['placements'] for r in b])
            self.assertEqual(len(list((base/'a_debug').glob('*_debug.png'))), 20)
            with self.assertRaises(FileExistsError):
                generate(source, poc, base/'a', base/'unused')
            with self.assertRaises(ValueError):
                generate(source, poc, source/'new_output', base/'unused')


if __name__ == '__main__':
    unittest.main()
