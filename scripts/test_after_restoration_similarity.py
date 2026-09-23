import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import diagnose_after_restoration_similarity as diagnostic


class AfterOnlyTests(unittest.TestCase):
    def test_ring_clips_image_boundary_and_excludes_mask(self):
        mask = np.zeros((9,9),bool); mask[0,0] = True
        ring = diagnostic.make_ring(mask,5)
        self.assertEqual(ring.shape, mask.shape)
        self.assertFalse((ring & mask).any())
        self.assertTrue(ring[0,1])
        self.assertFalse(ring[-1,-1])
        with self.assertRaises(ValueError):
            diagnostic.make_ring(mask,0)

    def test_boundary_crossing_edges_and_known_intensity(self):
        rgb = np.full((9,9,3),10,np.uint8)
        mask = np.zeros((9,9),bool); mask[4,4] = True
        rgb[mask] = 30
        maps = diagnostic.after_maps(rgb)
        values,boundary = diagnostic.boundary_values(maps[0],mask)
        self.assertEqual(len(values),4)
        self.assertTrue((values == 20).all())
        self.assertEqual(boundary.sum(),1)
        result = diagnostic.measure_region(maps,mask,diagnostic.make_ring(mask,2))
        self.assertEqual(result['intensity_difference'],20)
        self.assertEqual(result['intensity_ratio'],3)
        self.assertIsNone(result['intensity_zscore'])
        self.assertEqual(result['boundary_discontinuity_mean'],20)

    def test_empty_ring_zero_denominators_and_hollow_mask(self):
        rgb = np.zeros((7,7,3),np.uint8)
        mask = np.ones((7,7),bool)
        result = diagnostic.measure_region(diagnostic.after_maps(rgb),mask,diagnostic.make_ring(mask))
        self.assertEqual(result['ring_area'],0)
        self.assertIsNone(result['texture_ratio'])
        self.assertIsNone(result['boundary_discontinuity_mean'])
        self.assertTrue(all(v is None or np.isfinite(v) for v in result.values()))
        mask[2:5,2:5] = False
        ring = diagnostic.make_ring(mask,1)
        self.assertTrue(ring[2,3])
        self.assertFalse(ring[3,3])

    def test_effect_size_direction_and_ties(self):
        self.assertEqual(diagnostic.cliffs_delta([3,4],[1,2]),1)
        self.assertEqual(diagnostic.cliffs_delta([1,2],[3,4]),-1)
        self.assertEqual(diagnostic.cliffs_delta([1,2],[1,2]),0)

    def test_classifier_group_isolation_and_reproducibility(self):
        rows = []
        for i in range(10):
            for j,group in enumerate(('real','fake')):
                row = {f:float(i+j) for f in diagnostic.FEATURES}
                row.update(asset_id=str(i),region_id=1,group=group,split='train')
                rows.append(row)
        a, ap = diagnostic.classifier_diagnostic(rows,42)
        b, bp = diagnostic.classifier_diagnostic(rows,42)
        self.assertEqual(a,b); self.assertEqual(ap,bp)
        for fold in a['folds']:
            self.assertFalse(set(fold['train_groups']) & set(fold['test_groups']))
        for asset in {r['asset_id'] for r in ap}:
            self.assertEqual(len({r['fold'] for r in ap if r['asset_id']==asset}),1)

    def test_full_dataset_after_only_reproducibility_and_integrity(self):
        root = diagnostic.ROOT
        manifest = root/'outputs/yolo_poc_20_fake_restoration/manifest.csv'
        if not manifest.exists():
            self.skipTest('Existing augmented dataset not available')
        expected = {Path(r['output_image_path']).resolve() for r in diagnostic.read_csv(manifest)}
        original_loader = diagnostic.load_rgb
        def after_only_loader(path):
            self.assertIn(Path(path).resolve(),expected)
            return original_loader(path)
        with tempfile.TemporaryDirectory(prefix='after_similarity_test_',dir=root/'outputs') as tmp:
            results = []
            for name in ('first','second'):
                with patch.object(diagnostic,'load_rgb',side_effect=after_only_loader), contextlib.redirect_stdout(io.StringIO()):
                    results.append(diagnostic.analyze(manifest,root/'outputs/fake_restoration_debug',
                        root/'outputs/restoration_diagnostics',Path(tmp)/name))
            self.assertEqual(results[0],results[1])
            for name in ('real_after_features.csv','fake_after_features.csv','comparison_summary.csv',
                         'classifier_oof_predictions.csv','classifier_diagnostic.json'):
                self.assertEqual((Path(tmp)/'first'/name).read_bytes(),(Path(tmp)/'second'/name).read_bytes())
            rows = results[0][0]
            self.assertEqual(sum(r['group']=='real' for r in rows),34)
            self.assertEqual(sum(r['group']=='fake' for r in rows),19)
            self.assertEqual(len(list((Path(tmp)/'first/distributions').glob('*.png'))),6)
            with self.assertRaises(FileExistsError):
                diagnostic.analyze(manifest,root/'outputs/fake_restoration_debug',root/'outputs/restoration_diagnostics',Path(tmp)/'first')


if __name__ == '__main__':
    unittest.main()
