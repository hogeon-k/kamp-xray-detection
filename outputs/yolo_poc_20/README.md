# YOLO TELEA PoC dataset (20 images)

This is a small pipeline-validation split, not a statistically reliable evaluation split.

## Provenance

- Source images: TELEA, dilation=0 px, radius=3 only
- GT priority: representative Pascal VOC XML, then verified YOLO TXT
- Class mapping: `defect = 0`
- Split sizes: train=14, val=3, test=3
- Deterministic split seed: 42
- Color-overlay components were not used as labels.
- Original images and existing inpainting outputs were read only.

## Summary

| split | images | bboxes | machine 1 | machine 2 | machine 3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 14 | 24 | 2 | 3 | 9 |
| val | 3 | 5 | 1 | 1 | 1 |
| test | 3 | 5 | 1 | 1 | 1 |
| total | 20 | 34 | 4 | 5 | 11 |

- Total images: 20
- Total bbox: 34
- Mean bbox/image: 1.700
- Min bbox/image: 1
- Max bbox/image: 3
- Review required: 0
- Maximum GT -> YOLO -> pixel round-trip error: 0.0000000384 px

`previews/` contains cyan-box visual checks. The clean images under `images/` contain no drawn boxes.
