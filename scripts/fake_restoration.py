"""Seeded, GT-aware relocation of actual restoration mask components."""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

COUNT_PROBABILITIES = (0.25, 0.35, 0.25, 0.15)
METHODS = {"TELEA": cv2.INPAINT_TELEA, "NS": cv2.INPAINT_NS}


@dataclass(frozen=True)
class RestorationSettings:
    method: str
    radius: float
    dilation: int

    def __post_init__(self):
        if self.method not in METHODS or not math.isfinite(self.radius) or self.radius <= 0 or self.dilation < 0:
            raise ValueError("Unsupported restoration settings")


def inpaint_rgb(rgb, mask, settings):
    """Same RGB -> BGR -> cv2.inpaint -> RGB call as run_poc.py.

    The supplied mask already includes the real pipeline's dilation.
    """
    if not np.any(mask):
        return rgb.copy()
    return cv2.cvtColor(cv2.inpaint(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        (mask > 0).astype(np.uint8) * 255,
        settings.radius, METHODS[settings.method]), cv2.COLOR_BGR2RGB)


def bbox_mask(shape, boxes, margin=0):
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("Safety margin must be finite and nonnegative")
    h, w = shape
    result = np.zeros(shape, bool)
    for x0, y0, x1, y1 in boxes:
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1)) or not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError(f"Invalid GT box: {(x0, y0, x1, y1)}")
        result[max(0, math.floor(y0-margin)):min(h, math.ceil(y1+margin)),
               max(0, math.floor(x0-margin)):min(w, math.ceil(x1+margin))] = True
    return result


def mask_templates(real_mask):
    """Crop each 8-connected real component without changing its holes/shape."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats((real_mask > 0).astype(np.uint8), 8)
    result = []
    for i in range(1, count):
        x, y, w, h, _ = stats[i]
        result.append((labels[y:y+h, x:x+w] == i).astype(np.uint8) * 255)
    return result


def augment_fake_restoration(rgb, real_mask, boxes, *, split, rng, settings,
                             safety_margin=10, scale_range=(1.0, 1.0), max_attempts=500):
    """Return image, union fake mask, and placement audit. Never produces labels.

    Reject the entire rectangular footprint, including holes, when it contains
    protected pixels. This also prevents a hollow fake marking enclosing a GT.
    Exhausted searches skip the requested fake region; constraints are never relaxed.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {split}")
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or real_mask.shape != rgb.shape[:2]:
        raise ValueError("Expected uint8 RGB image and matching 2D restoration mask")
    lo, hi = scale_range
    if not (0.8 <= lo <= hi <= 1.2) or max_attempts < 1:
        raise ValueError("Scale must be within [0.8, 1.2]; max_attempts must be positive")
    gt = bbox_mask(real_mask.shape, boxes)
    safety = bbox_mask(real_mask.shape, boxes, safety_margin)
    real = real_mask > 0
    fake = np.zeros(real.shape, bool)
    audit = dict(requested_count=0, applied_count=0, skipped_count=0,
                 rejected_gt_or_safety=0, rejected_real=0, rejected_fake=0,
                 rejected_size=0, placements=[])
    if split == "train":
        audit["requested_count"] = int(rng.choice(4, p=COUNT_PROBABILITIES))
        templates = mask_templates(real_mask)
        h, w = real.shape
        for _ in range(audit["requested_count"]):
            if not templates:
                break
            for _attempt in range(max_attempts):
                template_id = int(rng.integers(len(templates)))
                patch = templates[template_id]
                scale = float(rng.uniform(lo, hi)) if lo != hi else lo
                if scale != 1:
                    patch = cv2.resize(patch, (max(1, round(patch.shape[1]*scale)),
                                              max(1, round(patch.shape[0]*scale))), interpolation=cv2.INTER_NEAREST)
                ph, pw = patch.shape
                if ph > h or pw > w or not np.any(patch):
                    audit["rejected_size"] += 1
                    continue
                x, y = int(rng.integers(w-pw+1)), int(rng.integers(h-ph+1))
                roi = np.s_[y:y+ph, x:x+pw]
                collisions = (safety[roi].any(), real[roi].any(), fake[roi].any())
                for key, hit in zip(("rejected_gt_or_safety", "rejected_real", "rejected_fake"), collisions):
                    audit[key] += int(hit)
                if any(collisions):
                    continue
                fake[roi] |= patch > 0
                audit["placements"].append(dict(template_id=template_id, x=x, y=y, width=pw, height=ph, scale=scale))
                break
        audit["applied_count"] = len(audit["placements"])
        audit["skipped_count"] = audit["requested_count"] - audit["applied_count"]
    for name, forbidden in (("gt", gt), ("safety", safety), ("real", real)):
        audit[f"overlap_{name}_pixels"] = int(np.count_nonzero(fake & forbidden))
        audit[f"overlap_{name}_count"] = int(audit[f"overlap_{name}_pixels"] > 0)
        if audit[f"overlap_{name}_pixels"]:
            raise RuntimeError(f"Unsafe fake restoration overlaps {name}")
    return inpaint_rgb(rgb, fake, settings), fake.astype(np.uint8)*255, audit
