"""
Debug visualisation for depth-aware local part ordering.

Left panel : body-HD crop with phalanx-level footprints (left hand warm colours,
             right hand cool colours), each interaction region (white contour),
             A-only VDA samples (green), B-only samples (magenta), label per pair.
Right panel: VDA map resampled onto the same window (brighter = closer).
"""
from typing import List, Optional

import cv2
import numpy as np

from .hand_parts import SEGMENTS, segment_part, segment_level

_PART_HUE = {'palm': 0, 'thumb': 1, 'index': 2, 'middle': 3, 'ring': 4, 'pinky': 5}
_LEFT = [(230, 80, 60), (245, 150, 40), (240, 210, 60), (200, 90, 150), (170, 60, 60), (250, 120, 120)]
_RIGHT = [(60, 110, 230), (40, 190, 220), (70, 200, 140), (120, 90, 220), (40, 140, 160), (130, 170, 250)]
_LEVEL_SCALE = {None: 1.0, 'prox': 0.65, 'mid': 0.85, 'dist': 1.0}


def segment_color(side: str, seg: str):
    base = (_LEFT if side == 'left' else _RIGHT)[_PART_HUE[segment_part(seg)]]
    k = _LEVEL_SCALE[segment_level(seg)]
    return tuple(int(c * k) for c in base)


def _window(instances, rasters, pad=30):
    pts = [np.array(i.center_px)[None] for i in instances]
    for r in rasters:
        H, W = r.shape
        pts.append(r.origin[None]); pts.append((r.origin + np.array([W, H]) * r.stride)[None])
    p = np.concatenate(pts, 0)
    lo, hi = p.min(0) - pad, p.max(0) + pad
    c, half = (lo + hi) / 2, max(hi - lo) / 2
    return c[0] - half, c[1] - half, c[0] + half, c[1] + half


def draw_depth_order_debug(image: np.ndarray, rasters, instances: List, depth_source=None, key: str = '',
                           M_c2o=None, out_size: int = 640, title: str = '') -> np.ndarray:
    """image: HxWx3 uint8 RGB body-HD crop. Returns an RGB panel image."""
    img = image.copy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    over = img.copy()
    for r in rasters:
        for (side, seg), m in r.masks.items():
            if not m.any():
                continue
            ys, xs = np.nonzero(m)
            for y, x in zip(ys, xs):
                x0, y0 = int(r.origin[0] + x * r.stride), int(r.origin[1] + y * r.stride)
                cv2.rectangle(over, (x0, y0), (x0 + r.stride - 1, y0 + r.stride - 1), segment_color(side, seg), -1)
    img = cv2.addWeighted(img, 0.45, over, 0.55, 0)
    for i in instances:
        col = (255, 255, 255) if i.confidence > 0 else (140, 140, 140)
        for c in (i.vis_region_contours or []):
            cv2.polylines(img, [np.round(c).astype(np.int32)], True, col, 1, cv2.LINE_AA)
        for pts, pc in ((i.vis_samples_a, (0, 255, 0)), (i.vis_samples_b, (255, 0, 255))):
            if pts is not None:
                for p in pts:
                    cv2.circle(img, (int(p[0]), int(p[1])), 1, pc, -1)
    x0, y0, x1, y1 = _window(instances, rasters) if (instances or rasters) else (0, 0, img.shape[1], img.shape[0])
    M = np.float32([[out_size / (x1 - x0), 0, -x0 * out_size / (x1 - x0)], [0, out_size / (y1 - y0), -y0 * out_size / (y1 - y0)]])
    left = cv2.warpAffine(img, M, (out_size, out_size), flags=cv2.INTER_NEAREST)
    right = np.zeros_like(left)
    if depth_source is not None:
        d = depth_source.crop_for_vis(key, M_c2o, out_size, x0, y0, x1, y1)
        if d is not None and np.isfinite(d).any():
            lo, hi = np.nanpercentile(d, 2), np.nanpercentile(d, 98)
            dn = np.nan_to_num(np.clip((d - lo) / (hi - lo + 1e-8), 0, 1))
            right = cv2.cvtColor(cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
            for i in instances:
                for c in (i.vis_region_contours or []):
                    cc = (c - [x0, y0]) * out_size / (x1 - x0)
                    cv2.polylines(right, [np.round(cc).astype(np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA)
    # labels
    for i in instances:
        c = (np.array(i.center_px) - [x0, y0]) * out_size / (x1 - x0)
        cv2.drawMarker(left, (int(c[0]), int(c[1])), (255, 255, 0), cv2.MARKER_CROSS, 10, 1)
    lines = [title] if title else []
    for i in sorted(instances, key=lambda t: -t.confidence)[:8]:
        if i.confidence > 0:
            tr = 'n/a' if not np.isfinite(i.tracker_dz) else ('agrees' if i.sign * i.tracker_dz < 0 else 'VIOLATES')
            lines.append(f'{i.name_a} vs {i.name_b} [{i.kind},{i.pair_type}] VDA front: {i.front_name()} '
                         f'conf {i.confidence:.2f} | tracker {tr} (dz_AB={i.tracker_dz * 1e3:+.1f}e-3)')
        else:
            lines.append(f'{i.name_a} vs {i.name_b} [{i.kind}] skipped: {i.skip_reason}')
    panel = np.concatenate([left, right], 1)
    bar = np.full((16 * max(len(lines), 1) + 8, panel.shape[1], 3), 20, np.uint8)
    for k, t in enumerate(lines):
        cv2.putText(bar, t[:150], (6, 16 * (k + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
    return np.concatenate([bar, panel], 0)
