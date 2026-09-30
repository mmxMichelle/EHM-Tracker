"""
Interface for PRECOMPUTED Video-Depth-Anything (VDA) relative depth maps.

VDA is NOT executed inside the tracker. The tracker only consumes arrays.

Convention used everywhere downstream: LARGER value = CLOSER to the camera
(VDA's raw disparity-like output). If your maps are metric-like depth
(larger = farther), pass larger_is_closer=False and they are negated on load.
Values are only ever compared ORDINALLY within a frame; they are never regressed
onto SMPL-X/MANO z.

Supported inputs
----------------
* ndarray / tensor [T, H, W]
* .npy  [T, H, W]  (memory-mapped)
* .npz  with key 'depths' (VDA default), 'depth', or the first array
* directory of per-frame .npy / .png / .exr files (sorted by name)

Frame mapping
-------------
Tracker frame keys (``frame_000123``) are NOT reliably ``video_idx / interval``
because the tracker skips frames that fail detection / hand checks.
``frame_map`` maps tracker key -> VDA frame index. For the first GPU experiment,
pass an explicit JSON: {"frame_000000": 0, "frame_000001": 6, ...} or
{"frames": {...}, "image_size": [H, W]}. When depth ordering is enabled the
pipeline writes ``<saving_root>/frame_index_map.json`` during extraction.

Coordinate spaces
-----------------
* space='body_hd'  : maps are in the 1024x1024 body-HD crop (same pixels as proj_vertices[..., :2]).
* space='original' : maps are in original video frame pixels (possibly resized);
                     crop pixels are mapped back with body_crop['M_c2o-hd'] (3x3),
                     then scaled by depth_res / original_size.
"""
import glob
import json
import os
import os.path as osp
from typing import Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch


def _load_array(path: str) -> np.ndarray:
    if osp.isdir(path):
        files = sorted([f for f in glob.glob(osp.join(path, '*')) if f.split('.')[-1].lower() in ('npy', 'png', 'exr', 'tiff', 'tif')])
        if not files:
            raise FileNotFoundError(f'no depth frames in {path}')
        frames = []
        for f in files:
            if f.endswith('.npy'):
                frames.append(np.load(f).astype(np.float32))
            else:
                frames.append(cv2.imread(f, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH).astype(np.float32))
        return np.stack(frames, 0)
    if path.endswith('.npy'):
        return np.load(path, mmap_mode='r')
    if path.endswith('.npz'):
        z = np.load(path)
        for k in ('depths', 'depth', 'disparity'):
            if k in z.files:
                return z[k]
        return z[z.files[0]]
    raise ValueError(f'unsupported depth file: {path}')


def bilinear(img: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Bilinear lookup of img [H,W] at float pixel coords p [N,2] (x,y); border-clamped."""
    H, W = img.shape
    x = np.clip(p[:, 0], 0, W - 1)
    y = np.clip(p[:, 1], 0, H - 1)
    x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, W - 1); y1 = np.minimum(y0 + 1, H - 1)
    wx = (x - x0).astype(np.float32); wy = (y - y0).astype(np.float32)
    return ((img[y0, x0] * (1 - wx) + img[y0, x1] * wx) * (1 - wy) +
            (img[y1, x0] * (1 - wx) + img[y1, x1] * wx) * wy).astype(np.float32)


def load_frame_map(path_or_dict) -> Tuple[Optional[Dict[str, int]], Optional[Tuple[int, int]]]:
    if path_or_dict is None:
        return None, None
    d = path_or_dict
    if isinstance(path_or_dict, str):
        with open(path_or_dict, 'r') as f:
            d = json.load(f)
    if 'frames' in d:
        size = tuple(d['image_size']) if d.get('image_size') is not None else None
        return {k: int(v) for k, v in d['frames'].items()}, size
    return {k: int(v) for k, v in d.items()}, None


class DepthSource:
    def __init__(self, depth: Union[np.ndarray, torch.Tensor, str], frame_map=None, space: str = 'original',
                 larger_is_closer: bool = True, original_size: Optional[Tuple[int, int]] = None):
        if isinstance(depth, str):
            depth = _load_array(depth)
        if torch.is_tensor(depth):
            depth = depth.detach().cpu().numpy()
        assert depth.ndim == 3, f'depth maps must be [T,H,W], got {depth.shape}'
        assert space in ('original', 'body_hd'), space
        self.depth = depth
        self.sign = 1.0 if larger_is_closer else -1.0
        self.space = space
        fmap, size = load_frame_map(frame_map)
        self.frame_map = fmap
        self.original_size = tuple(original_size) if original_size is not None else size  # (H, W)

    def __len__(self):
        return self.depth.shape[0]

    def index_of(self, key: str) -> Optional[int]:
        if self.frame_map is None:
            return None
        idx = self.frame_map.get(key)
        if idx is None or idx < 0 or idx >= len(self):
            return None
        return idx

    def has(self, key: str) -> bool:
        return self.index_of(key) is not None

    def frame(self, key: str) -> Optional[np.ndarray]:
        idx = self.index_of(key)
        if idx is None:
            return None
        return self.sign * np.asarray(self.depth[idx], dtype=np.float32)

    def to_depth_pixels(self, uv_hd: np.ndarray, M_c2o_hd: Optional[np.ndarray]) -> np.ndarray:
        """Map body-HD crop pixel coords [N,2] to depth-map pixel coords [N,2]."""
        uv = np.asarray(uv_hd, dtype=np.float64)
        if self.space == 'body_hd':
            return uv
        assert M_c2o_hd is not None, "space='original' needs body_crop['M_c2o-hd']"
        M = np.asarray(M_c2o_hd, dtype=np.float64).reshape(3, 3)
        uvo = uv @ M[:2, :2].T + M[:2, 2]
        H, W = self.depth.shape[1:]
        if self.original_size is not None:
            h0, w0 = self.original_size
            uvo = uvo * np.array([W / w0, H / h0])
        return uvo

    def sample(self, key: str, uv_hd: np.ndarray, M_c2o_hd: Optional[np.ndarray] = None):
        """Bilinear VDA values at body-HD pixel coords. Returns (values [N], valid [N])."""
        d = self.frame(key)
        n = len(uv_hd)
        if d is None or n == 0:
            return np.zeros(n, np.float32), np.zeros(n, bool)
        p = self.to_depth_pixels(uv_hd, M_c2o_hd).astype(np.float32)
        H, W = d.shape
        valid = (p[:, 0] >= 0) & (p[:, 0] <= W - 1) & (p[:, 1] >= 0) & (p[:, 1] <= H - 1)
        vals = bilinear(d, p)
        valid &= np.isfinite(vals)
        return vals, valid

    def crop_for_vis(self, key: str, M_c2o_hd, out_size: int, x0: float, y0: float, x1: float, y1: float):
        """Resample the depth map onto a body-HD window [x0,x1]x[y0,y1] (for debug images)."""
        d = self.frame(key)
        if d is None:
            return None
        xs = np.linspace(x0, x1, out_size)
        ys = np.linspace(y0, y1, out_size)
        gx, gy = np.meshgrid(xs, ys)
        p = self.to_depth_pixels(np.stack([gx.ravel(), gy.ravel()], 1), M_c2o_hd).astype(np.float32)
        v = cv2.remap(d, p[:, 0].reshape(out_size, out_size), p[:, 1].reshape(out_size, out_size),
                      interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=float('nan'))
        return v


def load_depth_source_for_video(cfg, video_name: str, saving_root: str, log=print) -> Optional[DepthSource]:
    """
    Resolve precomputed VDA maps + frame map for one video from DataPreparationConfig fields.
    Returns None (depth ordering skipped for this video) when anything is missing.
    """
    root = getattr(cfg, 'vda_depth_root', '') or ''
    if not root:
        log('depth-order: vda_depth_root not set; depth ordering disabled for this video')
        return None
    cands = [root] if osp.isfile(root) else [
        osp.join(root, f'{video_name}{s}') for s in ('.npy', '.npz', '_depths.npz', '_depths.npy', '')]
    depth_path = next((c for c in cands if osp.exists(c)), None)
    if depth_path is None:
        log(f'depth-order: no VDA depth found for {video_name} under {root}; disabled for this video')
        return None
    fm = getattr(cfg, 'vda_frame_map', '') or ''
    if fm and osp.isdir(fm):
        fm = osp.join(fm, f'{video_name}.json')
    if not fm:
        fm = osp.join(saving_root, 'frame_index_map.json')
    if not osp.exists(fm):
        log(f'depth-order: frame map {fm} not found; disabled for this video (pass vda_frame_map explicitly)')
        return None
    src = DepthSource(depth_path, frame_map=fm, space=getattr(cfg, 'vda_depth_space', 'original'),
                      larger_is_closer=getattr(cfg, 'vda_larger_is_closer', True))
    log(f'depth-order: VDA {depth_path} {tuple(src.depth.shape)} | frame map {fm} ({len(src.frame_map)} keys) | space={src.space}')
    return src
