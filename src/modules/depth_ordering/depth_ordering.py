"""
Depth-aware LOCAL part ordering from relative (VDA) depth.

Pipeline (per frame, no gradient; rebuilt every `refresh_every` steps):
  1. rasterise the 2D footprint of every phalanx-level segment of both hands
     (projected MANO triangles, see hand_parts.py) on a local canvas;
  2. find_interacting_part_pairs(): segment pairs whose footprints OVERLAP
     (each connected overlap component = one crossing instance) or, for
     inter-hand pairs, come within `interaction_2d_threshold` px;
  3. sample_local_vda(): inside a small region around that crossing, sample VDA
     ONLY at pixels covered by exactly one hand segment (A-only / B-only),
     eroded away from footprint borders and intersected with the matting mask.
     Overlap pixels are NEVER used as if both surfaces were visible (only the
     front surface is visible there); they are used solely for a consistency check;
  4. robust stats: median + IQR per side -> delta = d_A - d_B, confidence;
  5. the local tracker vertices of A and B = segment vertices projecting into
     the same local region (not the whole part).

Loss (differentiable, every step):
  z = proj_vertices[..., 2]  == camera-space z_view (VERIFIED in src/utils/graphics.py:
      GS_Camera.transform_points_to_ndc sets ndc[...,2] = w = (R X + T)_z because
      proj[3,2] = +1; screen transform only rescales x,y. Visible points have z > 0,
      LARGER z = FARTHER.)
  VDA: LARGER value = CLOSER.  s = +1 if d_A > d_B (A in front) else -1.
  A in front  <=>  z_A < z_B.
      l_k = relu(margin + s_k * (median(z_A,local) - median(z_B,local)))
      L_depth = sum_k c_k l_k / sum_k c_k

Only ORDINAL information of VDA is used. Nothing is regressed onto metric z.
LIMITATION: this constrains ordering only where both surfaces have visible
local support. It does NOT recover fully occluded fingers.
"""
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from .hand_parts import (HandPartIndex, SEGMENTS, SIDES, SIDE_SHORT, segment_part, segment_level)


@dataclass
class DepthOrderConfig:
    enable: bool = False
    lambda_depth_order: float = 100.0      # UNTUNED. z is in SMPL-X world units (~m); violations ~1e-2
    margin: float = 0.01                   # desired z gap (world units, ~1 cm)
    interaction_2d_threshold: float = 12.0 # px (body-HD crop) for inter-hand proximity pairs
    raster_stride: int = 2                 # px per footprint cell
    min_overlap_px: float = 24.0           # px^2, inter-hand overlap component
    min_overlap_px_intra: float = 48.0     # px^2, same-hand overlap component (stricter)
    overlap_full_px: float = 200.0         # px^2 at which overlap score saturates
    region_dilate_px: float = 10.0         # local region = overlap component dilated by this
    proximity_radius_px: float = 14.0      # local region radius for proximity pairs
    edge_erode_px: float = 2.0             # erode exclusive footprints (avoid border / mixed pixels)
    min_samples: int = 6                   # per side; fewer -> confidence 0
    full_samples: int = 30
    tie_sep: float = 0.5                   # |delta|/(iqrA+iqrB) below this -> confidence 0
    full_sep: float = 2.5
    rel_eps: float = 0.02                  # eps and tie floor relative to frame hand-depth spread
    rel_tie_delta: float = 0.01
    use_overlap_consistency: bool = True
    min_local_verts: int = 3
    refresh_every: int = 100               # rebuild pairs/targets every N steps (0 = once per batch)
    hand_pose_lr: Optional[float] = None   # override for left/right_hand_pose lr ONLY when enabled (e.g. 1e-5, 5e-5, 1e-4)
    debug_vis: bool = True
    max_vis_frames: int = 6
    grad_diag: bool = True
    log_every: int = 50
    max_canvas_cells: int = 640


# ----------------------------------------------------------------------------- utils

def _disk(r: int):
    r = max(int(r), 0)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _median_iqr(v: np.ndarray):
    if len(v) == 0:
        return np.nan, np.nan
    q25, q50, q75 = np.percentile(v, [25, 50, 75])
    return float(q50), float(q75 - q25)


def pair_part_type(seg_a: str, seg_b: str) -> str:
    pa, pb = segment_part(seg_a) == 'palm', segment_part(seg_b) == 'palm'
    if pa and pb:
        return 'palm-palm'
    if pa or pb:
        return 'palm-finger'
    return 'finger-finger'


def intra_pair_allowed(seg_a: str, seg_b: str) -> bool:
    """Same-hand pairs: skip kinematically adjacent segments that always touch in 2D."""
    pa, pb = segment_part(seg_a), segment_part(seg_b)
    if pa == pb:                       # same finger (adjacent phalanges) or palm-palm
        return False
    if pa == 'palm' and segment_level(seg_b) == 'prox':
        return False                   # palm vs proximal segment: joined at MCP / thenar
    if pb == 'palm' and segment_level(seg_a) == 'prox':
        return False
    return True


def _cross(u, v, w):
    return (v[..., 0] - u[..., 0]) * (w[..., 1] - u[..., 1]) - (v[..., 1] - u[..., 1]) * (w[..., 0] - u[..., 0])


def _raster_bucket(masks, tri, fs, lo, hi, K, H, W):
    oy, ox = np.meshgrid(np.arange(K), np.arange(K), indexing='ij')
    off = np.stack([ox.ravel(), oy.ravel()], 1)                 # [K*K,2] (x,y)
    cells = lo[:, None, :] + off[None]                          # [F,K*K,2]
    p = cells + 0.5
    a, b, c = tri[:, 0][:, None], tri[:, 1][:, None], tri[:, 2][:, None]
    w0, w1, w2 = _cross(a, b, p), _cross(b, c, p), _cross(c, a, p)
    inside = ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) | ((w0 <= 0) & (w1 <= 0) & (w2 <= 0))
    inside &= (np.abs(_cross(a, b, c)[:, 0]) > 1e-9)[:, None]
    inside &= (cells[..., 0] <= hi[:, None, 0]) & (cells[..., 1] <= hi[:, None, 1])
    inside &= (cells[..., 0] >= 0) & (cells[..., 0] < W) & (cells[..., 1] >= 0) & (cells[..., 1] < H)
    fi, ci = np.nonzero(inside)
    masks[fs[fi], cells[fi, ci, 1], cells[fi, ci, 0]] = True


def rasterize_segments(tri: np.ndarray, face_seg: np.ndarray, n_seg: int, H: int, W: int) -> np.ndarray:
    """
    Vectorised triangle rasteriser (cell-centre sampling, union semantics).
    tri: [F,3,2] in cell units (cell (i,j) centre at (i+.5, j+.5)); face_seg: [F] (-1 ignored).
    Returns masks [n_seg, H, W] bool.
    """
    masks = np.zeros((n_seg, H, W), dtype=bool)
    keep = face_seg >= 0
    tri, fs = tri[keep], face_seg[keep]
    if len(tri) == 0:
        return masks
    lo = np.ceil(tri.min(1) - 0.5).astype(np.int64)            # [F,2]
    hi = np.floor(tri.max(1) - 0.5).astype(np.int64)
    ext = hi - lo + 1
    ok = (ext > 0).all(1) & (hi[:, 0] >= 0) & (hi[:, 1] >= 0) & (lo[:, 0] < W) & (lo[:, 1] < H)
    tri, fs, lo, hi, ext = tri[ok], fs[ok], lo[ok], hi[ok], ext[ok]
    if len(tri) == 0:
        return masks
    # bucket triangles by bounding-box size so a few large triangles do not blow up memory
    emax = ext.max(1)
    lo_b = 0
    Kb = 8
    while lo_b < emax.max():
        sel = (emax > lo_b) & (emax <= Kb)
        if sel.any():
            _raster_bucket(masks, tri[sel], fs[sel], lo[sel], hi[sel], Kb, H, W)
        lo_b, Kb = Kb, Kb * 2
    return masks


# ----------------------------------------------------------------------------- data

@dataclass
class PairInstance:
    frame: int
    key: str
    side_a: str
    seg_a: str
    side_b: str
    seg_b: str
    kind: str                      # 'overlap' | 'proximity'
    pair_type: str                 # 'inter' | 'self'
    part_type: str                 # 'palm-finger' | 'finger-finger' | 'palm-palm'
    score: float                   # overlap / proximity score in [0,1]
    center_px: Tuple[float, float]
    area_px: float = 0.0
    dist_px: float = 0.0
    d_a: float = float('nan')
    d_b: float = float('nan')
    iqr_a: float = float('nan')
    iqr_b: float = float('nan')
    n_a: int = 0
    n_b: int = 0
    d_overlap: float = float('nan')
    n_overlap: int = 0
    delta: float = float('nan')    # d_a - d_b ; > 0 => A predicted CLOSER by VDA
    sign: float = 0.0              # +1 A front, -1 B front
    consistency: float = 1.0
    confidence: float = 0.0
    skip_reason: str = ''
    tracker_dz: float = float('nan')   # median z_A - median z_B at build time (>0 => tracker puts A behind)
    mano_ids_a: np.ndarray = field(default=None, repr=False)
    mano_ids_b: np.ndarray = field(default=None, repr=False)
    vis_samples_a: np.ndarray = field(default=None, repr=False)   # [n,2] px
    vis_samples_b: np.ndarray = field(default=None, repr=False)
    vis_region_contours: list = field(default=None, repr=False)   # list of [n,2] px polylines

    @property
    def name_a(self):
        return f'{SIDE_SHORT[self.side_a]}-{self.seg_a}'

    @property
    def name_b(self):
        return f'{SIDE_SHORT[self.side_b]}-{self.seg_b}'

    def front_name(self):
        return self.name_a if self.sign > 0 else self.name_b

    def to_dict(self):
        skip = {'mano_ids_a', 'mano_ids_b', 'vis_samples_a', 'vis_samples_b', 'vis_region_contours'}
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in skip}
        d['n_local_verts_a'] = 0 if self.mano_ids_a is None else int(len(self.mano_ids_a))
        d['n_local_verts_b'] = 0 if self.mano_ids_b is None else int(len(self.mano_ids_b))
        return d


@dataclass
class FrameRaster:
    origin: np.ndarray             # [2] px of canvas cell (0,0) corner
    stride: int
    masks: Dict[Tuple[str, str], np.ndarray]
    shape: Tuple[int, int]


@dataclass
class DepthOrderTargets:
    instances: List[PairInstance]            # all detected instances (active and skipped)
    active: List[PairInstance]
    frame: torch.Tensor = None               # [K]
    ids_a: torch.Tensor = None               # [K,Va] SMPL-X vertex ids
    valid_a: torch.Tensor = None
    ids_b: torch.Tensor = None
    valid_b: torch.Tensor = None
    sign: torch.Tensor = None                # [K]
    conf: torch.Tensor = None                # [K]
    type_codes: Dict[str, torch.Tensor] = None

    def __len__(self):
        return len(self.active)


# ----------------------------------------------------------------------------- builder

class DepthOrderTargetBuilder:
    def __init__(self, cfg: DepthOrderConfig, part_index: HandPartIndex, smplx2mano_ind: dict):
        self.cfg = cfg
        self.index = part_index
        self.m2s = {s: np.asarray(smplx2mano_ind[f'{s}_hand'], dtype=np.int64) for s in SIDES}
        self.seg_ids = {s: i for i, s in enumerate(SEGMENTS)}
        self.labels = part_index.labels
        self.faces = part_index.faces_all
        self.face_seg = part_index.face_labels

    # ---- step 1: footprints
    def rasterize_frame(self, hv: Dict[str, np.ndarray], sides: Sequence[str]) -> Optional[FrameRaster]:
        cfg, s = self.cfg, self.cfg.raster_stride
        pts = np.concatenate([hv[sd][self.labels >= 0, :2] for sd in sides], 0)
        pts = pts[np.isfinite(pts).all(1)]
        if len(pts) == 0:
            return None
        pad = cfg.region_dilate_px + cfg.proximity_radius_px + cfg.interaction_2d_threshold + 4 * s
        lo = pts.min(0) - pad
        hi = pts.max(0) + pad
        W, H = [int(np.ceil(v)) for v in (hi - lo) / s]
        if max(W, H) > cfg.max_canvas_cells:
            return None
        masks = {}
        for sd in sides:
            q = (hv[sd][:, :2] - lo) / s
            valid_v = hv[sd][:, 2] > 0
            fvalid = valid_v[self.faces].all(1)
            fseg = np.where(fvalid, self.face_seg, -1)
            m = rasterize_segments(q[self.faces], fseg, len(SEGMENTS), H, W)
            for i, seg in enumerate(SEGMENTS):
                masks[(sd, seg)] = m[i]
        return FrameRaster(origin=lo, stride=s, masks=masks, shape=(H, W))

    def cells_to_px(self, r: FrameRaster, cy: np.ndarray, cx: np.ndarray) -> np.ndarray:
        return np.stack([r.origin[0] + (cx + 0.5) * r.stride, r.origin[1] + (cy + 0.5) * r.stride], 1)

    # ---- step 2: interaction detection
    def find_interacting_part_pairs(self, r: FrameRaster, sides: Sequence[str], frame: int = 0, key: str = '') -> List[PairInstance]:
        cfg, s = self.cfg, r.stride
        nonempty = {k: m for k, m in r.masks.items() if m.any()}
        bbox = {}
        for k, m in nonempty.items():
            ys, xs = np.nonzero(m)
            bbox[k] = (xs.min(), ys.min(), xs.max(), ys.max())
        thr_c = cfg.interaction_2d_threshold / s
        cands = []
        if len(sides) == 2:
            for sa in SEGMENTS:
                for sb in SEGMENTS:
                    ka, kb = ('left', sa), ('right', sb)
                    if ka in bbox and kb in bbox:
                        cands.append((ka, kb, 'inter'))
        for sd in sides:
            for i, sa in enumerate(SEGMENTS):
                for sb in SEGMENTS[i + 1:]:
                    if intra_pair_allowed(sa, sb) and (sd, sa) in bbox and (sd, sb) in bbox:
                        cands.append(((sd, sa), (sd, sb), 'self'))
        out = []
        dist_cache = {}
        for ka, kb, ptype in cands:
            ba, bb = bbox[ka], bbox[kb]
            gap = max(ba[0] - bb[2], bb[0] - ba[2], ba[1] - bb[3], bb[1] - ba[3], 0)
            if gap > thr_c + 1:
                continue
            mA, mB = nonempty[ka], nonempty[kb]
            ov = mA & mB
            min_ov = (cfg.min_overlap_px if ptype == 'inter' else cfg.min_overlap_px_intra) / (s * s)
            common = dict(frame=frame, key=key, side_a=ka[0], seg_a=ka[1], side_b=kb[0], seg_b=kb[1],
                          pair_type=ptype, part_type=pair_part_type(ka[1], kb[1]))
            if ov.sum() >= min_ov:
                n, lab, stats, cents = cv2.connectedComponentsWithStats(ov.astype(np.uint8), connectivity=8)
                for c in range(1, n):
                    area = stats[c, cv2.CC_STAT_AREA]
                    if area < min_ov:
                        continue
                    cx, cy = cents[c]
                    ctr = (float(r.origin[0] + (cx + 0.5) * s), float(r.origin[1] + (cy + 0.5) * s))
                    inst = PairInstance(kind='overlap', score=float(min(1.0, area * s * s / cfg.overlap_full_px)),
                                        center_px=ctr, area_px=float(area * s * s), **common)
                    inst._comp = (lab == c)
                    out.append(inst)
            elif ptype == 'inter':
                if kb not in dist_cache:
                    dist_cache[kb] = cv2.distanceTransform((~mB).astype(np.uint8), cv2.DIST_L2, 5)
                dA = np.where(mA, dist_cache[kb], np.inf)
                j = int(np.argmin(dA))
                dmin = float(dA.flat[j])
                if dmin > thr_c:
                    continue
                ay, ax = divmod(j, mA.shape[1])
                by, bx = np.nonzero(mB)
                k2 = int(np.argmin((by - ay) ** 2 + (bx - ax) ** 2))
                cy, cx = (ay + by[k2]) / 2.0, (ax + bx[k2]) / 2.0
                ctr = (float(r.origin[0] + (cx + 0.5) * s), float(r.origin[1] + (cy + 0.5) * s))
                inst = PairInstance(kind='proximity', score=float(max(0.0, 1.0 - dmin / max(thr_c, 1e-6))),
                                    center_px=ctr, dist_px=dmin * s, **common)
                inst._center_cell = (cy, cx)
                out.append(inst)
        return out

    # ---- step 3/4: local visible VDA sampling + confidence
    def sample_local_vda(self, inst: PairInstance, r: FrameRaster, count_map: np.ndarray, body_cells: np.ndarray,
                         hv: Dict[str, np.ndarray], depth_source, M_c2o, s_frame: float, keep_vis: bool = False):
        cfg, s = self.cfg, r.stride
        H, W = r.shape
        if inst.kind == 'overlap':
            comp = inst._comp
            region = cv2.dilate(comp.astype(np.uint8), _disk(round(cfg.region_dilate_px / s))).astype(bool)
        else:
            cy, cx = inst._center_cell
            yy, xx = np.mgrid[0:H, 0:W]
            region = (yy - cy) ** 2 + (xx - cx) ** 2 <= (cfg.proximity_radius_px / s) ** 2
            comp = None
        er = _disk(round(cfg.edge_erode_px / s)) if cfg.edge_erode_px > 0 else None
        res = {}
        for tag, side, seg in (('a', inst.side_a, inst.seg_a), ('b', inst.side_b, inst.seg_b)):
            excl = r.masks[(side, seg)] & (count_map == 1)          # covered by THIS segment only
            if er is not None:
                excl = cv2.erode(excl.astype(np.uint8), er).astype(bool)
            vis = excl & region & body_cells
            cy_, cx_ = np.nonzero(vis)
            px = self.cells_to_px(r, cy_, cx_)
            vals, ok = depth_source.sample(inst.key, px, M_c2o)
            vals, px = vals[ok], px[ok]
            med, iqr = _median_iqr(vals)
            # local tracker vertices: segment vertices projecting into the local region
            vids = self.index.verts[seg]
            q = np.floor((hv[side][vids, :2] - r.origin) / s).astype(np.int64)
            inb = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H) & (hv[side][vids, 2] > 0)
            loc = np.zeros(len(vids), bool)
            loc[inb] = region[q[inb, 1], q[inb, 0]]
            res[tag] = dict(med=med, iqr=iqr, n=len(vals), vids=vids[loc], px=px)
        inst.d_a, inst.iqr_a, inst.n_a = res['a']['med'], res['a']['iqr'], res['a']['n']
        inst.d_b, inst.iqr_b, inst.n_b = res['b']['med'], res['b']['iqr'], res['b']['n']
        inst.mano_ids_a, inst.mano_ids_b = res['a']['vids'], res['b']['vids']
        if len(inst.mano_ids_a) and len(inst.mano_ids_b):
            inst.tracker_dz = float(np.median(hv[inst.side_a][inst.mano_ids_a, 2]) - np.median(hv[inst.side_b][inst.mano_ids_b, 2]))
        # overlap pixels: front surface only -> consistency check, never used as A or B depth
        if comp is not None and cfg.use_overlap_consistency:
            ce = cv2.erode(comp.astype(np.uint8), _disk(1)).astype(bool) & body_cells
            cy_, cx_ = np.nonzero(ce)
            vals, ok = depth_source.sample(inst.key, self.cells_to_px(r, cy_, cx_), M_c2o)
            vals = vals[ok]
            inst.n_overlap = int(len(vals))
            if len(vals):
                inst.d_overlap = float(np.median(vals))
        if keep_vis:
            inst.vis_samples_a, inst.vis_samples_b = res['a']['px'][::2], res['b']['px'][::2]
            cs, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            inst.vis_region_contours = [r.origin + (c[:, 0, :] + 0.5) * s for c in cs]
        self.compute_confidence(inst, s_frame)
        return inst

    def compute_confidence(self, inst: PairInstance, s_frame: float):
        """
        conf = c_sep * c_n * c_geo * c_cons  in [0,1]
          sep   = |d_A - d_B| / (iqr_A + iqr_B + eps),  eps = rel_eps * s_frame
          c_sep = clip((sep - tie_sep) / (full_sep - tie_sep), 0, 1)     (near tie -> 0)
          c_n   = 0 if min(n_A, n_B) < min_samples else min(1, min(n)/full_samples)
          c_geo = overlap / proximity score
          c_cons= for overlap pairs: |d_O - d_back| / (|d_O - d_front| + |d_O - d_back|);
                  set to 0 if < 0.5 (overlap looks like the BACK surface -> contradiction)
        s_frame = p90 - p10 of VDA over all visible hand pixels in the frame (scale-invariance).
        """
        cfg = self.cfg
        n = min(inst.n_a, inst.n_b)
        if n < cfg.min_samples or not np.isfinite(inst.d_a) or not np.isfinite(inst.d_b):
            inst.skip_reason = 'too_few_visible_samples'
            inst.confidence = 0.0
            return
        if len(inst.mano_ids_a) < cfg.min_local_verts or len(inst.mano_ids_b) < cfg.min_local_verts:
            inst.skip_reason = 'too_few_local_vertices'
            inst.confidence = 0.0
            return
        inst.delta = inst.d_a - inst.d_b
        inst.sign = 1.0 if inst.delta > 0 else -1.0
        sf = max(s_frame, 1e-8)
        if abs(inst.delta) < cfg.rel_tie_delta * sf:
            inst.skip_reason = 'near_tie'
            inst.confidence = 0.0
            return
        sep = abs(inst.delta) / (inst.iqr_a + inst.iqr_b + cfg.rel_eps * sf)
        c_sep = float(np.clip((sep - cfg.tie_sep) / (cfg.full_sep - cfg.tie_sep), 0, 1))
        c_n = float(min(1.0, n / cfg.full_samples))
        c_cons = 1.0
        if inst.kind == 'overlap' and cfg.use_overlap_consistency and np.isfinite(inst.d_overlap) \
                and inst.n_overlap >= max(1, cfg.min_samples // 2):
            d_front, d_back = (inst.d_a, inst.d_b) if inst.sign > 0 else (inst.d_b, inst.d_a)
            ef, eb = abs(inst.d_overlap - d_front), abs(inst.d_overlap - d_back)
            c_cons = float(eb / (ef + eb + 1e-8))
            if c_cons < 0.5:
                c_cons = 0.0
        inst.consistency = c_cons
        inst.confidence = float(c_sep * c_n * inst.score * c_cons)
        if inst.confidence <= 0:
            inst.skip_reason = 'near_tie' if c_sep == 0 else ('overlap_inconsistent' if c_cons == 0 else 'low_score')

    # ---- per frame
    def process_frame(self, f: int, key: str, hv: Dict[str, np.ndarray], depth_source, M_c2o=None,
                      body_mask: Optional[np.ndarray] = None, keep_vis: bool = False):
        cfg = self.cfg
        # decide canvases: one shared canvas if the hands are close, else one per hand (intra only)
        bb = {}
        for sd in SIDES:
            p = hv[sd][self.labels >= 0, :2]
            p = p[np.isfinite(p).all(1)]
            if len(p):
                bb[sd] = (p.min(0), p.max(0))
        groups = []
        if len(bb) == 2:
            (la, ha), (lb, hb) = bb['left'], bb['right']
            gap = max(la[0] - hb[0], lb[0] - ha[0], la[1] - hb[1], lb[1] - ha[1], 0)
            groups = [('left', 'right')] if gap <= cfg.interaction_2d_threshold + 2 * cfg.raster_stride else [('left',), ('right',)]
        else:
            groups = [(sd,) for sd in bb]
        insts, rasters = [], []
        for sides in groups:
            r = self.rasterize_frame(hv, sides)
            if r is None:
                continue
            count_map = np.zeros(r.shape, np.int16)
            for m in r.masks.values():
                count_map += m
            if body_mask is not None:
                H, W = r.shape
                yy, xx = np.mgrid[0:H, 0:W]
                px = np.stack([r.origin[0] + (xx.ravel() + 0.5) * r.stride, r.origin[1] + (yy.ravel() + 0.5) * r.stride], 1)
                bx = np.clip(np.round(px[:, 0]).astype(int), 0, body_mask.shape[1] - 1)
                by = np.clip(np.round(px[:, 1]).astype(int), 0, body_mask.shape[0] - 1)
                inside = (px[:, 0] >= 0) & (px[:, 0] < body_mask.shape[1]) & (px[:, 1] >= 0) & (px[:, 1] < body_mask.shape[0])
                body_cells = ((body_mask[by, bx] > 0.5) & inside).reshape(H, W)
            else:
                body_cells = np.ones(r.shape, bool)
            pairs = self.find_interacting_part_pairs(r, sides, f, key)
            if pairs:
                hy, hx = np.nonzero((count_map > 0) & body_cells)
                sel = np.linspace(0, len(hy) - 1, min(len(hy), 4000)).astype(int) if len(hy) else np.zeros(0, int)
                vals, ok = depth_source.sample(key, self.cells_to_px(r, hy[sel], hx[sel]), M_c2o)
                vals = vals[ok]
                s_frame = float(np.percentile(vals, 90) - np.percentile(vals, 10)) if len(vals) > 10 else 0.0
                for inst in pairs:
                    self.sample_local_vda(inst, r, count_map, body_cells, hv, depth_source, M_c2o, s_frame, keep_vis)
                    for attr in ('_comp', '_center_cell'):
                        if hasattr(inst, attr):
                            delattr(inst, attr)
                insts.extend(pairs)
            rasters.append(r)
        return insts, rasters

    # ---- whole batch
    @torch.no_grad()
    def build(self, proj_vertices: torch.Tensor, frame_keys: Sequence[str], depth_source, M_c2o_hd=None,
              body_masks=None, keep_vis_frames: Sequence[int] = ()) -> DepthOrderTargets:
        pv = proj_vertices.detach().float().cpu().numpy()
        M = None if M_c2o_hd is None else (M_c2o_hd.detach().cpu().numpy() if torch.is_tensor(M_c2o_hd) else np.asarray(M_c2o_hd))
        all_insts = []
        keep_vis_frames = set(keep_vis_frames)
        for f, key in enumerate(frame_keys):
            if not depth_source.has(key):
                continue
            hv = {sd: pv[f, self.m2s[sd]] for sd in SIDES}
            bm = None if body_masks is None else body_masks[f]
            insts, _ = self.process_frame(f, key, hv, depth_source, None if M is None else M[f], bm, keep_vis=True)
            all_insts.extend(insts)
        return self.pack(all_insts, proj_vertices.device)

    def pack(self, insts: List[PairInstance], device) -> DepthOrderTargets:
        active = [i for i in insts if i.confidence > 0]
        t = DepthOrderTargets(instances=insts, active=active)
        if not active:
            return t
        K = len(active)
        Va = max(len(i.mano_ids_a) for i in active)
        Vb = max(len(i.mano_ids_b) for i in active)
        ids_a = np.zeros((K, Va), np.int64); va = np.zeros((K, Va), bool)
        ids_b = np.zeros((K, Vb), np.int64); vb = np.zeros((K, Vb), bool)
        for k, i in enumerate(active):
            ids_a[k, :len(i.mano_ids_a)] = self.m2s[i.side_a][i.mano_ids_a]; va[k, :len(i.mano_ids_a)] = True
            ids_b[k, :len(i.mano_ids_b)] = self.m2s[i.side_b][i.mano_ids_b]; vb[k, :len(i.mano_ids_b)] = True
        t.frame = torch.tensor([i.frame for i in active], dtype=torch.long, device=device)
        t.ids_a, t.valid_a = torch.from_numpy(ids_a).to(device), torch.from_numpy(va).to(device)
        t.ids_b, t.valid_b = torch.from_numpy(ids_b).to(device), torch.from_numpy(vb).to(device)
        t.sign = torch.tensor([i.sign for i in active], dtype=torch.float32, device=device)
        t.conf = torch.tensor([i.confidence for i in active], dtype=torch.float32, device=device)
        t.type_codes = {
            'inter': torch.tensor([i.pair_type == 'inter' for i in active], device=device),
            'self': torch.tensor([i.pair_type == 'self' for i in active], device=device),
            'palm-finger': torch.tensor([i.part_type == 'palm-finger' for i in active], device=device),
            'finger-finger': torch.tensor([i.part_type == 'finger-finger' for i in active], device=device),
        }
        return t


# ----------------------------------------------------------------------------- loss

def local_median_z(z: torch.Tensor, frame: torch.Tensor, ids: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Median camera z of each instance's local vertex set. z: [B,V] -> [K]. Differentiable (torch.nanquantile)."""
    zz = z[frame[:, None], ids]
    zz = torch.where(valid, zz, torch.full_like(zz, float('nan')))
    return torch.nanquantile(zz, 0.5, dim=1)


def depth_order_loss(proj_vertices: torch.Tensor, targets: DepthOrderTargets, margin: float):
    """
    L = sum_k c_k * relu(margin + s_k * (medz_A - medz_B)) / sum_k c_k
    s_k = +1 when VDA says A is in front (A must have SMALLER camera z).
    """
    stats = {'n_pairs': len(targets), 'mean_conf': 0.0, 'n_wrong_order': 0, 'n_margin_violated': 0,
             'n_inter': 0, 'n_self': 0, 'n_palm_finger': 0, 'n_finger_finger': 0}
    if len(targets) == 0:
        return proj_vertices.new_zeros(()), stats
    z = proj_vertices[..., 2]
    za = local_median_z(z, targets.frame, targets.ids_a, targets.valid_a)
    zb = local_median_z(z, targets.frame, targets.ids_b, targets.valid_b)
    signed = targets.sign * (za - zb)                 # > 0  <=> tracker order contradicts VDA
    per = torch.relu(margin + signed)
    loss = (targets.conf * per).sum() / (targets.conf.sum() + 1e-8)
    with torch.no_grad():
        stats.update({
            'mean_conf': float(targets.conf.mean()),
            'n_wrong_order': int((signed > 0).sum()),
            'n_margin_violated': int((per > 0).sum()),
            'n_inter': int(targets.type_codes['inter'].sum()),
            'n_self': int(targets.type_codes['self'].sum()),
            'n_palm_finger': int(targets.type_codes['palm-finger'].sum()),
            'n_finger_finger': int(targets.type_codes['finger-finger'].sum()),
            'mean_signed_dz': float(signed.mean()),
        })
    return loss, stats
