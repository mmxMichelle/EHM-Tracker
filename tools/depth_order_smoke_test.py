"""
Smoke tests for depth-aware local part ordering.

  # CPU, no licensed SMPL-X/FLAME assets, no GPU needed:
  python tools/depth_order_smoke_test.py --synthetic --out /tmp/depth_order_synth

  # GPU, on an already-tracked clip (EHM output dir of ONE video) + precomputed VDA:
  python tools/depth_order_smoke_test.py --clip \
      --track_dir outputs/test_data/<video_name> \
      --vda /path/<video_name>_depths.npz --frame_map /path/frame_map.json --vda_space original \
      --n_frames 8 --steps 60 --hand_pose_lrs 1e-5 5e-5 1e-4

The clip mode checks:
  1. new code with depth disabled == ORIGINAL baseline refiner (loaded from git commit 55cfe1c)
  2. enabling depth ordering does not crash
  3. loss_depth_order > 0 when an active pair is violated (also forced with flipped VDA)
  4. gradient norms reach left/right_hand_pose
  5. debug images are written to <out>/visual_results/vis_depth_order
  + camera-z convention: proj[...,2] == (R X + T)_z and > 0
"""
import argparse
import ast
import copy
import json
import os
import os.path as osp
import pickle
import subprocess
import sys

import numpy as np
import torch

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.modules.depth_ordering import (DepthOrderConfig, DepthOrderTargetBuilder, DepthSource, HandPartIndex,
                                        SEGMENTS, build_mano_part_labels, depth_order_loss, gather_hand_parts,
                                        local_median_z, rasterize_segments)
from src.modules.depth_ordering.depth_ordering import FrameRaster, PairInstance
from src.modules.depth_ordering.depth_vis import draw_depth_order_debug

PASS, FAIL = [], []


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")


# ============================================================================ helpers

def load_mano_pkl(path):
    """MANO pkl without chumpy (stub unpickler). Only used by this test."""
    class Ch:
        def __setstate__(self, st):
            self.__dict__.update(st if isinstance(st, dict) else {'x': st})

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            return Ch if mod.startswith('chumpy') else super().find_class(mod, name)

    d = U(open(path, 'rb'), encoding='latin1').load()

    def A(v):
        if isinstance(v, Ch):
            v = v.__dict__.get('x', v.__dict__.get('a'))
        if hasattr(v, 'todense'):
            v = v.todense()
        return np.asarray(v)
    out = {k: A(v) for k, v in d.items() if k in ('weights', 'f', 'v_template', 'J_regressor', 'kintree_table', 'posedirs')}
    out['shapedirs'] = np.zeros((778, 3, 10))   # chumpy Select object; betas are zero in these tests
    return out


class _Obj:
    pass


def fake_ehm(mano_dir):
    m = load_mano_pkl(osp.join(mano_dir, 'MANO_RIGHT.pkl'))
    ehm = _Obj(); ehm.mano = _Obj(); ehm.smplx = _Obj()
    ehm.mano.lbs_weights = torch.tensor(m['weights'], dtype=torch.float32)
    ehm.mano.faces_tensor = torch.tensor(m['f'].astype(np.int64))
    ehm.mano.selected_vert_ids = np.load(osp.join(mano_dir, 'selected_hand_ver.npy'))
    # synthetic 'SMPL-X' vertex space: left hand = 0..777, right hand = 778..1555
    ehm.smplx.smplx2mano_ind = {'left_hand': np.arange(778), 'right_hand': np.arange(778, 1556)}
    return ehm, m


def rot_from_axes(src_axes, dst_axes):
    S = np.stack(src_axes, 1); D = np.stack(dst_axes, 1)
    return D @ np.linalg.inv(S)


def orthonormal_frame(m):
    v = m['v_template']; J = m['J_regressor'] @ v
    df = v[443] - J[0]; df /= np.linalg.norm(df)                     # wrist -> middle tip
    ds = J[1] - J[7]; ds -= ds.dot(df) * df; ds /= np.linalg.norm(ds)  # pinky MCP -> index MCP
    n = np.cross(df, ds)
    return df, ds, n


def zbuffer_render(verts_list, faces, f, c, size, bg_z=3.0, n_sub=36):
    """Dense barycentric point-splat z-buffer (test only)."""
    zb = np.full((size, size), bg_z, np.float64)
    u = np.array([(i, j) for i in range(n_sub + 1) for j in range(n_sub + 1 - i)], float) / n_sub
    bary = np.stack([u[:, 0], u[:, 1], 1 - u.sum(1)], 1)                # [P,3]
    for v in verts_list:
        tri = v[faces]                                                   # [F,3,3]
        pts = np.einsum('pk,fkd->fpd', bary, tri).reshape(-1, 3)
        x = np.round(f * pts[:, 0] / pts[:, 2] + c - 0.5).astype(int)
        y = np.round(f * pts[:, 1] / pts[:, 2] + c - 0.5).astype(int)
        ok = (x >= 0) & (x < size) & (y >= 0) & (y < size)
        np.minimum.at(zb, (y[ok], x[ok]), pts[ok, 2])
    return zb


# ============================================================================ synthetic

def run_synthetic(out_dir, mano_dir):
    import cv2
    from src.modules.smplx.lbs import lbs
    os.makedirs(out_dir, exist_ok=True)
    ehm, m = fake_ehm(mano_dir)

    # ---------------------------------------------------------------- T1 part labels (P0)
    labels = build_mano_part_labels(ehm.mano.lbs_weights, ehm.mano.selected_vert_ids)
    index = HandPartIndex.from_mano(ehm.mano)
    summ = index.summary()
    print('segment vertex counts:', summ)
    check('T1a all 16 phalanx-level segments non-empty', all(summ[s] > 0 for s in SEGMENTS) and len(SEGMENTS) == 16)
    check('T1b wrist-cut vertices ignored (-1)', int((labels < 0).sum()) == 778 - len(ehm.mano.selected_vert_ids),
          f'({int((labels < 0).sum())} ignored)')
    ml = load_mano_pkl(osp.join(mano_dir, 'MANO_LEFT.pkl'))
    check('T1c MANO_LEFT skinning argmax == MANO_RIGHT', np.array_equal(ml['weights'].argmax(1), m['weights'].argmax(1)))
    raw = build_mano_part_labels(ehm.mano.lbs_weights)
    exp = {'palm': 199, 'index_prox': 33, 'index_mid': 34, 'index_dist': 54, 'thumb_prox': 20, 'thumb_mid': 31, 'thumb_dist': 54}
    check('T1d counts match audited skinning argmax', all(int((raw == SEGMENTS.index(k)).sum()) == v for k, v in exp.items()))
    tips = {'thumb': 744, 'index': 320, 'middle': 443, 'ring': 554, 'pinky': 671}   # MANO.py fingertip vertex ids
    check('T1e MANO.py fingertip vertices land in *_dist segments',
          all(SEGMENTS[labels[v]] == f'{k}_dist' for k, v in tips.items()),
          str({k: SEGMENTS[labels[v]] for k, v in tips.items()}))

    # ---------------------------------------------------------------- T2 rasteriser union semantics
    rng = np.random.RandomState(0)
    tri = rng.rand(60, 1, 2) * 40 + (rng.rand(60, 3, 2) - 0.5) * 14   # mesh-like small triangles
    tri[:3] = rng.rand(3, 3, 2) * 40                                 # plus a few large ones (> kmax bucket)
    fs = rng.randint(0, 2, 60)
    mk = rasterize_segments(tri, fs, 2, 40, 40)
    ref = np.zeros((2, 40, 40), bool)   # brute-force reference: cell centre inside triangle
    for t, s_ in zip(tri, fs):
        for y in range(40):
            for x in range(40):
                pc = np.array([x + 0.5, y + 0.5])
                w = [np.cross(t[(k + 1) % 3] - t[k], pc - t[k]) for k in range(3)]
                if (min(w) >= 0 or max(w) <= 0) and abs(np.cross(t[1] - t[0], t[2] - t[0])) > 1e-9:
                    ref[s_, y, x] = True
    agree = (mk == ref).mean()
    check('T2 vectorised rasteriser == brute-force union reference (incl. large triangles)', agree == 1.0, f'(pixel agreement {agree:.3f})')

    # ---------------------------------------------------------------- scene: A(left) index crosses B(right) middle
    df, ds, n = orthonormal_frame(m)
    RA = rot_from_axes([df, ds, n], [np.array([1., 0, 0]), np.array([0, 1., 0]), np.array([0, 0, 1.])])
    # A: fingers -> +x, index on the +y side.  B: fingers -> -y (coming from below), palm below the crossing,
    # so B's fingertips lie across A's index finger only.
    RB = rot_from_axes([df, ds, n], [np.array([0, -1., 0]), np.array([1., 0, 0]), np.array([0, 0, 1.])])
    size, f, c = 1024, 1400.0, 512.0
    v0 = m['v_template']
    segs = labels
    vA = v0 @ RA.T
    vA = vA - vA[segs == SEGMENTS.index('index_dist')].mean(0) + np.array([0.0, 0.0, 0.60])
    vB_can = v0 @ RB.T
    cB = vB_can[segs == SEGMENTS.index('middle_dist')].mean(0)
    ray = np.array([0.0, 0.0, 1.0])  # A's index_dist centroid is on the optical axis

    def place_B(zc):
        return vB_can - cB + ray * zc

    Z_TRUE_B, Z_WRONG_B = 0.63, 0.57            # truth: A (0.60) in front of B ; tracker init: B in front
    vB_true = place_B(Z_TRUE_B)
    zb = zbuffer_render([vA, vB_true], m['f'], f, c, size)
    vda = 1.0 / zb                                                  # disparity-like: larger = closer
    vda = cv2.GaussianBlur(vda.astype(np.float32), (5, 5), 1.0) + rng.normal(0, 1e-3, vda.shape).astype(np.float32)
    # matting-like mask = true silhouette (holes closed); background pixels are never sampled
    body_mask = cv2.morphologyEx((zb < 2.9).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(np.float32)
    key = 'frame_000000'
    src = DepthSource(vda[None], frame_map={key: 0}, space='body_hd')

    # differentiable tracker: B = R_B * LBS(pose_B) + t_B, perspective projection
    th = {k: torch.tensor(v, dtype=torch.float32) for k, v in m.items() if k != 'kintree_table'}
    parents = torch.tensor(m['kintree_table'][0].astype(np.int64)); parents[0] = -1
    posedirs = th['posedirs'].reshape(-1, th['posedirs'].shape[-1]).T
    RB_t = torch.tensor(RB, dtype=torch.float32)

    def hand_B(pose45, tz):
        full = torch.cat([torch.zeros(1, 1, 3), pose45.view(1, 15, 3)], 1)
        v, _ = lbs(torch.zeros(1, 10), full, th['v_template'], th['shapedirs'], posedirs, th['J_regressor'],
                   parents, th['weights'], pose2rot=True)
        v = v[0] @ RB_t.T - torch.tensor(cB, dtype=torch.float32) + torch.stack([torch.zeros(()), torch.zeros(()), tz])
        return v

    vA_t = torch.tensor(vA, dtype=torch.float32)

    def project(vA_, vB_):
        V = torch.cat([vA_, vB_], 0)
        xy = f * V[:, :2] / V[:, 2:3] + c
        return torch.cat([xy, V[:, 2:3]], 1)[None]                 # [1,1556,3], z = camera z (larger = farther)

    pose_B = torch.zeros(45, requires_grad=True)
    tz_B = torch.tensor(Z_WRONG_B, requires_grad=True)
    cfg = DepthOrderConfig(enable=True, margin=0.005)
    builder = DepthOrderTargetBuilder(cfg, index, ehm.smplx.smplx2mano_ind)

    # ---------------------------------------------------------------- T3 gather_hand_parts API
    proj = project(vA_t, hand_B(pose_B, tz_B))
    parts = gather_hand_parts(proj, ehm.smplx.smplx2mano_ind, index)
    p = parts['left']['index_dist']
    check('T3 gather_hand_parts exposes xy/z/ids/valid', p.xy.shape[-1] == 2 and p.z.shape == p.valid.shape and
          len(p.vert_ids) == summ['index_dist'] and bool(p.valid.all()))

    # ---------------------------------------------------------------- T4 detection + local sampling (P1,P2)
    targets = builder.build(proj, [key], src, None, [body_mask])
    names = [(i.name_a, i.name_b, i.kind, round(i.confidence, 2), i.sign) for i in targets.instances]
    print('detected instances:', len(targets.instances), 'active:', len(targets.active))
    for i in sorted(targets.instances, key=lambda t: -t.confidence)[:12]:
        print(f'   {i.name_a:>14s} vs {i.name_b:<14s} {i.kind:9s} {i.pair_type:5s} {i.part_type:13s} '
              f'nA={i.n_a:3d} nB={i.n_b:3d} vA={len(i.mano_ids_a):2d} vB={len(i.mano_ids_b):2d} '
              f'dA={i.d_a:.4f} dB={i.d_b:.4f} conf={i.confidence:.2f} cons={i.consistency:.2f} '
              f'front={i.front_name() if i.confidence > 0 else "-"} tracker_dz={i.tracker_dz * 1e3:+.1f}e-3 {i.skip_reason}')
    tgt = [i for i in targets.active if i.name_a == 'L-index_dist' and i.name_b.startswith('R-middle')]
    check('T4a expected local crossing L-index_dist vs R-middle_* detected & active', len(tgt) > 0)
    check('T4b VDA says L-index_dist in FRONT (sign=+1)', len(tgt) > 0 and all(i.sign > 0 for i in tgt))
    check('T4c A-only / B-only samples exist and exclude overlap', len(tgt) > 0 and all(i.n_a >= cfg.min_samples and i.n_b >= cfg.min_samples for i in tgt))
    check('T4d overlap depth consistent with front surface', len(tgt) > 0 and all(i.consistency > 0.5 for i in tgt))
    check('T4e tracker initially VIOLATES (B in front)', len(tgt) > 0 and all(i.sign * i.tracker_dz > 0 for i in tgt))
    inter_far = [i for i in targets.active if i.pair_type == 'inter' and np.hypot(i.center_px[0] - c, i.center_px[1] - c) > 250]
    check('T4f no far-away unrelated inter-hand pairs', len(inter_far) == 0)
    # self-occlusion pairs (e.g. MANO rest-pose thumb over thenar): the tracker's INTRA-hand order equals the truth
    # here (hand A is exact, hand B is only translated), so every active self pair must agree with the tracker
    selfp = [i for i in targets.active if i.pair_type == 'self']
    check('T4g active same-hand (self-occlusion) pairs agree with ground-truth ordering',
          all(i.sign * i.tracker_dz < 0 for i in selfp), f'({[(i.name_a, i.name_b) for i in selfp]})')
    adj = [i for i in targets.instances if i.pair_type == 'self' and
           (i.seg_a.split('_')[0] == i.seg_b.split('_')[0] or ('palm' in (i.seg_a, i.seg_b) and 'prox' in i.seg_a + i.seg_b))]
    check('T4h kinematically adjacent same-hand segments never paired', len(adj) == 0)

    # ---------------------------------------------------------------- T5 loss + gradients (P3)
    loss, st = depth_order_loss(proj, targets, cfg.margin)
    check('T5a loss_depth_order > 0 when violated', float(loss) > 0, f'(loss={float(loss):.4e}, {st})')
    g_pose, g_tz = torch.autograd.grad(loss, [pose_B, tz_B])
    gp = g_pose.view(15, 3).norm(dim=1)
    print('   grad |dL/dpose_B| per MANO joint:', np.round(gp.numpy(), 5).tolist())
    check('T5b gradient reaches hand pose', float(g_pose.norm()) > 0, f'(|g_pose|={float(g_pose.norm()):.3e}, |g_tz|={float(g_tz):.3e})')
    check('T5c gradient sign pushes B farther (dL/dtz_B < 0)', float(g_tz) < 0)

    # ---------------------------------------------------------------- T6 optimisation resolves the order
    opt = torch.optim.Adam([{'params': [pose_B], 'lr': 1e-2}, {'params': [tz_B], 'lr': 2e-3}])
    hist = []
    for it in range(120):
        proj = project(vA_t, hand_B(pose_B, tz_B))
        if it % 30 == 0 and it > 0:
            targets = builder.build(proj.detach(), [key], src, None, [body_mask])
        loss, st = depth_order_loss(proj, targets, cfg.margin)
        hist.append((float(loss), st['n_wrong_order']))
        opt.zero_grad(); loss.backward(); opt.step()
    print('   loss/wrong-order trajectory:', hist[0], '->', hist[-1], f'tz_B {Z_WRONG_B} -> {float(tz_B):.4f}')
    check('T6 depth-only optimisation fixes ordering (wrong-order -> 0)', hist[-1][1] == 0 and hist[0][1] > 0)

    # visualisation of the initial state
    proj0 = project(vA_t, hand_B(torch.zeros(45), torch.tensor(Z_WRONG_B)))
    hv = {sd: proj0[0].numpy()[builder.m2s[sd]] for sd in ('left', 'right')}
    insts, rasters = builder.process_frame(0, key, hv, src, None, body_mask, keep_vis=True)
    img = np.full((size, size, 3), 90, np.uint8)
    panel = draw_depth_order_debug(img, rasters, insts, src, key, None, title='synthetic: L-index over R-middle (truth: L in front)')
    vis_fp = osp.join(out_dir, 'synthetic_crossing_debug.png')
    cv2.imwrite(vis_fp, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    check('T7 debug visualisation written', osp.exists(vis_fp), vis_fp)

    # ---------------------------------------------------------------- T8 flipped VDA -> flipped sign
    src_flip = DepthSource(vda[None], frame_map={key: 0}, space='body_hd', larger_is_closer=False)
    b_nc = DepthOrderTargetBuilder(DepthOrderConfig(enable=True, margin=0.005, use_overlap_consistency=False), index,
                                   ehm.smplx.smplx2mano_ind)
    tf = b_nc.build(proj0, [key], src_flip, None, [body_mask])
    tgt_f = [i for i in tf.active if i.name_a == 'L-index_dist' and i.name_b.startswith('R-middle')]
    check('T8a flipped VDA convention flips predicted front (consistency check off)', len(tgt_f) > 0 and all(i.sign < 0 for i in tgt_f))
    tf2 = builder.build(proj0, [key], src_flip, None, [body_mask])
    rej = [i for i in tf2.instances if i.name_a == 'L-index_dist' and i.name_b == 'R-middle_dist']
    check('T8b physically impossible (flipped) VDA rejected by overlap-consistency check',
          len(rej) > 0 and all(i.confidence == 0 and i.skip_reason == 'overlap_inconsistent' for i in rej))

    # ---------------------------------------------------------------- T9 near tie -> zero confidence
    flat = np.where(zb < 2.9, 1.0 / 0.6, 1.0 / 3.0).astype(np.float32) + rng.normal(0, 1e-4, zb.shape).astype(np.float32)
    tt = builder.build(proj0, [key], DepthSource(flat[None], frame_map={key: 0}, space='body_hd'), None, [body_mask])
    check('T9 near-tie VDA -> no active pairs (confidence 0)', len(tt.active) == 0 and len(tt.instances) > 0,
          f"(skip reasons: {sorted(set(i.skip_reason for i in tt.instances))})")

    # ---------------------------------------------------------------- T10 original-frame space mapping
    M_c2o = np.array([[2.0, 0, 100.0], [0, 2.0, 60.0], [0, 0, 1]])       # crop px -> original px
    big = cv2.warpAffine(vda, np.float32([[2, 0, 100], [0, 2, 60]]), (2248, 2168), flags=cv2.INTER_LINEAR)
    small = cv2.resize(big, (1124, 1084), interpolation=cv2.INTER_AREA)  # depth stored at half the video resolution
    src_o = DepthSource(small[None], frame_map={key: 0}, space='original', original_size=(2168, 2248))
    to = builder.build(proj0, [key], src_o, torch.tensor(M_c2o[None]), [body_mask])
    tgt_o = [i for i in to.active if i.name_a == 'L-index_dist' and i.name_b.startswith('R-middle')]
    check("T10 space='original' + M_c2o-hd + resize gives same ordering", len(tgt_o) > 0 and all(i.sign > 0 for i in tgt_o))

    # ---------------------------------------------------------------- T11 fully covered part -> no visible support
    H = W = 40
    mA = np.zeros((H, W), bool); mA[15:22, 15:22] = True            # A completely inside B in 2D
    mB = np.zeros((H, W), bool); mB[8:30, 8:30] = True
    masks = {(sd, s_): np.zeros((H, W), bool) for sd in ('left', 'right') for s_ in SEGMENTS}
    masks[('left', 'index_dist')] = mA; masks[('right', 'palm')] = mB
    r = FrameRaster(origin=np.zeros(2), stride=2, masks=masks, shape=(H, W))
    cm = sum(v.astype(np.int16) for v in masks.values())
    hv_dummy = {sd: np.concatenate([np.full((778, 2), 36.0), np.full((778, 1), 0.6)], 1) for sd in ('left', 'right')}
    pairs = builder.find_interacting_part_pairs(r, ('left', 'right'), 0, key)
    pr = [pp for pp in pairs if pp.seg_a == 'index_dist' and pp.seg_b == 'palm']
    for pp in pr:
        builder.sample_local_vda(pp, r, cm, np.ones((H, W), bool), hv_dummy, src, None, 1.0)
    check('T11 fully-covered part -> A-only samples = 0 -> confidence 0',
          len(pr) == 1 and pr[0].n_a == 0 and pr[0].confidence == 0 and pr[0].skip_reason == 'too_few_visible_samples')

    # ---------------------------------------------------------------- T12 median z gradient is well defined
    z = torch.randn(2, 20, requires_grad=True)
    ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 0]]); val = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.bool)
    med = local_median_z(z, torch.tensor([0, 1]), ids, val)
    med.sum().backward()
    gz = z.grad
    check('T12 local median z: finite grads, none to padded entries', torch.isfinite(gz).all().item() and gz[0, 3].item() == 0 and gz[1, 0].item() == 0)

    # ---------------------------------------------------------------- T13 baseline path is structurally unchanged
    same, info = baseline_ast_equivalent()
    check('T13 refiner with depth_ctx=None is AST-identical to baseline 55cfe1c after removing depth hooks', same, info)
    return vis_fp


class _StripDepth(ast.NodeTransformer):
    """Remove every depth-ordering hook so the remaining AST can be compared with the baseline."""
    DEPTH_NAMES = ('depth_ctx', 'depth_kw', 'depth_source', 'depth_order_cfg', 'depth_stats', 'loss_depth_order')

    def _mentions(self, node):
        return any(isinstance(n, ast.Name) and n.id in self.DEPTH_NAMES or isinstance(n, ast.Attribute) and n.attr in self.DEPTH_NAMES
                   for n in ast.walk(node))

    def visit_If(self, node):
        return None if self._mentions(node.test) else self.generic_visit(node)

    def visit_Assign(self, node):
        if any(self._mentions(tg) for tg in node.targets):
            return None                    # e.g. self.depth_order_cfg = ..., depth_kw = {}
        return self.generic_visit(node)     # keep the statement, strip **depth_kw from calls

    def visit_FunctionDef(self, node):
        if node.name == 'build_depth_ctx':
            return None
        a = node.args
        keep = [(x, d) for x, d in zip(a.args[len(a.args) - len(a.defaults):], a.defaults)]
        a.args = [x for x in a.args if x.arg not in self.DEPTH_NAMES]
        a.defaults = [d for x, d in keep if x.arg not in self.DEPTH_NAMES]
        return self.generic_visit(node)

    def visit_Call(self, node):
        node.keywords = [k for k in node.keywords if not (k.arg is None and self._mentions(k.value))
                         and k.arg not in self.DEPTH_NAMES]
        return self.generic_visit(node)


def baseline_ast_equivalent():
    fp = osp.join(ROOT, 'src/modules/refiner/ehm_refiner.py')
    try:
        base_src = subprocess.check_output(['git', '-C', ROOT, 'show', '55cfe1c:src/modules/refiner/ehm_refiner.py'], text=True)
    except Exception as e:
        return False, f'(cannot read baseline: {e})'
    new = _StripDepth().visit(ast.parse(open(fp).read()))
    ast.fix_missing_locations(new)
    a, b = ast.dump(ast.parse(base_src)), ast.dump(new)
    return a == b, '' if a == b else '(ASTs differ)'


# ============================================================================ clip (GPU)

def run_clip(args):
    import importlib
    import lmdb  # noqa: F401  (fail early)
    from src.configs.data_prepare_config import DataPreparationConfig
    from src.modules.refiner.ehm_refiner import EhmOptimizer
    from src.modules.refiner.flame_refiner import FlameOptimizer
    from src.utils.io import load_dict_pkl
    from src.utils.lmdb import LMDBEngine
    from src.utils.graphics import GS_Camera

    cfg = DataPreparationConfig()
    dev = cfg.device
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tdir = args.track_dir
    track_fp = osp.join(tdir, 'optim_tracking_flame.pkl')
    if not osp.exists(track_fp):
        track_fp = osp.join(tdir, 'base_tracking.pkl')
    tracked_all = load_dict_pkl(track_fp)
    id_share0 = load_dict_pkl(osp.join(tdir, 'id_share_params.pkl'))
    keys = sorted(tracked_all.keys())
    if args.keys:
        keys = [k for k in keys if k in set(args.keys)]
    keys = keys[args.start:args.start + args.n_frames]
    print('frames:', keys)
    lmdb_engine = LMDBEngine(osp.join(tdir, 'img_lmdb'), write=False)
    out = args.out or osp.join(tdir, 'depth_order_smoke')
    os.makedirs(out, exist_ok=True)

    def make_opt(cls, depth_cfg=None):
        flame_opt = FlameOptimizer(cfg.flame_assets_dir, device=dev, image_size=cfg.head_crop_size, tanfov=cfg.tanfov)
        kw = {} if depth_cfg is None else {'depth_order_cfg': depth_cfg}
        o = cls(cfg.flame_assets_dir, cfg.smplx_assets_dir, cfg.mano_assets_dir, device=dev,
                body_image_size=cfg.body_hd_size, head_image_size=cfg.head_crop_size, tanfov=cfg.tanfov,
                vposer_ckpt=cfg.vposer_ckpt_dir, **kw)
        o.ehm.flame = flame_opt.flame; o.head_renderer = flame_opt.renderer
        o.saving_root = out
        return o

    def run(o, **kw):
        torch.manual_seed(0); np.random.seed(0)
        tr = copy.deepcopy({k: tracked_all[k] for k in keys})
        res, _ = o.run(tr, copy.deepcopy(id_share0), lmdb_engine, 1, steps=args.steps, **kw)
        return res

    def maxdiff(a, b):
        return max(float(np.abs(np.asarray(a[k][p]) - np.asarray(b[k][p])).max()) for k in a for p in a[k])

    report = {}
    # ---- 1. baseline equivalence: original refiner source from 55cfe1c vs new code with depth disabled
    tmp = osp.join(ROOT, 'src/modules/refiner/_baseline_ehm_refiner_55cfe1c.py')
    with open(tmp, 'w') as fh:
        fh.write(subprocess.check_output(['git', '-C', ROOT, 'show', '55cfe1c:src/modules/refiner/ehm_refiner.py'], text=True))
    try:
        Base = importlib.import_module('src.modules.refiner._baseline_ehm_refiner_55cfe1c').EhmOptimizer
        r_base = run(make_opt(Base))
    finally:
        os.remove(tmp)
    r_new_off = run(make_opt(EhmOptimizer))
    d = maxdiff(r_base, r_new_off)
    report['baseline_vs_new_disabled_maxabs'] = d
    check('C1 depth disabled reproduces original baseline', d == 0.0 or d < args.tol, f'(max|diff|={d:.3e})')

    # ---- camera z convention on real data
    o = make_opt(EhmOptimizer)
    check_camera_convention(o, tracked_all, keys, id_share0, dev, GS_Camera)

    # ---- 2-4. enabled runs
    src = DepthSource(args.vda, frame_map=args.frame_map, space=args.vda_space, larger_is_closer=not args.vda_farther_larger)
    for lr in args.hand_pose_lrs:
        dc = DepthOrderConfig(enable=True, lambda_depth_order=args.lambda_depth_order, margin=args.margin,
                              hand_pose_lr=lr, refresh_every=args.refresh_every, log_every=10)
        o = make_opt(EhmOptimizer, dc)
        o.saving_root = osp.join(out, f'lr_{lr:g}')
        r_on = run(o, depth_source=src)
        check(f'C2 enabled run completes (hand_pose_lr={lr:g})', r_on is not None)
        report[f'lr_{lr:g}'] = summarize_logs(o.saving_root, r_on, r_new_off)
    # forced violation: flipped VDA
    src_f = DepthSource(args.vda, frame_map=args.frame_map, space=args.vda_space, larger_is_closer=args.vda_farther_larger)
    dc = DepthOrderConfig(enable=True, lambda_depth_order=args.lambda_depth_order, margin=args.margin,
                          hand_pose_lr=max(args.hand_pose_lrs), refresh_every=args.refresh_every, log_every=10,
                          use_overlap_consistency=False)  # flipped maps would otherwise be rejected as inconsistent
    o = make_opt(EhmOptimizer, dc); o.saving_root = osp.join(out, 'flipped_vda')
    r_f = run(o, depth_source=src_f)
    report['flipped_vda'] = summarize_logs(o.saving_root, r_f, r_new_off)
    any_pairs = any(v.get('max_n_pairs', 0) > 0 for v in report.values() if isinstance(v, dict))
    check('C3 active pairs found on this clip', any_pairs)
    nz = [v for v in report.values() if isinstance(v, dict) and v.get('max_loss_depth_order', 0) > 0]
    check('C3b loss_depth_order > 0 in at least one run (flipped VDA forces violations)', len(nz) > 0)
    g = [v for v in report.values() if isinstance(v, dict) and v.get('max_grad_hand_pose', 0) > 0]
    check('C4 gradients of depth loss reach left/right_hand_pose', len(g) > 0)
    json.dump(report, open(osp.join(out, 'smoke_report.json'), 'w'), indent=2)
    print('report:', json.dumps(report, indent=2))
    print('debug images under', out, '*/visual_results/vis_depth_order')


def check_camera_convention(o, tracked_all, keys, id_share0, dev, GS_Camera):
    """Verify proj[...,2] == (R X + T)_z (camera z, larger = farther) on the real camera code."""
    from src.modules.refiner.ehm_refiner import data_to_device
    b = torch.utils.data.default_collate([copy.deepcopy(tracked_all[k]) for k in keys[:2]])
    b = data_to_device(b, dev)
    X = torch.randn(2, 50, 3, device=dev) * 0.3
    R = b['smplx_coeffs']['camera_RT_params'][:, 0, :3, :3] if b['smplx_coeffs']['camera_RT_params'].dim() == 4 else b['smplx_coeffs']['camera_RT_params'][:, :3, :3]
    T = b['smplx_coeffs']['camera_RT_params'][..., :3, 3].reshape(2, 3)
    cam = GS_Camera(**o.build_cameras_kwargs(2, o.body_focal_length)).to(dev)
    p = cam.transform_points_screen(X, R=R.float(), T=T.float())
    z_manual = (torch.einsum('bij,bnj->bni', R.float(), X) + T.float()[:, None])[..., 2]
    err = float((p[..., 2] - z_manual).abs().max())
    check('C0 proj_vertices[...,2] is camera-space z = (R X + T)_z', err < 1e-4, f'(max err {err:.2e}, median z {float(z_manual.median()):.3f})')


def summarize_logs(root, res_on, res_off):
    d = osp.join(root, 'depth_order_logs')
    recs = []
    for fn in sorted(os.listdir(d)) if osp.isdir(d) else []:
        if fn.endswith('.jsonl'):
            recs += [json.loads(l) for l in open(osp.join(d, fn))]
    gmax = 0.0
    for r in recs:
        for v in (r.get('grad_norm', {}).get('weighted_loss_depth_order', {}) or {}).values():
            gmax = max(gmax, v)
    lh = max(float(np.abs(np.asarray(res_on[k]['left_hand_pose']) - np.asarray(res_off[k]['left_hand_pose'])).max()) for k in res_on)
    rh = max(float(np.abs(np.asarray(res_on[k]['right_hand_pose']) - np.asarray(res_off[k]['right_hand_pose'])).max()) for k in res_on)
    first, last = (recs[0] if recs else {}), (recs[-1] if recs else {})
    keys = ('loss_depth_order', 'weighted_loss_depth_order', 'loss_3d_z', 'loss_3d_hand_l', 'loss_3d_hand_r',
            'n_pairs', 'n_wrong_order', 'mean_conf', 'n_inter', 'n_self', 'n_palm_finger', 'n_finger_finger')
    return {'max_n_pairs': max([r.get('n_pairs', 0) for r in recs] or [0]),
            'max_loss_depth_order': max([r.get('loss_depth_order', 0) for r in recs] or [0]),
            'max_grad_hand_pose': gmax,
            'first': {k: first.get(k) for k in keys}, 'last': {k: last.get(k) for k in keys},
            'grad_norms': [r['grad_norm'] | {'step': r['step']} for r in recs if 'grad_norm' in r],
            'hand_pose_change_vs_baseline': {'left': lh, 'right': rh}}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--clip', action='store_true')
    ap.add_argument('--out', default='')
    ap.add_argument('--mano_dir', default=osp.join(ROOT, 'assets/MANO'))
    ap.add_argument('--track_dir', default='')
    ap.add_argument('--vda', default='')
    ap.add_argument('--frame_map', default=None)
    ap.add_argument('--vda_space', default='original', choices=['original', 'body_hd'])
    ap.add_argument('--vda_farther_larger', action='store_true', help='set if your maps are depth (larger = farther)')
    ap.add_argument('--keys', nargs='*', default=None)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--n_frames', type=int, default=8)
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--hand_pose_lrs', type=float, nargs='+', default=[1e-5, 5e-5, 1e-4])
    ap.add_argument('--lambda_depth_order', type=float, default=100.0)
    ap.add_argument('--margin', type=float, default=0.01)
    ap.add_argument('--refresh_every', type=int, default=20)
    ap.add_argument('--tol', type=float, default=0.0, help='allowed |diff| for C1 (CUDA nondeterminism)')
    a = ap.parse_args()
    if a.synthetic:
        run_synthetic(a.out or '/tmp/depth_order_synth', a.mano_dir)
    if a.clip:
        run_clip(a)
    print(f'\n{len(PASS)} passed, {len(FAIL)} failed' + (f': {FAIL}' if FAIL else ''))
    sys.exit(1 if FAIL else 0)
