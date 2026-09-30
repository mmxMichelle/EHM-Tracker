"""
Thin adapter between EhmOptimizer.optimize() and the depth-ordering module.

EhmOptimizer only calls into this when a DepthOrderContext is passed
(enable_depth_ordering=True and a DepthSource exists). With depth_ctx=None the
refiner executes exactly the baseline code path.
"""
import json
import os
import os.path as osp
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch

from .depth_ordering import DepthOrderConfig, DepthOrderTargetBuilder, DepthOrderTargets, depth_order_loss
from .depth_vis import draw_depth_order_debug
from .hand_parts import HandPartIndex, SIDES


def _to_mask(t, size):
    if t is None:
        return None
    m = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    m = np.squeeze(m).astype(np.float32)
    if m.ndim == 3:
        m = m.mean(0) if m.shape[0] in (1, 3, 4) else m.mean(-1)
    if m.max() > 1.5:
        m = m / 255.0
    if m.shape != (size, size):
        return None
    return m


class DepthOrderContext:
    def __init__(self, cfg: DepthOrderConfig, depth_source, frame_keys: Sequence[str], ehm, M_c2o_hd=None,
                 body_masks=None, image_size: int = 1024, save_root: Optional[str] = None, log=print):
        self.cfg = cfg
        self.src = depth_source
        self.keys = list(frame_keys)
        self.M = M_c2o_hd
        self.log = log
        self.image_size = image_size
        self.body_masks = None
        if body_masks is not None:
            bm = [_to_mask(m, image_size) for m in body_masks]
            if any(m is None for m in bm):
                log('depth-order: body mask shape mismatch; matting mask not used')
            else:
                self.body_masks = bm
        self.index = HandPartIndex.from_mano(ehm.mano)
        self.builder = DepthOrderTargetBuilder(cfg, self.index, ehm.smplx.smplx2mano_ind)
        self.targets: Optional[DepthOrderTargets] = None
        self.initial_targets: Optional[DepthOrderTargets] = None
        self.save_root = save_root
        self.batch_id = 0
        self.history: List[Dict] = []
        self.last_proj = None
        n_have = sum(self.src.has(k) for k in self.keys)
        log(f'depth-order: enabled: {n_have}/{len(self.keys)} frames have VDA | parts {self.index.summary()}')

    # ------------------------------------------------------------------ optimizer
    def attach_optimizer(self, opt, left_hand_pose, right_hand_pose):
        """Optional hand-pose LR override (only when depth ordering is enabled)."""
        if self.cfg.hand_pose_lr is None:
            return
        for g in opt.param_groups:
            if any(p is left_hand_pose or p is right_hand_pose for p in g['params']):
                self.log(f'depth-order: hand_pose lr {g["lr"]:.1e} -> {self.cfg.hand_pose_lr:.1e}')
                g['lr'] = float(self.cfg.hand_pose_lr)

    # ------------------------------------------------------------------ targets
    def refresh(self, proj_vertices):
        self.targets = self.builder.build(proj_vertices, self.keys, self.src, self.M, self.body_masks)
        if self.initial_targets is None:
            self.initial_targets = self.targets
        t = self.targets
        by_reason = {}
        for i in t.instances:
            if i.confidence <= 0:
                by_reason[i.skip_reason] = by_reason.get(i.skip_reason, 0) + 1
        self.log(f'depth-order: targets: {len(t.instances)} detected, {len(t.active)} active, skipped={by_reason}')

    def _grad_norms(self, terms: Dict[str, torch.Tensor], params) -> Dict[str, Dict[str, float]]:
        out = {}
        for name, v in terms.items():
            if not torch.is_tensor(v) or not v.requires_grad:
                out[name] = {'left_hand_pose': 0.0, 'right_hand_pose': 0.0}
                continue
            g = torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
            out[name] = {n: (0.0 if gg is None else float(gg.norm())) for n, gg in zip(('left_hand_pose', 'right_hand_pose'), g)}
        return out

    # ------------------------------------------------------------------ per step
    def step(self, i_step: int, steps: int, proj_vertices, loss_terms: Dict, hand_poses, batch_id: int = 0):
        self.batch_id = batch_id
        cfg = self.cfg
        if self.targets is None or (cfg.refresh_every > 0 and i_step > 0 and i_step % cfg.refresh_every == 0):
            self.refresh(proj_vertices)
        loss, stats = depth_order_loss(proj_vertices, self.targets, cfg.margin)
        self.last_proj = proj_vertices.detach()
        rec = None
        diag = i_step in (0, steps // 2, steps - 1)
        if i_step % max(cfg.log_every, 1) == 0 or diag:
            rec = {'step': i_step, 'loss_depth_order': float(loss),
                   'weighted_loss_depth_order': float(loss) * cfg.lambda_depth_order,
                   **{k: float(v) for k, v in loss_terms.items()}, **stats}
            if cfg.grad_diag and diag:
                terms = {'weighted_loss_depth_order': loss * cfg.lambda_depth_order, **loss_terms}
                rec['grad_norm'] = self._grad_norms(terms, list(hand_poses))
                g = rec['grad_norm']
                self.log('depth-order grad: step %d | ' % i_step + ' | '.join(
                    f"{k}: L {v['left_hand_pose']:.3e} R {v['right_hand_pose']:.3e}" for k, v in g.items()))
            self.history.append(rec)
            self._write_jsonl(rec)
        return loss, stats

    def format_log(self, stats) -> str:
        return f"| dord: n={stats['n_pairs']} c={stats['mean_conf']:.2f} wrong={stats['n_wrong_order']} "

    # ------------------------------------------------------------------ outputs
    def _log_dir(self, sub):
        if self.save_root is None:
            return None
        d = osp.join(self.save_root, sub)
        os.makedirs(d, exist_ok=True)
        return d

    def _write_jsonl(self, rec):
        d = self._log_dir('depth_order_logs')
        if d is None:
            return
        with open(osp.join(d, f'batch_{self.batch_id:02d}.jsonl'), 'a') as f:
            f.write(json.dumps(rec) + '\n')

    def maybe_visualize(self, i_step: int, steps: int, batch_imgs, batch_id: int = 0):
        if not self.cfg.debug_vis or batch_imgs is None or self.targets is None or self.last_proj is None:
            return
        if i_step not in (0, steps - 1):
            return
        d = self._log_dir(osp.join('visual_results', 'vis_depth_order'))
        if d is None:
            return
        by_frame = {}
        for i in self.targets.instances:
            by_frame.setdefault(i.frame, []).append(i)
        order = sorted(by_frame, key=lambda f: -sum(i.confidence for i in by_frame[f]))[:self.cfg.max_vis_frames]
        pv = self.last_proj.float().cpu().numpy()
        for f in order:
            hv = {sd: pv[f, self.builder.m2s[sd]] for sd in SIDES}
            M = None if self.M is None else self.M[f].detach().cpu().numpy() if torch.is_tensor(self.M) else self.M[f]
            bm = None if self.body_masks is None else self.body_masks[f]
            insts, rasters = self.builder.process_frame(f, self.keys[f], hv, self.src, M, bm, keep_vis=True)
            img = batch_imgs[f]
            img = img.clone().numpy().transpose(1, 2, 0) if torch.is_tensor(img) else np.asarray(img)
            panel = draw_depth_order_debug(img, rasters, insts, self.src, self.keys[f], M,
                                           title=f'{self.keys[f]} | batch {batch_id} step {i_step}')
            cv2.imwrite(osp.join(d, f'vis_depth_order_bid-{batch_id}_stp-{i_step}_{self.keys[f]}.png'),
                        cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    def finalize(self, batch_id: int = 0):
        """Before/after summary on the INITIAL targets (did the tracker order change?)."""
        if self.initial_targets is None or self.last_proj is None:
            return None
        l0 = self.history[0] if self.history else {}
        _, s1 = depth_order_loss(self.last_proj, self.initial_targets, self.cfg.margin)
        summ = {'batch_id': batch_id, 'initial_targets_n': len(self.initial_targets),
                'initial_wrong_order': l0.get('n_wrong_order'), 'final_wrong_order_on_initial_targets': s1['n_wrong_order'],
                'final_margin_violated_on_initial_targets': s1['n_margin_violated'],
                'first_record': l0, 'last_record': self.history[-1] if self.history else {}}
        self.log(f"depth-order: batch {batch_id}: wrong-order pairs {summ['initial_wrong_order']} -> "
                 f"{summ['final_wrong_order_on_initial_targets']} (of {summ['initial_targets_n']} initial active pairs)")
        d = self._log_dir('depth_order_logs')
        if d is not None:
            with open(osp.join(d, f'batch_{batch_id:02d}_summary.json'), 'w') as f:
                json.dump(summ, f, indent=2)
            with open(osp.join(d, f'batch_{batch_id:02d}_pairs_initial.json'), 'w') as f:
                json.dump([i.to_dict() for i in self.initial_targets.instances], f, indent=1)
        return summ
