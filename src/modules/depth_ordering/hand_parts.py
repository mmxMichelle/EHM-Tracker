"""
MANO phalanx-level semantic hand-part mapping for depth-aware part ordering.

WHAT IS EXACT (model-derived) AND WHAT IS HEURISTIC
---------------------------------------------------
* EXACT / MODEL-DERIVED:
  Each MANO vertex is assigned to the MANO joint with the largest linear-blend-
  skinning (LBS) weight, i.e. ``argmax_j lbs_weights[v, j]``.  The weights are the
  ones shipped in ``assets/MANO/MANO_RIGHT.pkl`` and already loaded by
  ``src/modules/mano/MANO.py`` as ``self.ehm.mano.lbs_weights`` (778 x 16).
  MANO_LEFT.pkl yields an identical assignment (verified: 100% agreement), and
  EHM indexes both hands in MANO vertex order through
  ``self.ehm.smplx.smplx2mano_ind['left_hand' | 'right_hand']`` (778 ids each),
  so one label array serves both hands.

* HEURISTIC:
  Interpreting the dominant-skinning region of a joint as an anatomical surface
  region (e.g. "index distal phalanx") is an APPROXIMATION.  LBS regions are
  smooth-blended near joints, so vertices close to a joint may belong to either
  neighbouring bone anatomically.  In particular:
    - MANO joint j drives the bone from joint j to its child, so
        index1/middle1/ring1/pinky1 -> proximal phalanx   ("*_prox")
        index2/...                  -> middle phalanx     ("*_mid")
        index3/...                  -> distal phalanx     ("*_dist")
    - The thumb has only two phalanges.  We keep the requested naming
        thumb_prox = MANO thumb1 region  (anatomically ~ thumb METACARPAL / thenar)
        thumb_mid  = MANO thumb2 region  (anatomically ~ proximal phalanx)
        thumb_dist = MANO thumb3 region  (anatomically ~ distal phalanx)
    - "palm" = vertices dominated by the MANO root/wrist joint, MINUS the wrist-cut
      ring that EHM already excludes from its hand losses
      (``assets/MANO/selected_hand_ver.npy``: 671 of 778 kept; the 107 removed
      are 92 wrist-dominated + 15 thumb1/thumb2-dominated vertices at the cut).
      Removed vertices get label -1 (ignored everywhere).

The API (``HandPartIndex``) is designed so these labels can later be replaced by
exact, hand-annotated semantic vertex labels without touching the rest of the
depth-ordering code: pass ``labels=`` explicitly.

Verified counts on the shipped MANO_RIGHT.pkl (before wrist-cut removal):
    wrist/palm 199 | index 33/34/54 | middle 35/30/52 | pinky 23/39/52 |
    ring 30/38/54 | thumb 20/31/54   (prox/mid/dist)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

# MANO joint order (as in MANO_RIGHT.pkl kintree / J_regressor, 16 joints):
#  0 wrist | 1-3 index | 4-6 middle | 7-9 pinky | 10-12 ring | 13-15 thumb
# (consistent with MANO.py `mano_to_openpose` and SMPLX_names left_index1..left_thumb3)
MANO_JOINT_TO_SEGMENT = {
    0: 'palm',
    1: 'index_prox', 2: 'index_mid', 3: 'index_dist',
    4: 'middle_prox', 5: 'middle_mid', 6: 'middle_dist',
    7: 'pinky_prox', 8: 'pinky_mid', 9: 'pinky_dist',
    10: 'ring_prox', 11: 'ring_mid', 12: 'ring_dist',
    13: 'thumb_prox', 14: 'thumb_mid', 15: 'thumb_dist',
}

FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
LEVELS = ('prox', 'mid', 'dist')
# Canonical segment order used for integer labels 0..15.
SEGMENTS: List[str] = ['palm'] + [f'{f}_{l}' for f in FINGERS for l in LEVELS]
SEGMENT_ID: Dict[str, int] = {s: i for i, s in enumerate(SEGMENTS)}
SIDES = ('left', 'right')
SIDE_SHORT = {'left': 'L', 'right': 'R'}


def segment_part(seg: str) -> str:
    """Coarse part ('palm', 'thumb', 'index', ...) of a phalanx-level segment."""
    return seg.split('_')[0]


def segment_level(seg: str) -> Optional[str]:
    return seg.split('_')[1] if '_' in seg else None


def build_mano_part_labels(lbs_weights, selected_vert_ids=None) -> np.ndarray:
    """
    Per-MANO-vertex segment label (index into SEGMENTS), -1 = ignored.

    lbs_weights: [778, 16] MANO skinning weights (numpy or torch).
    selected_vert_ids: optional ids of vertices to keep (EHM's selected_hand_ver.npy);
                       all other vertices -> -1 (wrist cut).
    """
    W = lbs_weights.detach().cpu().numpy() if torch.is_tensor(lbs_weights) else np.asarray(lbs_weights)
    assert W.ndim == 2 and W.shape[1] == 16, f'expected [V,16] MANO weights, got {W.shape}'
    dom = W.argmax(1)                                   # EXACT: dominant skinning joint
    labels = np.array([SEGMENT_ID[MANO_JOINT_TO_SEGMENT[int(j)]] for j in dom], dtype=np.int64)
    if selected_vert_ids is not None:
        keep = np.zeros(len(labels), dtype=bool)
        keep[np.asarray(selected_vert_ids, dtype=np.int64)] = True
        labels[~keep] = -1
    return labels


def face_labels_from_vertex_labels(faces: np.ndarray, vlabels: np.ndarray) -> np.ndarray:
    """Majority vote of the 3 vertex labels (ties -> label of first vertex). -1 if majority is -1."""
    fl = vlabels[faces]                                  # [F,3]
    out = fl[:, 0].copy()
    agree12 = fl[:, 1] == fl[:, 2]
    out[agree12] = fl[agree12, 1]                        # if v1==v2 they form the majority
    return out


@dataclass
class HandPartIndex:
    """
    Semantic index over the MANO topology (shared by left and right hands in EHM).

    verts[seg]  -> MANO vertex ids of that segment
    faces[seg]  -> MANO face ids of that segment (for 2D footprint rasterisation)
    """
    labels: np.ndarray                    # [778] segment id or -1
    faces_all: np.ndarray                 # [F,3] MANO faces
    face_labels: np.ndarray = field(init=False)
    verts: Dict[str, np.ndarray] = field(init=False)
    faces: Dict[str, np.ndarray] = field(init=False)

    def __post_init__(self):
        self.labels = np.asarray(self.labels, dtype=np.int64)
        self.faces_all = np.asarray(self.faces_all, dtype=np.int64)
        self.face_labels = face_labels_from_vertex_labels(self.faces_all, self.labels)
        self.verts = {s: np.nonzero(self.labels == i)[0] for i, s in enumerate(SEGMENTS)}
        self.faces = {s: np.nonzero(self.face_labels == i)[0] for i, s in enumerate(SEGMENTS)}

    @property
    def segments(self) -> List[str]:
        return list(SEGMENTS)

    @classmethod
    def from_mano(cls, mano_module, labels=None):
        """Build from EHM's MANO module (self.ehm.mano). Pass `labels` to override with exact labels."""
        if labels is None:
            labels = build_mano_part_labels(mano_module.lbs_weights, getattr(mano_module, 'selected_vert_ids', None))
        faces = mano_module.faces_tensor.detach().cpu().numpy()
        return cls(labels=labels, faces_all=faces)

    def summary(self) -> Dict[str, int]:
        return {s: int(len(v)) for s, v in self.verts.items()}


@dataclass
class PartData:
    """Projected data of one semantic part of one hand, for a batch of frames."""
    side: str
    segment: str
    xy: torch.Tensor          # [B, Nv, 2] projected pixel coords (body-HD crop space)
    z: torch.Tensor           # [B, Nv]   camera-space z (larger = farther; see depth_ordering.py)
    vert_ids: np.ndarray      # [Nv] MANO vertex ids
    smplx_vert_ids: np.ndarray  # [Nv] SMPL-X vertex ids
    valid: torch.Tensor       # [B, Nv] bool (in front of camera & finite)


def hand_vertices_from_proj(proj_vertices: torch.Tensor, smplx2mano_ind: dict, side: str) -> torch.Tensor:
    """[B, 10475, 3] -> [B, 778, 3] in MANO order (xy pixels, z camera depth)."""
    return proj_vertices[:, np.asarray(smplx2mano_ind[f'{side}_hand'], dtype=np.int64)]


def gather_hand_parts(proj_vertices: torch.Tensor, smplx2mano_ind: dict, index: HandPartIndex,
                      sides: Sequence[str] = SIDES) -> Dict[str, Dict[str, PartData]]:
    """
    parts[side][segment] -> PartData, e.g. parts['left']['index_dist'], parts['right']['palm'].
    """
    out = {}
    for side in sides:
        m2s = np.asarray(smplx2mano_ind[f'{side}_hand'], dtype=np.int64)
        hv = proj_vertices[:, m2s]
        out[side] = {}
        for seg in SEGMENTS:
            vids = index.verts[seg]
            p = hv[:, vids]
            z = p[..., 2]
            out[side][seg] = PartData(side=side, segment=seg, xy=p[..., :2], z=z, vert_ids=vids,
                                      smplx_vert_ids=m2s[vids], valid=torch.isfinite(z) & (z > 0))
    return out
