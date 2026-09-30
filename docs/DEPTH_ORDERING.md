# Depth-aware local hand-part ordering (experimental MVP)

Branch: `experiment/depth-aware-part-ordering` (baseline commit `55cfe1c`).
Default: **disabled**. With `enable_depth_ordering=False` (or no VDA maps) the
refiner runs the original code path; `tools/depth_order_smoke_test.py` checks this
statically (AST equivalence with 55cfe1c) and, on GPU, numerically.

## What it does

Uses precomputed Video Depth Anything (VDA) **relative** depth to add a
confidence-weighted **ordinal** (hinge) constraint on the camera-space z of
**local** hand-surface regions at 2D crossings / near-contacts, inside
`EhmOptimizer.optimize()`:

```
total_loss = loss_3d + loss_2d + loss_prior + mtn_reg_loss            # unchanged baseline
           + lambda_depth_order * loss_depth_order                    # only when enabled
```

All baseline terms (including `loss_3d_z` and `loss_3d_hand_l/r`) are untouched.

## Conventions (verified)

* `proj_vertices[..., 2]` is **camera-space z**: `GS_Camera.transform_points_to_ndc`
  (src/utils/graphics.py) computes `X_view = R X + T`, the projection has `P[3,2]=+1`
  so `w = z_view`, and it sets `ndc[...,2] = w`; the screen transform only rescales x,y.
  Visible points have z > 0; **larger z = farther**.
  (`tools/depth_order_smoke_test.py --clip` re-checks this numerically: C0.)
* VDA: **larger value = closer** (`vda_larger_is_closer=True`; set False for metric-like depth).
* So if VDA says A is in front: `z_A < z_B`, hinge `relu(margin + z_A - z_B)`.

## Semantic parts (src/modules/depth_ordering/hand_parts.py)

16 phalanx-level segments per hand: `palm`, and `{thumb,index,middle,ring,pinky}_{prox,mid,dist}`.

* **Model-derived (exact):** each MANO vertex is assigned to the joint with the
  largest MANO LBS skinning weight (`argmax lbs_weights`, MANO_RIGHT.pkl; MANO_LEFT
  gives an identical assignment). EHM indexes both hands in MANO vertex order via
  `smplx2mano_ind`, so one label array serves both hands.
* **Heuristic:** reading a joint's dominant-skinning region as an anatomical
  surface region (e.g. "index distal phalanx") is an approximation; LBS regions
  blend near joints. MANO joint j drives the bone j -> child, hence
  `*1 -> prox`, `*2 -> mid`, `*3 -> dist`. The thumb has two phalanges:
  `thumb_prox` = MANO thumb1 region (~ metacarpal / thenar), `thumb_mid` = proximal
  phalanx, `thumb_dist` = distal phalanx.
* The wrist-cut ring excluded by EHM (`selected_hand_ver.npy`, 107 vertices) is
  ignored. Consequence: `thumb_prox` keeps only 8 vertices, `palm` 107.
* Counts: palm 107 | thumb 8/28/54 | index 33/34/54 | middle 35/30/52 | ring 30/38/54 | pinky 23/39/52.
* Replaceable: `HandPartIndex(labels=<exact labels>, faces_all=...)`.

## Local interaction detection (depth_ordering.py)

Per frame (no gradient; rebuilt every `refresh_every` steps from the current fit):

1. Rasterise each segment's projected MANO triangles (union, cell-centre sampling,
   stride 2 px) on a local canvas; one canvas for both hands if they are within
   the interaction threshold, otherwise one per hand.
2. Candidate pairs: inter-hand (16x16) and intra-hand (self-occlusion). Only pairs whose
   footprints **overlap** (each connected overlap component = one instance), or,
   inter-hand only, come within `interaction_2d_threshold` px. Same-hand pairs
   must overlap, and kinematically adjacent segments (same finger; palm vs any
   `*_prox`) are never paired, because they always touch in 2D.

## Local VDA sampling + confidence

* Region: overlap component dilated by `region_dilate_px` (proximity pairs: disk
  of `proximity_radius_px` around the closest points).
* A-samples: pixels in the region covered **only** by segment A (no other hand
  segment), eroded by `edge_erode_px`, inside the matting mask. Same for B.
  **Overlap pixels are never used as A or B depth** (only the front surface is
  visible there). They are used only for a consistency check: VDA at the overlap
  must resemble the claimed *front* side; otherwise confidence = 0 (this also
  rejects physically impossible depth, e.g. a flipped map).
* Statistics: median and IQR per side; `delta = d_A - d_B` (> 0 means A is closer).
* Confidence `c = c_sep * c_n * c_geo * c_cons`:
  `c_sep = clip((|delta|/(iqr_A+iqr_B+eps) - tie)/(full - tie), 0, 1)` (a near tie gives 0),
  `c_n = 0 if min(n_A, n_B) < min_samples else min(1, n/full_samples)`,
  `c_geo` = overlap/proximity score, `c_cons` = overlap consistency.
  `eps` and the absolute tie floor are relative to the frame's VDA hand spread (p90 - p10).

## Loss

```
z_A = median(z of A's vertices projecting into the local region)   # torch.nanquantile, differentiable
s_k = +1 if VDA puts A in front, else -1
L_depth = sum_k c_k * relu(margin + s_k (z_A - z_B)) / sum_k c_k
```

## Usage

```
python tracking_video.py -i <video> -o <out> --enable_depth_ordering \
    --vda_depth_root <dir or file> --vda_frame_map <frame_map.json> --vda_depth_space original \
    --lambda_depth_order 100 --depth_pair_margin 0.01 --depth_hand_pose_lr 5e-5
```

* Depth maps: `[T,H,W]` `.npy` / `.npz` (`depths` key) / directory of frames, looked
  up as `<root>/<video_name>{.npy,.npz,_depths.npz}` or `root` itself if it is a file.
* Frame map: JSON `{"frame_000000": 0, ...}` or `{"frames": {...}, "image_size": [H, W]}`.
  Tracker keys are *not* reliably `video_idx / interval` (skipped frames). When enabled,
  extraction writes `<saving_root>/frame_index_map.json`. For already-extracted clips, pass an explicit map.
* `vda_depth_space`: `original` (video pixels, mapped via `body_crop['M_c2o-hd']`, with
  the depth map resized relative to `image_size`) or `body_hd` (1024 crop pixels).
* `depth_hand_pose_lr`: overrides the 1e-5 LR of `left/right_hand_pose` **only when
  enabled** (try 1e-5 / 5e-5 / 1e-4). The baseline LR is unchanged.

Outputs (under `<saving_root>`):
`depth_order_logs/batch_XX.jsonl` (per `log_every` steps: `loss_depth_order`,
weighted loss, `loss_3d_z`, `loss_3d_hand_l`, `loss_3d_hand_r`, pair counts by type,
mean confidence, wrong-order count; gradient norms of each of these terms w.r.t.
left/right hand pose at steps 0, mid, last), `batch_XX_summary.json` (wrong-order
before/after on the initial targets), `batch_XX_pairs_initial.json`, and
`visual_results/vis_depth_order/*.png`.

## Tests

```
python tools/depth_order_smoke_test.py --synthetic            # CPU, no licensed assets
python tools/depth_order_smoke_test.py --clip --track_dir ... --vda ... --frame_map ...   # GPU
```

## Limitations (do not overclaim)

* It only constrains front/back **ordering where VDA gives reliable visible local
  support on both sides**. It does **not** recover fully occluded fingers: if one
  side has no exclusive visible pixels, confidence is 0. Those need temporal / learned priors later.
* Pairs and visibility come from the *current* tracker footprint. If the 2D fit is
  badly wrong, samples can fall on the wrong surface. The matting mask, the edge erosion,
  the IQR and the overlap-consistency check reduce this but do not remove it.
* The SMPL-X forearm/body is not in the footprint, so it is not treated as an occluder.
* `loss_3d_z` (anchors all vertex z to the initialisation) and `loss_3d_hand_*`
  (HaMeR wrist-relative 3D) are kept unchanged and compete with this term. If
  `hand_pose_change_vs_baseline` stays ~0, use the logged relative magnitudes and
  gradient norms to decide whether local z anchoring should be relaxed.
* `lambda_depth_order` and `margin` are untuned.
