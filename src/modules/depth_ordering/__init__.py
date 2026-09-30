"""
Depth-aware local hand-part ordering (experimental, branch experiment/depth-aware-part-ordering).
Consumes PRECOMPUTED Video-Depth-Anything relative depth; uses ordinal constraints only.
See docs/DEPTH_ORDERING.md.
"""
from .hand_parts import (HandPartIndex, PartData, SEGMENTS, build_mano_part_labels, gather_hand_parts,
                         segment_part, segment_level)
from .depth_io import DepthSource, load_depth_source_for_video, load_frame_map
from .depth_ordering import (DepthOrderConfig, DepthOrderTargetBuilder, DepthOrderTargets, PairInstance,
                             depth_order_loss, local_median_z, rasterize_segments)
