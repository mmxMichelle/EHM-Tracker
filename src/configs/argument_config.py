import os.path as osp
from dataclasses import dataclass,field
import tyro
from typing_extensions import Annotated
from .base_config import PrintableConfig
from typing import List, Optional

@dataclass(repr=False)
class ArgumentConfig(PrintableConfig):
    ########## input arguments ##########
    #source_dir: Annotated[str, tyro.conf.arg(aliases=["-s"])] ='assets/videos'         
    output_dir: Annotated[str, tyro.conf.arg(aliases=["-o"])] = 'outputs/test_data/'    # path to driving video or template (.pkl format)

    save_vis_video: bool = False
    tracking_with_interval : bool =False
    save_images: bool = False
    save_visual_render: bool = False
    check_hand_score: float = 0.7
    not_check_hand: bool = False

    # ---- [depth-ordering] experimental; defaults keep the baseline unchanged ----
    enable_depth_ordering: bool = False
    vda_depth_root: str = ''              # file, or dir with <video_name>.npy|.npz|_depths.npz
    vda_frame_map: str = ''               # json (or dir of <video_name>.json); default <saving_root>/frame_index_map.json
    vda_depth_space: str = 'original'     # 'original' (video frame pixels) | 'body_hd' (1024 crop pixels)
    vda_larger_is_closer: bool = True
    lambda_depth_order: float = 100.0
    depth_pair_margin: float = 0.01
    interaction_2d_threshold: float = 12.0
    depth_refresh_every: int = 100
    depth_hand_pose_lr: Optional[float] = None   # e.g. 1e-5 / 5e-5 / 1e-4; None keeps baseline 1e-5
    depth_debug_vis: bool = True
    
    visible_gpus: Annotated[str, tyro.conf.arg(aliases=["-v"])] = '0,'        # visible gpus, separated by `,`, e.g. 0, 1
    part_lst: Annotated[str, tyro.conf.arg(aliases=["-p"])] = 'nan'           # starts and ends for subprocessing, e.g. 20,40, default: None
    n_divide: Annotated[str, tyro.conf.arg(aliases=["-n"])] = 8               # max divide number
    in_root: Annotated[str, tyro.conf.arg(aliases=["-i"])]  = 'assets/videos' # the input video file paths
    more_in_root: Annotated[List[str], tyro.conf.arg(aliases=["-m"])] = field(default_factory=list)