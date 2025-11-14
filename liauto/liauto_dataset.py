import os
import io
import math
import numpy as np
import pandas as pd
import torch
import cv2
import random
import json
import logging
from contextlib import redirect_stdout
from os import path as osp
from typing import Dict, List, Tuple, OrderedDict
from torch.utils.data import Dataset

import asset
from liauto.utils.config import Config
from liauto.rewards import Rewarder
from copy import deepcopy
from liauto.pipelines import LoadPointsFromLiautoFiles, LoadNavigation

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

class DiffSimConfig:
    """
    Configuration for LiDAR point cloud processing.
    """
    def __init__(self):
        # LiDAR configuration
        self.lidar_min_x = -32
        self.lidar_max_x = 32
        self.lidar_min_y = -32
        self.lidar_max_y = 32
        self.pixels_per_meter = 4
        self.hist_max_per_pixel = 15
        self.max_height_lidar = 3.5
        self.lidar_split_height = 0.5
        self.use_ground_plane = False


class OnlineDataset(Dataset):
    """
    Dataset for online data loading without caching.
    Loads and processes data on-the-fly during training.
    """
    def __init__(self, 
                dataset_config, 
                whitelist_clips=None, 
                blacklist_clips=None,
                max_clip_len=int(os.environ.get("MAX_CLIP_LEN", 1000000)),
                clip_start_idx=int(os.environ.get("CLIP_START_IDX", 0)),
                min_clip_len=int(os.environ.get("MIN_CLIP_LEN", 2)),
                test_mode=False,
                history_frames=0,
                filter_info_list=None,
                jsonl_npz_file=None):
        """
        Initialize the online dataset.
        
        Args:
            dataset_config: Configuration for datasets to load.
            whitelist_clips: List of clip IDs to include. If None, all clips are included.
            blacklist_clips: List of clip IDs to exclude.
            max_clip_len: Maximum length of clips to use.
            clip_start_idx: Starting index for clip subsetting.
            min_clip_len: Minimum length of clips to use.
            test_mode: Whether to run in test mode.
        """
        self.dataset_config = dataset_config
        self.load_interval = 1
        self.seq_split_num = 1
        self.max_clip_len = max_clip_len
        self.clip_start_idx = clip_start_idx
        self.min_clip_len = min_clip_len
        self.test_mode = test_mode
        self.history_frames = history_frames
        self.scene_to_label_pathes = {}
        self.filter_info_set = set()
        # --- New JSONL‑NPZ direct loading support ---
        self.use_jsonl_npz = jsonl_npz_file is not None
        if self.use_jsonl_npz:
            # Load records from the JSONL file where each line has {"npz_path": ..., "scene_token": ...}
            self.jsonl_records = []
            with open(jsonl_npz_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    if "npz_path" not in rec:
                        raise ValueError(f"Missing 'npz_path' in JSONL record: {line}")
                    self.jsonl_records.append(rec)

            # Build a minimal scene/group mapping (each sample is its own group)
            self.flag = list(range(len(self.jsonl_records)))
            self.scene_to_idx_range = OrderedDict(
                (i, (i, i)) for i in range(len(self.jsonl_records))
            )
        else:
            self.use_jsonl_npz = False

        if filter_info_list:
            # 如果传入的是 JSON 文件路径，则先加载
            if isinstance(filter_info_list, str):
                try:
                    with open(filter_info_list, 'r') as f:
                        filter_info_list = json.load(f)
                except Exception as e:
                    raise ValueError(f"无法从 {filter_info_list} 加载过滤列表: {e}")
            # 构建查找用的集合
            self.filter_info_set = {
                (entry.get('scene_token'), str(entry.get('timestamp')).replace('.', ''))
                for entry in filter_info_list
            }
        # Load clip lists
        self.blacklist_clips = self.load_cliplist(blacklist_clips)
        self.whitelist_clips = self.load_cliplist(whitelist_clips)
        
        # Load annotations and set up dataset structure
        if not self.use_jsonl_npz:
            self.load_weighted_annotations()
            self.set_sequence_group_and_flag()
        
        # Initialize additional components
        self.rewarder = Rewarder()
        self.global_path_config = None
        self.load_global_path_config()
        self.path_mapping = self.preprocess_paths()
        self.lidar_p = LoadPointsFromLiautoFiles(coord_type='LIDAR')
        self.diff_navi = LoadNavigation(sample_pts_num=8, pc_range=[-19.8, -24.0, -3.5, 119.4, 24.0, 4.5])
        
        # For processing features
        self.diff_sim_config = DiffSimConfig()
        
        # JWT token for asset access
        if self.use_jsonl_npz:
            self.valid_indices = None  # handled uniformly later
        self.jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJyZW5ob25ncWlhbkBsaXhpYW5nLmNvbSIsImlzcyI6ImxwYWkiLCJpYXQiOjE3MjEwMTAzMTEsImp0aSI6IjE4NmY0OGZhLWIyZTYtNDZhYy1iMTQxLThjNGYzZWVlM2RlZCJ9.kyaLXG9ZqSzJJSH_PZeK7_SrE7Xfn1T5LhYz03dK5MM"

        if filter_info_list:
            # Fast filtering: scan each scene file only once and map to global index
            valid = []
            from tqdm import tqdm
            pbar = tqdm(total=len(self.flag), desc="Filtering scenes", unit="items")
            filter_set = self.filter_info_set

            for scene_idx, (start_idx, end_idx) in self.scene_to_idx_range.items():
                # Load the per‑scene label paths if not already cached
                if scene_idx not in self.scene_to_label_pathes:
                    with open(self.scene_path_list[scene_idx], "r") as f:
                        self.scene_to_label_pathes[scene_idx] = [
                            line.strip() for line in f
                        ]
                label_paths = self.scene_to_label_pathes[scene_idx]

                for local_i, label_path in enumerate(label_paths):
                    pbar.update(1)
                    fname = os.path.basename(label_path)
                    scene_token = fname.split("**")[0]
                    # strip the .npz extension after the timestamp
                    timestamp = fname.split("_")[-1].split(".")[0]
                    if (scene_token, timestamp) in filter_set:
                        valid.append(start_idx + local_i)

            pbar.close()
            self.valid_indices = valid
            print(
                f"✅ 过滤完成：共找到 {len(self.valid_indices)} / {len(self.flag)} 个有效索引"
            )

        self.all_indices = self.valid_indices if self.valid_indices else list(range(len(self.flag)))
        self.bad_indices = set()          # 永久跳过的索引

    def load_cliplist(self, cliplist_files):
        """
        Load a list of clip IDs from files.
        
        Args:
            cliplist_files: Path to clip list files or list of paths.
            
        Returns:
            set: Set of clip IDs.
        """
        if cliplist_files is None:
            return set()
        elif isinstance(cliplist_files, str):
            cliplist_files = [cliplist_files]

        clips = set()
        for cliplist_file in cliplist_files:
            with open(cliplist_file, "r") as f:
                clips.update(set([line.strip() for line in f.readlines()]))
        return clips
    
    def filter_mini_clip(self, num_frames_labeled):
        """
        Filter clip lengths based on max_clip_len setting.
        
        Args:
            num_frames_labeled: List of frame counts for each clip.
            
        Returns:
            list: Filtered frame counts.
        """
        if (not self.test_mode) or self.max_clip_len <= 0:
            return num_frames_labeled
        mini_clip_num_frames_labeled = []
        for num_labeled in num_frames_labeled:
            end = min(num_labeled, self.clip_start_idx + self.max_clip_len)
            start = max(0, end - self.max_clip_len)
            mini_clip_num_frames_labeled.append(end - start)
        return mini_clip_num_frames_labeled

    def load_clip_labels_info(self, ann_file):
        """
        Load clip label information from annotation file.
        
        Args:
            ann_file: Path to annotation file.
            
        Returns:
            tuple: Tuple of (clips, num_frames, num_frames_labeled).
        """
        df = pd.read_csv(ann_file, header=None)
        if df is None:
            print(f"empty! {ann_file}")
            return [], [], []

        if not osp.exists(df[0].tolist()[0]):
            clip_labels_root = osp.dirname(osp.dirname(ann_file))
            clips = [
                clip_label_file
                if "s3://s3" in clip_label_file
                else osp.join(clip_labels_root, clip_label_file)
                for clip_label_file in df[0].tolist()
            ]
        else:
            clips = df[0].tolist()

        if len(df.columns) == 3 or len(df.columns) == 4:
            num_frames = [int(num) for num in df[1].tolist()]
            num_frames_labeled = [int(num) for num in df[2].tolist()]

            clip_infos = zip(clips, num_frames, num_frames_labeled)

            # Extract whitelist clips
            def filter_white_list(x):
                return osp.basename(osp.dirname(x[0])) in self.whitelist_clips

            if self.whitelist_clips:
                clip_infos = list(filter(filter_white_list, clip_infos))

            # Filter clips by min length and blacklist
            def filter_mini_clip(x):
                return x[2] > self.min_clip_len

            def filter_black_list(x):
                return osp.basename(osp.dirname(x[0])) not in self.blacklist_clips

            clip_infos = list(filter(filter_mini_clip, clip_infos))
            clip_infos = list(filter(filter_black_list, clip_infos))

            if len(list(clip_infos)) == 0:
                clips, num_frames, num_frames_labeled = [], [], []
            else:
                clips, num_frames, num_frames_labeled = zip(*clip_infos)
        else:
            raise ValueError(
                f"Unsupported format! ann_file should be 3 or 4 columns: {ann_file}"
            )

        print(
            f"load {len(clips)} clips, {sum(num_frames_labeled)} labeled-frames from {ann_file}"
        )
        return clips, num_frames, num_frames_labeled
    
    def load_weighted_annotations(self):
        """
        1. load cbgs annotations
        Implementation of
        paper `Class-balanced Grouping and Sampling for Point Cloud 3D Object
        Detection <https://arxiv.org/abs/1908.09492>`_.
        Balance the classes of scenes.
        2. set sequence group flag
        """

        # load cbgs annotations
        data_infos_all = []
        txt2dataset_weight_list = []
        txt2interval_list = []
        txt2min_interval_list = []
        txt2max_interval_list = []
        txt2data_list = []

        self.scene_path_list = []
        num_frames_list = []
        num_frames_labeled_list = []

        for dataset_group in self.dataset_config:
            dataset_weight = dataset_group.get("dataset_weight", 1.0)
            interval = dataset_group.get("interval", self.load_interval)
            min_interval = dataset_group.get("min_interval", 5)
            max_interval = dataset_group.get("max_interval", 7)
            label_files = dataset_group["label_files"]
            for ann_file in label_files:
                # meta = self.load_annotation_meta(ann_file) if self.with_meta else None
                (
                    scene_pathes,
                    num_frames,
                    num_frames_labeled,
                ) = self.load_clip_labels_info(ann_file)
                num_frames_labeled = self.filter_mini_clip(num_frames_labeled)
                self.scene_path_list += scene_pathes
                num_frames_list += num_frames
                num_frames_labeled_list += num_frames_labeled

        # build scene_to_idx_range from scene_path_list, num_frames_labeled
        self.scene_to_idx_range = OrderedDict()
        scene_start_idx = 0
        for idx, num_frames_labeled in enumerate(num_frames_labeled_list):
            self.scene_to_idx_range[idx] = (
                scene_start_idx,
                scene_start_idx + num_frames_labeled - 1,
            )
            scene_start_idx += num_frames_labeled

        print(
            f"load {len(self.scene_to_idx_range)} clips with num_frames_labeled > {self.min_clip_len}"
        )

        if self.seq_split_num != 1:
            if self.seq_split_num == "all":
                self.flag = np.array(range(len(self.flag)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(
                            range(
                                0,
                                bin_counts[curr_flag],
                                math.ceil(bin_counts[curr_flag] / self.seq_split_num),
                            )
                        )
                        + [bin_counts[curr_flag]]
                    )

                    for sub_seq_idx in (
                        curr_sequence_length[1:] - curr_sequence_length[:-1]
                    ):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert (
                    len(np.bincount(new_flags))
                    == len(np.bincount(self.flag)) * self.seq_split_num
                )
                self.flag = np.array(new_flags, dtype=np.int64)
        return

    @staticmethod
    def load_np_file(np_file):
        """
        Load a NumPy file, handling both .npy and .npz formats.
        
        Args:
            np_file: Path to NumPy file.
            
        Returns:
            dict or list: Loaded data.
        """
        if np_file.endswith(".npz"):
            return np.load(np_file, allow_pickle=True)["data"].tolist()
        else:
            return np.load(np_file, allow_pickle=True).tolist()
        
    def set_sequence_group_and_flag(self):
        """
        Set sequence groups and flags for dataset indexing.
        """
        self.flag = [
            scene_idx
            for scene_idx, (start_idx, end_idx) in self.scene_to_idx_range.items()
            for _ in range(end_idx - start_idx + 1)
        ]
        self.group_idx_to_sample_idxs = {
            scene_idx: list(range(start_idx, end_idx + 1))
            for scene_idx, (start_idx, end_idx) in self.scene_to_idx_range.items()
        }
 
    def get_info_path(self, index):
        # Shortcut for direct‑JSONL mode
        if getattr(self, "use_jsonl_npz", False):
            return self.jsonl_records[index]["npz_path"]
        scene_ind = self.flag[index]
        scene_path = self.scene_path_list[scene_ind]
        if scene_ind not in self.scene_to_label_pathes:
            with open(scene_path, "r") as f:
                lines = f.readlines()
            scene_label_pathes = [line.strip() for line in lines]
            if self.test_mode and self.max_clip_len > 0:
                end = min(
                    len(scene_label_pathes), self.clip_start_idx + self.max_clip_len
                )
                start = max(0, end - self.max_clip_len)
                scene_label_pathes = scene_label_pathes[start:end]
            self.scene_to_label_pathes[scene_ind] = scene_label_pathes
        else:
            scene_label_pathes = self.scene_to_label_pathes[scene_ind]

        sample_ind = index - self.scene_to_idx_range[scene_ind][0]
        scene_dir = os.path.dirname(scene_path)
        info_path = os.path.join(scene_dir, scene_label_pathes[sample_ind])

        return info_path
    
    def __len__(self):
        """
        Get the total number of samples in the dataset.
        
        Returns:
            int: Dataset length.
        """
        if self.valid_indices:
            # return len(self.valid_indices)
            return len(self.all_indices)
        else:
            return len(self.flag)
        
    def prepare_raw_data(self, index):
        """
        Prepare raw data for a specific sample index.
        
        Args:
            index: Dataset index.
            
        Returns:
            dict: Raw data dictionary with scene information.
        """
        info_path = self.get_info_path(index)
        info = self.load_np_file(info_path.strip())
        # Ensure a scene_token exists even if the npz content lacks it
        if "scene_token" not in info:
            info["scene_token"] = os.path.basename(info_path).split("**")[0]
        # Filter out entries whose (scene_token, timestamp) is not in the provided list

        result = {}

        config = asset.config(
            env="prod",
            jwt_token=self.jwt_token,
        )
        image_path = info['cams']
        dataset_name = image_path["CAM_FRONT"]["data_path"].split("/")[3]
        dataset_version = image_path["CAM_FRONT"]["data_path"].split("/")[4]
        dataset_resource = (
            "datasets/" + dataset_name + "/versions/" + dataset_version
        )
        dataset = asset.resource(dataset_resource, config=config)
        scene_token = info["scene_token"]
        result[scene_token] = {}

        def load_img(path):
            with redirect_stdout(io.StringIO()):
                image_data = dataset.get_file(s3_path=path).read()
            image_array = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            return img

        front_img = load_img(info['cams']['CAM_FRONT']['data_s3_path'])
        left_img = load_img(info['cams']['CAM_FRONT_LEFT']['data_s3_path'])
        right_img = load_img(info['cams']['CAM_FRONT_RIGHT']['data_s3_path'])
        result[scene_token]["images"] = {
            "left": left_img,
            "front": front_img,
            "right": right_img,
        }

        result[scene_token]["velocity"] = np.array(
            [info['ego_info']['velocity']['x'], info['ego_info']['velocity']['y']]
        )

        result[scene_token]["acceleration"] = np.array(
            [info['ego_info']['acc']['x'], info['ego_info']['acc']['y']]
        )

        # Process past trajectories
        past_trajs = info['sdc_motion']['past_traj']  # [20, 2]
        past_trajs_extended = np.hstack((past_trajs, np.zeros((past_trajs.shape[0], 1))))  # [20, 3]

        # Process future trajectories
        motion_trajs = info['sdc_motion']['motion_traj']  # [60, 2]
        motion_trajs_mask = info['sdc_motion']['motion_traj_mask']
        motion_trajs_extended = np.hstack((motion_trajs, np.zeros((motion_trajs.shape[0], 1))))  # [60, 3]
        result[scene_token]["history_poses"] = past_trajs_extended  # [20, 3]
        result[scene_token]["future_trajectory"] = motion_trajs_extended  # [60, 3]
        result[scene_token]["future_trajectory_mask"] = motion_trajs_mask # [60, 2]

        # Process LiDAR data
        at128_pts_filename = self.get_path(
                info, "lidar_at128_path", "lidar_at128_s3_path", ""
        )
        lidar_dict = {
                    'at128_pts_filename': at128_pts_filename,
                    'lidar2ego_rotation_at128': info['lidar2ego_rotation_at128'],
                    'lidar2ego_translation_at128': info['lidar2ego_translation_at128']
        }
        lidar_dict = self.lidar_p(lidar_dict)
        lidar_points = lidar_dict['points'].tensor
        result[scene_token]["point_cloud"] = lidar_points
        result[scene_token]["log_name"] = info['token']
        result[scene_token]["npz_path"] = info_path.strip()

        # Process navigation data
        info["navigation_info"] = deepcopy(info["navigation_prior"])
        diffusion_navi = self.diff_navi(info)['diffusion_navi']
        result[scene_token]["diffusion_navi"] = diffusion_navi
        
        # Return only the data for this scene (not the dict of scenes)
        return result[scene_token]
    
    def prepare_camera_feature(self, images: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Process camera images into model-ready format.
        
        Args:
            images: Dictionary containing 'left', 'front', 'right' camera images.
            
        Returns:
            torch.Tensor: Processed camera tensor of shape [3, 256, 1024].
        """
        # Check input images
        required_keys = ['left', 'front', 'right']
        for key in required_keys:
            if key not in images:
                raise ValueError(f"Missing required key: {key} in images dictionary")

        left = images['left']
        front = images['front']
        right = images['right']

        # Scale front image proportionally
        scale = left.shape[0] / front.shape[0]  # Calculate height scale ratio
        new_width = int(front.shape[1] * scale)  # Calculate new width with same ratio
        front_resized = cv2.resize(front, (new_width, left.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Calculate target width for 4:1 aspect ratio
        height = left.shape[0]
        required_width = height * 4  # Target aspect ratio 4:1

        # Calculate width for each part
        side_width = (required_width - front_resized.shape[1]) // 2
        
        # Center crop each image
        def center_crop(img, target_width):
            start = (img.shape[1] - target_width) // 2
            end = start + target_width
            return img[:, start:end]

        # Crop each image
        l0 = center_crop(left, side_width)
        f0 = front_resized
        r0 = center_crop(right, side_width)

        # Concatenate images
        stitched = np.concatenate([l0, f0, r0], axis=1)

        # Scale proportionally to target size
        scale = min(1024 / stitched.shape[1], 256 / stitched.shape[0])
        new_width = int(stitched.shape[1] * scale)
        new_height = int(stitched.shape[0] * scale)
        
        # First scale proportionally
        resized = cv2.resize(stitched, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create final image with padding
        final = np.zeros((256, 1024, 3), dtype=np.uint8)
        y_offset = (256 - new_height) // 2
        x_offset = (1024 - new_width) // 2
        final[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized

        # Convert to tensor, change shape to [C,H,W], normalize
        tensor = torch.from_numpy(final).permute(2, 0, 1).float() / 255.0

        return tensor

    def prepare_lidar_feature(self, point_cloud: np.ndarray) -> torch.Tensor:
        """
        Process LiDAR point cloud into model-ready format.
        
        Args:
            point_cloud: Point cloud array of shape [N,3].
            
        Returns:
            torch.Tensor: Processed LiDAR tensor of shape [C,256,256].
        """
        config = self.diff_sim_config
        
        def splat_points(points):
            # Create 256x256 grid
            xbins = np.linspace(
                config.lidar_min_x,
                config.lidar_max_x,
                (config.lidar_max_x - config.lidar_min_x) * config.pixels_per_meter + 1,
            )
            ybins = np.linspace(
                config.lidar_min_y,
                config.lidar_max_y,
                (config.lidar_max_y - config.lidar_min_y) * config.pixels_per_meter + 1,
            )
            hist = np.histogramdd(points[:, :2], bins=(xbins, ybins))[0]
            hist[hist > config.hist_max_per_pixel] = config.hist_max_per_pixel
            return hist / config.hist_max_per_pixel

        # Remove points above vehicle
        point_cloud = point_cloud[point_cloud[..., 2] < config.max_height_lidar]

        # Separate points above and below ground plane
        below = point_cloud[point_cloud[..., 2] <= config.lidar_split_height]
        above = point_cloud[point_cloud[..., 2] > config.lidar_split_height]

        # Generate features
        above_features = splat_points(above)
        if config.use_ground_plane:
            below_features = splat_points(below)
            features = np.stack([below_features, above_features], axis=0)
        else:
            features = np.expand_dims(above_features, axis=0)

        return torch.from_numpy(features).float()

    def prepare_features(self, data: Dict) -> Dict[str, torch.Tensor]:
        """
        Prepare model input features from raw data.
        
        Args:
            data: Raw data dictionary.
            
        Returns:
            dict: Dictionary of preprocessed features ready for model input.
        """
        features = {}

        # 1. Camera feature [3, 256, 1024]
        features["camera_feature"] = self.prepare_camera_feature(data["images"])

        # 2. LiDAR feature [C, 256, 256]
        features["lidar_feature"] = self.prepare_lidar_feature(data["point_cloud"])

        # 3. Status feature [driving_command(4) + velocity(2) + acceleration(2)]
        features["status_feature"] = torch.cat([
            torch.tensor([0, 0, 0, 1], dtype=torch.float32),  # Default driving command
            torch.tensor(data["velocity"], dtype=torch.float32),
            torch.tensor(data["acceleration"], dtype=torch.float32),
        ])

        # 4. Past trajectory [4, 3] - (x,y,yaw) x 4 historical frames
        features["past_trajectory"] = torch.tensor(
            data["history_poses"][::5][:3],  # Should be shape [3,3]
            dtype=torch.float32
        )
        features["past_trajectory"] = features["past_trajectory"].flip(dims=[0])
        features["past_trajectory"] = torch.cat([features["past_trajectory"], 
                                                torch.tensor([[0.0, 0.0, 0.0]])], dim=0)

        # 5. Navigation feature
        features["diffusion_navi"] = torch.tensor(data["diffusion_navi"], dtype=torch.float32)
        
        return features

    def prepare_targets(self, data: Dict) -> Dict[str, torch.Tensor]:
        """
        Prepare model training targets from raw data.
        
        Args:
            data: Raw data dictionary.
            
        Returns:
            dict: Dictionary of target tensors for model training.
        """
        return {
            "trajectory": torch.tensor(
                data["future_trajectory"][::5][:8],  # Should be shape [8,3]
                dtype=torch.float32
            ),
            "traj_mask": torch.tensor(
                data["future_trajectory_mask"][::5][:8],
                dtype=torch.float32
            ),
            "npz_path": data["npz_path"],
        }

    def get_history_indices(self, index):
        scene_ind = self.flag[index]
        sample_ind = index - self.scene_to_idx_range[scene_ind][0]
        
        indices = []
        for i in range(self.history_frames, -1, -1):
            hist_sample_ind = sample_ind - i
            if hist_sample_ind >= 0:
                hist_index = self.scene_to_idx_range[scene_ind][0] + hist_sample_ind
                indices.append(hist_index)
            else:
                # 如果历史帧超出场景边界,使用第一帧填充
                indices.append(self.scene_to_idx_range[scene_ind][0])
        return indices

    def prepare_features_with_history(self, history_data):
        # 处理当前帧(最后一帧)的特征
        current_features = self.prepare_features(history_data[-1])
        
        # 添加历史帧特征
        features = current_features.copy()
        
        # 创建历史帧特征列表
        history_camera = []
        history_lidar = []
        history_status = []
        
        # 处理所有帧(包括当前帧)
        for data in history_data:
            # 相机特征
            camera_feature = self.prepare_camera_feature(data["images"])
            history_camera.append(camera_feature)
            
            # 激光雷达特征
            lidar_feature = self.prepare_lidar_feature(data["point_cloud"])
            history_lidar.append(lidar_feature)
            
            # 状态特征
            status = torch.cat([
                torch.tensor([0, 0, 0, 1], dtype=torch.float32),  # 默认驾驶命令
                torch.tensor(data["velocity"], dtype=torch.float32),
                torch.tensor(data["acceleration"], dtype=torch.float32),
            ])
            history_status.append(status)
        
        # 将历史特征堆叠起来
        features["history_camera_feature"] = torch.stack(history_camera)  # [T, 3, 256, 1024]
        features["history_lidar_feature"] = torch.stack(history_lidar)    # [T, C, 256, 256]
        features["history_status_feature"] = torch.stack(history_status)  # [T, 8]
        
        return features

    def __getitem__(self, index):

        logger = logging.getLogger(__name__)

        while True:
            if index >= len(self.all_indices):
                raise IndexError("no valid samples left")

            real_idx = self.all_indices[index]
            if real_idx in self.bad_indices:          # 理论上不会发生，稳妥再检查
                self.all_indices.pop(index)
                continue

            try:
                if self.history_frames == 0:
                    raw = self.prepare_raw_data(real_idx)
                    return self.prepare_features(raw), self.prepare_targets(raw)
                else:
                    hist_idxs = self.get_history_indices(real_idx)
                    hist_raw = [self.prepare_raw_data(i) for i in hist_idxs]
                    feats = self.prepare_features_with_history(hist_raw)
                    targs = self.prepare_targets(hist_raw[-1])
                    return feats, targs
            except Exception as e:
                logger.warning(f"bad sample {real_idx}: {e} — skipping permanently")
                self.bad_indices.add(real_idx)
                self.all_indices.pop(index)
                continue

    def load_global_path_config(self):
        """
        Load global path configuration, only load once.
        """
        if self.global_path_config is None:
            path_config_str = os.environ.get("LPAI_OFS_PATH_CONVERT", "{}")
            self.global_path_config = json.loads(path_config_str)

    def preprocess_paths(self):
        """
        Preprocess path mappings for efficient replacement.
        
        Returns:
            dict: Sorted path mapping dictionary.
        """
        sorted_paths = sorted(self.global_path_config.keys(), key=len, reverse=True)
        return {k: self.global_path_config[k] for k in sorted_paths}

    def get_path(self, info, default_key, ofs_key, default_value=""):
        """
        Get path from info using mapping.
        
        Args:
            info: Data info dictionary.
            default_key: Default key to use.
            ofs_key: OFS key to use.
            default_value: Default value if path not found.
            
        Returns:
            str: Converted path.
        """
        # Replace OFS path
        ofs_path = info[ofs_key]
        for s3_path, ofs_replacement in self.path_mapping.items():
            if ofs_path.startswith(s3_path):
                ofs_path = ofs_path.replace(s3_path, ofs_replacement, 1)
                break

        return ofs_path