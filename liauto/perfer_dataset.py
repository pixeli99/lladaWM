import os
import math
import numpy as np
import pandas as pd
from os import path as osp

from typing import OrderedDict
from utils.config import Config
from rewards import Rewarder

import asset
import cv2
import json
from copy import deepcopy
from pipelines import LoadPointsFromLiautoFiles, LoadNavigation

def get_global_path_mapping():
    # 加载全局路径配置，只加载一次
    path_config_str = os.getenv("LPAI_OFS_PATH_CONVERT", "{}")
    global_path_config = json.loads(path_config_str)
    # 预处理路径映射，生成替换前缀的映射关系
    sorted_paths = sorted(global_path_config.keys(), key=len, reverse=True)
    return {k: global_path_config[k] for k in sorted_paths}


def get_ofs_path(path, path_mapping):
    # 根据info获取路径
    ofs_path = None
    for s3_path, ofs_replacement in path_mapping.items():
        if path.startswith(s3_path):
            ofs_path = path.replace(s3_path, ofs_replacement, 1)
            break
    return ofs_path


class AllinOneDataset():
    def __init__(self, 
                dataset_config, 
                whitelist_clips, 
                blacklist_clips,
                max_clip_len=int(os.environ.get("MAX_CLIP_LEN", 1000000)),
                clip_start_idx=int(os.environ.get("CLIP_START_IDX", 0)),
                min_clip_len=int(os.environ.get("MIN_CLIP_LEN", 2))):
        self.dataset_config = dataset_config
        self.load_interval = 1
        self.seq_split_num = 1
        self.max_clip_len = max_clip_len
        self.clip_start_idx = clip_start_idx
        self.min_clip_len = min_clip_len
        self.test_mode = False
        self.scene_to_label_pathes = {}
        self.blacklist_clips = self.load_cliplist(blacklist_clips)
        self.whitelist_clips = self.load_cliplist(whitelist_clips)
        self.load_weighted_annotations()
        self.set_sequence_group_and_flag()

        self.rewarder = Rewarder()
        self.global_path_config = None
        self.load_global_path_config()
        self.path_mapping = self.preprocess_paths()
        self.lidar_p = LoadPointsFromLiautoFiles(coord_type='LIDAR')
        self.diff_navi = LoadNavigation(sample_pts_num=8)

        # ----
        self.global_path_mapping = get_global_path_mapping()

    def load_cliplist(self, cliplist_files):
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
        if (not self.test_mode) or self.max_clip_len <= 0:
            return num_frames_labeled
        mini_clip_num_frames_labeled = []
        for num_labeled in num_frames_labeled:
            end = min(num_labeled, self.clip_start_idx + self.max_clip_len)
            start = max(0, end - self.max_clip_len)
            mini_clip_num_frames_labeled.append(end - start)
        return mini_clip_num_frames_labeled

    
    def load_clip_labels_info(self, ann_file):
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

            # extract whitelist clips, num_frames, num_frames_labeled
            def filter_white_list(x):
                return osp.basename(osp.dirname(x[0])) in self.whitelist_clips

            if self.whitelist_clips:
                clip_infos = list(filter(filter_white_list, clip_infos))

            # filter clips, num_frames, num_frames_labeled
            # when  num_frames_labeled < self.min_clip_len or clip in blacklist_clips
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
                if "s3://s3" in ann_file:
                    ann_file = get_ofs_path(ann_file, self.global_path_mapping)
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
        if np_file.endswith(".npz"):
            return np.load(np_file, allow_pickle=True)["data"].tolist()
        else:
            return np.load(np_file, allow_pickle=True).tolist()
        
    def set_sequence_group_and_flag(self):
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
        scene_ind = self.flag[index]
        scene_path = self.scene_path_list[scene_ind]
        if scene_ind not in self.scene_to_label_pathes:
            if "s3://s3" in scene_path:
                scene_path = get_ofs_path(scene_path, self.global_path_mapping)

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
        info_path = scene_label_pathes[sample_ind]

        if "s3://s3" in info_path:
            info_path = get_ofs_path(info_path, self.global_path_mapping)
        else:
            scene_dir = os.path.dirname(scene_path)
            if "0219_csv_data_20w_from_v23.0.0_balance/" in scene_dir:
                scene_dir = scene_dir.replace(
                    "zl-mini-tain-vla/0-1-1/0219_samplev4_20w_from_250w/0219_csv_data_20w_from_v23.0.0_balance",
                    "",
                )
            info_path = os.path.join(scene_dir, scene_label_pathes[sample_ind])

        return info_path
    
    def __len__(self):
        return len(self.flag)
        
    def get_data_info(self, index, vis=False):
        info_path = self.get_info_path(index)
        info = self.load_np_file(info_path.strip())

        result = {}

        """
        {
            "scene_token": {
                "log_name": str,
                "images": {
                    "left": np.ndarray,    # [H,W,3] uint8
                    "front": np.ndarray,   # [H,W,3] uint8
                    "right": np.ndarray,   # [H,W,3] uint8
                },
                "point_cloud": np.ndarray,  # [N,3] float32
                "driving_command": float,   # 单个浮点数
                "velocity": np.ndarray,     # [vx,vy] float32
                "acceleration": np.ndarray, # [ax,ay] float32
                "history_poses": np.ndarray,# [4,3] float32, 4个历史帧的[x,y,yaw]
                "future_trajectory": np.ndarray, # [N,3] float32, N个未来帧的[x,y,yaw]
            },
            ...
        }
        """
        jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjdWlzb25nQGxpeGlhbmcuY29tIiwiaXNzIjoibHBhaSIsImlhdCI6MTcwNDE2MTc5OCwianRpIjoiMmU3ODQ5ODgtNTc3Ni00ODk3LTliOTUtOWM4ZDkwMzg4ZTRiIn0.CijUMFHBtc4gdTdUTF4TdUH_LCmhJjLe8xyUKFTX7_w"
        config = asset.config(
            env="prod",
            jwt_token=jwt_token,
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
            image_data = dataset.get_file(s3_path=path).read()
            image_array = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
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

        # result[scene_token]["history_poses"] = info['past_trajs'][0] # 20, 2
        # result[scene_token]["future_trajectory"] = info['motion_trajs'][0] # 60, 2
        # 处理 past_trajs
        past_trajs = info['sdc_motion']['past_traj']  # [20, 2]
        past_trajs_extended = np.hstack((past_trajs, np.zeros((past_trajs.shape[0], 1))))  # [20, 3]

        # 处理 motion_trajs
        motion_trajs = info['sdc_motion']['motion_traj']  # [60, 2]
        motion_trajs_mask = info['sdc_motion']['motion_traj_mask']
        motion_trajs_extended = np.hstack((motion_trajs, np.zeros((motion_trajs.shape[0], 1))))  # [60, 3]
        result[scene_token]["history_poses"] = past_trajs_extended  # [20, 3]
        result[scene_token]["future_trajectory"] = motion_trajs_extended  # [60, 3]
        result[scene_token]["future_trajectory_mask"] = motion_trajs_mask # [60, 2]

        # --- - - - -- - -
        at128_pts_filename=self.get_path(
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

        info["navigation_info"] = deepcopy(info["navigation_prior"])
        diffusion_navi = self.diff_navi(info)['diffusion_navi']
        result[scene_token]["diffusion_navi"] = diffusion_navi
        return result
    
    def __getitem__(self, index):
        return self.get_data_info(index)

    def load_global_path_config(self):
        """加载全局路径配置，只加载一次。"""
        if self.global_path_config is None:
            path_config_str = os.environ.get("LPAI_OFS_PATH_CONVERT", "{}")
            self.global_path_config = json.loads(path_config_str)

    def preprocess_paths(self):
        sorted_paths = sorted(self.global_path_config.keys(), key=len, reverse=True)
        return {k: self.global_path_config[k] for k in sorted_paths}

    def get_path(self, info, default_key, ofs_key, default_value=""):
        """根据info获取路径"""
        # replace ofs path
        ofs_path = info[ofs_key]
        for s3_path, ofs_replacement in self.path_mapping.items():
            if ofs_path.startswith(s3_path):
                ofs_path = ofs_path.replace(s3_path, ofs_replacement, 1)
                break

        return ofs_path

    
if __name__ == "__main__":
    # readconfig
    # init dataset
    config_path = "/lpai/proj_rewarder/policy_rewarder/projects/end2end/all_in_one.py"
    cfg = Config.fromfile(config_path)

    dataset = AllinOneDataset(dataset_config=cfg.txt_root,
                              whitelist_clips=cfg.whitelist_files,
                              blacklist_clips=cfg.blacklist_files)
    
    rewards_dict = dataset.get_data_info(0)

    print(cfg)
