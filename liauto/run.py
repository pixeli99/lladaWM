import os
import cv2
import torch
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import gzip
import pickle
import signal
import sys
import torch.multiprocessing as mp

# from dataset import AllinOneDataset
from perfer_dataset import AllinOneDataset
from utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INTERVAL = 1
class DiffSimConfig:
    """模拟 DiffSimConfig"""

    def __init__(self):
        # LiDAR配置
        self.lidar_min_x = -32
        self.lidar_max_x = 32
        self.lidar_min_y = -32
        self.lidar_max_y = 32
        self.pixels_per_meter = 4
        self.hist_max_per_pixel = 15
        self.max_height_lidar = 3.5
        self.lidar_split_height = 0.5
        self.use_ground_plane = False


def dump_feature_target_to_pickle(path: Path, data_dict: Dict[str, torch.Tensor]) -> None:
    """保存数据到gzip压缩的pickle文件"""
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(data_dict, f)


def prepare_camera_feature(images: Dict[str, np.ndarray]) -> torch.Tensor:
    """
    处理相机图像
    :param images: 包含 'left', 'front', 'right' 三个视角图像的字典
    :return: shape 为 [3, 256, 1024] 的 tensor
    """
    # 检查输入图像
    required_keys = ['left', 'front', 'right']
    for key in required_keys:
        if key not in images:
            raise ValueError(f"Missing required key: {key} in images dictionary")

    left = images['left']
    front = images['front']
    right = images['right']

    # 等比例缩放 front 图像
    scale = left.shape[0] / front.shape[0]  # 计算高度缩放比例
    new_width = int(front.shape[1] * scale)  # 按相同比例计算新的宽度
    front_resized = cv2.resize(front, (new_width, left.shape[0]), interpolation=cv2.INTER_LINEAR)

    # 计算目标宽高比 4:1 所需的总宽度
    height = left.shape[0]
    required_width = height * 4  # 目标宽高比 4:1

    # 计算每个部分应该占用的宽度
    side_width = (required_width - front_resized.shape[1]) // 2
    
    # 从中心裁剪每个图像
    def center_crop(img, target_width):
        start = (img.shape[1] - target_width) // 2
        end = start + target_width
        return img[:, start:end]

    # 裁剪每个图像
    l0 = center_crop(left, side_width)
    f0 = front_resized
    r0 = center_crop(right, side_width)

    # 拼接图像
    stitched = np.concatenate([l0, f0, r0], axis=1)

    # 等比例缩放到目标尺寸
    scale = min(1024 / stitched.shape[1], 256 / stitched.shape[0])
    new_width = int(stitched.shape[1] * scale)
    new_height = int(stitched.shape[0] * scale)
    
    # 首先等比例缩放
    resized = cv2.resize(stitched, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    final = np.zeros((256, 1024, 3), dtype=np.uint8)
    y_offset = (256 - new_height) // 2
    x_offset = (1024 - new_width) // 2
    final[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized

    tensor = torch.from_numpy(final).permute(2, 0, 1).float() / 255.0

    return tensor



def prepare_lidar_feature(point_cloud: np.ndarray, config: DiffSimConfig) -> torch.Tensor:
    """
    处理LiDAR点云
    :param point_cloud: shape为[N,3]的点云数组
    :return: shape为[2,256,256]的tensor(如果use_ground_plane=True)
    """

    def splat_points(points):
        # 创建256x256网格
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

    # 移除高于车辆的点
    point_cloud = point_cloud[point_cloud[..., 2] < config.max_height_lidar]

    # 分离地平面上下的点
    below = point_cloud[point_cloud[..., 2] <= config.lidar_split_height]
    above = point_cloud[point_cloud[..., 2] > config.lidar_split_height]

    # 生成特征
    above_features = splat_points(above)
    if config.use_ground_plane:
        below_features = splat_points(below)
        features = np.stack([below_features, above_features], axis=0)
    else:
        features = np.expand_dims(above_features, axis=0)

    return torch.from_numpy(features).float()


def prepare_features(data: Dict) -> Dict[str, torch.Tensor]:
    """
    准备模型输入特征
    """
    config = DiffSimConfig()
    features = {}

    # 1. 相机特征 [3, 256, 1024]
    features["camera_feature"] = prepare_camera_feature(data["images"])

    # 2. LiDAR特征 [2, 256, 256]
    features["lidar_feature"] = prepare_lidar_feature(data["point_cloud"], config)

    # 3. 状态特征 [driving_command(1) + velocity(2) + acceleration(2)]
    features["status_feature"] = torch.cat([
        torch.tensor([0, 0, 0, 1], dtype=torch.float32),
        torch.tensor(data["velocity"], dtype=torch.float32),
        torch.tensor(data["acceleration"], dtype=torch.float32),
    ])

    # 4. 历史轨迹 [3, 4] - (x,y,yaw) x 4个历史帧
    features["past_trajectory"] = torch.tensor(
        data["history_poses"][::5][:3],  # 应为shape [4,3]的数组
        dtype=torch.float32
    )
    features["past_trajectory"] = features["past_trajectory"].flip(dims=[0])
    features["past_trajectory"] = torch.cat([features["past_trajectory"], torch.tensor([[0.0, 0.0, 0.0]])], dim=0)

    features["diffusion_navi"] = torch.tensor(data["diffusion_navi"], dtype=torch.float32)
    return features


def prepare_targets(data: Dict) -> Dict[str, torch.Tensor]:
    """
    准备模型训练目标
    """
    return {
        "trajectory": torch.tensor(
            data["future_trajectory"][::5][:8],  # 应为shape [N,3]的数组,N为预测步数
            dtype=torch.float32
        ),
        "traj_mask": torch.tensor(
            data["future_trajectory_mask"][::5][:8],
            dtype=torch.float32
        ),
        "npz_path": data["npz_path"],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", type=str, default="/lpai/volumes/ad-vla-vol-ga/lipengxiang/liauto_perfer_data_cache_v3_train",
                        help="缓存路径")
    parser.add_argument("--config", type=str, default='projects/end2end/all_in_one.py', help="配置文件")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker threads")
    return parser.parse_args()


def process_worker(config_path, cache_path, rank, num_workers):
    """
    将配置文件的加载移到worker内部
    """
    # 在worker中加载配置
    cfg = Config.fromfile(config_path)
    
    # 创建数据集
    dataset = AllinOneDataset(
        dataset_config=cfg.txt_root,
        whitelist_clips=cfg.whitelist_files,
        blacklist_clips=cfg.blacklist_files)
    
    data_len = len(dataset)
    chunk_size = data_len // num_workers
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank != num_workers-1 else data_len

    for idx in range(start_idx, end_idx, INTERVAL):
        res_dict = dataset.get_data_info(idx, vis=False)
        for token, data in res_dict.items():
            try:
                scene_path = cache_path / token / data['log_name']
                os.makedirs(scene_path, exist_ok=True)

                features = prepare_features(data)
                feature_path = scene_path / "transfuser_feature.gz"
                dump_feature_target_to_pickle(feature_path, features)

                targets = prepare_targets(data)
                target_path = scene_path / "transfuser_target.gz"
                dump_feature_target_to_pickle(target_path, targets)
                
                print(f"进程 {rank}: 场景 {token} 处理完成!")
            except Exception as e:
                print(f"进程 {rank}: 处理场景 {token} 失败: {str(e)}")

def main():
    args = parse_args()
    cache_path = Path(args.cache_path)
    
    if args.num_workers > 1:
        # 将cfg对象的创建移出多进程部分
        processes = []
        
        def signal_handler(signum, frame):
            print("\n清理进程...")
            for p in processes:
                p.terminate()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            for rank in range(args.num_workers):
                p = mp.Process(
                    target=process_worker,
                    args=(args.config, cache_path, rank, args.num_workers)
                )
                p.start()
                processes.append(p)
                
            for p in processes:
                p.join()
                
        except KeyboardInterrupt:
            print("\n清理进程...")
            for p in processes:
                p.terminate()
            sys.exit(0)
    else:
        # 单进程处理
        process_worker(args.config, cache_path, 0, 1)

    print(f"数据处理完成! 缓存保存在: {cache_path}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()

