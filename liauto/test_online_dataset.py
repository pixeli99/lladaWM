import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from utils.config import Config
from liauto_dataset import OnlineDataset
import unittest
import cv2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import time

class TestOnlineDataset(unittest.TestCase):
    """测试OnlineDataset类的功能"""
    
    @classmethod
    def setUpClass(cls):
        """初始化测试环境"""
        # 加载配置
        config_path = "/lpai/volumes/ad-vla-vol-ga/lipengxiang/lipx_dev/TrajHF/liauto/projects/end2end/all_in_one.py"
        cls.cfg = Config.fromfile(config_path)
        
        # 创建测试输出目录
        cls.output_dir = "/lpai/volumes/ad-vla-vol-ga/lipengxiang/lipx_dev/TrajHF/liauto/test_outputs"
        os.makedirs(cls.output_dir, exist_ok=True)
        
        # 创建在线数据集
        cls.dataset = OnlineDataset(
            dataset_config=cls.cfg.txt_root,
            whitelist_clips=cls.cfg.whitelist_files,
            blacklist_clips=cls.cfg.blacklist_files,
            max_clip_len=100  # 限制片段长度以加快测试
        )
        print(f"Dataset loaded with {len(cls.dataset)} samples")
    
    def test_dataset_length(self):
        """测试数据集长度是否大于0"""
        self.assertGreater(len(self.dataset), 0, "Dataset should have at least one sample")
    
    def test_get_single_item(self):
        """测试获取单个样本"""
        idx = 0
        start_time = time.time()
        features, targets = self.dataset[idx]
        end_time = time.time()
        print(f"Time to load single sample: {end_time - start_time:.2f} seconds")
        
        # 检查特征字典中的键
        expected_features = ["camera_feature", "lidar_feature", "status_feature", 
                            "past_trajectory", "diffusion_navi"]
        for key in expected_features:
            self.assertIn(key, features, f"Missing feature key: {key}")
        
        # 检查目标字典中的键
        expected_targets = ["trajectory", "traj_mask", "npz_path"]
        for key in expected_targets:
            self.assertIn(key, targets, f"Missing target key: {key}")
    
    def test_dataloader(self):
        """测试数据加载器的功能"""
        # 创建数据加载器
        dataloader = DataLoader(
            self.dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,  # 单线程以便调试
            pin_memory=False
        )
        
        # 获取一个批次
        start_time = time.time()
        batch = next(iter(dataloader))
        end_time = time.time()
        print(f"Time to load batch: {end_time - start_time:.2f} seconds")
        
        features, targets = batch
        
        # 检查批次维度
        self.assertEqual(features["camera_feature"].shape[0], 2, "Batch size should be 2")
        self.assertEqual(targets["trajectory"].shape[0], 2, "Batch size should be 2")
    
    def visualize_sample(self, idx=0):
        """可视化单个样本的数据"""
        raw_data = self.dataset.prepare_raw_data(idx)
        features = self.dataset.prepare_features(raw_data)
        targets = self.dataset.prepare_targets(raw_data)
        
        # 创建多个子图
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 可视化相机图像
        ax1 = fig.add_subplot(3, 2, 1)
        self._visualize_camera(raw_data["images"], ax1)
        
        # 2. 可视化点云数据
        ax2 = fig.add_subplot(3, 2, 2, projection='3d')
        self._visualize_point_cloud(raw_data["point_cloud"], ax2)
        
        # 3. 可视化激光雷达特征
        ax3 = fig.add_subplot(3, 2, 3)
        self._visualize_lidar_feature(features["lidar_feature"], ax3)
        
        # 4. 可视化轨迹
        ax4 = fig.add_subplot(3, 2, 4)
        self._visualize_trajectory(raw_data["history_poses"], raw_data["future_trajectory"], 
                                  raw_data["future_trajectory_mask"], ax4)
        
        # 5. 可视化导航信息
        ax5 = fig.add_subplot(3, 2, 5)
        self._visualize_navigation(features["diffusion_navi"], ax5)
        
        # 6. 可视化状态特征
        ax6 = fig.add_subplot(3, 2, 6)
        self._visualize_status(features["status_feature"], ax6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"sample_{idx}_visualization.png"), dpi=150)
        plt.close()
        print(f"Visualization saved to {os.path.join(self.output_dir, f'sample_{idx}_visualization.png')}")
    
    def _visualize_camera(self, images, ax):
        """可视化相机图像"""
        # 拼接左、前、右三个图像
        left = images["left"]
        front = images["front"]
        right = images["right"]
        
        # 确保所有图像高度相同
        height = min(left.shape[0], front.shape[0], right.shape[0])
        
        # 调整大小
        left_resized = cv2.resize(left, (int(left.shape[1]*height/left.shape[0]), height))
        front_resized = cv2.resize(front, (int(front.shape[1]*height/front.shape[0]), height))
        right_resized = cv2.resize(right, (int(right.shape[1]*height/right.shape[0]), height))
        
        # 横向拼接
        stitched = np.hstack([left_resized, front_resized, right_resized])
        
        # 显示图像
        ax.imshow(stitched)
        ax.set_title("Camera Images (Left, Front, Right)")
        ax.axis('off')
    
    def _visualize_point_cloud(self, point_cloud, ax):
        """可视化点云数据"""
        # 限制点的数量以加快渲染速度
        max_points = 5000
        if len(point_cloud) > max_points:
            indices = np.random.choice(len(point_cloud), max_points, replace=False)
            point_cloud = point_cloud[indices]
            
        # 设置点的颜色基于高度
        colors = point_cloud[:, 2]
        norm = Normalize(vmin=min(colors), vmax=max(colors))
        
        # 绘制点云
        scatter = ax.scatter(
            point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
            c=colors, s=1, cmap='viridis', norm=norm
        )
        
        # 设置坐标轴标签和标题
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("LiDAR Point Cloud")
        
        # 加入色条
        plt.colorbar(scatter, ax=ax, label='Height')
        
        # 设置视角
        ax.view_init(elev=20, azim=30)
    
    def _visualize_lidar_feature(self, lidar_feature, ax):
        """可视化激光雷达特征"""
        # 转换为numpy数组并压缩为2D
        lidar_np = lidar_feature.numpy()
        if lidar_np.ndim > 2:
            lidar_np = np.mean(lidar_np, axis=0)
            
        # 显示热图
        im = ax.imshow(lidar_np, cmap='viridis', origin='lower')
        ax.set_title("LiDAR Features")
        plt.colorbar(im, ax=ax)
    
    def _visualize_trajectory(self, history_poses, future_trajectory, future_mask, ax):
        """可视化历史和未来轨迹"""
        # 绘制历史轨迹
        history_x = history_poses[:, 0]
        history_y = history_poses[:, 1]
        ax.plot(history_x, history_y, 'b-', linewidth=2, label='History')
        ax.scatter(history_x, history_y, c='blue', s=30)
        
        # 绘制未来轨迹
        future_x = future_trajectory[:, 0]
        future_y = future_trajectory[:, 1]
        
        # 使用掩码区分有效和无效轨迹点
        if future_mask is not None:
            valid_mask = future_mask[:, 0] > 0.5  # 假设掩码>0.5为有效点
            
            # 绘制有效轨迹点
            ax.plot(future_x[valid_mask], future_y[valid_mask], 'g-', linewidth=2, label='Future (valid)')
            ax.scatter(future_x[valid_mask], future_y[valid_mask], c='green', s=30)
            
            # 绘制无效轨迹点（如果存在）
            invalid_mask = ~valid_mask
            if np.any(invalid_mask):
                ax.plot(future_x[invalid_mask], future_y[invalid_mask], 'r--', linewidth=1, label='Future (invalid)')
                ax.scatter(future_x[invalid_mask], future_y[invalid_mask], c='red', s=20)
        else:
            # 没有掩码，绘制所有未来轨迹点
            ax.plot(future_x, future_y, 'g-', linewidth=2, label='Future')
            ax.scatter(future_x, future_y, c='green', s=30)
        
        # 设置坐标轴等比例
        ax.set_aspect('equal')
        ax.set_title("Vehicle Trajectory")
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend()
        
        # 添加网格
        ax.grid(True)
    
    def _visualize_navigation(self, navigation_data, ax):
        """可视化导航信息"""
        nav_np = navigation_data.numpy()
        
        # 绘制导航点
        ax.plot(nav_np[:, 0], nav_np[:, 1], 'r-', linewidth=2)
        ax.scatter(nav_np[:, 0], nav_np[:, 1], c='red', s=30)
        
        # 突出显示起始点
        ax.scatter([nav_np[0, 0]], [nav_np[0, 1]], c='blue', s=100, marker='o', label='Start')
        
        # 突出显示结束点
        ax.scatter([nav_np[-1, 0]], [nav_np[-1, 1]], c='green', s=100, marker='*', label='End')
        
        ax.set_aspect('equal')
        ax.set_title("Navigation Points")
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend()
        ax.grid(True)
    
    def _visualize_status(self, status_feature, ax):
        """可视化状态特征"""
        status_np = status_feature.numpy()
        
        # 创建标签
        labels = ["Cmd1", "Cmd2", "Cmd3", "Cmd4", "Vx", "Vy", "Ax", "Ay"]
        
        # 绘制条形图
        bars = ax.bar(labels, status_np)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        ax.set_title("Status Features")
        ax.set_ylabel("Value")
        ax.grid(axis='y')


def run_tests_and_visualize():
    """运行测试并可视化样本"""
    # 运行单元测试
    suite = unittest.TestSuite()
    suite.addTest(TestOnlineDataset("test_dataset_length"))
    suite.addTest(TestOnlineDataset("test_get_single_item"))
    suite.addTest(TestOnlineDataset("test_dataloader"))
    
    runner = unittest.TextTestRunner()
    runner.run(suite)
    
    # 可视化多个样本
    test_instance = TestOnlineDataset()
    test_instance.setUpClass()
    
    # 可视化前5个样本
    num_samples = min(5, len(test_instance.dataset))
    for i in range(num_samples):
        print(f"Visualizing sample {i}...")
        test_instance.visualize_sample(i)


if __name__ == "__main__":
    run_tests_and_visualize()