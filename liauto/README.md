# OnlineDataset 使用指南

## 简介

该数据集类支持：
- 加载相机图像（左/前/右视角）
- 处理LiDAR点云数据
- 提取车辆状态信息（速度、加速度等）
- 获取历史和未来轨迹信息
- 导航信息处理

## 基本用法

### 1. 初始化数据集

```python
from liauto_dataset import OnlineDataset
from utils.config import Config

# 加载配置
config_path = "/path/to/config.py"
cfg = Config.fromfile(config_path)

# 创建数据集实例
dataset = OnlineDataset(
    dataset_config=cfg.txt_root,
    whitelist_clips=cfg.whitelist_files,  # 可选
    blacklist_clips=cfg.blacklist_files,  # 可选
    max_clip_len=100,  # 可选，限制片段长度
    min_clip_len=2     # 可选，最小片段长度
)
```

### 2. 获取单个样本

```python
# 获取索引为0的样本
features, targets = dataset[0]

# 特征包含以下键
camera_feature = features["camera_feature"]     # [3, 256, 1024] 相机特征张量
lidar_feature = features["lidar_feature"]       # [C, 256, 256] 雷达特征张量
status_feature = features["status_feature"]     # [8] 状态特征（驾驶命令、速度、加速度）
past_trajectory = features["past_trajectory"]   # [4, 3] 历史轨迹
diffusion_navi = features["diffusion_navi"]     # 导航特征

# 目标包含以下键
trajectory = targets["trajectory"]              # [8, 3] 未来轨迹
traj_mask = targets["traj_mask"]                # [8, 2] 轨迹掩码
npz_path = targets["npz_path"]                  # 数据文件路径
```

### 3. 使用DataLoader进行批处理

```python
from torch.utils.data import DataLoader

# 创建数据加载器
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# 训练循环
for batch_idx, (features, targets) in enumerate(dataloader):
    # 每个特征和目标现在都是批量形式
    # features["camera_feature"].shape == [批量大小, 3, 256, 1024]
    
    # 在这里进行模型训练
    ...
```

## 数据格式详解

### 输入特征

| 特征名称 | 形状 | 描述 |
|---------|------|------|
| camera_feature | [3, 256, 1024] | 拼接和处理后的左/前/右相机RGB图像 |
| lidar_feature | [C, 256, 256] | 处理后的LiDAR点云特征图 |
| status_feature | [8] | 包含驾驶命令(4)、速度(2)和加速度(2)的状态向量 |
| past_trajectory | [4, 3] | 4个历史帧的位置和朝向 (x,y,yaw) |
| diffusion_navi | [N, 2] | 导航路径点 |

### 输出目标

| 目标名称 | 形状 | 描述 |
|---------|------|------|
| trajectory | [8, 3] | 8个未来帧的预测位置和朝向 (x,y,yaw) |
| traj_mask | [8, 2] | 轨迹掩码，指示哪些未来点是有效的 |
| npz_path | str | 原始数据文件的路径 |

## 数据可视化

数据集提供了可视化工具，以便于理解和调试：

```python
from test_online_dataset import TestOnlineDataset

# 创建测试实例
test = TestOnlineDataset()
test.setUpClass()

# 可视化第0个样本
test.visualize_sample(0)
```

可视化结果包括：
1. 相机图像（左/前/右）
2. 3D点云数据展示
3. 处理后的LiDAR特征图
4. 历史和未来轨迹
5. 导航路径点
6. 车辆状态特征

## 高级用法

### 1. 自定义数据过滤

```python
dataset = OnlineDataset(
    dataset_config=cfg.txt_root,
    whitelist_clips=['clip_id_1', 'clip_id_2'],  # 只包含这些片段
    blacklist_clips=['clip_id_3'],               # 排除这些片段
    max_clip_len=200,                            # 限制每个片段的最大长度
    min_clip_len=5                               # 最小片段长度要求
)
```

### 2. 访问原始数据

```python
# 获取单个样本的原始数据
idx = 0
raw_data = dataset.prepare_raw_data(idx)

# 原始数据包含：
images = raw_data["images"]                  # 原始相机图像
point_cloud = raw_data["point_cloud"]        # 原始点云数据
velocity = raw_data["velocity"]              # 速度信息
acceleration = raw_data["acceleration"]      # 加速度信息
history_poses = raw_data["history_poses"]    # 历史轨迹
future_trajectory = raw_data["future_trajectory"]  # 未来轨迹
```