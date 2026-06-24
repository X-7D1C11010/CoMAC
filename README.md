# CoMAC: Continual Cross-Modal Adaptive Clustering

这是 ICCV 2023 论文 "Multi-Modal Continual Test-Time Adaptation for 3D Semantic Segmentation" 的复现代码。

论文链接: https://openaccess.thecvf.com/content/ICCV2023/papers/Cao_Multi-Modal_Continual_Test-Time_Adaptation_for_3D_Semantic_Segmentation_ICCV_2023_paper.pdf

## 项目结构

```
CoMAC/
├── configs/           # 配置文件
│   └── default.py     # 默认配置
├── models/            # 模型定义
│   ├── image_model.py        # 2D图像分割网络 (ResNet34+U-Net)
│   ├── pointcloud_model.py   # 3D点云分割网络 (PointNet++)
│   └── comac.py              # CoMAC核心模块
├── datasets/          # 数据集处理
│   ├── semantickitti.py      # SemanticKITTI数据集
│   ├── synthia.py            # Synthia数据集
│   ├── waymo.py              # Waymo数据集
│   └── datamodule.py         # 数据加载器
├── utils/             # 工具函数
│   ├── losses.py             # 损失函数
│   ├── metrics.py            # 评估指标
│   ├── augmentations.py      # 数据增强
│   └── utils.py              # 通用工具
├── scripts/           # 运行脚本
│   ├── train.py              # 预训练脚本
│   └── tta.py                # 测试时间适应脚本
├── checkpoints/       # 模型检查点
├── logs/              # 日志目录
└── requirements.txt   # 依赖列表
```

## 方法概述

CoMAC 是一种用于多模态持续测试时间适应（MM-CTTA）的方法，主要包含以下核心组件：

### 1. 自适应双阶段机制 (Adaptive Dual-Stage Mechanism)
- **iMPA (Intra-Modal Prediction Aggregation)**: 模态内预测聚合，基于特征中心距离加权融合增强平均和原始预测
- **xMPF (Cross-Modal Prediction Fusion)**: 跨模态预测融合，基于模态可靠性进行自适应加权融合

### 2. 类级动量队列 (Class-Wise Momentum Queues)
- 维护每个类别的特征队列，捕捉目标域中置信度高的特征
- 通过动量更新策略保持队列的稳定性

### 3. 随机恢复机制 (Stochastic Restoration)
- 随机将网络参数恢复到源域预训练状态
- 防止持续适应过程中的灾难性遗忘

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据准备

### SemanticKITTI
数据集应放置在 `data/SemanticKITTI/` 目录下，结构如下：
```
SemanticKITTI/
├── 00/
│   ├── velodyne/     # .bin 文件
│   ├── labels/       # .label 文件
│   ├── image_2/      # .png 文件
│   ├── poses.txt
│   └── calib.txt
├── 01/
└── ...
```

### Synthia
数据集应放置在 `data/Synthia/` 目录下，结构如下：
```
Synthia/
└── sequences/
    ├── 001/
    │   ├── RGB/      # RGB图像
    │   ├── Depth/    # 深度图
    │   └── Labels/   # 标签图
    └── ...
```

### Waymo
数据集应放置在 `data/Waymo/` 目录下，包含 `.tfrecord` 文件。

## 训练

### 预训练阶段
在 SemanticKITTI 上预训练 2D 和 3D 网络：

```bash
python scripts/train.py \
    --dataset semantickitti \
    --batch_size 8 \
    --num_epochs 10000 \
    --lr 0.001 \
    --device cuda
```

### 测试时间适应
在目标域上进行持续测试时间适应：

```bash
python scripts/tta.py \
    --source_dataset semantickitti \
    --target_dataset synthia \
    --batch_size 1 \
    --lr 1.25e-5 \
    --queue_size 2000 \
    --restore_rate 0.01 \
    --device cuda
```

## 配置说明

主要配置参数（见 `configs/default.py`）：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| NUM_CLASSES | 类别数量 | 11 |
| batch_size | 训练批次大小 | 8 |
| base_lr | 基础学习率 | 0.001 |
| lr (TTA) | TTA学习率 | 1.25e-5 |
| queue_size | 动量队列大小 | 2000 |
| restore_rate | 随机恢复率 | 0.01 |
| momentum | 动量系数 | 0.999 |

## 评估指标

- **mIoU (Mean Intersection over Union)**: 平均交并比，语义分割的标准评估指标

## 论文中的基准

论文提出了两个基准：
1. **SemanticKITTI-to-Synthia (S-to-S)**: 从真实数据集到合成数据集
2. **SemanticKITTI-to-Waymo (S-to-W)**: 从SemanticKITTI到Waymo

## 注意事项

1. 本实现基于 PyTorch，建议使用 GPU 进行训练和推理
2. 预训练阶段需要大量计算资源，建议使用 NVIDIA RTX 3090 或更高配置
3. 数据集下载需要根据官方网站的要求进行

## 参考文献

```
@inproceedings{cao2023multi,
    title={Multi-Modal Continual Test-Time Adaptation for 3D Semantic Segmentation},
    author={Cao, Haozhi and Xu, Yuecong and Yang, Jianfei and Yin, Pengyu and Yuan, Shenghai and Xie, Lihua},
    booktitle={ICCV},
    year={2023}
}
```
