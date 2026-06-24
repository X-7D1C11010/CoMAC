import os
import sys
import argparse
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim

from configs.default import config
from models.image_model import ImageSegmentationModel
from models.pointcloud_model import PointCloudSegmentationModel
from models.comac import CoMAC
from datasets.datamodule import DataModule
from utils.losses import CrossEntropyLoss, EntropyLoss
from utils.metrics import SegmentationMetrics
from utils.utils import set_seed, load_checkpoint, create_log_dir

def parse_args():
    parser = argparse.ArgumentParser(description='CoMAC测试时间适应')
    parser.add_argument('--source_dataset', type=str, default='semantickitti', help='源数据集')
    parser.add_argument('--target_dataset', type=str, default='synthia', help='目标数据集')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    parser.add_argument('--lr', type=float, default=1.25e-5, help='TTA学习率')
    parser.add_argument('--num_augmentations', type=int, default=3, help='增强次数')
    parser.add_argument('--queue_size', type=int, default=2000, help='动量队列大小')
    parser.add_argument('--restore_rate', type=float, default=0.01, help='随机恢复率')
    parser.add_argument('--momentum', type=float, default=0.999, help='动量系数')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default=None, help='日志目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备类型')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    return parser.parse_args()

def compute_source_centroids(model, dataloader, num_classes, feature_dim, device):
    model.eval()
    
    class_features = [[] for _ in range(num_classes)]
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            points = batch['points'].to(device)
            labels = batch['labels'].to(device)
            
            _, features_2d = model.image_model(images, return_features=True)
            _, features_3d = model.pointcloud_model(points, return_features=True)
            
            features_2d_flat = features_2d.flatten(2).transpose(1, 2)
            features_3d_flat = features_3d
            
            labels_flat = labels.view(-1)
            
            for c in range(num_classes):
                mask = (labels_flat == c)
                if mask.sum() > 0:
                    class_features[c].append(features_2d_flat[mask])
    
    centroids = []
    for c in range(num_classes):
        if len(class_features[c]) > 0:
            feat = torch.cat(class_features[c], dim=0)
            centroid = feat.mean(dim=0)
            centroid = centroid / (centroid.norm() + 1e-8)
        else:
            centroid = torch.zeros(feature_dim)
        centroids.append(centroid)
    
    return torch.stack(centroids)

def test_time_adaptation(comac, dataloader, optimizer, loss_fn, metrics, device, 
                          num_augmentations=3, restore_rate=0.01):
    comac.train()
    metrics.reset()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        images = batch['image'].to(device)
        points = batch['points'].to(device)
        labels = batch['labels'].to(device) if 'labels' in batch else None
        
        optimizer.zero_grad()
        
        logits_2d, features_2d_flat = comac.predict_2d(images)
        logits_3d, features_3d_flat = comac.predict_3d(points)
        
        pseudo_labels_2d = torch.argmax(logits_2d, dim=1)
        pseudo_labels_3d = torch.argmax(logits_3d, dim=1)
        
        probs_2d = torch.softmax(logits_2d, dim=1)
        probs_3d = torch.softmax(logits_3d, dim=1)
        
        max_probs_2d = torch.max(probs_2d, dim=1)[0]
        max_probs_3d = torch.max(probs_3d, dim=1)[0]
        
        confidence_mask_2d = (max_probs_2d > 0.9).float()
        confidence_mask_3d = (max_probs_3d > 0.9).float()
        
        loss_2d = loss_fn(logits_2d, pseudo_labels_2d) * confidence_mask_2d.mean()
        loss_3d = loss_fn(logits_3d, pseudo_labels_3d) * confidence_mask_3d.mean()
        
        loss = loss_2d + loss_3d
        
        loss.backward()
        optimizer.step()
        
        comac.apply_stochastic_restoration()
        
        comac.update_momentum_queues(
            features_2d_flat.flatten(0, 1), 
            pseudo_labels_2d.flatten(),
            features_3d_flat.flatten(0, 1),
            pseudo_labels_3d.flatten()
        )
        
        comac.update_centroids()
        
        preds_3d = torch.argmax(logits_3d, dim=1)
        
        if labels is not None:
            metrics.update(preds_3d, labels)
        
        total_loss += loss.item()
        num_batches += 1
        
        if num_batches % 100 == 0:
            miou, _ = metrics.compute_miou()
            avg_loss = total_loss / num_batches
            print(f"  Batch {num_batches}: Loss = {avg_loss:.4f}, mIoU = {miou:.4f}")
    
    avg_loss = total_loss / num_batches
    miou, _ = metrics.compute_miou()
    
    return avg_loss, miou

def main():
    args = parse_args()
    
    set_seed(args.seed)
    
    if args.log_dir is None:
        args.log_dir = create_log_dir(config.LOG_DIR)
    
    if args.checkpoint_dir is None:
        args.checkpoint_dir = config.CHECKPOINT_DIR
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, 'tta.log')),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"开始测试时间适应，配置: {args}")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    image_model = ImageSegmentationModel(num_classes=config.NUM_CLASSES)
    pointcloud_model = PointCloudSegmentationModel(num_classes=config.NUM_CLASSES)
    
    image_model_path = os.path.join(args.checkpoint_dir, 'best_image_model.pth')
    pointcloud_model_path = os.path.join(args.checkpoint_dir, 'best_pointcloud_model.pth')
    
    if os.path.exists(image_model_path):
        load_checkpoint(image_model, None, image_model_path)
        logger.info(f"加载预训练2D模型: {image_model_path}")
    
    if os.path.exists(pointcloud_model_path):
        load_checkpoint(pointcloud_model, None, pointcloud_model_path)
        logger.info(f"加载预训练3D模型: {pointcloud_model_path}")
    
    comac = CoMAC(
        image_model,
        pointcloud_model,
        num_classes=config.NUM_CLASSES,
        feature_dim=128,
        queue_size=args.queue_size,
        momentum=args.momentum,
        restore_rate=args.restore_rate
    )
    
    comac.to(device)
    comac.register_source_params()
    
    datamodule = DataModule(config.__dict__)
    
    source_dataloader = datamodule.get_train_dataloader(args.source_dataset, batch_size=1)
    target_dataloader = datamodule.get_test_dataloader(args.target_dataset, batch_size=args.batch_size)
    
    logger.info(f"计算源域特征中心...")
    source_centroids_2d = compute_source_centroids(
        comac, source_dataloader, config.NUM_CLASSES, 128, device
    )
    source_centroids_3d = compute_source_centroids(
        comac, source_dataloader, config.NUM_CLASSES, 128, device
    )
    
    comac.set_source_centroids(source_centroids_2d, source_centroids_3d)
    logger.info(f"源域特征中心计算完成")
    
    parameters = list(image_model.parameters()) + list(pointcloud_model.parameters()) + \
                 list(comac.iMPA_2d.parameters()) + list(comac.iMPA_3d.parameters()) + \
                 list(comac.xMPF.parameters())
    
    optimizer = optim.Adam(
        parameters,
        lr=args.lr,
        betas=(config.TRAIN_CONFIG['beta1'], config.TRAIN_CONFIG['beta2']),
        weight_decay=config.TRAIN_CONFIG['weight_decay']
    )
    
    loss_fn = CrossEntropyLoss(ignore_index=-100)
    metrics = SegmentationMetrics(num_classes=config.NUM_CLASSES, ignore_index=-100)
    
    logger.info(f"目标域数据集大小: {len(target_dataloader.dataset)}")
    
    logger.info(f"\n开始测试时间适应...")
    tta_loss, tta_miou = test_time_adaptation(
        comac, target_dataloader, optimizer, loss_fn, metrics, device,
        num_augmentations=args.num_augmentations,
        restore_rate=args.restore_rate
    )
    
    logger.info(f"测试时间适应完成")
    logger.info(f"TTA结果: Loss = {tta_loss:.4f}, mIoU = {tta_miou:.4f}")
    
    save_path = os.path.join(args.checkpoint_dir, 'comac_tta.pth')
    torch.save({
        'image_model_state_dict': image_model.state_dict(),
        'pointcloud_model_state_dict': pointcloud_model.state_dict(),
        'comac_state_dict': comac.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_path)
    
    logger.info(f"保存TTA后的模型: {save_path}")

if __name__ == '__main__':
    main()
