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
from datasets.datamodule import DataModule
from utils.losses import CrossEntropyLoss, DiceLoss
from utils.metrics import SegmentationMetrics
from utils.utils import set_seed, save_checkpoint, create_log_dir

def parse_args():
    parser = argparse.ArgumentParser(description='训练CoMAC模型')
    parser.add_argument('--dataset', type=str, default='semantickitti', help='数据集名称')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=10000, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--log_dir', type=str, default=None, help='日志目录')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='保存目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备类型')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    return parser.parse_args()

def train_one_epoch(model, dataloader, optimizer, loss_fn, metrics, device):
    model.train()
    metrics.reset()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        images = batch['image'].to(device)
        points = batch['points'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        logits_2d = model.image_model(images)
        logits_3d = model.pointcloud_model(points)
        
        loss_2d = loss_fn(logits_2d, labels)
        loss_3d = loss_fn(logits_3d, labels)
        
        loss = loss_2d + loss_3d
        
        loss.backward()
        optimizer.step()
        
        preds_2d = torch.argmax(logits_2d, dim=1)
        preds_3d = torch.argmax(logits_3d, dim=1)
        
        metrics.update(preds_2d, labels)
        
        total_loss += loss.item()
        num_batches += 1
        
        if num_batches % 100 == 0:
            miou, _ = metrics.compute_miou()
            avg_loss = total_loss / num_batches
            print(f"  Batch {num_batches}: Loss = {avg_loss:.4f}, mIoU = {miou:.4f}")
    
    avg_loss = total_loss / num_batches
    miou, _ = metrics.compute_miou()
    
    return avg_loss, miou

def validate(model, dataloader, loss_fn, metrics, device):
    model.eval()
    metrics.reset()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            points = batch['points'].to(device)
            labels = batch['labels'].to(device)
            
            logits_2d = model.image_model(images)
            logits_3d = model.pointcloud_model(points)
            
            loss_2d = loss_fn(logits_2d, labels)
            loss_3d = loss_fn(logits_3d, labels)
            
            loss = loss_2d + loss_3d
            
            preds_2d = torch.argmax(logits_2d, dim=1)
            preds_3d = torch.argmax(logits_3d, dim=1)
            
            metrics.update(preds_2d, labels)
            
            total_loss += loss.item()
            num_batches += 1
    
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
            logging.FileHandler(os.path.join(args.log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"开始训练，配置: {args}")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    image_model = ImageSegmentationModel(num_classes=config.NUM_CLASSES)
    pointcloud_model = PointCloudSegmentationModel(num_classes=config.NUM_CLASSES)
    
    image_model.to(device)
    pointcloud_model.to(device)
    
    parameters = list(image_model.parameters()) + list(pointcloud_model.parameters())
    
    optimizer = optim.Adam(
        parameters,
        lr=args.lr,
        betas=(config.TRAIN_CONFIG['beta1'], config.TRAIN_CONFIG['beta2']),
        weight_decay=config.TRAIN_CONFIG['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 0.1 if epoch >= 8000 else 1.0
    )
    
    loss_fn = CrossEntropyLoss(ignore_index=-100)
    metrics = SegmentationMetrics(num_classes=config.NUM_CLASSES, ignore_index=-100)
    
    datamodule = DataModule(config.__dict__)
    
    train_dataloader = datamodule.get_train_dataloader(args.dataset, batch_size=args.batch_size)
    val_dataloader = datamodule.get_val_dataloader(args.dataset)
    
    best_miou = 0.0
    
    logger.info(f"训练集大小: {len(train_dataloader.dataset)}")
    logger.info(f"验证集大小: {len(val_dataloader.dataset)}")
    
    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        train_loss, train_miou = train_one_epoch(
            image_model, train_dataloader, optimizer, loss_fn, metrics, device
        )
        
        logger.info(f"训练结果: Loss = {train_loss:.4f}, mIoU = {train_miou:.4f}")
        
        scheduler.step()
        
        if (epoch + 1) % 1000 == 0:
            val_loss, val_miou = validate(
                image_model, val_dataloader, loss_fn, metrics, device
            )
            
            logger.info(f"验证结果: Loss = {val_loss:.4f}, mIoU = {val_miou:.4f}")
            
            if val_miou > best_miou:
                best_miou = val_miou
                save_checkpoint(image_model, optimizer, epoch, 
                               os.path.join(args.checkpoint_dir, 'best_image_model.pth'), is_best=True)
                save_checkpoint(pointcloud_model, optimizer, epoch, 
                               os.path.join(args.checkpoint_dir, 'best_pointcloud_model.pth'), is_best=True)
                
                logger.info(f"保存最佳模型，mIoU = {best_miou:.4f}")
        
        if (epoch + 1) % 5000 == 0:
            save_checkpoint(image_model, optimizer, epoch, 
                           os.path.join(args.checkpoint_dir, f'image_model_epoch_{epoch+1}.pth'))
            save_checkpoint(pointcloud_model, optimizer, epoch, 
                           os.path.join(args.checkpoint_dir, f'pointcloud_model_epoch_{epoch+1}.pth'))
    
    logger.info(f"训练完成，最佳验证mIoU = {best_miou:.4f}")

if __name__ == '__main__':
    main()
