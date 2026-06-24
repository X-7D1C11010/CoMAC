import os
import sys
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, '/home/lixiang/lx/CoMAC')

from datasets.weather_dataset import get_domain_data, DOMAINS, NUM_CLASSES
from utils.metrics import SegmentationMetrics
from utils.utils import set_seed, save_checkpoint, load_checkpoint

DOMAIN_MAP = {
    '晴天': 'sunny',
    '黑天': 'night',
    '逆光': 'backlight',
    '雾天': 'foggy',
    '雨天': 'rainy'
}

class MultiModalClassifier(nn.Module):
    def __init__(self, num_classes=14):
        super(MultiModalClassifier, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(512 * 12 * 12, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        features_flat = features.view(features.size(0), -1)
        features_512 = self.fc1(features_flat)
        logits = self.fc2(features_512)
        return logits, features_512

class DomainAdaptationModel(nn.Module):
    def __init__(self, num_classes=14):
        super(DomainAdaptationModel, self).__init__()
        
        self.infrared_extractor = MultiModalClassifier(num_classes)
        self.visible_extractor = MultiModalClassifier(num_classes)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(num_classes * 2, num_classes),
            nn.ReLU(inplace=True),
            nn.Linear(num_classes, num_classes)
        )
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
    
    def forward(self, infrared, visible, return_features=False):
        logits_ir, features_ir = self.infrared_extractor(infrared)
        logits_vis, features_vis = self.visible_extractor(visible)
        
        fused_logits = torch.cat([logits_ir, logits_vis], dim=1)
        fused_logits = self.fusion_layer(fused_logits)
        
        combined_features = torch.cat([features_ir, features_vis], dim=1)
        domain_logits = self.domain_classifier(combined_features)
        
        if return_features:
            return fused_logits, domain_logits, features_ir, features_vis
        return fused_logits, domain_logits

def train_source(model, source_loader, criterion_cls, optimizer, device, epoch, logger):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, data in enumerate(source_loader):
        infrared = data['infrared'].to(device)
        visible = data['visible'].to(device)
        labels = data['label'].to(device)
        
        optimizer.zero_grad()
        
        logits, _ = model(infrared, visible)
        loss = criterion_cls(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * infrared.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += infrared.size(0)
        
        if (batch_idx + 1) % 20 == 0:
            logger.info(f'Epoch [{epoch}], Batch [{batch_idx+1}/{len(source_loader)}], '
                       f'Loss: {loss.item():.4f}, Acc: {total_correct/total_samples:.4f}')
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    logger.info(f'Source Training - Epoch [{epoch}], Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}')
    return avg_loss, avg_acc

def train_domain_adaptation(model, source_loader, target_loader, criterion_cls, 
                            criterion_domain, optimizer, device, epoch, logger, lambda_domain=0.1):
    model.train()
    total_loss_cls = 0.0
    total_loss_domain = 0.0
    total_correct = 0
    total_samples = 0
    
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    
    for batch_idx in range(max(len(source_loader), len(target_loader))):
        try:
            source_data = next(source_iter)
            src_infrared = source_data['infrared'].to(device)
            src_visible = source_data['visible'].to(device)
            src_labels = source_data['label'].to(device)
            src_domains = torch.zeros(src_infrared.size(0), dtype=torch.long).to(device)
        except StopIteration:
            source_iter = iter(source_loader)
            source_data = next(source_iter)
            src_infrared = source_data['infrared'].to(device)
            src_visible = source_data['visible'].to(device)
            src_labels = source_data['label'].to(device)
            src_domains = torch.zeros(src_infrared.size(0), dtype=torch.long).to(device)
        
        try:
            target_data = next(target_iter)
            tgt_infrared = target_data['infrared'].to(device)
            tgt_visible = target_data['visible'].to(device)
            tgt_domains = torch.ones(tgt_infrared.size(0), dtype=torch.long).to(device)
        except StopIteration:
            target_iter = iter(target_loader)
            target_data = next(target_iter)
            tgt_infrared = target_data['infrared'].to(device)
            tgt_visible = target_data['visible'].to(device)
            tgt_domains = torch.ones(tgt_infrared.size(0), dtype=torch.long).to(device)
        
        optimizer.zero_grad()
        
        src_logits, src_domain_logits = model(src_infrared, src_visible)
        tgt_logits, tgt_domain_logits = model(tgt_infrared, tgt_visible)
        
        loss_cls = criterion_cls(src_logits, src_labels)
        
        domain_logits = torch.cat([src_domain_logits, tgt_domain_logits], dim=0)
        domain_labels = torch.cat([src_domains, tgt_domains], dim=0)
        loss_domain = criterion_domain(domain_logits, domain_labels)
        
        loss = loss_cls + lambda_domain * loss_domain
        
        loss.backward()
        optimizer.step()
        
        total_loss_cls += loss_cls.item() * src_infrared.size(0)
        total_loss_domain += loss_domain.item() * (src_infrared.size(0) + tgt_infrared.size(0))
        
        preds = torch.argmax(src_logits, dim=1)
        total_correct += (preds == src_labels).sum().item()
        total_samples += src_infrared.size(0)
        
        if (batch_idx + 1) % 20 == 0:
            logger.info(f'Epoch [{epoch}], Batch [{batch_idx+1}], '
                       f'Cls Loss: {loss_cls.item():.4f}, Domain Loss: {loss_domain.item():.4f}')
    
    avg_loss_cls = total_loss_cls / total_samples
    avg_loss_domain = total_loss_domain / (total_samples * 2)
    avg_acc = total_correct / total_samples
    
    logger.info(f'DA Training - Epoch [{epoch}], Cls Loss: {avg_loss_cls:.4f}, '
               f'Domain Loss: {avg_loss_domain:.4f}, Acc: {avg_acc:.4f}')
    return avg_loss_cls, avg_loss_domain, avg_acc

def test(model, data_loader, device, metrics):
    model.eval()
    metrics.reset()
    
    with torch.no_grad():
        for data in data_loader:
            infrared = data['infrared'].to(device)
            visible = data['visible'].to(device)
            labels = data['label'].to(device)
            
            logits, _ = model(infrared, visible)
            preds = torch.argmax(logits, dim=1)
            
            metrics.update(preds.cpu().numpy(), labels.cpu().numpy())
    
    results = metrics.compute()
    return results

def run_single_experiment(source_domain, target_domain, data_root, output_dir, 
                          seed, num_epochs=50, batch_size=32, logger=None):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    data_loaders = get_domain_data(data_root, source_domain, target_domain, batch_size, transform)
    
    model = DomainAdaptationModel(num_classes=NUM_CLASSES).to(device)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    metrics = SegmentationMetrics(num_classes=NUM_CLASSES)
    
    best_acc = 0.0
    best_results = None
    best_epoch = 0
    
    logger.info(f'=== 开始源域训练 (种子 {seed}) ===')
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_source(model, data_loaders['source_train'], 
                                              criterion_cls, optimizer, device, epoch, logger)
        
        val_results = test(model, data_loaders['source_val'], device, metrics)
        logger.info(f'Source Val - Epoch [{epoch}], Acc: {val_results["accuracy"]:.4f}, '
                   f'F1: {val_results["f1_score"]:.4f}')
        
        if val_results['accuracy'] > best_acc:
            best_acc = val_results['accuracy']
            best_results = val_results
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, output_dir, f'best_source_{seed}.pth')
    
    logger.info(f'=== 开始目标域适应 (种子 {seed}) ===')
    best_tgt_acc = -1.0
    best_tgt_results = None
    best_tgt_epoch = 0
    
    for epoch in range(1, num_epochs + 1):
        cls_loss, domain_loss, train_acc = train_domain_adaptation(
            model, data_loaders['source_train'], data_loaders['target_train'],
            criterion_cls, criterion_domain, optimizer, device, epoch, logger
        )
        
        tgt_results = test(model, data_loaders['target_val'], device, metrics)
        logger.info(f'Target Val - Epoch [{epoch}], Acc: {tgt_results["accuracy"]:.4f}, '
                   f'F1: {tgt_results["f1_score"]:.4f}')
        
        if best_tgt_results is None or tgt_results['accuracy'] > best_tgt_acc:
            best_tgt_acc = tgt_results['accuracy']
            best_tgt_results = tgt_results
            best_tgt_epoch = epoch
            save_checkpoint(model, optimizer, epoch, output_dir, f'best_target_{seed}.pth')
    
    return {
        'seed': seed,
        'best_source_epoch': best_epoch,
        'best_source_results': best_results,
        'best_target_epoch': best_tgt_epoch,
        'best_target_results': best_tgt_results
    }

def run_experiments(source_domain, target_domains, data_root, output_root, 
                    num_runs=5, num_epochs=50, batch_size=32):
    os.makedirs(output_root, exist_ok=True)
    
    all_results = {}
    
    for target_domain in target_domains:
        src_en = DOMAIN_MAP.get(source_domain, source_domain)
        tgt_en = DOMAIN_MAP.get(target_domain, target_domain)
        
        logger = logging.getLogger(f'{src_en}_to_{tgt_en}')
        logger.setLevel(logging.INFO)
        
        log_path = os.path.join(output_root, f'{src_en}_to_{tgt_en}.log')
        file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        console_handler = logging.StreamHandler()
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.info(f'========== 实验: {source_domain} -> {target_domain} ==========')
        logger.info(f'重复次数: {num_runs}')
        logger.info(f'训练轮数: {num_epochs}')
        logger.info(f'批大小: {batch_size}')
        
        domain_output_dir = os.path.join(output_root, f'{src_en}_to_{tgt_en}')
        os.makedirs(domain_output_dir, exist_ok=True)
        
        run_results = []
        
        for run_idx in range(num_runs):
            seed = run_idx + 42
            logger.info(f'\n---------- 第 {run_idx + 1}/{num_runs} 次重复 (种子 {seed}) ----------')
            
            result = run_single_experiment(
                source_domain, target_domain, data_root, domain_output_dir,
                seed, num_epochs, batch_size, logger
            )
            
            run_results.append(result)
            
            logger.info(f'第 {run_idx + 1} 次重复完成')
            logger.info(f'源域最佳: Epoch {result["best_source_epoch"]}, '
                       f'Acc: {result["best_source_results"]["accuracy"]:.4f}, '
                       f'F1: {result["best_source_results"]["f1_score"]:.4f}')
            logger.info(f'目标域最佳: Epoch {result["best_target_epoch"]}, '
                       f'Acc: {result["best_target_results"]["accuracy"]:.4f}, '
                       f'F1: {result["best_target_results"]["f1_score"]:.4f}')
        
        all_results[target_domain] = run_results
        
        acc_list = [r['best_target_results']['accuracy'] for r in run_results]
        prec_list = [r['best_target_results']['precision'] for r in run_results]
        rec_list = [r['best_target_results']['recall'] for r in run_results]
        f1_list = [r['best_target_results']['f1_score'] for r in run_results]
        
        logger.info(f'\n========== {source_domain} -> {target_domain} 实验汇总 ==========')
        logger.info(f'ACC: 均值 = {np.mean(acc_list):.4f}, 标准差 = {np.std(acc_list):.4f}')
        logger.info(f'Precision: 均值 = {np.mean(prec_list):.4f}, 标准差 = {np.std(prec_list):.4f}')
        logger.info(f'Recall: 均值 = {np.mean(rec_list):.4f}, 标准差 = {np.std(rec_list):.4f}')
        logger.info(f'F1: 均值 = {np.mean(f1_list):.4f}, 标准差 = {np.std(f1_list):.4f}')
        
        summary = {
            'source_domain': source_domain,
            'target_domain': target_domain,
            'num_runs': num_runs,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'results': run_results,
            'statistics': {
                'accuracy': {'mean': float(np.mean(acc_list)), 'std': float(np.std(acc_list))},
                'precision': {'mean': float(np.mean(prec_list)), 'std': float(np.std(prec_list))},
                'recall': {'mean': float(np.mean(rec_list)), 'std': float(np.std(rec_list))},
                'f1_score': {'mean': float(np.mean(f1_list)), 'std': float(np.std(f1_list))}
            }
        }
        
        summary_path = os.path.join(domain_output_dir, 'summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.removeHandler(file_handler)
        logger.removeHandler(console_handler)
    
    overall_summary = {
        'source_domain': source_domain,
        'target_domains': target_domains,
        'num_runs': num_runs,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'all_results': all_results
    }
    
    overall_path = os.path.join(output_root, 'overall_summary.json')
    with open(overall_path, 'w', encoding='utf-8') as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)
    
    print('\n========== 所有实验完成 ==========')
    for target_domain in target_domains:
        stats = all_results[target_domain][0]['best_target_results']
        print(f'{source_domain} -> {target_domain}:')
        acc_list = [r['best_target_results']['accuracy'] for r in all_results[target_domain]]
        f1_list = [r['best_target_results']['f1_score'] for r in all_results[target_domain]]
        print(f'  ACC: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}')
        print(f'  F1: {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='域适应实验')
    parser.add_argument('--data_root', type=str, default='/home/lixiang/lx/Data',
                        help='数据集根目录')
    parser.add_argument('--output_root', type=str, default='/home/lixiang/lx/CoMAC/experiments',
                        help='输出目录')
    parser.add_argument('--source_domain', type=str, default='晴天',
                        help='源域名称')
    parser.add_argument('--target_domains', type=str, nargs='+', 
                        default=['黑天', '逆光', '雾天', '雨天'],
                        help='目标域列表')
    parser.add_argument('--num_runs', type=int, default=5,
                        help='重复实验次数')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批大小')
    
    args = parser.parse_args()
    
    run_experiments(
        source_domain=args.source_domain,
        target_domains=args.target_domains,
        data_root=args.data_root,
        output_root=args.output_root,
        num_runs=args.num_runs,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size
    )
