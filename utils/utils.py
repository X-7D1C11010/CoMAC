import os
import torch
import numpy as np
import random
from datetime import datetime

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model, optimizer, epoch, save_dir, filename):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, filename)
    torch.save(state, save_path)

def load_checkpoint(model, optimizer, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    return epoch

def get_lr_scheduler(optimizer, config):
    lr_decay_epochs = config['lr_decay_epochs']
    lr_decay_factor = config['lr_decay_factor']
    
    def lr_lambda(epoch):
        decay = 1.0
        for decay_epoch in lr_decay_epochs:
            if epoch >= decay_epoch:
                decay *= lr_decay_factor
        return decay
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def create_log_dir(base_dir):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(base_dir, timestamp)
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    return log_dir

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(item, device) for item in data]
    else:
        return data

def normalize_features(features, dim=1):
    norms = torch.norm(features, dim=dim, keepdim=True)
    return features / (norms + 1e-8)

def compute_class_centroids(features, labels, num_classes):
    centroids = []
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            class_features = features[mask]
            centroid = class_features.mean(dim=0)
            centroids.append(centroid)
        else:
            centroids.append(torch.zeros_like(features[0]))
    
    return torch.stack(centroids)

def project_3d_to_2d(points, intrinsic_matrix):
    points_homogeneous = np.hstack([points[:, :3], np.ones((points.shape[0], 1))])
    pixels = points_homogeneous @ intrinsic_matrix.T
    pixels = pixels[:, :2] / pixels[:, 2, np.newaxis]
    
    return pixels

def filter_points_in_fov(points, fov=30, max_distance=100):
    distances = np.linalg.norm(points[:, :3], axis=1)
    mask = distances <= max_distance
    
    theta = np.arctan2(points[:, 1], points[:, 0])
    phi = np.arcsin(points[:, 2] / (distances + 1e-8))
    
    fov_rad = fov * np.pi / 180
    mask &= (np.abs(phi) <= fov_rad / 2)
    
    return points[mask]

def sample_lidar_lines(points, num_lines=64, fov=30):
    phi = np.arcsin(points[:, 2] / np.linalg.norm(points[:, :3], axis=1))
    phi_deg = phi * 180 / np.pi
    
    fov_range = [-fov/2, fov/2]
    mask = (phi_deg >= fov_range[0]) & (phi_deg <= fov_range[1])
    points = points[mask]
    phi_deg = phi_deg[mask]
    
    bins = np.linspace(fov_range[0], fov_range[1], num_lines + 1)
    labels = np.digitize(phi_deg, bins) - 1
    
    sampled_points = []
    for line_idx in range(num_lines):
        line_mask = labels == line_idx
        line_points = points[line_mask]
        
        if len(line_points) > 0:
            num_samples = max(1, len(line_points) // 312)
            indices = np.random.choice(len(line_points), min(num_samples, len(line_points)), replace=False)
            sampled_points.append(line_points[indices])
    
    if len(sampled_points) > 0:
        return np.vstack(sampled_points)
    else:
        return points
