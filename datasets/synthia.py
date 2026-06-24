import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SynthiaDataset(Dataset):
    def __init__(self, root_dir, split='train', class_mapping=None, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.class_mapping = class_mapping if class_mapping is not None else {}
        self.transform = transform
        
        self.data = self._load_data()
    
    def _load_data(self):
        data = []
        
        sequences_dir = os.path.join(self.root_dir, 'sequences')
        if not os.path.exists(sequences_dir):
            sequences_dir = self.root_dir
        
        sequences = sorted([d for d in os.listdir(sequences_dir) if os.path.isdir(os.path.join(sequences_dir, d))])
        
        for seq in sequences:
            seq_dir = os.path.join(sequences_dir, seq)
            
            rgb_dir = os.path.join(seq_dir, 'RGB')
            depth_dir = os.path.join(seq_dir, 'Depth')
            label_dir = os.path.join(seq_dir, 'Labels')
            pose_dir = os.path.join(seq_dir, 'poses')
            
            if os.path.exists(rgb_dir):
                rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
            else:
                rgb_files = []
            
            if os.path.exists(depth_dir):
                depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])
            else:
                depth_files = []
            
            if os.path.exists(label_dir):
                label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])
            else:
                label_files = []
            
            for i in range(len(rgb_files)):
                rgb_file = os.path.join(rgb_dir, rgb_files[i]) if i < len(rgb_files) else None
                depth_file = os.path.join(depth_dir, depth_files[i]) if i < len(depth_files) else None
                label_file = os.path.join(label_dir, label_files[i]) if i < len(label_files) else None
                
                data.append({
                    'sequence': seq,
                    'frame': i,
                    'rgb': rgb_file,
                    'depth': depth_file,
                    'labels': label_file
                })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        image = self._load_image(item['rgb']) if item['rgb'] else None
        depth = self._load_depth(item['depth']) if item['depth'] else None
        labels = self._load_labels(item['labels']) if item['labels'] else None
        
        points = self._generate_point_cloud(depth) if depth is not None else None
        
        if self.class_mapping and labels is not None:
            labels = self._remap_labels(labels)
        
        if self.transform:
            if points is not None:
                points = self.transform['3d'](points)
            if image is not None:
                image = self.transform['2d'](image)
        
        return {
            'points': points,
            'labels': labels,
            'image': image,
            'depth': depth,
            'sequence': item['sequence'],
            'frame': item['frame']
        }
    
    def _load_image(self, file_path):
        import cv2
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image
    
    def _load_depth(self, file_path):
        import cv2
        depth = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if depth is not None:
            depth = depth.astype(np.float32) / 1000.0
        return depth
    
    def _load_labels(self, file_path):
        import cv2
        labels = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        return labels
    
    def _generate_point_cloud(self, depth, fov=30, max_distance=100, num_lines=64):
        height, width = depth.shape
        
        fx = 700.0
        fy = 700.0
        cx = width / 2.0
        cy = height / 2.0
        
        u = np.arange(width)
        v = np.arange(height)
        u, v = np.meshgrid(u, v)
        
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        
        distances = np.linalg.norm(points, axis=1)
        mask = distances <= max_distance
        
        phi = np.arcsin(points[:, 2] / (distances + 1e-8))
        fov_rad = fov * np.pi / 180
        mask &= (np.abs(phi) <= fov_rad / 2)
        
        points = points[mask]
        phi = phi[mask]
        
        bins = np.linspace(-fov_rad / 2, fov_rad / 2, num_lines + 1)
        labels = np.digitize(phi, bins) - 1
        
        sampled_points = []
        for line_idx in range(num_lines):
            line_mask = labels == line_idx
            line_points = points[line_mask]
            
            if len(line_points) > 0:
                num_samples = max(1, len(line_points) // 312)
                indices = np.random.choice(len(line_points), min(num_samples, len(line_points)), replace=False)
                sampled_points.append(line_points[indices])
        
        if len(sampled_points) > 0:
            result_points = np.vstack(sampled_points)
        else:
            result_points = points
        
        if result_points.shape[0] < 20000:
            padding = np.zeros((20000 - result_points.shape[0], 3))
            result_points = np.vstack([result_points, padding])
        else:
            indices = np.random.choice(result_points.shape[0], 20000, replace=False)
            result_points = result_points[indices]
        
        return result_points.astype(np.float32)
    
    def _remap_labels(self, labels):
        remapped = np.zeros_like(labels)
        for orig_label, new_label in self.class_mapping.items():
            remapped[labels == orig_label] = new_label
        return remapped
    
    def get_intrinsic_matrix(self):
        return np.array([
            [700.0, 0, 600.0, 0],
            [0, 700.0, 300.0, 0],
            [0, 0, 1, 0]
        ])

if __name__ == '__main__':
    dataset = SynthiaDataset(
        root_dir='/home/lixiang/lx/CoMAC/data/Synthia',
        split='train'
    )
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    if sample['points'] is not None:
        print(f"Points shape: {sample['points'].shape}")
    if sample['labels'] is not None:
        print(f"Labels shape: {sample['labels'].shape}")
    if sample['image'] is not None:
        print(f"Image shape: {sample['image'].shape}")
