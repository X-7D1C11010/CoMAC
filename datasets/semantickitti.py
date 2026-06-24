import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SemanticKITTIDataset(Dataset):
    def __init__(self, root_dir, split='train', class_mapping=None, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.class_mapping = class_mapping if class_mapping is not None else {}
        self.transform = transform
        
        self.sequences = self._get_sequences(split)
        self.data = self._load_data()
    
    def _get_sequences(self, split):
        if split == 'train':
            return ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
        elif split == 'val':
            return ['08']
        elif split == 'test':
            return ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        else:
            raise ValueError(f"Unknown split: {split}")
    
    def _load_data(self):
        data = []
        
        for seq in self.sequences:
            seq_dir = os.path.join(self.root_dir, seq)
            
            if not os.path.exists(seq_dir):
                continue
            
            pose_file = os.path.join(seq_dir, 'poses.txt')
            calib_file = os.path.join(seq_dir, 'calib.txt')
            
            points_dir = os.path.join(seq_dir, 'velodyne')
            label_dir = os.path.join(seq_dir, 'labels')
            image_dir = os.path.join(seq_dir, 'image_2')
            
            if os.path.exists(points_dir):
                point_files = sorted([f for f in os.listdir(points_dir) if f.endswith('.bin')])
            else:
                point_files = []
            
            if os.path.exists(label_dir):
                label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.label')])
            else:
                label_files = []
            
            if os.path.exists(image_dir):
                image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
            else:
                image_files = []
            
            for i in range(len(point_files)):
                point_file = os.path.join(points_dir, point_files[i]) if i < len(point_files) else None
                label_file = os.path.join(label_dir, label_files[i]) if i < len(label_files) else None
                image_file = os.path.join(image_dir, image_files[i]) if i < len(image_files) else None
                
                data.append({
                    'sequence': seq,
                    'frame': i,
                    'points': point_file,
                    'labels': label_file,
                    'image': image_file,
                    'pose_file': pose_file,
                    'calib_file': calib_file
                })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        points = self._load_points(item['points']) if item['points'] else None
        labels = self._load_labels(item['labels']) if item['labels'] else None
        image = self._load_image(item['image']) if item['image'] else None
        
        calib = self._load_calib(item['calib_file']) if item['calib_file'] else None
        
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
            'calib': calib,
            'sequence': item['sequence'],
            'frame': item['frame']
        }
    
    def _load_points(self, file_path):
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        return points
    
    def _load_labels(self, file_path):
        labels = np.fromfile(file_path, dtype=np.int32)
        labels = labels & 0xFFFF
        return labels
    
    def _load_image(self, file_path):
        import cv2
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image
    
    def _load_calib(self, file_path):
        calib = {}
        with open(file_path, 'r') as f:
            for line in f:
                key, value = line.strip().split(':')
                calib[key] = np.array([float(v) for v in value.split()]).reshape(3, 4)
        return calib
    
    def _remap_labels(self, labels):
        remapped = np.zeros_like(labels)
        for orig_label, new_label in self.class_mapping.items():
            remapped[labels == orig_label] = new_label
        return remapped
    
    def get_intrinsic_matrix(self):
        return np.array([
            [704.062866, 0, 607.1928, 0],
            [0, 704.062866, 185.2157, 0],
            [0, 0, 1, 0]
        ])

if __name__ == '__main__':
    dataset = SemanticKITTIDataset(
        root_dir='/home/lixiang/lx/CoMAC/data/SemanticKITTI',
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
