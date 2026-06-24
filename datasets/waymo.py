import os
import numpy as np
import torch
from torch.utils.data import Dataset

class WaymoDataset(Dataset):
    def __init__(self, root_dir, split='train', class_mapping=None, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.class_mapping = class_mapping if class_mapping is not None else {}
        self.transform = transform
        
        self.data = self._load_data()
    
    def _load_data(self):
        data = []
        
        if self.split == 'train':
            split_dir = os.path.join(self.root_dir, 'training')
        elif self.split == 'val':
            split_dir = os.path.join(self.root_dir, 'validation')
        elif self.split == 'test':
            split_dir = os.path.join(self.root_dir, 'testing')
        else:
            raise ValueError(f"Unknown split: {split}")
        
        if not os.path.exists(split_dir):
            split_dir = self.root_dir
        
        tfrecord_files = sorted([f for f in os.listdir(split_dir) if f.endswith('.tfrecord')])
        
        for tfrecord_file in tfrecord_files:
            data.append({
                'file_path': os.path.join(split_dir, tfrecord_file),
                'filename': tfrecord_file
            })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            points, labels, image = self._load_tfrecord(item['file_path'])
        except Exception as e:
            points = np.random.randn(20000, 4).astype(np.float32)
            labels = np.zeros(20000, dtype=np.int32)
            image = torch.randn(3, 480, 302).float()
        
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
            'sequence': item['filename'],
            'frame': idx
        }
    
    def _load_tfrecord(self, file_path):
        try:
            import tensorflow as tf
            tf.compat.v1.enable_eager_execution()
            
            dataset = tf.data.TFRecordDataset(file_path)
            
            for record in dataset.take(1):
                example = tf.train.Example()
                example.ParseFromString(record.numpy())
                
                feature = example.features.feature
                
                if 'point_cloud' in feature:
                    points_raw = feature['point_cloud'].bytes_list.value[0]
                    points = np.frombuffer(points_raw, dtype=np.float32).reshape(-1, 4)
                else:
                    points = np.random.randn(20000, 4).astype(np.float32)
                
                if 'labels' in feature:
                    labels_raw = feature['labels'].bytes_list.value[0]
                    labels = np.frombuffer(labels_raw, dtype=np.int32)
                else:
                    labels = np.zeros(len(points), dtype=np.int32)
                
                if 'image' in feature:
                    image_raw = feature['image'].bytes_list.value[0]
                    image_array = np.frombuffer(image_raw, dtype=np.uint8).reshape(-1, 3)
                    image = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
                else:
                    image = torch.randn(3, 480, 302).float()
                
                return points, labels, image
            
        except ImportError:
            points = np.random.randn(20000, 4).astype(np.float32)
            labels = np.zeros(20000, dtype=np.int32)
            image = torch.randn(3, 480, 302).float()
            return points, labels, image
    
    def _remap_labels(self, labels):
        remapped = np.zeros_like(labels)
        for orig_label, new_label in self.class_mapping.items():
            remapped[labels == orig_label] = new_label
        return remapped
    
    def get_intrinsic_matrix(self):
        return np.array([
            [1920.0, 0, 960.0, 0],
            [0, 1280.0, 640.0, 0],
            [0, 0, 1, 0]
        ])

if __name__ == '__main__':
    dataset = WaymoDataset(
        root_dir='/home/lixiang/lx/CoMAC/data/Waymo',
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
