import torch
from torch.utils.data import DataLoader, ConcatDataset
from .semantickitti import SemanticKITTIDataset
from .synthia import SynthiaDataset
from .waymo import WaymoDataset

class DataModule:
    def __init__(self, config):
        self.config = config
        self.class_mapping = config.get('SEMANTICKITTI_CLASS_MAPPING', {})
        
    def get_dataloader(self, dataset_name, split='train', batch_size=1, shuffle=False, num_workers=4):
        if dataset_name == 'semantickitti':
            dataset = SemanticKITTIDataset(
                root_dir=self.config['DATASET_PATHS']['semantickitti'],
                split=split,
                class_mapping=self.class_mapping
            )
        elif dataset_name == 'synthia':
            class_mapping = self.config.get('SYNTHIA_CLASS_MAPPING', {})
            dataset = SynthiaDataset(
                root_dir=self.config['DATASET_PATHS']['synthia'],
                split=split,
                class_mapping=class_mapping
            )
        elif dataset_name == 'waymo':
            class_mapping = self.config.get('WAYMO_CLASS_MAPPING', {})
            dataset = WaymoDataset(
                root_dir=self.config['DATASET_PATHS']['waymo'],
                split=split,
                class_mapping=class_mapping
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
        
        return dataloader
    
    def get_mixed_dataloader(self, source_dataset='semantickitti', target_dataset='synthia', 
                            split='train', batch_size=1, shuffle=False, num_workers=4):
        source_ds = SemanticKITTIDataset(
            root_dir=self.config['DATASET_PATHS'][source_dataset],
            split=split,
            class_mapping=self.class_mapping
        )
        
        if target_dataset == 'synthia':
            class_mapping = self.config.get('SYNTHIA_CLASS_MAPPING', {})
            target_ds = SynthiaDataset(
                root_dir=self.config['DATASET_PATHS']['synthia'],
                split=split,
                class_mapping=class_mapping
            )
        elif target_dataset == 'waymo':
            class_mapping = self.config.get('WAYMO_CLASS_MAPPING', {})
            target_ds = WaymoDataset(
                root_dir=self.config['DATASET_PATHS']['waymo'],
                split=split,
                class_mapping=class_mapping
            )
        else:
            raise ValueError(f"Unknown target dataset: {target_dataset}")
        
        mixed_dataset = ConcatDataset([source_ds, target_ds])
        
        dataloader = DataLoader(
            mixed_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
        
        return dataloader
    
    def _collate_fn(self, batch):
        points = []
        labels = []
        images = []
        sequences = []
        frames = []
        
        for item in batch:
            if item['points'] is not None:
                if isinstance(item['points'], np.ndarray):
                    points.append(torch.from_numpy(item['points']))
                else:
                    points.append(item['points'])
            else:
                points.append(torch.randn(20000, 4))
            
            if item['labels'] is not None:
                if isinstance(item['labels'], np.ndarray):
                    labels.append(torch.from_numpy(item['labels']))
                else:
                    labels.append(item['labels'])
            else:
                labels.append(torch.zeros(20000, dtype=torch.long))
            
            if item['image'] is not None:
                images.append(item['image'])
            else:
                images.append(torch.randn(3, 480, 302))
            
            sequences.append(item['sequence'])
            frames.append(item['frame'])
        
        points = torch.stack(points)
        labels = torch.stack(labels)
        images = torch.stack(images)
        
        return {
            'points': points,
            'labels': labels,
            'image': images,
            'sequence': sequences,
            'frame': frames
        }
    
    def get_train_dataloader(self, dataset_name, batch_size=None):
        batch_size = batch_size or self.config['TRAIN_CONFIG']['batch_size']
        return self.get_dataloader(
            dataset_name,
            split='train',
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
    
    def get_val_dataloader(self, dataset_name, batch_size=1):
        return self.get_dataloader(
            dataset_name,
            split='val',
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
    
    def get_test_dataloader(self, dataset_name, batch_size=1):
        return self.get_dataloader(
            dataset_name,
            split='test',
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
