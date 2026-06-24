import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    NUM_CLASSES = 11
    NUM_MODALITIES = 2
    
    S_TO_S_CLASSES = {
        'car': 0,
        'bike': 1,
        'person': 2,
        'road': 3,
        'sidewalk': 4,
        'building': 5,
        'nature': 6,
        'poles': 7,
        'fence': 8,
        'traffic-sign': 9,
        'other-objects': 10
    }
    
    S_TO_W_CLASSES = {
        'car': 0,
        'bike': 1,
        'person': 2,
        'road': 3,
        'sidewalk': 4,
        'building': 5,
        'nature': 6,
        'poles': 7,
        'trunk': 8,
        'traffic-sign': 9,
        'other-objects': 10
    }
    
    SEMANTICKITTI_CLASS_MAPPING = {
        0: 10, 1: 10, 10: 0, 11: 5, 12: 3, 13: 8, 14: 7, 15: 10,
        16: 10, 17: 6, 18: 10, 20: 4, 30: 0, 31: 1, 32: 2, 40: 3,
        44: 4, 48: 5, 49: 5, 50: 6, 51: 10, 52: 10, 60: 0, 70: 6,
        71: 7, 72: 7, 80: 8, 81: 8, 99: 10, 252: 0, 253: 1, 254: 2,
        255: 5, 256: 3, 257: 6, 258: 8, 259: 10
    }
    
    SYNTHIA_CLASS_MAPPING = {
        0: 10, 1: 5, 2: 6, 3: 8, 4: 3, 5: 9, 6: 7, 7: 10, 8: 4,
        9: 10, 10: 10, 11: 10, 12: 10, 13: 10, 14: 10, 15: 10,
        16: 10, 17: 10, 18: 10, 19: 10, 20: 10, 21: 10, 22: 10,
        23: 10, 24: 10, 25: 10, 26: 10, 27: 10, 28: 10, 29: 10,
        30: 10, 31: 0, 32: 1, 33: 2, 34: 10, 35: 10, 36: 10,
        37: 10, 38: 10, 39: 10, 40: 10, 41: 10, 42: 10, 43: 10,
        44: 10, 45: 10, 46: 10, 47: 10, 48: 10, 49: 10, 50: 10,
        51: 10, 52: 10, 53: 10, 54: 10, 55: 10, 56: 10, 57: 10,
        58: 10, 59: 10, 60: 10, 61: 10, 62: 10, 63: 10, 64: 10,
        65: 10, 66: 10, 67: 10, 68: 10, 69: 10, 70: 10, 71: 10,
        72: 10, 73: 10, 74: 10, 75: 10, 76: 10, 77: 10, 78: 10,
        79: 10, 80: 10, 81: 10, 82: 10, 83: 10
    }
    
    WAYMO_CLASS_MAPPING = {
        0: 10, 1: 0, 2: 1, 3: 2, 4: 5, 5: 4, 6: 3, 7: 6, 8: 7,
        9: 8, 10: 9, 11: 10, 12: 10, 13: 10, 14: 10, 15: 10,
        16: 10, 17: 10, 18: 10, 19: 10, 20: 10, 21: 10, 22: 10,
        23: 10, 24: 10, 25: 10, 26: 10, 27: 10, 28: 10, 29: 10,
        30: 10, 31: 10, 32: 10, 33: 10, 34: 10, 35: 10, 36: 10,
        37: 10, 38: 10, 39: 10, 40: 10, 41: 10, 42: 10, 43: 10,
        44: 10, 45: 10, 46: 10, 47: 10, 48: 10, 49: 10, 50: 10,
        51: 10, 52: 10, 53: 10, 54: 10, 55: 10, 56: 10, 57: 10,
        58: 10, 59: 10, 60: 10, 61: 10, 62: 10, 63: 10, 64: 10,
        65: 10, 66: 10, 67: 10, 68: 10, 69: 10, 70: 10, 71: 10,
        72: 10, 73: 10, 74: 10, 75: 10, 76: 10, 77: 10, 78: 10,
        79: 10, 80: 10, 81: 10, 82: 10, 83: 10, 84: 10, 85: 10,
        86: 10, 87: 10, 88: 10, 89: 10, 90: 10, 91: 10, 92: 10,
        93: 10, 94: 10, 95: 10, 96: 10, 97: 10, 98: 10, 99: 10
    }
    
    DATASET_PATHS = {
        'semantickitti': os.path.join(BASE_DIR, 'data', 'SemanticKITTI'),
        'synthia': os.path.join(BASE_DIR, 'data', 'Synthia'),
        'waymo': os.path.join(BASE_DIR, 'data', 'Waymo')
    }
    
    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    EXPERIMENT_DIR = os.path.join(BASE_DIR, 'experiments')
    
    TRAIN_CONFIG = {
        'batch_size': 8,
        'base_lr': 0.001,
        'num_epochs': 10000,
        'lr_decay_epochs': [8000, 9000],
        'lr_decay_factor': 0.1,
        'optimizer': 'Adam',
        'beta1': 0.9,
        'beta2': 0.999,
        'weight_decay': 0.0001
    }
    
    TTA_CONFIG = {
        'batch_size': 1,
        'lr': 1.25e-5,
        'momentum': 0.999,
        'queue_size': 2000,
        'restore_rate': 0.01,
        'confidence_threshold': 0.9,
        'num_augmentations': 3
    }
    
    SPARSE_CONV_CONFIG = {
        'voxel_scale': 20,
        'full_scale': 4096,
        'num_kernel': [16, 32, 64, 128],
        'kernel_size': [3, 3, 3, 3],
        'stride': [2, 2, 2, 2]
    }
    
    RESNET_CONFIG = {
        'backbone': 'resnet34',
        'pretrained': True,
        'out_channels': [64, 128, 256, 512]
    }
    
    AUGMENTATION_CONFIG = {
        '2d': {
            'bottom_crop': (480, 302),
            'horizontal_flip_prob': 0.5,
            'color_jitter': (0.4, 0.4, 0.4)
        },
        '3d': {
            'noisy_rotation': 0.1,
            'y_flip_prob': 0.5,
            'z_rotation_range': (0, 360),
            'translation_range': 0.5
        }
    }
    
    SEED = 42
    DEVICE = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'

config = Config()
