import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

class ImageAugmentation:
    def __init__(self, bottom_crop=(480, 302), horizontal_flip_prob=0.5, color_jitter=(0.4, 0.4, 0.4)):
        self.bottom_crop = bottom_crop
        self.horizontal_flip_prob = horizontal_flip_prob
        self.color_jitter = color_jitter
        
        self.transform = transforms.Compose([
            transforms.ColorJitter(brightness=color_jitter[0], 
                                  contrast=color_jitter[1], 
                                  saturation=color_jitter[2]),
            transforms.RandomHorizontalFlip(p=horizontal_flip_prob)
        ])
    
    def __call__(self, image):
        h, w = image.shape[-2:]
        if self.bottom_crop is not None:
            crop_h, crop_w = self.bottom_crop
            start_h = max(0, h - crop_h)
            start_w = max(0, w - crop_w)
            image = image[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        image = self.transform(image)
        return image
    
    def apply_multiple(self, image, num_augmentations=3):
        augmented_images = []
        for _ in range(num_augmentations):
            augmented = self.__call__(image.clone())
            augmented_images.append(augmented)
        return augmented_images

class PointCloudAugmentation:
    def __init__(self, noisy_rotation=0.1, y_flip_prob=0.5, z_rotation_range=(0, 360), translation_range=0.5):
        self.noisy_rotation = noisy_rotation
        self.y_flip_prob = y_flip_prob
        self.z_rotation_range = z_rotation_range
        self.translation_range = translation_range
    
    def __call__(self, points):
        augmented = points.clone()
        
        if np.random.rand() < self.y_flip_prob:
            augmented[:, 1] = -augmented[:, 1]
        
        z_angle = np.random.uniform(*self.z_rotation_range) * np.pi / 180
        rotation_matrix = np.array([
            [np.cos(z_angle), -np.sin(z_angle), 0],
            [np.sin(z_angle), np.cos(z_angle), 0],
            [0, 0, 1]
        ])
        augmented[:, :3] = augmented[:, :3] @ rotation_matrix.T
        
        translation = np.random.uniform(-self.translation_range, self.translation_range, size=3)
        augmented[:, :3] += translation
        
        noise = np.random.normal(0, self.noisy_rotation, size=augmented[:, :3].shape)
        augmented[:, :3] += noise
        
        return augmented
    
    def apply_multiple(self, points, num_augmentations=3):
        augmented_points = []
        for _ in range(num_augmentations):
            augmented = self.__call__(points.clone())
            augmented_points.append(augmented)
        return augmented_points

class VoxelAugmentation:
    def __init__(self, voxel_scale=20, full_scale=4096):
        self.voxel_scale = voxel_scale
        self.full_scale = full_scale
    
    def voxelize(self, points):
        coords = np.floor(points[:, :3] * self.voxel_scale)
        coords -= coords.min(axis=0)
        
        coords = coords.astype(np.int32)
        mask = (coords >= 0).all(axis=1) & (coords < self.full_scale).all(axis=1)
        coords = coords[mask]
        
        unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
        
        return torch.from_numpy(unique_coords), inverse

def random_flip_horizontal(image):
    if np.random.rand() < 0.5:
        return torch.flip(image, dims=[2])
    return image

def random_color_jitter(image, brightness=0.4, contrast=0.4, saturation=0.4):
    transform = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)
    return transform(image)

def random_rotation_z(points, angle_range=(0, 360)):
    angle = np.random.uniform(*angle_range) * np.pi / 180
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    points[:, :3] = points[:, :3] @ rotation_matrix.T
    return points

def random_translation(points, range_=0.5):
    translation = np.random.uniform(-range_, range_, size=3)
    points[:, :3] += translation
    return points

def noisy_rotation(points, std=0.1):
    noise = np.random.normal(0, std, size=points[:, :3].shape)
    points[:, :3] += noise
    return points
