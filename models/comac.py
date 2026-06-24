import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassWiseMomentumQueue:
    def __init__(self, num_classes, feature_dim, queue_size=2000, momentum=0.999):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.queue_size = queue_size
        self.momentum = momentum
        
        self.queues = []
        self.centroids = []
        
        for _ in range(num_classes):
            queue = torch.randn(queue_size, feature_dim)
            queue = F.normalize(queue, dim=1)
            self.queues.append(queue)
            
            centroid = torch.zeros(feature_dim)
            self.centroids.append(centroid)
        
        self.queues = nn.ParameterList([nn.Parameter(q, requires_grad=False) for q in self.queues])
        self.centroids = nn.ParameterList([nn.Parameter(c, requires_grad=False) for c in self.centroids])
        
        self.pointer = 0
    
    def enqueue(self, features, labels):
        features = F.normalize(features, dim=1)
        
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                class_features = features[mask]
                
                for feat in class_features:
                    self.queues[c][self.pointer] = self.momentum * self.queues[c][self.pointer] + (1 - self.momentum) * feat
                    self.queues[c][self.pointer] = F.normalize(self.queues[c][self.pointer], dim=0)
        
        self.pointer = (self.pointer + 1) % self.queue_size
        
        self.update_centroids()
    
    def update_centroids(self):
        for c in range(self.num_classes):
            valid_mask = torch.norm(self.queues[c], dim=1) > 0
            if valid_mask.sum() > 0:
                self.centroids[c] = F.normalize(self.queues[c][valid_mask].mean(dim=0), dim=0)
    
    def get_centroids(self):
        return torch.stack([c.clone() for c in self.centroids])
    
    def get_queue_features(self, class_id, num_samples=100):
        queue = self.queues[class_id]
        valid_mask = torch.norm(queue, dim=1) > 0
        
        if valid_mask.sum() > 0:
            valid_features = queue[valid_mask]
            if len(valid_features) > num_samples:
                indices = torch.randperm(len(valid_features))[:num_samples]
                return valid_features[indices]
            else:
                return valid_features
        else:
            return queue[:num_samples]

class IntraModalPredictionAggregation(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        self.register_buffer('source_centroids', torch.zeros(num_classes, feature_dim))
        self.register_buffer('target_centroids', torch.zeros(num_classes, feature_dim))
    
    def forward(self, raw_logits, features, num_augmentations=3):
        probs = F.softmax(raw_logits, dim=1)
        max_probs, preds = torch.max(probs, dim=1)
        
        centroid_distances = self.compute_centroid_distances(features, preds)
        
        weights = 1.0 / (centroid_distances + 1e-8)
        weights = F.softmax(weights, dim=0)
        
        augmented_logits_list = []
        for _ in range(num_augmentations):
            noise = torch.randn_like(raw_logits) * 0.1
            augmented_logits = raw_logits + noise
            augmented_logits_list.append(augmented_logits)
        
        if augmented_logits_list:
            avg_augmented_logits = torch.stack(augmented_logits_list).mean(dim=0)
        else:
            avg_augmented_logits = raw_logits
        
        aggregated_logits = weights * avg_augmented_logits + (1 - weights) * raw_logits
        
        return aggregated_logits
    
    def compute_centroid_distances(self, features, preds):
        distances = torch.zeros(len(preds), device=features.device)
        
        for c in range(self.num_classes):
            mask = (preds == c)
            if mask.sum() > 0:
                class_features = features[mask]
                if self.source_centroids[c].norm() > 0:
                    centroid_dist = torch.norm(class_features - self.source_centroids[c], dim=1)
                    distances[mask] = centroid_dist
        
        return distances
    
    def set_source_centroids(self, centroids):
        self.source_centroids.copy_(centroids)
    
    def update_target_centroids(self, features, labels):
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                class_features = features[mask]
                self.target_centroids[c] = F.normalize(class_features.mean(dim=0), dim=0)

class CrossModalPredictionFusion(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        self.register_buffer('modal_centroids_2d', torch.zeros(num_classes, feature_dim))
        self.register_buffer('modal_centroids_3d', torch.zeros(num_classes, feature_dim))
        
        self.attention_weights = nn.Parameter(torch.ones(2) / 2)
    
    def forward(self, logits_2d, logits_3d, features_2d, features_3d):
        probs_2d = F.softmax(logits_2d, dim=1)
        probs_3d = F.softmax(logits_3d, dim=1)
        
        preds_2d = torch.argmax(probs_2d, dim=1)
        preds_3d = torch.argmax(probs_3d, dim=1)
        
        reliability_2d = self.compute_modality_reliability(features_2d, preds_2d, is_2d=True)
        reliability_3d = self.compute_modality_reliability(features_3d, preds_3d, is_2d=False)
        
        reliability_combined = torch.stack([reliability_2d, reliability_3d], dim=-1)
        attention = F.softmax(reliability_combined * self.attention_weights, dim=-1)
        
        fused_logits = attention[..., 0].unsqueeze(1) * logits_2d + attention[..., 1].unsqueeze(1) * logits_3d
        
        return fused_logits, attention
    
    def compute_modality_reliability(self, features, preds, is_2d=True):
        if is_2d:
            centroids = self.modal_centroids_2d
        else:
            centroids = self.modal_centroids_3d
        
        reliability = torch.zeros(len(preds), device=features.device)
        
        for c in range(self.num_classes):
            mask = (preds == c)
            if mask.sum() > 0:
                class_features = features[mask]
                if centroids[c].norm() > 0:
                    dist = torch.norm(class_features - centroids[c], dim=1)
                    reliability[mask] = 1.0 / (dist + 1e-8)
        
        return reliability
    
    def update_modal_centroids(self, features_2d, labels_2d, features_3d, labels_3d):
        for c in range(self.num_classes):
            mask_2d = (labels_2d == c)
            if mask_2d.sum() > 0:
                class_features = features_2d[mask_2d]
                self.modal_centroids_2d[c] = F.normalize(class_features.mean(dim=0), dim=0)
            
            mask_3d = (labels_3d == c)
            if mask_3d.sum() > 0:
                class_features = features_3d[mask_3d]
                self.modal_centroids_3d[c] = F.normalize(class_features.mean(dim=0), dim=0)

class StochasticRestoration(nn.Module):
    def __init__(self, restore_rate=0.01):
        super().__init__()
        self.restore_rate = restore_rate
        self.source_parameters = {}
    
    def register_source_params(self, model):
        for name, param in model.named_parameters():
            self.source_parameters[name] = param.data.clone()
    
    def forward(self, model):
        if not self.training:
            return
        
        for name, param in model.named_parameters():
            if name in self.source_parameters:
                if torch.rand(1) < self.restore_rate:
                    param.data.copy_(self.source_parameters[name])

class CoMAC(nn.Module):
    def __init__(self, image_model, pointcloud_model, num_classes=11, feature_dim=128, 
                 queue_size=2000, momentum=0.999, restore_rate=0.01):
        super().__init__()
        
        self.image_model = image_model
        self.pointcloud_model = pointcloud_model
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        self.iMPA_2d = IntraModalPredictionAggregation(num_classes, feature_dim)
        self.iMPA_3d = IntraModalPredictionAggregation(num_classes, feature_dim)
        
        self.xMPF = CrossModalPredictionFusion(num_classes, feature_dim)
        
        self.momentum_queue_2d = ClassWiseMomentumQueue(num_classes, feature_dim, queue_size, momentum)
        self.momentum_queue_3d = ClassWiseMomentumQueue(num_classes, feature_dim, queue_size, momentum)
        
        self.stochastic_restoration_2d = StochasticRestoration(restore_rate)
        self.stochastic_restoration_3d = StochasticRestoration(restore_rate)
        
        self.register_buffer('source_centroids_2d', torch.zeros(num_classes, feature_dim))
        self.register_buffer('source_centroids_3d', torch.zeros(num_classes, feature_dim))
    
    def forward(self, image, points, return_features=False):
        logits_2d, features_2d = self.image_model(image, return_features=True)
        logits_3d, features_3d = self.pointcloud_model(points, return_features=True)
        
        num_points = points.shape[1]
        
        features_2d_reshaped = features_2d.permute(0, 2, 3, 1).contiguous().view(1, -1, features_2d.shape[1])
        logits_2d_reshaped = logits_2d.permute(0, 2, 3, 1).contiguous().view(1, -1, logits_2d.shape[1])
        
        if features_2d_reshaped.shape[1] != num_points:
            features_2d_sampled = F.interpolate(
                features_2d_reshaped.transpose(1, 2), 
                size=num_points, 
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            
            logits_2d_sampled = F.interpolate(
                logits_2d_reshaped.transpose(1, 2), 
                size=num_points, 
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            features_2d_sampled = features_2d_reshaped
            logits_2d_sampled = logits_2d_reshaped
        
        agg_logits_2d = self.iMPA_2d(logits_2d_sampled, features_2d_sampled)
        agg_logits_3d = self.iMPA_3d(logits_3d, features_3d)
        
        fused_logits, attention = self.xMPF(
            agg_logits_2d, 
            agg_logits_3d, 
            features_2d_sampled, 
            features_3d
        )
        
        if return_features:
            return fused_logits, features_2d_sampled, features_3d, attention
        else:
            return fused_logits
    
    def predict_2d(self, image):
        logits, features = self.image_model(image, return_features=True)
        features_flat = features.flatten(2).transpose(1, 2)
        agg_logits = self.iMPA_2d(logits.flatten(2).transpose(1, 2), features_flat)
        agg_logits = agg_logits.transpose(1, 2).view(logits.shape)
        return agg_logits, features_flat
    
    def predict_3d(self, points):
        logits, features = self.pointcloud_model(points, return_features=True)
        agg_logits = self.iMPA_3d(logits, features)
        return agg_logits, features
    
    def update_momentum_queues(self, features_2d, labels_2d, features_3d, labels_3d):
        self.momentum_queue_2d.enqueue(features_2d, labels_2d)
        self.momentum_queue_3d.enqueue(features_3d, labels_3d)
    
    def update_centroids(self):
        centroids_2d = self.momentum_queue_2d.get_centroids()
        centroids_3d = self.momentum_queue_3d.get_centroids()
        
        self.iMPA_2d.update_target_centroids(centroids_2d, torch.arange(self.num_classes))
        self.iMPA_3d.update_target_centroids(centroids_3d, torch.arange(self.num_classes))
        
        self.xMPF.update_modal_centroids(
            centroids_2d, torch.arange(self.num_classes),
            centroids_3d, torch.arange(self.num_classes)
        )
    
    def set_source_centroids(self, centroids_2d, centroids_3d):
        self.source_centroids_2d.copy_(centroids_2d)
        self.source_centroids_3d.copy_(centroids_3d)
        
        self.iMPA_2d.set_source_centroids(centroids_2d)
        self.iMPA_3d.set_source_centroids(centroids_3d)
    
    def register_source_params(self):
        self.stochastic_restoration_2d.register_source_params(self.image_model)
        self.stochastic_restoration_3d.register_source_params(self.pointcloud_model)
    
    def apply_stochastic_restoration(self):
        self.stochastic_restoration_2d(self.image_model)
        self.stochastic_restoration_3d(self.pointcloud_model)

if __name__ == '__main__':
    from image_model import ImageSegmentationModel
    from pointcloud_model import PointCloudSegmentationModel
    
    image_model = ImageSegmentationModel(num_classes=11)
    pointcloud_model = PointCloudSegmentationModel(num_classes=11)
    
    comac = CoMAC(image_model, pointcloud_model, num_classes=11)
    
    image = torch.randn(1, 3, 480, 302)
    points = torch.randn(1, 20000, 6)
    
    output = comac(image, points)
    print(f"Image input shape: {image.shape}")
    print(f"Point cloud input shape: {points.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in comac.parameters() if p.requires_grad)}")
