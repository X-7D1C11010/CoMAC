import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
    
    def forward(self, logits, targets):
        return self.loss(logits, targets)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, ignore_index=-100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        mask = (targets != self.ignore_index).float()
        
        intersection = torch.sum(probs * mask.unsqueeze(1), dim=(2, 3))
        union = torch.sum(probs * mask.unsqueeze(1), dim=(2, 3)) + torch.sum(mask, dim=(1, 2)).unsqueeze(1)
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - torch.mean(dice)

class EntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, logits):
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -torch.sum(probs * log_probs, dim=1)
        
        if self.reduction == 'mean':
            return torch.mean(entropy)
        elif self.reduction == 'sum':
            return torch.sum(entropy)
        else:
            return entropy

class ConsistencyLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, logits1, logits2):
        probs1 = F.softmax(logits1 / self.temperature, dim=1)
        probs2 = F.softmax(logits2 / self.temperature, dim=1)
        
        loss = -torch.mean(torch.sum(probs1 * torch.log(probs2 + 1e-8), dim=1))
        return loss

class FeatureCentroidLoss(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.register_buffer('centroids', torch.zeros(num_classes, feature_dim))
    
    def forward(self, features, logits, confidence_threshold=0.9):
        probs = F.softmax(logits, dim=1)
        max_probs, preds = torch.max(probs, dim=1)
        
        mask = (max_probs > confidence_threshold).float()
        
        loss = 0.0
        for c in range(self.num_classes):
            class_mask = (preds == c) & (mask == 1)
            if class_mask.sum() > 0:
                class_features = features[class_mask]
                centroid = class_features.mean(dim=0)
                
                diff = class_features - centroid
                loss += torch.mean(torch.norm(diff, dim=1))
        
        return loss / self.num_classes

def get_loss_function(loss_name, **kwargs):
    loss_map = {
        'cross_entropy': CrossEntropyLoss,
        'dice': DiceLoss,
        'entropy': EntropyLoss,
        'consistency': ConsistencyLoss,
        'feature_centroid': FeatureCentroidLoss
    }
    
    if loss_name not in loss_map:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    return loss_map[loss_name](**kwargs)
