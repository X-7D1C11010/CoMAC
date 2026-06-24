import torch
import numpy as np

class SegmentationMetrics:
    def __init__(self, num_classes, ignore_index=-100):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def update(self, pred, target):
        if hasattr(pred, 'cpu'):
            pred = pred.cpu().numpy()
        if hasattr(target, 'cpu'):
            target = target.cpu().numpy()
        
        mask = (target != self.ignore_index)
        pred = pred[mask]
        target = target[mask]
        
        indices = target * self.num_classes + pred
        unique, counts = np.unique(indices, return_counts=True)
        
        for idx, cnt in zip(unique, counts):
            row = idx // self.num_classes
            col = idx % self.num_classes
            if row < self.num_classes and col < self.num_classes:
                self.confusion_matrix[row, col] += cnt
    
    def compute_miou(self):
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=0) + np.sum(self.confusion_matrix, axis=1) - intersection
        
        valid_classes = union > 0
        if np.any(valid_classes):
            iou = intersection[valid_classes] / union[valid_classes]
            miou = np.mean(iou)
            return miou, iou
        else:
            return 0.0, np.zeros(self.num_classes)
    
    def compute_accuracy(self):
        correct = np.sum(np.diag(self.confusion_matrix))
        total = np.sum(self.confusion_matrix)
        
        if total > 0:
            return correct / total
        else:
            return 0.0
    
    def compute_precision(self):
        precision = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            tp = self.confusion_matrix[c, c]
            fp = np.sum(self.confusion_matrix[:, c]) - tp
            if tp + fp > 0:
                precision[c] = tp / (tp + fp)
        
        valid_classes = np.sum(self.confusion_matrix, axis=1) > 0
        if np.any(valid_classes):
            return np.mean(precision[valid_classes])
        else:
            return 0.0
    
    def compute_recall(self):
        recall = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            tp = self.confusion_matrix[c, c]
            fn = np.sum(self.confusion_matrix[c, :]) - tp
            if tp + fn > 0:
                recall[c] = tp / (tp + fn)
        
        valid_classes = np.sum(self.confusion_matrix, axis=1) > 0
        if np.any(valid_classes):
            return np.mean(recall[valid_classes])
        else:
            return 0.0
    
    def compute_f1_score(self):
        precision = np.zeros(self.num_classes)
        recall = np.zeros(self.num_classes)
        
        for c in range(self.num_classes):
            tp = self.confusion_matrix[c, c]
            fp = np.sum(self.confusion_matrix[:, c]) - tp
            fn = np.sum(self.confusion_matrix[c, :]) - tp
            
            if tp + fp > 0:
                precision[c] = tp / (tp + fp)
            if tp + fn > 0:
                recall[c] = tp / (tp + fn)
        
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        valid_classes = np.sum(self.confusion_matrix, axis=1) > 0
        if np.any(valid_classes):
            return np.mean(f1[valid_classes])
        else:
            return 0.0
    
    def compute(self):
        return {
            'accuracy': self.compute_accuracy(),
            'precision': self.compute_precision(),
            'recall': self.compute_recall(),
            'f1_score': self.compute_f1_score(),
            'miou': self.compute_miou()[0]
        }

def compute_miou(preds, targets, num_classes, ignore_index=-100):
    metrics = SegmentationMetrics(num_classes, ignore_index)
    metrics.update(preds, targets)
    miou, _ = metrics.compute_miou()
    return miou

def compute_accuracy(preds, targets, ignore_index=-100):
    mask = (targets != ignore_index)
    correct = torch.sum(preds[mask] == targets[mask])
    total = torch.sum(mask)
    
    if total > 0:
        return correct.float() / total.float()
    else:
        return torch.tensor(0.0)

def compute_entropy(logits):
    probs = torch.softmax(logits, dim=1)
    log_probs = torch.log_softmax(logits, dim=1)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return torch.mean(entropy)
