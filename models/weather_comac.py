import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def _load_resnet(backbone, pretrained):
    if backbone == "resnet18":
        constructor = models.resnet18
        weights_attr = "ResNet18_Weights"
    elif backbone == "resnet34":
        constructor = models.resnet34
        weights_attr = "ResNet34_Weights"
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    if pretrained and hasattr(models, weights_attr):
        try:
            weights = getattr(models, weights_attr).DEFAULT
            return constructor(weights=weights)
        except Exception:
            pass

    try:
        return constructor(pretrained=pretrained)
    except Exception:
        try:
            return constructor(weights=None)
        except TypeError:
            return constructor(pretrained=False)


def _adapt_first_conv(module, in_channels):
    old_conv = module.conv1
    if old_conv.in_channels == in_channels:
        return

    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    with torch.no_grad():
        if in_channels == 1 and old_conv.weight.shape[1] == 3:
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        if new_conv.bias is not None:
            new_conv.bias.zero_()

    module.conv1 = new_conv


class SmallCNNBackbone(nn.Module):
    def __init__(self, in_channels, feature_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(192, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = feature_dim

    def forward(self, x):
        return self.net(x).flatten(1)


class ModalityEncoder(nn.Module):
    def __init__(self, in_channels, num_classes, backbone="resnet18", feature_dim=512, pretrained=False):
        super().__init__()
        if backbone == "cnn":
            self.backbone = SmallCNNBackbone(in_channels, feature_dim=feature_dim)
            out_dim = self.backbone.out_dim
        else:
            resnet = _load_resnet(backbone, pretrained)
            _adapt_first_conv(resnet, in_channels)
            out_dim = resnet.fc.in_features
            resnet.fc = nn.Identity()
            self.backbone = resnet

        self.projector = nn.Sequential(
            nn.Linear(out_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features = self.projector(self.backbone(x))
        logits = self.classifier(features)
        return logits, features


class CoMACWeatherClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone="resnet18",
        feature_dim=512,
        pretrained=False,
        reliability_temperature=0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.reliability_temperature = reliability_temperature

        self.infrared_encoder = ModalityEncoder(
            1,
            num_classes,
            backbone=backbone,
            feature_dim=feature_dim,
            pretrained=pretrained,
        )
        self.visible_encoder = ModalityEncoder(
            3,
            num_classes,
            backbone=backbone,
            feature_dim=feature_dim,
            pretrained=pretrained,
        )

        self.register_buffer("centroids_ir", torch.zeros(num_classes, feature_dim))
        self.register_buffer("centroids_vis", torch.zeros(num_classes, feature_dim))
        self.register_buffer("source_centroids_ir", torch.zeros(num_classes, feature_dim))
        self.register_buffer("source_centroids_vis", torch.zeros(num_classes, feature_dim))
        self.source_state = None

    def forward(self, infrared, visible, return_features=False):
        logits_ir, features_ir = self.infrared_encoder(infrared)
        logits_vis, features_vis = self.visible_encoder(visible)
        fused_logits, modal_weights = self.fuse_logits(logits_ir, logits_vis, features_ir, features_vis)

        output = {
            "logits": fused_logits,
            "logits_ir": logits_ir,
            "logits_vis": logits_vis,
            "features_ir": features_ir,
            "features_vis": features_vis,
            "modal_weights": modal_weights,
        }
        if return_features:
            return output
        return fused_logits

    def fuse_logits(self, logits_ir, logits_vis, features_ir, features_vis):
        reliability_ir = self.modality_reliability(logits_ir, features_ir, self.centroids_ir)
        reliability_vis = self.modality_reliability(logits_vis, features_vis, self.centroids_vis)
        reliability = torch.stack([reliability_ir, reliability_vis], dim=1)
        weights = F.softmax(reliability / self.reliability_temperature, dim=1)
        fused = weights[:, 0:1] * logits_ir + weights[:, 1:2] * logits_vis
        return fused, weights

    def modality_reliability(self, logits, features, centroids):
        probs = F.softmax(logits, dim=1)
        confidence, predictions = probs.max(dim=1)
        centroid_norms = centroids.norm(dim=1)
        selected_centroids = centroids[predictions]
        selected_valid = centroid_norms[predictions] > 0

        normalized_features = F.normalize(features, dim=1)
        normalized_centroids = F.normalize(selected_centroids, dim=1)
        cosine = (normalized_features * normalized_centroids).sum(dim=1)
        cosine = torch.where(selected_valid, (cosine + 1.0) * 0.5, torch.ones_like(cosine))
        return confidence * cosine.clamp(min=1e-4)

    def centroid_logits(self, features, modality, temperature=0.1):
        centroids = self.centroids_ir if modality == "infrared" else self.centroids_vis
        valid = centroids.norm(dim=1) > 0
        if not valid.any():
            return None
        logits = F.normalize(features, dim=1) @ F.normalize(centroids, dim=1).t()
        logits[:, ~valid] = -1e4
        return logits / temperature

    def center_contrastive_loss(self, features, labels, modality, temperature=0.1):
        centroids = self.centroids_ir if modality == "infrared" else self.centroids_vis
        valid_classes = centroids.norm(dim=1) > 0
        valid_samples = valid_classes[labels]
        if valid_samples.sum() == 0:
            return features.sum() * 0.0

        logits = self.centroid_logits(features[valid_samples], modality, temperature=temperature)
        if logits is None:
            return features.sum() * 0.0
        return F.cross_entropy(logits, labels[valid_samples])

    @torch.no_grad()
    def set_centroids(self, centroids_ir, centroids_vis, set_source=False):
        self.centroids_ir.copy_(F.normalize(centroids_ir, dim=1))
        self.centroids_vis.copy_(F.normalize(centroids_vis, dim=1))
        if set_source:
            self.source_centroids_ir.copy_(self.centroids_ir)
            self.source_centroids_vis.copy_(self.centroids_vis)

    @torch.no_grad()
    def update_centroids(self, features_ir, features_vis, labels, mask=None, momentum=0.95):
        if mask is None:
            mask = torch.ones_like(labels, dtype=torch.bool)
        labels = labels[mask]
        features_ir = F.normalize(features_ir[mask], dim=1)
        features_vis = F.normalize(features_vis[mask], dim=1)

        for class_id in labels.unique():
            class_mask = labels == class_id
            class_index = int(class_id.item())
            ir_mean = F.normalize(features_ir[class_mask].mean(dim=0), dim=0)
            vis_mean = F.normalize(features_vis[class_mask].mean(dim=0), dim=0)

            if self.centroids_ir[class_index].norm() == 0:
                self.centroids_ir[class_index].copy_(ir_mean)
            else:
                updated = momentum * self.centroids_ir[class_index] + (1.0 - momentum) * ir_mean
                self.centroids_ir[class_index].copy_(F.normalize(updated, dim=0))

            if self.centroids_vis[class_index].norm() == 0:
                self.centroids_vis[class_index].copy_(vis_mean)
            else:
                updated = momentum * self.centroids_vis[class_index] + (1.0 - momentum) * vis_mean
                self.centroids_vis[class_index].copy_(F.normalize(updated, dim=0))

    @torch.no_grad()
    def restore_source_centroids(self, restore_strength=0.1):
        valid_ir = self.source_centroids_ir.norm(dim=1) > 0
        valid_vis = self.source_centroids_vis.norm(dim=1) > 0
        if valid_ir.any():
            mixed = (1.0 - restore_strength) * self.centroids_ir[valid_ir] + restore_strength * self.source_centroids_ir[valid_ir]
            self.centroids_ir[valid_ir].copy_(F.normalize(mixed, dim=1))
        if valid_vis.any():
            mixed = (1.0 - restore_strength) * self.centroids_vis[valid_vis] + restore_strength * self.source_centroids_vis[valid_vis]
            self.centroids_vis[valid_vis].copy_(F.normalize(mixed, dim=1))

    def snapshot_source_state(self):
        self.source_state = copy.deepcopy(self.state_dict())

    @torch.no_grad()
    def stochastic_restore(self, restore_rate=0.001):
        if self.source_state is None or restore_rate <= 0:
            return
        for name, value in self.named_parameters():
            if name not in self.source_state:
                continue
            mask = torch.rand_like(value) < restore_rate
            value.copy_(torch.where(mask, self.source_state[name].to(value.device), value))


@torch.no_grad()
def update_ema_model(student, teacher, momentum=0.999):
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        teacher_buffer.copy_(student_buffer)
