import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels_x1, in_channels_x2, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels_x1 + in_channels_x2, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels_x1, in_channels_x1 // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels_x1 // 2 + in_channels_x2, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

class ResNetEncoder(nn.Module):
    def __init__(self, backbone='resnet34', pretrained=True):
        super().__init__()
        
        if backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=pretrained)
            self.out_channels = [64, 64, 128, 256, 512]
        elif backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained)
            self.out_channels = [64, 64, 128, 256, 512]
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.maxpool = self.resnet.maxpool
        
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
    
    def forward(self, x):
        features = []
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)
        
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)
        
        x = self.layer2(x)
        features.append(x)
        
        x = self.layer3(x)
        features.append(x)
        
        x = self.layer4(x)
        features.append(x)
        
        return features

class ImageSegmentationModel(nn.Module):
    def __init__(self, num_classes=11, backbone='resnet34', pretrained=True, bilinear=True):
        super().__init__()
        self.num_classes = num_classes
        
        self.encoder = ResNetEncoder(backbone=backbone, pretrained=pretrained)
        self.out_channels = self.encoder.out_channels
        
        factor = 2 if bilinear else 1
        
        self.up1 = Up(self.out_channels[-1], self.out_channels[-2], self.out_channels[-2], bilinear)
        self.up2 = Up(self.out_channels[-2], self.out_channels[-3], self.out_channels[-3], bilinear)
        self.up3 = Up(self.out_channels[-3], self.out_channels[-4], self.out_channels[-4], bilinear)
        self.up4 = Up(self.out_channels[-4], self.out_channels[-5], self.out_channels[-5], bilinear)
        
        self.out_conv = OutConv(self.out_channels[-5], num_classes)
    
    def forward(self, x, return_features=False):
        features = self.encoder(x)
        
        x1, x2, x3, x4, x5 = features
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.out_conv(x)
        
        if return_features:
            return logits, x
        else:
            return logits

if __name__ == '__main__':
    model = ImageSegmentationModel(num_classes=11)
    input_tensor = torch.randn(1, 3, 480, 302)
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
