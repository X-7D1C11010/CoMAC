import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, out_channels=128):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv1d(128, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        return x.transpose(1, 2)

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.npoint
        
        if self.group_all:
            new_xyz = xyz
            grouped_xyz = xyz.view(B, 1, N, C) - xyz.view(B, N, 1, C)
            if points is not None:
                grouped_points = points.view(B, 1, N, -1)
            else:
                grouped_points = grouped_xyz
        else:
            fps_idx = self.farthest_point_sample(xyz, S)
            new_xyz = self.gather_points(xyz, fps_idx)
            
            idx = self.ball_query(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = self.group_points(xyz, idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            
            if points is not None:
                grouped_points = self.group_points(points, idx)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz
        
        grouped_points = grouped_points.permute(0, 3, 2, 1)
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points = self.relu(bn(conv(grouped_points)))
        
        new_points = torch.max(grouped_points, 2)[0]
        new_points = new_points.permute(0, 2, 1)
        
        return new_xyz, new_points
    
    @staticmethod
    def farthest_point_sample(xyz, npoint):
        device = xyz.device
        B, N, C = xyz.shape
        
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            distance = torch.min(distance, dist)
            farthest = torch.max(distance, -1)[1]
        
        return centroids
    
    @staticmethod
    def gather_points(points, idx):
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        
        batch_indices = torch.arange(B, dtype=torch.long).view(view_shape).repeat(repeat_shape)
        
        return points[batch_indices, idx, :]
    
    @staticmethod
    def ball_query(radius, nsample, xyz, new_xyz):
        device = xyz.device
        B, N, C = xyz.shape
        _, S, _ = new_xyz.shape
        
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        dist_sq = torch.sum((xyz.unsqueeze(1) - new_xyz.unsqueeze(2)) ** 2, dim=-1)
        
        group_idx[dist_sq > radius ** 2] = N
        group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
        
        group_first = group_idx[:, :, 0:1].repeat([1, 1, nsample])
        mask = group_idx == N
        group_idx[mask] = group_first[mask]
        
        return group_idx
    
    @staticmethod
    def group_points(points, idx):
        B = points.shape[0]
        C = points.shape[-1]
        S = idx.shape[1]
        N = idx.shape[-1]
        
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        
        batch_indices = torch.arange(B, dtype=torch.long).view(view_shape).repeat(repeat_shape)
        
        return points[batch_indices, idx, :]

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, xyz1, xyz2, points1, points2):
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape
        
        if points2 is None:
            points2 = torch.zeros(B, S, 0).to(xyz1.device)
        
        dist_sq = torch.sum((xyz1.unsqueeze(1) - xyz2.unsqueeze(2)) ** 2, dim=-1)
        dist_sq = dist_sq.permute(0, 2, 1)
        dist_sq, idx = dist_sq.sort(dim=-1)
        dist_sq, idx = dist_sq[:, :, :3], idx[:, :, :3]
        
        dist_recip = 1.0 / (dist_sq + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        
        if points2.size(-1) > 0:
            interpolated_points = torch.zeros(B, N, points2.size(-1)).to(xyz1.device)
            for b in range(B):
                for n in range(N):
                    ws = weight[b, n]
                    nearest_idx = idx[b, n]
                    for i in range(3):
                        s_idx = int(nearest_idx[i])
                        if s_idx < S:
                            interpolated_points[b, n] += ws[i] * points2[b, s_idx]
        else:
            interpolated_points = torch.zeros(B, N, 0).to(xyz1.device)
        
        if points1 is not None:
            if interpolated_points.size(-1) > 0:
                new_points = torch.cat([points1, interpolated_points], dim=-1)
            else:
                new_points = points1
        else:
            new_points = interpolated_points
        
        new_points = new_points.permute(0, 2, 1)
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = self.relu(bn(conv(new_points)))
        
        new_points = new_points.permute(0, 2, 1)
        
        return new_points

class PointCloudSegmentationModel(nn.Module):
    def __init__(self, num_classes=11, feature_dim=128):
        super().__init__()
        self.num_classes = num_classes
        
        self.sa1 = PointNetSetAbstraction(
            npoint=2048, radius=0.1, nsample=32,
            in_channel=9, mlp=[32, 32, 64], group_all=False
        )
        
        self.sa2 = PointNetSetAbstraction(
            npoint=1024, radius=0.2, nsample=32,
            in_channel=64 + 3, mlp=[64, 64, 128], group_all=False
        )
        
        self.sa3 = PointNetSetAbstraction(
            npoint=512, radius=0.4, nsample=32,
            in_channel=128 + 3, mlp=[128, 128, 256], group_all=False
        )
        
        self.sa4 = PointNetSetAbstraction(
            npoint=256, radius=0.8, nsample=32,
            in_channel=256 + 3, mlp=[256, 256, 512], group_all=False
        )
        
        self.fp4 = PointNetFeaturePropagation(
            in_channel=768, mlp=[256, 256]
        )
        
        self.fp3 = PointNetFeaturePropagation(
            in_channel=384, mlp=[256, 256]
        )
        
        self.fp2 = PointNetFeaturePropagation(
            in_channel=320, mlp=[256, 128]
        )
        
        self.fp1 = PointNetFeaturePropagation(
            in_channel=128 + 6, mlp=[128, 128, 128]
        )
        
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(128, num_classes, 1)
    
    def forward(self, points, return_features=False):
        xyz = points[:, :, :3]
        features = points[:, :, 3:] if points.shape[-1] > 3 else None
        
        l0_points = points
        
        l1_xyz, l1_points = self.sa1(xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, l0_points, l1_points)
        
        final_features = l0_points.transpose(1, 2)
        
        x = self.conv1(final_features)
        x = self.bn1(x)
        x = self.relu(x)
        
        logits = self.conv2(x)
        logits = logits.transpose(1, 2)
        
        if return_features:
            return logits, l0_points
        else:
            return logits

if __name__ == '__main__':
    model = PointCloudSegmentationModel(num_classes=11)
    input_tensor = torch.randn(1, 20000, 6)
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
