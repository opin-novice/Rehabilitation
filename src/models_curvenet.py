"""CurveNet point-cloud encoder for rehabilitation exercise scoring.

Clean, device-agnostic port of the CurveNet architecture from:
  King-Rafat/Transformer_Rehabilitation (Rafat et al., JBHI 2026)

Adapted for SSL compatibility: exposes `forward_features(x) -> (B, out_dim)`
and `.out_dim` so it plugs into `build_encoder()` in models_stgcn.py.

Original authors: Tiange Xiang (CurveNet), Kazi Rafat (rehab adaptation).
Ported and fixed for cross-device operation by the research team.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Geometry primitives (device-agnostic)
# ---------------------------------------------------------------------------

def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """K-nearest neighbors on point cloud x.

    Args:
        x: (B, C, N) point features/coordinates.
        k: number of neighbors (includes self, so k+1 returned).

    Returns:
        (B, N, k+1) neighbor indices.
    """
    k = k + 1
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance between src (B,N,3) and dst (B,M,3)."""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by index.

    Args:
        points: (B, N, C)
        idx: (B, S) or (B, S, K)

    Returns:
        Indexed points with same leading dims as idx.
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long, device=device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest point sampling.

    Args:
        xyz: (B, N, 3)
        npoint: number of points to select.

    Returns:
        (B, npoint) selected indices.
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)  # (B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)  # (B, N)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = distance.argmax(dim=-1)
    return centroids


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor,
                     new_xyz: torch.Tensor) -> torch.Tensor:
    """Find all points within radius of new_xyz.

    Args:
        radius: search radius.
        nsample: max points to return per query.
        xyz: (B, N, 3) all points.
        new_xyz: (B, S, 3) query points.

    Returns:
        (B, S, nsample) indices (padded with 0 for short neighborhoods).
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.zeros(B, S, nsample, dtype=torch.long, device=device)
    for i in range(S):
        diff = xyz - new_xyz[:, i:i + 1, :]
        dist = torch.sum(diff ** 2, dim=-1)
        group_idx[:, i] = dist.topk(nsample, dim=-1, largest=False)[1]
    return group_idx


def sample_and_group(npoint: int, radius: float, k: int,
                     xyz: torch.Tensor, points: torch.Tensor):
    """Sample points and group neighbors.

    Args:
        npoint: number of sampled points.
        radius: ball query radius.
        k: neighbors per sampled point.
        xyz: (B, N, 3)
        points: (B, N, D)

    Returns:
        new_xyz: (B, npoint, 3)
        new_points: (B, npoint, k, 3+D)
    """
    new_xyz = index_points(xyz, farthest_point_sample(xyz, npoint))
    idx = query_ball_point(radius, k, xyz, new_xyz)
    new_points = index_points(points, idx)
    return new_xyz, new_points


# ---------------------------------------------------------------------------
# LPFA — Local Point Feature Aggregation
# ---------------------------------------------------------------------------

class LPFA(nn.Module):
    """Initial local feature extraction via KNN grouping + 2D conv.

    For non-initial layers, computes relative features and combines with
    coordinate-derived features via learned projection.
    """

    def __init__(self, in_channel: int, out_channel: int, k: int,
                 mlp_num: int = 2, initial: bool = False):
        super().__init__()
        self.k = k
        self.initial = initial

        if not initial:
            self.xyz2feature = nn.Sequential(
                nn.Conv2d(9, in_channel, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channel),
            )

        layers = []
        for _ in range(mlp_num):
            layers.extend([
                nn.Conv2d(in_channel, out_channel, 1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.LeakyReLU(0.2),
            ])
            in_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, xyz: torch.Tensor,
                idx: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, N) point features.
            xyz: (B, N, 3) or (B, 3, N) coordinates.
            idx: optional pre-computed KNN indices (B, N, k).

        Returns:
            (B, out_channel, N) aggregated features.
        """
        x = self.group_feature(x, xyz, idx)
        x = self.mlp(x)
        if self.initial:
            return x.max(dim=-1, keepdim=False)[0]
        return x.mean(dim=-1, keepdim=False)

    def group_feature(self, x: torch.Tensor, xyz: torch.Tensor,
                      idx: torch.Tensor | None) -> torch.Tensor:
        """Group features around each point via KNN."""
        batch_size, num_dims, num_points = x.size()

        if idx is None:
            idx = knn(xyz, k=self.k)[:, :, :self.k]

        # Flatten batch indexing
        device = x.device
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = (idx + idx_base).view(-1)

        if self.initial:
            # xyz: (B, 3, N) -> (B, N, 3) for coordinate features
            xyz_t = xyz.transpose(2, 1).contiguous()  # (B, N, 3)
            # Neighbor coordinates: (B, N, K, 3)
            point_feature = xyz_t.view(batch_size * num_points, -1)[idx, :]
            point_feature = point_feature.view(batch_size, num_points, self.k, -1)
            # Center coordinates: (B, N, 1, 3)
            center = xyz_t.unsqueeze(2).expand(-1, -1, self.k, -1)
            # Relative coordinates
            relative = point_feature - center
            # Concatenate: (B, N, K, 9) = [xyz, relative, center]
            point_feature = torch.cat([point_feature, relative, center], dim=-1)
            # -> (B, 9, N, K) for Conv2d
            return point_feature.permute(0, 3, 1, 2).contiguous()

        # Non-initial: relative features from point features
        x_t = x.transpose(2, 1).contiguous()  # (B, N, C)
        feature = x_t.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, self.k, num_dims)
        x_expanded = x_t.view(batch_size, num_points, 1, num_dims)
        feature = feature - x_expanded
        feature = feature.permute(0, 3, 1, 2).contiguous()  # (B, C, N, K)

        # Coordinate features: same 9-channel format as initial path
        xyz_t = xyz.transpose(2, 1).contiguous()  # (B, N, 3)
        neighbor_xyz = xyz_t.view(batch_size * num_points, -1)[idx, :]
        neighbor_xyz = neighbor_xyz.view(batch_size, num_points, self.k, 3)
        center_xyz = xyz_t.unsqueeze(2).expand(-1, -1, self.k, -1)
        relative_xyz = neighbor_xyz - center_xyz
        coord_feature = torch.cat([neighbor_xyz, relative_xyz, center_xyz], dim=-1)  # (B, N, K, 9)
        coord_feature = coord_feature.permute(0, 3, 1, 2).contiguous()  # (B, 9, N, K)

        point_feature = self.xyz2feature(coord_feature)  # (B, C, N, K)
        feature = F.leaky_relu(feature + point_feature, 0.2)
        return feature  # (B, C, N, K)


# ---------------------------------------------------------------------------
# Walk — Curve walking through point cloud
# ---------------------------------------------------------------------------

class Walk(nn.Module):
    """Walk through point cloud along KNN neighbors to form curves.

    Uses momentum-based curve descriptors and crossover suppression
    to prevent curves from doubling back.
    """

    def __init__(self, in_channel: int, k: int, curve_num: int, curve_length: int):
        super().__init__()
        self.curve_num = curve_num
        self.curve_length = curve_length
        self.k = k

        self.agent_mlp = nn.Sequential(
            nn.Conv2d(in_channel * 2, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
        )
        self.momentum_mlp = nn.Sequential(
            nn.Conv1d(in_channel * 2, 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(2),
        )

    def crossover_suppression(self, cur, neighbor, bn, n, k):
        neighbor = neighbor.detach()
        cur = cur.unsqueeze(-1).detach()
        dot = torch.bmm(cur.transpose(1, 2), neighbor)
        norm1 = torch.norm(cur, dim=1, keepdim=True)
        norm2 = torch.norm(neighbor, dim=1, keepdim=True)
        divider = torch.clamp(norm1 * norm2, min=1e-8)
        ans = torch.div(dot, divider).squeeze()
        ans = 1.0 + ans
        return torch.clamp(ans, 0.0, 1.0).detach()

    def forward(self, xyz: torch.Tensor, x: torch.Tensor,
                adj: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: (B, 3, N) coordinates.
            x: (B, C, N) features.
            adj: (B, N, K) KNN adjacency indices (self-loop already removed).
            cur: (B, curve_num, 1) starting point indices.

        Returns:
            (B, C, curve_num, curve_length) curve features.
        """
        device = x.device
        bn, c, tot_points = x.size()

        xyz = xyz.transpose(1, 2).contiguous()  # (B, N, 3)
        x = x.transpose(1, 2).contiguous()       # (B, N, C)

        flatten_x = x.view(bn * tot_points, -1)
        batch_offset = torch.arange(0, bn, device=device).detach() * tot_points

        # Build full-point adjacency from KNN (adj already has self-loop removed)
        n_adj = adj.size(1)
        adj_offset = (adj + batch_offset.view(-1, 1, 1)).view(bn * n_adj, -1)

        # For points not in adj (if N_adj < N), pad with self
        full_adj = torch.zeros(bn * tot_points, adj.size(-1), dtype=torch.long, device=device)
        full_adj.view(bn, tot_points, -1)[:, :n_adj, :] = adj_offset.view(bn, n_adj, -1)

        # Adjust starting point indices
        flatten_cur = (cur + batch_offset.view(-1, 1, 1)).view(-1)

        curves = []

        for step in range(self.curve_length):
            if step == 0:
                starting_points = flatten_x[flatten_cur, :].contiguous()
                # (B, curve_num, C) -> (B, C, curve_num, 1)
                pre_feature = starting_points.view(bn, self.curve_num, -1, 1).transpose(1, 2)
            else:
                # Dynamic momentum: blend cur_feature and pre_feature
                cat_feature = torch.cat((cur_feature.squeeze(), pre_feature.squeeze()), dim=1)
                att_feature = F.softmax(self.momentum_mlp(cat_feature), dim=1).view(bn, 1, self.curve_num, 2)
                cat_feature = torch.cat((cur_feature, pre_feature), dim=-1)  # (B, C, curve_num, 2)
                pre_feature = torch.sum(cat_feature * att_feature, dim=-1, keepdim=True)  # (B, C, curve_num, 1)
                pre_feature_cos = pre_feature.transpose(1, 2).contiguous().view(bn * self.curve_num, -1)

            pick_idx = full_adj[flatten_cur]  # (B*curve_num, K)

            pick_values = flatten_x[pick_idx.view(-1), :]

            # Reshape for crossover suppression
            pick_values_cos = pick_values.view(bn * self.curve_num, self.k, c)
            pick_values = pick_values_cos.view(bn, self.curve_num, self.k, c)
            pick_values_cos = pick_values_cos.transpose(1, 2).contiguous()

            pick_values = pick_values.permute(0, 3, 1, 2)  # (B, C, curve_num, K)

            pre_feature_expand = pre_feature.expand_as(pick_values)

            # Concatenate candidate features with curve descriptor
            pre_feature_expand = torch.cat((pick_values, pre_feature_expand), dim=1)

            # Agent-based curve descriptor: which node to pick next?
            pre_feature_expand = self.agent_mlp(pre_feature_expand)  # (B, 1, curve_num, K)

            if step != 0:
                # Crossover suppression
                d = self.crossover_suppression(
                    cur_feature_cos - pre_feature_cos,
                    pick_values_cos - cur_feature_cos.unsqueeze(-1),
                    bn, self.curve_num, self.k
                )
                d = d.view(bn, self.curve_num, self.k).unsqueeze(1)
                pre_feature_expand = torch.mul(pre_feature_expand, d)

            # Gumbel softmax for hard assignment
            y = F.softmax(pre_feature_expand, dim=-1)
            shape = y.size()
            _, ind = y.max(dim=-1)
            y_hard = torch.zeros_like(y).view(-1, shape[-1])
            y_hard.scatter_(1, ind.view(-1, 1), 1)
            y_hard = y_hard.view(*shape)
            pre_feature_expand = (y_hard - y).detach() + y

            # Weighted sum over neighbors
            cur_feature = torch.sum(pick_values * pre_feature_expand, dim=-1, keepdim=True)  # (B, C, curve_num, 1)

            cur_feature_cos = cur_feature.transpose(1, 2).contiguous().view(bn * self.curve_num, c)

            # Advance to the selected neighbor
            cur = torch.argmax(pre_feature_expand, dim=-1).view(-1, 1)
            flatten_cur = torch.gather(pick_idx, 1, cur.view(bn * self.curve_num, 1)).squeeze()

            curves.append(cur_feature)

        return torch.cat(curves, dim=-1)  # (B, C, curve_num, curve_length)


# ---------------------------------------------------------------------------
# CurveGrouping — Select starting points + walk
# ---------------------------------------------------------------------------

class CurveGrouping(nn.Module):
    """Select curve starting points via self-attention, then walk."""

    def __init__(self, in_channel: int, k: int, curve_num: int,
                 curve_length: int, att_in_channels: int | None = None):
        super().__init__()
        self.curve_num = curve_num
        self.curve_length = curve_length
        self.k = k
        self.att = nn.Conv1d(att_in_channels or in_channel, 1, kernel_size=1, bias=False)
        self.walk = Walk(in_channel, k, curve_num, curve_length)

    def forward(self, x: torch.Tensor, xyz: torch.Tensor,
                idx: torch.Tensor) -> torch.Tensor:
        """Returns (B, C, curve_num, curve_length)."""
        x_att = torch.sigmoid(self.att(x))
        x = x * x_att
        _, start_index = torch.topk(x_att, self.curve_num, dim=2, sorted=False)
        start_index = start_index.squeeze().unsqueeze(2)
        return self.walk(xyz, x, idx, start_index)


# ---------------------------------------------------------------------------
# CurveAggregation — Mix features across and along curves
# ---------------------------------------------------------------------------

class CurveAggregation(nn.Module):
    """Aggregate features from curves using inter-curve and intra-curve cross-attention."""

    def __init__(self, in_channel: int):
        super().__init__()
        self.in_channel = in_channel
        mid_feature = in_channel // 2
        # Curve-side projections
        self.conva = nn.Conv1d(in_channel, mid_feature, kernel_size=1, bias=False)
        self.convb = nn.Conv1d(in_channel, mid_feature, kernel_size=1, bias=False)
        # Point-side projection (projects x to logits for bmm)
        self.convc = nn.Conv1d(in_channel, mid_feature, kernel_size=1, bias=False)
        # After-bmm projections
        self.convn = nn.Conv1d(mid_feature, mid_feature, kernel_size=1, bias=False)
        self.convl = nn.Conv1d(mid_feature, mid_feature, kernel_size=1, bias=False)
        # Final merge
        self.convd = nn.Sequential(
            nn.Conv1d(mid_feature * 2, in_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channel),
        )
        self.line_conv_att = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, curves: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, N) point features.
            curves: (B, C, curve_num, curve_length)

        Returns:
            (B, C, N) updated features.
        """
        curves_att = self.line_conv_att(curves)  # (B, 1, c_n, c_l)

        # Inter-curve: pool over curve_length -> (B, C, c_n)
        curver_inter = torch.sum(
            curves * F.softmax(curves_att, dim=-1), dim=-1
        )
        # Intra-curve: pool over curve_num -> (B, C, c_l)
        curves_intra = torch.sum(
            curves * F.softmax(curves_att, dim=-2), dim=-2
        )

        # Project curve features
        curver_inter = self.conva(curver_inter)  # (B, mid, c_n)
        curves_intra = self.convb(curves_intra)  # (B, mid, c_l)

        # Cross-attention: project points to logits and bmm with curves
        x_logits = self.convc(x).transpose(1, 2).contiguous()  # (B, N, mid)
        x_inter = F.softmax(torch.bmm(x_logits, curver_inter), dim=-1)  # (B, N, c_n)
        x_intra = F.softmax(torch.bmm(x_logits, curves_intra), dim=-1)  # (B, N, c_l)

        # Project back to curve space and bmm to get point features
        curver_inter = self.convn(curver_inter).transpose(1, 2).contiguous()  # (B, c_n, mid)
        curves_intra = self.convl(curves_intra).transpose(1, 2).contiguous()  # (B, c_l, mid)

        x_inter = torch.bmm(x_inter, curver_inter)  # (B, N, mid)
        x_intra = torch.bmm(x_intra, curves_intra)  # (B, N, mid)

        # Merge
        curve_features = torch.cat((x_inter, x_intra), dim=-1).transpose(1, 2).contiguous()  # (B, 2*mid, N)
        x = x + self.convd(curve_features)

        return F.leaky_relu(x, negative_slope=0.2)


# ---------------------------------------------------------------------------
# MaskedMaxPool
# ---------------------------------------------------------------------------

class MaskedMaxPool(nn.Module):
    """Max pooling with ball query grouping."""

    def __init__(self, npoint: int, radius: float, k: int):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.k = k

    def forward(self, xyz: torch.Tensor, features: torch.Tensor):
        sub_xyz, neighborhood_features = sample_and_group(
            self.npoint, self.radius, self.k, xyz, features.transpose(1, 2)
        )
        neighborhood_features = neighborhood_features.permute(0, 3, 1, 2).contiguous()
        sub_features = F.max_pool2d(
            neighborhood_features,
            kernel_size=[1, neighborhood_features.shape[3]],
        )
        sub_features = sub_features.squeeze(-1)
        return sub_xyz, sub_features


# ---------------------------------------------------------------------------
# CIC — Curve-Informed Convolution block
# ---------------------------------------------------------------------------

class CIC(nn.Module):
    """Curve-Informed Convolution: the core building block.

    Combines curve grouping, curve aggregation, and local point feature
    aggregation with a residual connection.
    """

    def __init__(self, npoint: int, radius: float, k: int,
                 in_channels: int, output_channels: int,
                 bottleneck_ratio: int = 2, mlp_num: int = 2,
                 curve_config: list | None = None):
        super().__init__()
        self.in_channels = in_channels
        self.output_channels = output_channels
        self.bottleneck_ratio = bottleneck_ratio
        self.radius = radius
        self.k = k
        self.npoint = npoint

        planes = in_channels // bottleneck_ratio

        self.use_curve = curve_config is not None
        if self.use_curve:
            self.curveaggregation = CurveAggregation(planes)
            self.curvegrouping = CurveGrouping(
                in_channel=planes, k=k,
                curve_num=curve_config[0], curve_length=curve_config[1],
                att_in_channels=planes,  # att receives post-conv1 features (planes channels)
            )

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, planes, kernel_size=1, bias=False),
            nn.BatchNorm1d(planes),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(planes, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_channels),
        )

        if in_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, output_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(output_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.maxpool = MaskedMaxPool(npoint, radius, k)
        self.lpfa = LPFA(planes, planes, k, mlp_num=mlp_num, initial=False)

    def forward(self, xyz: torch.Tensor, x: torch.Tensor):
        """
        Args:
            xyz: (B, 3, N) coordinates.
            x: (B, C, N) features.

        Returns:
            xyz: (B, 3, npoint) subsampled coordinates.
            x: (B, output_channels, npoint) features.
        """
        # Subsample if needed
        if xyz.size(-1) != self.npoint:
            xyz, x = self.maxpool(xyz.transpose(1, 2).contiguous(), x)
            xyz = xyz.transpose(1, 2)

        shortcut = x
        x = self.conv1(x)  # (B, C', N)

        idx = knn(xyz, self.k)
        if self.use_curve:
            curves = self.curvegrouping(x, xyz, idx[:, :, 1:])  # avoid self-loop
            x = self.curveaggregation(x, curves)

        x = self.lpfa(x, xyz, idx=idx[:, :, :self.k])  # (B, C', N, K)
        x = self.conv2(x)  # (B, C_out, N)

        shortcut = self.shortcut(shortcut)
        x = self.relu(x + shortcut)

        return xyz, x


# ---------------------------------------------------------------------------
# CurveNet Encoder — the rehabilitation-adapted version
# ---------------------------------------------------------------------------

class CurveNetEncoder(nn.Module):
    """CurveNet point-cloud encoder for 25-joint skeleton data.

    Pipeline:
        LPFA(initial) -> CIC11 -> CIC12 -> CIC21 -> CIC22 -> conv

    Input:  (B, C, N) where C=3 (xyz), N=25 (joints)
    Output: (B, 256, N) per-frame features

    For the full TemporalModel pipeline:
        Input:  (B*T, 3, 25)
        Output: (B*T, 256, 25) -> reshape to (B, T, 256, 25)
    """

    def __init__(self, k: int = 20, num_joints: int = 25,
                 setting: str = "default"):
        super().__init__()
        self.num_joints = num_joints
        self.k = k
        self.out_dim = 256  # final feature dimension per point

        # curve_config for 25-joint skeleton
        # [curve_num, curve_length] for each stage, None = no curves
        self._curve_config = {
            "default": [[15, 5], [15, 5], None, None],
            "long": [[10, 30], None, None, None],
        }
        assert setting in self._curve_config
        cfg = self._curve_config[setting]

        additional_channel = 32

        self.lpfa = LPFA(9, additional_channel, k=k, mlp_num=1, initial=True)

        # Encoder stages (25 joints throughout, no downsampling)
        self.cic11 = CIC(
            npoint=num_joints, radius=0.08, k=k,
            in_channels=additional_channel, output_channels=32,
            bottleneck_ratio=2, mlp_num=1, curve_config=cfg[0],
        )
        self.cic12 = CIC(
            npoint=num_joints, radius=0.08, k=k,
            in_channels=32, output_channels=64,
            bottleneck_ratio=4, mlp_num=1, curve_config=cfg[0],
        )
        self.cic21 = CIC(
            npoint=num_joints, radius=0.08, k=k,
            in_channels=64, output_channels=128,
            bottleneck_ratio=2, mlp_num=1, curve_config=cfg[1],
        )
        self.cic22 = CIC(
            npoint=num_joints, radius=0.16, k=k,
            in_channels=128, output_channels=256,
            bottleneck_ratio=4, mlp_num=1, curve_config=cfg[1],
        )

        self.conv0 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: (B, 3, N) or (B, C, N) point cloud per frame.
                 For skeletons: (B, 3, 25).

        Returns:
            (B, 256, N) per-point features.
        """
        l0_points = self.lpfa(xyz, xyz)
        l1_xyz, l1_points = self.cic11(xyz, l0_points)
        l1_xyz, l1_points = self.cic12(l1_xyz, l1_points)
        l2_xyz, l2_points = self.cic21(l1_xyz, l1_points)
        l2_xyz, l2_points = self.cic22(l2_xyz, l2_points)
        return self.conv0(l2_points)  # (B, 256, N)


# ---------------------------------------------------------------------------
# Full Temporal Model — CurveNet + Axial Transformer + Temporal Transformer
# ---------------------------------------------------------------------------

class PointCloudTransformerRegressor(nn.Module):
    """Full rehabilitation scoring model using point-cloud transformer.

    Pipeline:
        [B, T, N, C]  (batch, frames, joints, coords)
        -> reshape to [(B*T), C, N]
        -> CurveNetEncoder   [(B*T), 256, N]
        -> reshape to [B, T, 256, N]
        -> AxialTransformer  [B, T, D]  (spatial+temporal attention)
        -> TemporalPool      [B, D]
        -> MLP Head          [B, 1]

    This model is compatible with the SSL pipeline:
        - forward_features(x) returns the pooled embedding [B, D]
        - .out_dim exposes D for head construction
    """

    def __init__(
        self,
        seq_len: int = 100,
        num_joints: int = 25,
        num_channels: int = 3,
        dim: int = 256,
        spatial_depth: int = 6,
        temporal_depth: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        k: int = 20,
        curve_setting: str = "default",
        num_exercises: int = 0,
        multitask: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_joints = num_joints
        self.num_channels = num_channels
        self.dim = dim
        self.out_dim = dim  # SSL encoder output dimension
        self.multitask = multitask

        # Point-cloud encoder
        self.encoder = CurveNetEncoder(k=k, num_joints=num_joints, setting=curve_setting)

        # Downsample joints: Conv1d(N, 32) operates on transposed features
        self.downsample = nn.Sequential(
            nn.Conv1d(num_joints, 32, kernel_size=1, bias=False),
            nn.BatchNorm1d(32),
        )

        # Axial self-attention transformer
        self.spatial_transformer = _AxialTransformer(
            dim=dim, depth=spatial_depth, heads=heads,
            dim_head=dim // heads, mlp_dim=dim * 2, dropout=dropout,
        )

        # Temporal transformer
        self.temporal_transformer = _AxialTransformer(
            dim=dim, depth=temporal_depth, heads=heads,
            dim_head=dim // heads, mlp_dim=dim * 2, dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

        # Regression head
        def _head(d: int, p: float) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d // 2),
                nn.GELU(),
                nn.Dropout(p),
                nn.Linear(d // 2, 1),
            )

        self.head = _head(dim, dropout)
        if multitask:
            self.head_po = _head(dim, dropout)
            self.head_cf = _head(dim, dropout)
        else:
            self.head_po = None
            self.head_cf = None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """SSL encoder entry point: returns pooled embedding [B, D].

        Args:
            x: (B, T, N, C) input skeleton sequence.

        Returns:
            (B, dim) pooled embedding.
        """
        b, t, n, c = x.size()

        # Process each frame through CurveNet
        x_flat = x.reshape(b * t, c, n)  # (B*T, C, N)
        features = self.encoder(x_flat)   # (B*T, encoder_out_dim, N)
        features = self.dropout(features)

        # Rearrange: (B*T, C, N) -> (B*T, N, C) for downsample
        features = features.transpose(1, 2)  # (B*T, N, encoder_out_dim)
        # Downsample joints: (B*T, N, C) -> Conv1d sees C=N, L=C -> (B*T, 32, C)
        features = self.downsample(features)  # (B*T, 32, encoder_out_dim)
        # Reshape: (B, T, 32, encoder_out_dim)
        features = features.view(b, t, 32, -1)

        # Axial transformer: alternate spatial/temporal attention
        features = self.spatial_transformer(features, swap=True)
        # (B, T, 32, dim) -> mean over joints -> (B, T, dim)
        features = features.view(b, t, 32, self.dim).mean(dim=2)

        # Temporal transformer: attend over frames
        features = self.temporal_transformer(features, swap=False)
        # (B, T, dim) -> mean over time -> (B, dim)
        features = features.mean(dim=1)

        return features

    def forward(
        self,
        x: torch.Tensor,
        exercise_id: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> torch.Tensor | tuple:
        """
        Args:
            x: (B, T, N, C) input.
            exercise_id: optional (B,) exercise type indices.
            return_features: if True, return (score, embedding).

        Returns:
            (B, 1) quality score, or (score, embedding) if return_features.
        """
        feat = self.forward_features(x)  # (B, dim)

        ts = self.head(feat)  # (B, 1)

        if self.multitask and self.head_po is not None:
            po = self.head_po(feat)
            cf = self.head_cf(feat)
            if return_features:
                return ts, po, cf, feat
            return ts, po, cf

        if return_features:
            return ts, feat
        return ts


# ---------------------------------------------------------------------------
# Axial Transformer (clean, device-agnostic reimplementation)
# ---------------------------------------------------------------------------

class _AxialTransformer(nn.Module):
    """Transformer with alternating spatial/temporal attention.

    When swap=True:
        Even layers: attend across frames for each joint  [(B*N, T, D)]
        Odd layers:  attend across joints for each frame  [(B*T, N, D)]
    When swap=False:
        Standard self-attention over the sequence dimension.
    """

    def __init__(self, dim: int, depth: int, heads: int,
                 dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                _PreNorm(dim, _Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                _PreNorm(dim, _FeedForward(dim, mlp_dim, dropout=dropout)),
            ]))

    def forward(self, x: torch.Tensor, swap: bool = False) -> torch.Tensor:
        """
        Args:
            x: if swap=True: (B, T, N, D)
               if swap=False: (B, T, D) or (B, T, N, D)
        """
        if swap:
            b, t, n, c = x.size()

        for idx, (attn, ff) in enumerate(self.layers):
            if swap:
                if idx % 2 == 0:
                    x = x.reshape(b * n, t, c)
                else:
                    x = x.reshape(b * t, n, c)

            x = attn(x) + x
            x = ff(x) + x

            if swap:
                if idx % 2 == 0:
                    x = x.reshape(b, t, n, c)
                else:
                    x = x.reshape(b, t, n, c)

        return x


class _PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64,
                 dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == _Attention)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.pos_embedding = _SinCosPositionalEncoding(dim, dropout=dropout)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embedding.pe[:, :x.size(1)]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.reshape(x.size(0), -1, self.heads, t.size(-1) // self.heads).transpose(1, 2),
            qkv,
        )
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(x.size(0), -1, x.size(-1))
        return self.to_out(out)


class _FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SinCosPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        import math
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10_000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Parameter counter
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
