# Code partially adapted and modified from PTC-Net:
# https://github.com/LeegoChen/PTC-Net

import torch
import torch.nn as nn
from functools import partial

import MinkowskiEngine as ME

from typing import List

from libs.pointops.functions import pointops
from misc import pt_util


def sparse_to_tensor(tensorIn: ME.SparseTensor, batch_size: int, padding_value: float):
    points, feats = [], []
    for i in range(batch_size):
        points.append(tensorIn.C[tensorIn.C[:, 0] == i, :])
        feats.append(tensorIn.F[tensorIn.C[:, 0] == i, :])

    max_len = max([len(i) for i in feats])
    padding_num = [max_len - len(i) for i in feats]
    if padding_value is not None:
        padding_funcs = [nn.ConstantPad2d(padding=(0, 0, 0, i), value=padding_value) for i in padding_num]

        tensor_feats = torch.stack([pad_fun(e) for e, pad_fun in zip(feats, padding_funcs)], dim=0)
        tensor_coords = torch.stack([pad_fun(e) for e, pad_fun in zip(points, padding_funcs)], dim=0)
        mask = [torch.ones(len(i), 1) for i in feats]
        mask = torch.stack([pad_fun(e) for e, pad_fun in zip(mask, padding_funcs)], dim=0).bool().squeeze(dim=2)
    else:
        tensor_feats = torch.stack(
            [torch.cat((feats[i], feats[i][-1].repeat(num, 1)), dim=0) for i, num in enumerate(padding_num)], dim=0
        )
        tensor_coords = torch.stack(
            [torch.cat((points[i], points[i][-1].repeat(num, 1)), dim=0) for i, num in enumerate(padding_num)], dim=0
        )
        mask = [torch.ones(len(i), 1) for i in feats]
        mask = torch.stack(
            [torch.cat((mask[i], torch.zeros(1, 1).repeat(num, 1)), dim=0) for i, num in enumerate(padding_num)],
            dim=0,
        ).bool().squeeze(dim=2)
    return tensor_feats, tensor_coords, mask


class BasicSpconvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1):
        super(BasicSpconvBlock, self).__init__()
        D = 3
        self.conv = ME.MinkowskiConvolution(
            inplanes, outplanes, kernel_size=kernel_size, stride=stride, bias=False, dimension=D
        )
        self.bn = ME.MinkowskiBatchNorm(outplanes, eps=1e-6)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


class FTU(nn.Module):
    def __init__(
        self,
        inplanes,
        outplanes,
        act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        quantization_size=0.01,
    ):
        super(FTU, self).__init__()
        D = 3
        self.quantization_size = quantization_size
        self.conv1x1 = ME.MinkowskiConvolution(inplanes, outplanes, kernel_size=1, stride=1, dimension=D)

        self.norm = ME.MinkowskiBatchNorm(outplanes)
        self.act = ME.MinkowskiGELU()

        self.ln = norm_layer(outplanes)
        self.act = act_layer()
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

    def forward(self, sparse, xyz_t):
        B, N, _ = xyz_t.size()
        sparse = self.conv1x1(sparse)

        f_tensor, x_tensor, _ = sparse_to_tensor(sparse, B, 1e3)
        dist, idx = pointops.nearestneighbor(xyz_t, x_tensor[:, :, 1:] * self.quantization_size)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm

        interpolated_feats = pointops.interpolation(f_tensor.transpose(1, 2).contiguous(), idx, weight)
        interpolated_feats = self.act(self.ln(interpolated_feats.transpose(1, 2)))
        interpolated_feats = interpolated_feats.transpose(1, 2).contiguous()

        return interpolated_feats


class Sample_interpolated(nn.Module):
    def __init__(self, npoint, inplanes, outplanes, group=4, first_sample=False):
        super(Sample_interpolated, self).__init__()
        self.npoint = npoint
        self.get_interpolate = FTU(inplanes, outplanes)
        self.first_sample = first_sample

    def forward(self, xyz, sparse, feat_t=None):
        B, _, N = xyz.shape
        if self.first_sample:
            xyz_trans = xyz.transpose(1, 2).contiguous()
            center_idx = pointops.furthestsampling(xyz, self.npoint)
            new_xyz = (
                pointops.gathering(xyz_trans, center_idx).transpose(1, 2).contiguous()
                if self.npoint is not None
                else None
            )
            if feat_t is not None:
                new_features = pointops.gathering(feat_t, center_idx)
            else:
                new_features = None
        else:
            new_xyz = xyz[:, : self.npoint, :].contiguous()
            if feat_t is not None:
                new_features = feat_t[:, :, : self.npoint]
            else:
                new_features = None

        if sparse is not None:
            interpolated_feat = self.get_interpolate(sparse, new_xyz)
        else:
            interpolated_feat = None
        return new_xyz, interpolated_feat, new_features


class DeformablePointAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        num_points=16,
        dropout=0.1,
        offset_scale=1.0,
        use_relative_pos=True,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = dim // num_heads
        self.use_relative_pos = use_relative_pos
        self.scale = self.head_dim**-0.5

        self.offset_scale = nn.Parameter(torch.tensor(offset_scale))

        self.offset_net = nn.Sequential(
            nn.Linear(dim + 3, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_heads * num_points * 3),
        )

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        if use_relative_pos:
            self.rel_pos_enc = nn.Sequential(
                nn.Linear(3, dim // 4),
                nn.LayerNorm(dim // 4),
                nn.GELU(),
                nn.Linear(dim // 4, 1),
            )

        self.proj = nn.Linear(dim, dim, bias=False)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.proj]:
            nn.init.xavier_uniform_(m.weight)

        nn.init.normal_(self.offset_net[-1].weight, std=0.001)
        nn.init.zeros_(self.offset_net[-1].bias)

        nn.init.xavier_uniform_(self.ffn[0].weight)
        nn.init.xavier_uniform_(self.ffn[3].weight)
        nn.init.zeros_(self.ffn[0].bias)
        nn.init.zeros_(self.ffn[3].bias)

    def _compute_offsets(self, x: torch.Tensor, xyz: torch.Tensor):
        B, N, C = x.shape
        x_pos = torch.cat([x, xyz], dim=-1)
        offsets = self.offset_net(x_pos)
        offsets = offsets.view(B, N, self.num_heads, self.num_points, 3)
        scale = self.offset_scale.abs()
        offsets = torch.tanh(offsets) * scale
        return offsets

    def _interp_KV_once(self, K: torch.Tensor, V: torch.Tensor, xyz: torch.Tensor, sampled_xyz: torch.Tensor):
        B, N, H, D = K.shape
        P = sampled_xyz.size(3)

        KV = torch.cat([K, V], dim=-1)
        known_feats = KV.permute(0, 2, 3, 1).reshape(B * H, 2 * D, N)

        xyz_expand = xyz.unsqueeze(1).expand(B, H, N, 3).reshape(B * H, N, 3)
        q_xyz = sampled_xyz.permute(0, 2, 1, 3, 4).reshape(B * H, N * P, 3)

        dist, idx = pointops.nearestneighbor(q_xyz, xyz_expand)
        dist_recip = 1.0 / (dist + 1e-8)
        weight = dist_recip / dist_recip.sum(dim=2, keepdim=True)

        interp = pointops.interpolation(known_feats, idx, weight)
        interp = interp.view(B, H, 2 * D, N, P).permute(0, 3, 1, 4, 2).contiguous()

        sampled_K = interp[..., :D]
        sampled_V = interp[..., D:]

        return sampled_K, sampled_V

    def forward(self, x: torch.Tensor, xyz: torch.Tensor):
        B, C, N = x.shape
        x_t = x.transpose(1, 2)
        identity = x_t

        offsets = self._compute_offsets(x_t, xyz)
        sampled_xyz = xyz.unsqueeze(2).unsqueeze(3) + offsets

        Q = self.q_proj(x_t).view(B, N, self.num_heads, self.head_dim)
        K = self.k_proj(x_t).view(B, N, self.num_heads, self.head_dim)
        V = self.v_proj(x_t).view(B, N, self.num_heads, self.head_dim)

        sampled_K, sampled_V = self._interp_KV_once(K, V, xyz, sampled_xyz)

        attn = torch.matmul(Q.unsqueeze(3), sampled_K.transpose(-2, -1)).squeeze(3) * self.scale

        if self.use_relative_pos:
            rel_pos = xyz.unsqueeze(2).unsqueeze(3) - sampled_xyz
            B_rp, N_rp, H_rp, P_rp, _ = rel_pos.shape
            rel_pos_flat = rel_pos.reshape(B_rp * N_rp * H_rp * P_rp, 3)
            rel_pos_bias = self.rel_pos_enc(rel_pos_flat)
            rel_pos_bias = rel_pos_bias.reshape(B_rp, N_rp, H_rp, P_rp)
            attn = attn + rel_pos_bias

        attn_weights = torch.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = (sampled_V * attn_weights.unsqueeze(-1)).sum(dim=3)
        out = out.reshape(B, N, C)

        out = self.proj(out)
        out = self.dropout(out)

        out = self.norm1(identity + out)
        out = self.norm2(out + self.ffn(out))

        out = out.transpose(1, 2).contiguous()
        return out


class PointNet2FPModule(nn.Module):
    def __init__(self, *, mlp: List[int], bn: bool = True, groups: int = 1):
        super().__init__()
        self.mlp = pt_util.SharedMLP(mlp, bn=bn, groups=groups)

    def forward(
        self,
        unknown: torch.Tensor,
        known: torch.Tensor,
        unknow_feats: torch.Tensor,
        known_feats: torch.Tensor,
    ) -> torch.Tensor:
        if known is not None:
            dist, idx = pointops.nearestneighbor(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feats = pointops.interpolation(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)
        else:
            new_features = interpolated_feats
        return self.mlp(new_features.unsqueeze(-1)).squeeze(-1)


class WGeM(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Sequential(nn.Conv1d(channels, channels, 1), nn.Sigmoid())
        self.p = nn.Parameter(torch.ones(1) * 3)
        self.eps = eps
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        w = self.weight(x)
        x = torch.clamp(x * w, min=self.eps).pow(self.p)
        x = self.pool(x).pow(1.0 / self.p)
        return x


class DPA_Net(nn.Module):
    def __init__(self, params_mink):
        super().__init__()

        self.use_spare_tensor = False
        self.use_top_down = True
        self.quantization_size = params_mink.quantization_step
        self.conv0_kernel_size = 5
        base_channel = 16
        self.kernel_size = 3
        self.SPconv_Fun = BasicSpconvBlock

        c = 3
        self.num_top_down_trans = 1
        groups = [4, 4, 4, 4]
        sap = [1024, 256, 64, 16]
        fs = [256, 256, 256, 256]
        mlp_base_layers = [base_channel, 2 * base_channel]
        mlp_bn = True
        mlp_gp = 1

        self.spconv0 = BasicSpconvBlock(1, 32, kernel_size=self.conv0_kernel_size, stride=1)

        group = groups[0]
        self.spconvs = nn.ModuleList()

        if self.SPconv_Fun is not None:
            self.spconv1 = self.SPconv_Fun(32, 2 * base_channel, kernel_size=self.conv0_kernel_size, stride=1)
        else:
            self.spconv1 = None
        self.spconvs.append(self.spconv1)

        self.sample_inter0 = Sample_interpolated(sap[0], 2 * base_channel, 2 * base_channel, group=group, first_sample=True)

        self.dpa0 = DeformablePointAttention(
            dim=2 * base_channel,
            num_heads=8,
            num_points=8,
            dropout=0.1,
            offset_scale=1.5,
            use_relative_pos=True,
        )

        self.mlp0 = pt_util.SharedMLP_1d(mlp_base_layers, bn=mlp_bn, groups=mlp_gp)

        if self.use_top_down:
            self.FP_modules = nn.ModuleList()
            self.FP_modules.append(PointNet2FPModule(mlp=[32 + c, 256, fs[0]]))

        self.Gem = WGeM(channels=fs[0])

    def forward(self, batch):
        feature_maps_sparse = []
        feature_maps_t = []
        xyz_ts = []

        sparse_tensor = ME.SparseTensor(batch["features"], coordinates=batch["coords"])

        if self.use_spare_tensor:
            B = sparse_tensor.C[-1, 0].item() + 1
            _, bxyz_tensor, _ = sparse_to_tensor(sparse_tensor, B, padding_value=None)
            xyz_tensor = bxyz_tensor[:, :, 1:] * self.quantization_size
        else:
            xyz_tensor = batch["batch"]

        feats_t = xyz_tensor.transpose(1, 2).contiguous()
        feature_maps_t.append(feats_t)
        xyz_ts.append(xyz_tensor)

        sparse = self.spconv0(sparse_tensor)
        feature_maps_sparse.append(sparse)

        if self.SPconv_Fun is not None:
            for spconv in self.spconvs:
                sparse = spconv(sparse)
                feature_maps_sparse.append(sparse)
        else:
            for spconv in self.spconvs:
                feature_maps_sparse.append(None)

        if self.SPconv_Fun is not None:
            xyz_t, inter_feats_t, _ = self.sample_inter0(xyz_tensor, feature_maps_sparse[1], None)
        else:
            xyz_t, inter_feats_t, _ = self.sample_inter0(xyz_tensor, sparse, None)

        feats_t = self.dpa0(inter_feats_t, xyz_t)

        feature_maps_t.append(feats_t)
        xyz_ts.append(xyz_t)

        if self.use_top_down:
            for i in range(-1, -(len(self.FP_modules) + 1), -1):
                feature_maps_t[i - 1] = self.FP_modules[i](xyz_ts[i - 1], xyz_ts[i], feature_maps_t[i - 1], feature_maps_t[i])
            feats_t = feature_maps_t[len(self.FP_modules) - self.num_top_down_trans]

        out = self.Gem(feats_t).squeeze(-1)

        return {"global": out}
