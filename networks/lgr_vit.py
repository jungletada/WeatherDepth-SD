import math
import torch
import logging
import numpy as np

import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, trunc_normal_
from einops import rearrange
from functools import partial
from torch import nn, einsum
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.runner import load_checkpoint, load_state_dict
from mmcv.cnn import build_norm_layer

__all__ = [
    "mbmpvit_tiny",
    "mbmpvit_xsmall",
    "mbmpvit_small",
    "mbmpvit_base",
]


def _cfg_mpvit(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


class Mlp(nn.Module):
    """Feed-forward network (FFN, a.k.a. MLP) class."""

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Conv2d_BN(nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            pad=0,
            dilation=1,
            groups=1,
            bn_weight_init=1,
            act_layer=None,
            norm_cfg=dict(type="BN"),
    ):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_ch, out_ch, kernel_size, stride, pad, dilation, groups, bias=False
        )
        self.bn = build_norm_layer(norm_cfg, out_ch)[1]

        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_layer(x)

        return x


class DWConv2d_BN(nn.Module):
    """
    Depthwise Separable Conv
    """

    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.Hardswish,
            bn_weight_init=1,
            norm_cfg=dict(type="BN"),
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride,
            (kernel_size - 1) // 2,
            groups=out_ch,
            bias=False,
        )
        self.pwconv = nn.Conv2d(out_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = build_norm_layer(norm_cfg, out_ch)[1]
        self.act = act_layer() if act_layer is not None else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(bn_weight_init)
                m.bias.data.zero_()

    def forward(self, x):

        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class DWCPatchEmbed(nn.Module):
    """
    Depthwise Convolutional Patch Embedding layer
    Image to Patch Embedding
    """

    def __init__(
            self,
            in_chans=3,
            embed_dim=768,
            patch_size=16,
            stride=1,
            pad=0,
            act_layer=nn.Hardswish,
            norm_cfg=dict(type="BN"),
    ):
        super().__init__()

        self.patch_conv = DWConv2d_BN(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            act_layer=nn.Hardswish,
            norm_cfg=norm_cfg,
        )

    def forward(self, x):
        x = self.patch_conv(x)

        return x


class Patch_Embed_stage(nn.Module):
    def __init__(self, embed_dim, num_path=4, isPool=False, norm_cfg=dict(type="BN")):
        super(Patch_Embed_stage, self).__init__()

        self.patch_embeds = nn.ModuleList(
            [
                DWCPatchEmbed(
                    in_chans=embed_dim,
                    embed_dim=embed_dim,
                    patch_size=3,
                    stride=2 if isPool and idx == 0 else 1,
                    pad=1,
                    norm_cfg=norm_cfg,
                )
                for idx in range(num_path)
            ]
        )

    def forward(self, x):
        att_inputs = []
        for pe in self.patch_embeds:
            x = pe(x)
            att_inputs.append(x)

        return att_inputs


class ConvPosEnc(nn.Module):
    """Convolutional Position Encoding.
    Note: This module is similar to the conditional position encoding in CPVT.
    """

    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()

        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size

        feat = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2).contiguous()

        return x


class ConvRelPosEnc(nn.Module):
    """Convolutional relative position encoding."""

    def __init__(self, Ch, h, window):
        """Initialization.

        Ch: Channels per head.
        h: Number of heads.
        window: Window size(s) in convolutional relative positional encoding.
                It can have two forms:
                1. An integer of window size, which assigns all attention heads
                   with the same window size in ConvRelPosEnc.
                2. A dict mapping window size to #attention head splits
                   (e.g. {window size 1: #attention head split 1, window size
                                      2: #attention head split 2})
                   It will apply different window size to
                   the attention head splits.
        """
        super().__init__()

        if isinstance(window, int):
            # Set the same window size for all attention heads.
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) *
                            (dilation - 1)) // 2
            cur_conv = nn.Conv2d(
                cur_head_split * Ch,
                cur_head_split * Ch,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split * Ch,
            )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch for x in self.head_splits]

    def forward(self, q, v, size):
        """foward function"""
        B, h, N, Ch = q.shape
        H, W = size

        q_img = q
        v_img = v

        # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        v_img = rearrange(v_img, "B h (H W) Ch -> B (h Ch) H W", H=H, W=W)
        # Split according to channels.
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        conv_v_img_list = [
            conv(x) for conv, x in zip(self.conv_list, v_img_list)
        ]
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].
        conv_v_img = rearrange(conv_v_img, "B (h Ch) H W -> B h (H W) Ch", h=h)

        EV_hat_img = q_img * conv_v_img
        EV_hat = EV_hat_img
        return EV_hat


class FactorAtt_ConvRelPosEnc(nn.Module):
    """Factorized attention with convolutional relative position encoding class."""

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            shared_crpe=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)  # Note: attn_drop is actually not used.
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, x, size):
        B, N, C = x.shape

        # Generate Q, K, V.
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )  # Shape: [3, B, h, N, Ch].
        q, k, v = qkv[0], qkv[1], qkv[2]  # Shape: [B, h, N, Ch].

        # Factorized attention.
        k_softmax = k.softmax(dim=2)  # Softmax on dim N.
        k_softmax_T_dot_v = einsum(
            "b h n k, b h n v -> b h k v", k_softmax, v
        )  # Shape: [B, h, Ch, Ch].
        factor_att = einsum(
            "b h n k, b h k v -> b h n v", q, k_softmax_T_dot_v
        )  # Shape: [B, h, N, Ch].

        # Convolutional relative position encoding.
        crpe = self.crpe(q, v, size=size)  # Shape: [B, h, N, Ch].

        # Merge and reshape.
        x = self.scale * factor_att + crpe
        x = (
            x.transpose(1, 2).reshape(B, N, C).contiguous()
        )  # Shape: [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C].

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MHCABlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=3,
            drop_path=0.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            shared_cpe=None,
            shared_crpe=None,
    ):
        super().__init__()

        self.cpe = shared_cpe
        self.crpe = shared_crpe
        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            shared_crpe=shared_crpe,
        )
        self.mlp = Mlp(in_features=dim, hidden_features=dim * mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x, size):
        # x.shape = [B, N, C]

        if self.cpe is not None:
            x = self.cpe(x, size)
        cur = self.norm1(x)
        x = x + self.drop_path(self.factoratt_crpe(cur, size))

        cur = self.norm2(x)
        x = x + self.drop_path(self.mlp(cur))
        return x


class MHCAEncoder(nn.Module):
    def __init__(
            self,
            dim,
            num_layers=1,
            num_heads=8,
            mlp_ratio=3,
            drop_path_list=[],
            qk_scale=None,
            crpe_window={3: 2, 5: 3, 7: 3},
    ):
        super().__init__()

        self.num_layers = num_layers
        self.cpe = ConvPosEnc(dim, k=3)
        self.crpe = ConvRelPosEnc(Ch=dim // num_heads, h=num_heads, window=crpe_window)
        self.MHCA_layers = nn.ModuleList(
            [
                MHCABlock(
                    dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path_list[idx],
                    qk_scale=qk_scale,
                    shared_cpe=self.cpe,
                    shared_crpe=self.crpe,
                )
                for idx in range(self.num_layers)
            ]
        )

    def forward(self, x, size):
        H, W = size
        B = x.shape[0]
        # x' shape : [B, N, C]
        for layer in self.MHCA_layers:
            x = layer(x, (H, W))

        # return x's shape : [B, N, C] -> [B, C, H, W]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class ResBlock(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.Hardswish,
            norm_cfg=dict(type="BN"),
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = Conv2d_BN(
            in_features, hidden_features, act_layer=act_layer, norm_cfg=norm_cfg
        )
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            bias=False,
            groups=hidden_features,
        )
        # self.norm = norm_layer(hidden_features)
        self.norm = build_norm_layer(norm_cfg, hidden_features)[1]
        self.act = act_layer()
        self.conv2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, x):
        identity = x
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.norm(feat)
        feat = self.act(feat)
        feat = self.conv2(feat)

        return identity + feat


class MHCA_stage(nn.Module):
    def __init__(
            self,
            embed_dim,
            out_embed_dim,
            num_layers=1,
            num_heads=8,
            mlp_ratio=3,
            num_path=4,
            norm_cfg=dict(type="BN"),
            drop_path_list=[],
    ):
        super().__init__()

        self.mhca_blks = nn.ModuleList(
            [
                MHCAEncoder(
                    embed_dim,
                    num_layers,
                    num_heads,
                    mlp_ratio,
                    drop_path_list=drop_path_list,
                )
                for _ in range(num_path)
            ]
        )

        self.InvRes = ResBlock(
            in_features=embed_dim, out_features=embed_dim, norm_cfg=norm_cfg
        )
        self.aggregate = Conv2d_BN(
            embed_dim * (num_path + 1),
            out_embed_dim,
            act_layer=nn.Hardswish,
            norm_cfg=norm_cfg,
        )

    def forward(self, inputs):
        att_outputs = [self.InvRes(inputs[0])]
        for x, encoder in zip(inputs, self.mhca_blks):
            # [B, C, H, W] -> [B, N, C]
            _, _, H, W = x.shape
            x = x.flatten(2).transpose(1, 2).contiguous()
            att_outputs.append(encoder(x, size=(H, W)))

        out_concat = torch.cat(att_outputs, dim=1)
        out = self.aggregate(out_concat)

        return out, att_outputs


class GlobalAggregateAttention(nn.Module):
    def __init__(self, in_channels_list, embed_dim=128, num_heads=1):
        """
        Global Feature Aggregation
        Args:
            in_channels_list (list[int]): Number of channels for each input feature scale.
            embed_dim (int): Dimension to project features into for cross-scale interactions.
            num_heads (int): Number of attention heads (if using multihead attention). 
                              Use 1 for single-head (simple) attention.
        """
        super(GlobalAggregateAttention, self).__init__()
        self.num_scales = len(in_channels_list)
        self.embed_dim = embed_dim
        # Linear layers to embed each scale's descriptor to a common dimension
        self.embed_layers = nn.ModuleList([
            nn.Linear(ch, embed_dim) for ch in in_channels_list
        ])
        # Linear layers to project fused embeddings back to original channel dimensions
        self.output_layers = nn.ModuleList([
            nn.Linear(embed_dim, ch) for ch in in_channels_list
        ])
        # If using multi-head attention for more capacity
        if num_heads > 1:
            self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        else:
            self.attention = None

    def forward(self, features):
        """
        Fuse a list of multi-scale feature maps.
        Args:
            features (List[Tensor]): list of tensors [x0, x1, ..., xN], each of shape [B, C_i, H_i, W_i].
        Returns:
            List[Tensor]: list of fused tensors [y0, y1, ..., yN] with same shapes as inputs.
        """
        B = features[0].shape[0]
        # 1. Global average pool each feature map to get a [B, C_i] descriptor
        descriptors = [x.mean(dim=(2, 3)) for x in features]  # list of tensors [B, C_i]
        # 2. Embed each descriptor to [B, embed_dim]
        embedded = [embed_layer(desc) for embed_layer, desc in zip(self.embed_layers, descriptors)]
        embedded = torch.stack(embedded, dim=1)  # shape [B, N, embed_dim]
        # 3. Cross-scale attention to fuse embeddings
        if self.attention is not None:
            # Multi-head attention on scale dimension
            attn_output, _ = self.attention(embedded, embedded, embedded)  # [B, N, embed_dim]
        else:
            # Single-head scaled dot-product attention
            Q = embedded  # [B, N, d]
            K = embedded
            V = embedded
            # Compute attention weights across scales (N x N matrix for each batch)
            energy = torch.bmm(Q, K.transpose(1, 2)) / (self.embed_dim ** 0.5)  # [B, N, N]
            attn_weights = F.softmax(energy, dim=-1)  # normalize over source scale dim
            attn_output = torch.bmm(attn_weights, V)  # [B, N, d]

        # 4. Split fused embeddings and project back to original channels
        fused_features = []
        for i, orig_feature in enumerate(features):
            # Get fused embedding for scale i: shape [B, embed_dim]
            fused_emb_i = attn_output[:, i, :] 
            # Project to original channel count [B, C_i] and apply sigmoid gating
            scale_weights = self.output_layers[i](fused_emb_i) # torch.sigmoid(self.output_layers[i](fused_emb_i))  # [B, C_i]
            scale_weights = scale_weights.unsqueeze(-1).unsqueeze(-1)          # [B, C_i, 1, 1]
            # 5. Reweight original feature map with these channel weights
            fused_feature = orig_feature * scale_weights + orig_feature # broadcast multiplication
            fused_features.append(fused_feature)
        return fused_features
    

class LightweightScaleGCN(nn.Module):
    """
        Local Feature Refinement 
    """
    def __init__(self):
        super(LightweightScaleGCN, self).__init__()
        # 定义每个邻接尺度之间的1x1卷积，用于通道对齐和信息变换
        # 下采样邻居卷积 (conv_down{i} 表示将第{i-1}层高分辨率特征映射到第{i}层通道数)
        self.conv_down1 = nn.Conv2d(64, 128, kernel_size=1)   # x0 -> x1 通道变换
        self.conv_down2 = nn.Conv2d(128, 216, kernel_size=1)  # x1 -> x2
        self.conv_down3 = nn.Conv2d(216, 288, kernel_size=1)  # x2 -> x3
        self.conv_down4 = nn.Conv2d(288, 288, kernel_size=1)  # x3 -> x4
        # 上采样邻居卷积 (conv_up{i} 表示将第{i+1}层低分辨率特征映射到第{i}层通道数)
        self.conv_up0  = nn.Conv2d(128, 64, kernel_size=1)    # x1 -> x0
        self.conv_up1  = nn.Conv2d(216, 128, kernel_size=1)   # x2 -> x1
        self.conv_up2  = nn.Conv2d(288, 216, kernel_size=1)   # x3 -> x2
        self.conv_up3  = nn.Conv2d(288, 288, kernel_size=1)   # x4 -> x3
        # 自身特征通道变换卷积 (conv_self{i} 用于赋予每层特征一个可学习的权重变换)
        self.conv_self0 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv_self1 = nn.Conv2d(128, 128, kernel_size=1)
        self.conv_self2 = nn.Conv2d(216, 216, kernel_size=1)
        self.conv_self3 = nn.Conv2d(288, 288, kernel_size=1)
        self.conv_self4 = nn.Conv2d(288, 288, kernel_size=1)

    def forward(self, x_list):
        """
        参数:
            x_list: 长度为5的特征图列表 [x0, x1, x2, x3, x4]
        返回:
            增强后的5个特征图列表 [y0, y1, y2, y3, y4] (形状分别与输入对应)
        """
        # 解包输入特征
        x0, x1, x2, x3, x4 = x_list

        # 逐层融合相邻尺度特征
        # 对于每层yi_new，我们使用该层自身以及相邻上下层的信息
        # 1. 最底层 x0 仅与 x1 融合（x0 邻居只有上方的 x1）
        #    将 x1 上采样至 x0 大小，通过1x1卷积调整通道后与 x0 融合
        x1_up_to_x0 = F.interpolate(x1, size=x0.shape[2:], mode='bilinear', align_corners=False)
        y0 = self.conv_self0(x0) + self.conv_up0(x1_up_to_x0)

        # 2. 次底层 x1 与 x0、x2 融合
        #    下采样 x0 至 x1 大小，上采样 x2 至 x1 大小，分别1x1卷积后与 x1 融合
        x0_down_to_x1 = F.interpolate(x0, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x2_up_to_x1   = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        y1 = self.conv_self1(x1) + self.conv_down1(x0_down_to_x1) + self.conv_up1(x2_up_to_x1)

        # 3. 中间层 x2 与 x1、x3 融合
        #    下采样 x1 至 x2 大小，上采样 x3 至 x2 大小，1x1卷积后融合
        x1_down_to_x2 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x3_up_to_x2   = F.interpolate(x3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        y2 = self.conv_self2(x2) + self.conv_down2(x1_down_to_x2) + self.conv_up2(x3_up_to_x2)

        # 4. 次高层 x3 与 x2、x4 融合
        #    下采样 x2 至 x3 大小，上采样 x4 至 x3 大小，1x1卷积后融合
        x2_down_to_x3 = F.interpolate(x2, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x4_up_to_x3   = F.interpolate(x4, size=x3.shape[2:], mode='bilinear', align_corners=False)
        y3 = self.conv_self3(x3) + self.conv_down3(x2_down_to_x3) + self.conv_up3(x4_up_to_x3)

        # 5. 最顶层 x4 仅与 x3 融合（x4 邻居只有下方的 x3）
        #    将 x3 下采样至 x4 大小，1x1卷积后与 x4 融合
        x3_down_to_x4 = F.interpolate(x3, size=x4.shape[2:], mode='bilinear', align_corners=False)
        y4 = self.conv_self4(x4) + self.conv_down4(x3_down_to_x4)

        return [y0, y1, y2, y3, y4]


def dpr_generator(drop_path_rate, num_layers, num_stages):
    """
    Generate drop path rate list following linear decay rule
    """
    dpr_list = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))]
    dpr = []
    cur = 0
    for i in range(num_stages):
        dpr_per_stage = dpr_list[cur: cur + num_layers[i]]
        dpr.append(dpr_per_stage)
        cur += num_layers[i]

    return dpr


class LocalGlobalViT(nn.Module):
    """Multi-Path ViT class."""
    def __init__(
            self,
            num_classes=80,
            in_chans=3,
            num_stages=4,
            num_layers=[1, 1, 1, 1],
            mlp_ratios=[8, 8, 4, 4],
            num_path=[4, 4, 4, 4],
            embed_dims=[64, 128, 256, 512],
            num_heads=[8, 8, 8, 8],
            drop_path_rate=0.2,
            norm_cfg=dict(type="BN"),
            norm_eval=False,
            pretrained=None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_stages = num_stages
        self.conv_norm_cfg = norm_cfg
        self.norm_eval = norm_eval

        dpr = dpr_generator(drop_path_rate, num_layers, num_stages)

        self.stem = nn.Sequential(
            Conv2d_BN(
                in_chans,
                embed_dims[0] // 2,
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
                norm_cfg=self.conv_norm_cfg,
            ),
            Conv2d_BN(
                embed_dims[0] // 2,
                embed_dims[0],
                kernel_size=3,
                stride=1,
                pad=1,
                act_layer=nn.Hardswish,
                norm_cfg=self.conv_norm_cfg,
            ),
        )

        # Patch embeddings.
        self.patch_embed_stages = nn.ModuleList(
            [
                Patch_Embed_stage(
                    embed_dims[idx],
                    num_path=num_path[idx],
                    isPool=True,
                    norm_cfg=self.conv_norm_cfg,
                )
                for idx in range(self.num_stages)
            ]
        )

        # Multi-Head Convolutional Self-Attention (MHCA)
        self.mhca_stages = nn.ModuleList(
            [
                MHCA_stage(
                    embed_dims[idx],
                    embed_dims[idx + 1]
                    if not (idx + 1) == self.num_stages
                    else embed_dims[idx],
                    num_layers[idx],
                    num_heads[idx],
                    mlp_ratios[idx],
                    num_path[idx],
                    norm_cfg=self.conv_norm_cfg,
                    drop_path_list=dpr[idx],
                )
                for idx in range(self.num_stages)
            ]
        )

        self.gcn_local = LightweightScaleGCN()
        
        self.attn_global = GlobalAggregateAttention(
            in_channels_list=embed_dims+[embed_dims[3]],
            embed_dim=128,
            num_heads=4,
            )
    
    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")
    
    def forward_features(self, x):
        # x's shape : [B, C, H, W]
        outs = []
        x = self.stem(x)  # Shape : [B, C, H/4, W/4]
        outs.append(x)
        for idx in range(self.num_stages):
            att_inputs = self.patch_embed_stages[idx](x)
            x, ff = self.mhca_stages[idx](att_inputs)
            outs.append(x)
        
        outs = self.gcn_local(outs)
        outs = self.attn_global(outs)
        return outs

    def forward(self, x):
        x = self.forward_features(x)

        return x

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(LocalGlobalViT, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


def mbmpvit_tiny(**kwargs):
    """mpvit_tiny :

    - #paths : [2, 3, 3, 3]
    - #layers : [1, 2, 4, 1]
    - #channels : [64, 96, 176, 216]
    - MLP_ratio : 2
    Number of params: 5843736
    FLOPs : 1654163812
    Activations : 16641952
    """

    model = LocalGlobalViT(
        num_stages=4,
        num_path=[2, 3, 3, 3],
        num_layers=[1, 2, 4, 1],
        embed_dims=[64, 96, 176, 216],
        mlp_ratios=[2, 2, 2, 2],
        num_heads=[8, 8, 8, 8],
        **kwargs,
    )
    model.default_cfg = _cfg_mpvit()
    return model


def mbmpvit_xsmall(**kwargs):
    """mpvit_xsmall :

    - #paths : [2, 3, 3, 3]
    - #layers : [1, 2, 4, 1]
    - #channels : [64, 128, 192, 256]
    - MLP_ratio : 4
    Number of params : 10573448
    FLOPs : 2971396560
    Activations : 21983464
    """
    model = LocalGlobalViT(
        num_stages=4,
        num_path=[2, 3, 3, 3],
        num_layers=[1, 2, 4, 1],
        embed_dims=[64, 128, 192, 256],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[8, 8, 8, 8],
        **kwargs,
    )
    checkpoint = torch.load('./ckpt/mpvit_xsmall.pth', map_location=lambda storage, loc: storage)['model']
    logger = logging.getLogger()
    load_state_dict(model, checkpoint, strict=False, logger=logger)
    del checkpoint
    del logger
    model.default_cfg = _cfg_mpvit()
    return model


def mbmpvit_small(**kwargs):
    """ mpvit_small :
    - # paths : [2, 3, 3, 3]
    - # layers : [1, 3, 6, 3]
    - # channels : [64, 128, 216, 288]
    - MLP_ratio : 4
    Number of params : 22892400
    FLOPs : 4799650824
    Activations : 30601880
    """

    model = LocalGlobalViT(
        num_stages=4,
        num_path=[2, 3, 3, 3],
        num_layers=[1, 3, 6, 3],
        embed_dims=[64, 128, 216, 288],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[8, 8, 8, 8],
        **kwargs,
    )
    checkpoint = torch.load('ckpt/mpvit_small.pth', map_location=lambda storage, loc: storage)['model']
    logger = logging.getLogger()
    load_state_dict(model, checkpoint, strict=False, logger=logger)
    del checkpoint
    del logger
    model.default_cfg = _cfg_mpvit()
    return model


def mbmpvit_base(**kwargs):
    """mpvit_base :

    - #paths : [2, 3, 3, 3]
    - #layers : [1, 3, 8, 3]
    - #channels : [128, 224, 368, 480]
    - MLP_ratio : 4
    Number of params: 74845976
    FLOPs : 16445326240
    Activations : 60204392
    """

    model = LocalGlobalViT(
        num_stages=4,
        num_path=[2, 3, 3, 3],
        num_layers=[1, 3, 8, 3],
        embed_dims=[128, 224, 368, 480],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[8, 8, 8, 8],
        **kwargs,
    )
    model.default_cfg = _cfg_mpvit()
    return model


if __name__ == '__main__':
    model = mbmpvit_small()
    model.num_ch_enc = [64, 128, 216, 288, 288]
    x = torch.randn(2, 3, 192, 640)
    output_features = model(x)
    # 打印每个输出特征的张量形状，验证与输入形状一致
    for i, out in enumerate(output_features):
        print(f"Output feature {i} shape: {tuple(out.shape)}")

    """
    ([b, 64, 96, 320])
    ([b, 128, 48, 160])
    ([b, 216, 24, 80])
    ([b, 288, 12, 40])
    ([b, 288, 6, 20])
    """