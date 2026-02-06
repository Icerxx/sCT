import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from dinov3.models.vision_transformer import vit_base


class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # x: (N, C, H, W)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Upsample(nn.Module):
    def __init__(self, in_chans, out_chans, scale_factor=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        # self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
                                #   nn.BatchNorm2d(out_chans),
                                  LayerNorm2d(out_chans),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(self.up(x))

class Downsample(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.down = nn.Sequential(
            LayerNorm2d(in_dims),
            nn.Conv2d(in_dims, out_dims, kernel_size=2, stride=2)
        )
    def forward(self, x):
        return self.down(x)

class AdaptiveSkipProjector(nn.Module):
    def __init__(self, in_dim, out_dim, scale_factor):
        super().__init__()

        # self.compressor = nn.Sequential(
        #     nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
        #     LayerNorm2d(out_dim),
        #     nn.ReLU(inplace=True)
        # )
        self.compressor = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        
        layers = []
        num_stages = int(torch.log2(torch.tensor(scale_factor)))
        for _ in range(num_stages):
            layers.append(Upsample(out_dim, out_dim, scale_factor=2))
            
        self.upsampler = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 768, H, W)
        x = self.compressor(x) # -> (B, out_dim, H, W)
        x = self.upsampler(x)  # -> (B, out_dim, H*scale, W*scale)
        return x

class ConvStem(nn.Module):
    def __init__(self, in_chans=3, out_chans=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            LayerNorm2d(out_chans),
            nn.GELU(),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            LayerNorm2d(out_chans),
            nn.GELU()
        )
    def forward(self, x):
        return self.conv(x)

# class ResEncoderBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#         # 1. 卷积路径
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels) # Encoder通常用BN，也可以换LN
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#         # 2. 下采样路径 (Shortcut)
#         # 如果 stride > 1 或者通道数改变，shortcut 需要用 1x1 卷积调整
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#         else:
#             self.shortcut = nn.Identity()
#
#     def forward(self, x):
#         residual = self.shortcut(x)
#
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#
#         x = self.conv2(x)
#         x = self.bn2(x)
#
#         x += residual
#         x = self.relu(x)
#         return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm2d(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        # x = input + x
        return x

class CNNEncoder(nn.Module):
    def __init__(self, in_chans=32, base_dim=64, layers=[2, 2, 2, 2], drop_path_rate=0.):
        super().__init__()
        # # Input: 32 (来自 InputEncoder)
        self.layers = layers
        self.drop_path_rate = drop_path_rate
        self.dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]

        # Stage 1: 32 -> 96, 1/2 Res
        self.down1 = Downsample(in_chans, base_dim)
        self.stage1 = self._make_stage(base_dim, layers[0], self.dp_rates[:layers[0]])

        # Stage 2: 96 -> 192, 1/4 Res
        self.down2 = Downsample(base_dim, base_dim * 2)
        self.stage2 = self._make_stage(base_dim * 2, layers[1], self.dp_rates[layers[0]:layers[0] + layers[1]])

        # Stage 3: 192 -> 384, 1/8 Res
        self.down3 = Downsample(base_dim * 2, base_dim * 4)
        self.stage3 = self._make_stage(base_dim * 4, layers[2], self.dp_rates[layers[0] + layers[1]:sum(layers[:3])])

        # Stage 4: 384 -> 768, 1/16 Res
        self.down4 = Downsample(base_dim * 4, base_dim * 8)
        self.stage4 = self._make_stage(base_dim * 8, layers[3], self.dp_rates[sum(layers[:3]):])

    def _make_stage(self, dim, num_blocks, dp_rates):
        return nn.Sequential(*[Block(dim, drop_path=dp_rates[i]) for i in range(num_blocks)])

    def forward(self, x):
        # 逐层下采样和特征提取
        x = self.down1(x)
        c1 = self.stage1(x)  # 1/2

        x = self.down2(c1)
        c2 = self.stage2(x)  # 1/4

        x = self.down3(c2)
        c3 = self.stage3(x)  # 1/8

        x = self.down4(c3)
        c4 = self.stage4(x)  # 1/16

        return c1, c2, c3, c4

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, num_blocks=1):
        super().__init__()

        # 1. 上采样 (ConvTranspose 2x)
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        # 2. 融合层 (1x1 Conv)
        # 将 Concat 后的通道数降维到 out_channels
        concat_dim = (in_channels // 2) + skip_channels
        self.fusion = nn.Conv2d(concat_dim, out_channels, kernel_size=1)

        self.blocks = nn.Sequential(*[Block(out_channels) for _ in range(num_blocks)])

    def forward(self, x, skip=None):
        # Upsample
        x = self.up(x)

        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        # Fusion -> Deep Feature Extraction
        x = self.fusion(x)
        x = self.blocks(x)
        return x

class ViTFeatureAggregator(nn.Module):
    def __init__(self, in_dims=[768, 768, 768, 768], out_dim=768):
        super().__init__()
        total_dim = sum(in_dims)
        
        # 1. 压缩通道
        self.conv = nn.Conv2d(total_dim, out_dim, kernel_size=1)
        
        # 2. 通道注意力 (SE-Block Style)
        # 动态决定哪些层的特征更重要
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_dim, out_dim // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim // 16, out_dim, 1),
            nn.Sigmoid()
        )
        
        self.norm = LayerNorm2d(out_dim)

    def forward(self, features_list):
        # features_list: [z3, z6, z9, z12]
        
        # 拼接
        x = torch.cat(features_list, dim=1) # (B, 4*768, H, W)
        
        # 融合
        x = self.conv(x)
        
        # 加权
        w = self.se(x)
        x = x * w
        
        x = self.norm(x)
        return x

class ViTTokenMoE(nn.Module):
    def __init__(self, in_dim=768, num_experts=4, hidden_dim=256):
        """
        Args:
            in_dim: 单个 ViT 层的通道数 (768)
            num_experts: ViT 层数 (4: z3, z6, z9, z12)
            hidden_dim: Router 的隐藏层维度 (轻量化)
        """
        super().__init__()
        self.num_experts = num_experts
        
        # --- Router (Gating Network) ---
        # 输入: 所有专家的特征拼接 (B, 4*768, H, W)
        # 任务: 为每个像素点预测 4 个权重
        self.pre_reduce = nn.Conv2d(in_dim, 128, 1)
        self.router = nn.Sequential(
            # 1. 压缩特征，提取路由信息
            nn.Conv2d(128 * num_experts, hidden_dim, kernel_size=1, bias=False),
            LayerNorm2d(hidden_dim),
            nn.GELU(),
            
            # 2. 预测 Logits
            nn.Conv2d(hidden_dim, num_experts, kernel_size=1, bias=True)
        )
        
        # 最后的融合层 (可选，用于平滑聚合后的特征)
        self.project = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim), # Depthwise
            LayerNorm2d(in_dim),
            nn.GELU(),
            nn.Conv2d(in_dim, in_dim, kernel_size=1) # Pointwise
        )

    def forward(self, features_list):
        """
        Args:
            features_list: List of [z3, z6, z9, z12], each (B, C, H, W)
        Returns:
            z_agg: (B, C, H, W)
        """
        reduced_feats = []
        for feat in features_list:
            reduced_feats.append(self.pre_reduce(feat)) # 共享降维

        router_input = torch.cat(reduced_feats, dim=1) # (B, 256, H, W)
        
        # 计算路由权重 (Gating Weights)
        # logits: (B, 4, H, W)
        gate_logits = self.router(router_input)
        
        # Softmax 归一化: 在 Expert 维度 (dim=1) 做
        # weights: (B, 4, H, W)
        gate_weights = F.softmax(gate_logits, dim=1)
        
        # 加权聚合 (Weighted Sum)
        # (B, 4, 1, H, W) * (B, 4, C, H, W) -> sum dim 1 -> (B, C, H, W)
        # 这是一个 Token-wise (Pixel-wise) 的加权
        expert_features = torch.stack(features_list, dim=1) 
        z_agg = (gate_weights.unsqueeze(2) * expert_features).sum(dim=1)
        
        # 特征平滑/投影
        z_agg = self.project(z_agg)
        
        # (可选) 返回权重用于可视化，写论文时画热力图很有用！
        # return z_agg, gate_weights 
        return z_agg
    
class CNNAttendedViT(nn.Module):
    def __init__(self, dim, num_heads=8, feat_size=16):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.norm_q = LayerNorm2d(dim)
        self.norm_k = LayerNorm2d(dim)

        # Q, K, V Projections
        self.to_q = nn.Conv2d(dim, dim, 1, bias=True) # CNN Input
        self.to_k = nn.Conv2d(dim, dim, 1, bias=True) # ViT Input
        self.to_v = nn.Conv2d(dim, dim, 1, bias=True) # ViT Input

        self.pos_embed = nn.Parameter(torch.randn(1, dim, feat_size, feat_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # self.proj = nn.Conv2d(dim, dim, 1)
        # self.norm = LayerNorm2d(dim)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=False), # 3x3
            LayerNorm2d(dim),
            nn.GELU() # 增加非线性
        )

    def forward(self, cnn_feat, vit_feat):
        """
        cnn_feat: (B, C, H, W) -> Query
        vit_feat: (B, C, H, W) -> Key, Value
        """
        B, C, H, W = cnn_feat.shape
        residual = cnn_feat
        
        if self.pos_embed.shape[2:] != (H, W):
            pos_resized = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear')
        else:
            pos_resized = self.pos_embed

        q_in = self.norm_q(cnn_feat) + pos_resized
        k_in = self.norm_k(vit_feat) + pos_resized
        # 生成 Q, K, V
        # Reshape to (B, Heads, N, C_head)
        # N = H*W
        q = self.to_q(q_in).flatten(2).transpose(1, 2)
        k = self.to_k(k_in).flatten(2).transpose(1, 2)
        v = self.to_v(vit_feat).flatten(2).transpose(1, 2)
        
        q = q.reshape(B, H*W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, H*W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, H*W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # Attention Calculation
        # (B, Heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Weighted Sum
        out = (attn @ v).transpose(1, 2).reshape(B, H*W, C).transpose(1, 2).view(B, C, H, W)
        
        # out = self.proj(out)
        # out = self.norm(out)
        # or residual + lamba * out
        # return residual + out

        # return self.fusion_conv(torch.cat([cnn_feat, out], dim=1)) + residual
        return self.fusion_conv(torch.cat([cnn_feat, out], dim=1))

# =================================================================
# 2. MedDINO sCT Generator
# =================================================================
class MedDINOGenerator(nn.Module):
    def __init__(self, img_size=256, patch_size=16, out_chans=1, embed_dim=768, pretrained_path=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.vit_encoder = vit_base(img_size=img_size, patch_size=patch_size, 
                                drop_path_rate=0.2, layerscale_init=1.0e-05, 
                                n_storage_tokens=4, qkv_bias=False, mask_k_bias=True)
        # load pretrained weights
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading MedDINO weights from {pretrained_path}...")
            chkpt = torch.load(pretrained_path, map_location='cpu', weights_only=True)

            state_dict = chkpt['teacher']
            state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() 
                          if 'ibot' not in k and 'dino_head' not in k}  # 去掉 'backbone.' 前缀和 'ibot', 'dino_head' 相关参数
            
            missing, unexpected = self.vit_encoder.load_state_dict(state_dict, strict=True)
            print(f"Loaded MedDINO weights with {len(missing)} missing and {len(unexpected)} unexpected keys.")
        else:
            print("Warning: No pretrained weights loaded!")

        # Skip 3 (Layer 9): 16 -> 32 (2x)
        # self.skip_proj3 = AdaptiveSkipProjector(embed_dim, 256, scale_factor=2)
        # Skip 2 (Layer 6): 16 -> 64 (4x)
        # self.skip_proj2 = AdaptiveSkipProjector(embed_dim, 128, scale_factor=4)
        # Skip 1 (Layer 3): 16 -> 128 (8x)
        # self.skip_proj1 = AdaptiveSkipProjector(embed_dim, 64, scale_factor=8)

        self.input_skip = ConvStem(in_chans=3, out_chans=32)
        self.cnn_encoder = CNNEncoder(in_chans=32, base_dim=96, layers=[2, 2, 2, 2])

        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(embed_dim + 768, 768, kernel_size=3, padding=1),
            LayerNorm2d(768),
            nn.GELU()
        )
        # self.vit_aggregator = ViTFeatureAggregator(in_dims=[768]*4, out_dim=768)
        self.vit_aggregator = ViTTokenMoE(in_dim=768, num_experts=4, hidden_dim=256)
        # self.cross_fusion = CNNAttendedViT(dim=768)

        # Layer 4: 1/16 -> 1/8
        self.decoder4 = DecoderBlock(in_channels=768, skip_channels=384, out_channels=384, num_blocks=2)
        # Layer 3: 1/8 -> 1/4
        self.decoder3 = DecoderBlock(in_channels=384, skip_channels=192, out_channels=192, num_blocks=2)
        # Layer 2: 1/4 -> 1/2
        self.decoder2 = DecoderBlock(in_channels=192, skip_channels=96, out_channels=96, num_blocks=2)
        # Layer 1: 1/2 -> 1/1
        self.decoder1 = DecoderBlock(in_channels=96, skip_channels=32, out_channels=64, num_blocks=2)
        
        # 输出层
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            LayerNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, out_chans, kernel_size=1)
        )

    def forward(self, x):
        # x: (B, 1, H, W)
        # 如果输入是单通道 (医学图像)，但模型期望 3 通道 (DINO Pretrained)，则复制通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x_in = x
        x_feat = self.input_skip(x_in)

        # --- Encoder ---
        # 提取 ViT 中间层的索引 (用于 Skip Connection)
        # ViT-Base 有 12 层 transformer blocks。
        # 我们通常取第 3, 6, 9, 12 层的输出
        # ViT 的输出特征图尺寸: (B, Embed_Dim, H/16, W/16)
        z3, z6, z9, z12 = self.vit_encoder.get_intermediate_layers(x, n=[2, 5, 8, 11], reshape=True)
        # print(f"ViT features: z3:{z3.shape}, z6:{z6.shape}, z9:{z9.shape}, z12:{z12.shape}")
        c1, c2, c3, c4 = self.cnn_encoder(x_feat) # 128 -> 64 -> 32 -> 16
        # print(f"CNN features: c1:{c1.shape}, c2:{c2.shape}, c3:{c3.shape}, c4:{c4.shape}")
        
        # --- Project Skips (将 ViT 特征放大到对应尺寸) ---
        # skip3 = self.skip_proj3(z9) # 16 -> 32
        # skip2 = self.skip_proj2(z6) # 16 -> 64
        # skip1 = self.skip_proj1(z3) # 16 -> 128

        # --- Decoder ---
        z_agg = self.vit_aggregator([z3, z6, z9, z12])  # (B, 768, 16, 16)

        # bottleneck = self.bottleneck_conv(z12) # -> (B, 512, 16, 16)
        # bottleneck = self.bottleneck_conv(c4) # -> (B, 512, 16, 16)
        # bottleneck_fused = torch.cat([z12, c4], dim=1) # (B, 768+768, 16, 16)

        bottleneck = self.bottleneck_conv(torch.cat([z_agg, c4], dim=1)) # -> (B, 768, 16, 16)
        # bottleneck = self.cross_fusion(cnn_feat=c4, vit_feat=z_agg)  # (B, 768, 16, 16)
        
        x = self.decoder4(bottleneck, skip=c3) # Upsample to 32
        # print(f"decoder4: {x.shape}")
        x = self.decoder3(x, skip=c2) # Upsample to 64
        # print(f"decoder3: {x.shape}")
        x = self.decoder2(x, skip=c1) # Upsample to 128
        # print(f"decoder2: {x.shape}")
        x = self.decoder1(x, x_feat) # Upsample to 256
        # print(f"decoder1: {x.shape}")
        
        out = self.out_conv(x)
        # return out
        return out + x_in.mean(dim=1, keepdim=True)  # Residual connection


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MedDINOGenerator(img_size=256).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Params: {total_params/1e6:.2f}M | Trainable: \
          {(total_params-sum(p.numel() for p in model.vit_encoder.parameters()))/1e6:.2f}M")

    dummy_input = torch.randn(1, 1, 256, 256).to(device)
    output = model(dummy_input)
    print(f"Input: {dummy_input.shape}")
    print(f"Output: {output.shape}")
