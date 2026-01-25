import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class InputEncoder(nn.Module):
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
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

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

        # x = input + self.drop_path(x)
        x = input + x
        return x

class CNNEncoder(nn.Module):
    def __init__(self, in_chans=32, base_dim=64, layers=[2, 2, 2, 2]):
        super().__init__()
        # # Input: 32 (来自 InputEncoder)

        # Stage 1: 32 -> 96, 1/2 Res
        self.down1 = Downsample(in_chans, base_dim)
        self.stage1 = self._make_stage(base_dim, layers[0])

        # Stage 2: 96 -> 192, 1/4 Res
        self.down2 = Downsample(base_dim, base_dim * 2)
        self.stage2 = self._make_stage(base_dim * 2, layers[1])

        # Stage 3: 192 -> 384, 1/8 Res
        self.down3 = Downsample(base_dim * 2, base_dim * 4)
        self.stage3 = self._make_stage(base_dim * 4, layers[2])

        # Stage 4: 384 -> 768, 1/16 Res
        self.down4 = Downsample(base_dim * 4, base_dim * 8)
        self.stage4 = self._make_stage(base_dim * 8, layers[3])

    def _make_stage(self, dim, num_blocks):
        return nn.Sequential(*[Block(dim) for _ in range(num_blocks)])

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

        self.input_skip = InputEncoder(in_chans=3, out_chans=32)
        self.cnn_encoder = CNNEncoder(in_chans=32, base_dim=96, layers=[2, 2, 2, 2])

        # self.bottleneck_conv = nn.Conv2d(embed_dim, 512, kernel_size=3, padding=1)
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(embed_dim, 768, kernel_size=3, padding=1),
            LayerNorm2d(768),
            nn.GELU()
        )
        
        # Layer 4: 1/16 -> 1/8
        self.decoder4 = DecoderBlock(in_channels=768, skip_channels=0, out_channels=384, num_blocks=2)
        # Layer 3: 1/8 -> 1/4
        self.decoder3 = DecoderBlock(in_channels=384, skip_channels=0, out_channels=192, num_blocks=2)
        # Layer 2: 1/4 -> 1/2
        # self.decoder2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.decoder2 = DecoderBlock(in_channels=192, skip_channels=0, out_channels=96, num_blocks=2)
        # Layer 1: 1/2 -> 1/1
        # self.decoder1 = DecoderBlock(in_channels=64, skip_channels=32, out_channels=32)
        self.decoder1 = DecoderBlock(in_channels=96, skip_channels=32, out_channels=64, num_blocks=2)
        
        # 输出层
        # self.out_conv = nn.Conv2d(32, out_chans, kernel_size=1)
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
        # z3, z6, z9, z12 = self.vit_encoder.get_intermediate_layers(x, n=[2, 5, 8, 11], reshape=True)
        _, _, _, cnn_feat = self.cnn_encoder(x_feat) # (B, 512, 16, 16)
        
        # --- Project Skips (将 ViT 特征放大到对应尺寸) ---
        # skip3 = self.skip_proj3(z9) # 16 -> 32
        # skip2 = self.skip_proj2(z6) # 16 -> 64
        # skip1 = self.skip_proj1(z3) # 16 -> 128

        # --- Decoder ---
        # bottleneck_fused = torch.cat([z12, cnn_feat], dim=1) # (B, 768+512, 16, 16)

        # x = self.bottleneck_conv(z12) # -> (B, 512, 16, 16)
        x = self.bottleneck_conv(cnn_feat) # -> (B, 512, 16, 16)
        # x = self.bottleneck_conv(bottleneck_fused) # -> (B, 512, 16, 16)
        # print(f"bottleneck: {x.shape}")
        
        x = self.decoder4(x, None) # Upsample to 32
        # print(f"decoder4: {x.shape}")
        x = self.decoder3(x, None) # Upsample to 64
        # print(f"decoder3: {x.shape}")
        x = self.decoder2(x, None) # Upsample to 128
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
