import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dinov3.models.vision_transformer import vit_base


class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        # x: (N, C, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class Upsample(nn.Module):
    def __init__(self, in_chans, out_chans, scale_factor=2):
        super().__init__()
        # self.up = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
                                #   nn.BatchNorm2d(out_chans),
                                  LayerNorm2d(out_chans),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(self.up(x))

class PixelShuffleBlock(nn.Module):
    def __init__(self, dim, scale=2):
        super().__init__()
        # 1. 通道扩充: C -> C * scale^2
        self.expand = nn.Conv2d(dim, dim * scale * scale, kernel_size=1)
        # 2. PixelShuffle: 像素重排上采样
        self.ps = nn.PixelShuffle(scale)
        # 3. 深度卷积 (Depthwise Conv): 增加非线性并平滑伪影
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.norm = LayerNorm2d(dim)
        self.act = nn.GELU() # ViT 常用 GELU，比 ReLU 更平滑

    def forward(self, x):
        x = self.expand(x)
        x = self.ps(x)
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
    
class AttentionGate(nn.Module):
    """
    Attention Gate: 使用 Gating Signal (g) 来过滤 Skip Feature (x)
    原理: x * Sigmoid(Conv(ReLU(Conv(g) + Conv(x))))
    """
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: Gating 信号的通道数 (来自 Decoder)
            F_l: Skip 特征的通道数 (来自 Encoder/Projector)
            F_int: 中间层的通道数 (通常为 F_l / 2)
        """
        super(AttentionGate, self).__init__()
        
        # 对 Gating Signal 进行 1x1 卷积
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            LayerNorm2d(F_int) # 保持和你代码风格一致，用 LayerNorm
        )
        
        # 对 Skip Feature 进行 1x1 卷积
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            LayerNorm2d(F_int)
        )

        # 生成 Attention Map
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: Decoder 特征 (B, F_g, H, W)
        # x: Skip 特征 (B, F_l, H, W)
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
    
        # 1. 投影到中间空间
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # 2. 核心：相加融合 (广播机制)
        # 这一步让网络判断 Skip 中的哪些位置与 Decoder 当前关注的区域一致
        psi = self.relu(g1 + x1)
        
        # 3. 生成权重图 (0~1)
        psi = self.psi(psi)
        
        # 4. 加权 Skip 特征
        return x * psi
    
class AdaptiveSkipProjector(nn.Module):
    def __init__(self, in_dim, out_dim, scale_factor, gating_dim=None):
        super().__init__()

        # self.compressor = nn.Sequential(
        #     nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
        #     LayerNorm2d(out_dim),
        #     nn.ReLU(inplace=True)
        # )
        self.compressor = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        
        self.use_ag = gating_dim is not None
        if self.use_ag:
            self.ag = AttentionGate(F_g=gating_dim, F_l=out_dim, F_int=out_dim // 2)

        layers = []
        num_stages = int(torch.log2(torch.tensor(scale_factor)))
        for _ in range(num_stages):
            layers.append(Upsample(out_dim, out_dim, scale_factor=2))
            
        self.upsampler = nn.Sequential(*layers)

    def forward(self, x, gating_signal=None):
        # x: (B, 768, H, W)
        x = self.compressor(x) # -> (B, out_dim, H, W)

        if self.use_ag and gating_signal is not None:
            # # 如果因为 padding 导致尺寸微小不匹配，强制对齐 x 到 gating_signal
            # if x.shape[2:] != gating_signal.shape[2:]:
            #     x = F.interpolate(x, size=gating_signal.shape[2:], mode='bilinear', align_corners=False)          
            x = self.ag(g=gating_signal, x=x)

        x = self.upsampler(x)  # -> (B, out_dim, H*scale, W*scale)

        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # self.up = Upsample(in_channels, in_channels)
        self.up = PixelShuffleBlock(in_channels, scale=2)

        # 拼接后的通道数
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn1 = LayerNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn2 = LayerNorm2d(out_channels)
        
    def forward(self, x, skip=None):
        # upsample
        x = self.up(x)

        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

# =================================================================
# 2. MedDINO sCT Generator
# =================================================================
class MedDINOGenerator(nn.Module):
    def __init__(self, img_size=512, patch_size=16, out_chans=1, embed_dim=768, pretrained_path=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.encoder = vit_base(img_size=img_size, patch_size=patch_size, 
                                drop_path_rate=0.2, layerscale_init=1.0e-05, 
                                n_storage_tokens=4, qkv_bias=False, mask_k_bias=True)
        # load pretrained weights
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading MedDINO weights from {pretrained_path}...")
            chkpt = torch.load(pretrained_path, map_location='cpu', weights_only=True)

            state_dict = chkpt['teacher']
            state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() 
                          if 'ibot' not in k and 'dino_head' not in k}  # 去掉 'backbone.' 前缀和 'ibot', 'dino_head' 相关参数
            
            missing, unexpected = self.encoder.load_state_dict(state_dict, strict=True)
            print(f"Loaded MedDINO weights with {len(missing)} missing and {len(unexpected)} unexpected keys.")
        else:
            print("Warning: No pretrained weights loaded!")

        self.input_proj = nn.Conv2d(3, 16, kernel_size=3, padding=1)

        # self.bottleneck_conv = nn.Conv2d(embed_dim, 512, kernel_size=3, padding=1)
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(embed_dim, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            LayerNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Skip 3 (Layer 9): 22 -> 44 (2x)
        self.skip_proj3 = AdaptiveSkipProjector(embed_dim, 256, scale_factor=2, gating_dim=512)
        # Skip 2 (Layer 6): 22 -> 88 (4x)
        self.skip_proj2 = AdaptiveSkipProjector(embed_dim, 128, scale_factor=4, gating_dim=256)
        # Skip 1 (Layer 3): 22 -> 176 (8x)
        self.skip_proj1 = AdaptiveSkipProjector(embed_dim, 64, scale_factor=8, gating_dim=128)

        # Layer 4: 1/16 -> 1/8
        self.decoder4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        # Layer 3: 1/8 -> 1/4
        self.decoder3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        # Layer 2: 1/4 -> 1/2
        self.decoder2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        # Layer 1: 1/2 -> 1/1
        self.decoder1 = DecoderBlock(in_channels=64, skip_channels=16, out_channels=32)
        
        # 输出层
        self.out_conv = nn.Conv2d(32, out_chans, kernel_size=1)

    def forward(self, x):
        # x: (B, 1, H, W)
        # 如果输入是单通道 (医学图像)，但模型期望 3 通道 (DINO Pretrained)，则复制通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # x_in = x
        x_in = self.input_proj(x)  # -> (B, 16, H, W)

        # --- Encoder ---
        # 提取 ViT 中间层的索引 (用于 Skip Connection)
        # ViT-Base 有 12 层 transformer blocks。
        # 我们通常取第 3, 6, 9, 12 层的输出
        # ViT 的输出特征图尺寸: (B, Embed_Dim, H/16, W/16) -> (B, 768, 22, 22)
        z3, z6, z9, z12 = self.encoder.get_intermediate_layers(x, n=[2, 5, 8, 11], reshape=True)
        # bottleneck, skip3, skip2, skip1 = features[::-1]
        
        # --- Decoder ---
        x = self.bottleneck_conv(z12) # -> (B, 512, 22, 22)
        # print(f"bottleneck: {x.shape}")
        
        skip3 = self.skip_proj3(z9, gating_signal=x) # 22 -> 44
        x = self.decoder4(x, skip3) # Upsample to 44
        # print(f"decoder4: {x.shape}")

        skip2 = self.skip_proj2(z6, gating_signal=x) # 22 -> 88
        x = self.decoder3(x, skip2) # Upsample to 88
        # print(f"decoder3: {x.shape}")

        skip1 = self.skip_proj1(z3, gating_signal=x) # 22 -> 176
        x = self.decoder2(x, skip1) # Upsample to 176
        # print(f"decoder2: {x.shape}")
        
        x = self.decoder1(x, x_in) # Upsample to 352
        # print(f"decoder1: {x.shape}")
        
        out = self.out_conv(x)
        return out


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MedDINOGenerator(img_size=512).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Params: {total_params/1e6:.2f}M | Trainable: \
          {(total_params-sum(p.numel() for p in model.encoder.parameters()))/1e6:.2f}M")

    dummy_input = torch.randn(1, 1, 512, 512).to(device)
    output = model(dummy_input)
    print(f"Input: {dummy_input.shape}")
    print(f"Output: {output.shape}")
