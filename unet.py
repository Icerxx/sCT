import segmentation_models_pytorch as smp
import torch
from monai.networks.nets import UNETR


def get_unet(device):
    # 使用 ResNet34 作为 Encoder，加载 ImageNet 权重
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1, # 输出 sCT
        activation=None # 回归任务不需要 Sigmoid/Softmax
    )
    return model.to(device)

def get_unetpp(device):
    # 使用 ResNet34 作为 Encoder，加载 ImageNet 权重
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1, # 输出 sCT
        activation=None # 回归任务不需要 Sigmoid/Softmax
    )
    return model.to(device)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model = get_unet(device)
    # model = get_unetpp(device)
    model = UNETR(
        in_channels=3,
        out_channels=1,
        img_size=(256, 256), # 2D模式下
        feature_size=16,     # 控制网络宽度
        hidden_size=768,     # ViT hidden dim
        mlp_dim=3072,
        num_heads=12,
        norm_name="instance",
        res_block=True,
        spatial_dims=2       # 【关键】设为 2 表示 2D 网络
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Params: {total_params/1e6:.2f}M")

    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    output = model(dummy_input)
    print(f"Input: {dummy_input.shape}")
    print(f"Output: {output.shape}")