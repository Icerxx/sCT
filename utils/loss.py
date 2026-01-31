import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return torch.mean(loss)
    
class FFTLoss(nn.Module):
    def __init__(self, loss_weight=0.1):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        # 转换到频域 (Real + Imaginary)
        # rfft2 适用于实数输入，只计算一半的频域信息（对称性）
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        
        # 计算实部和虚部的差异，或者幅度和相位的差异
        loss = self.criterion(torch.real(pred_fft), torch.real(target_fft)) + \
               self.criterion(torch.imag(pred_fft), torch.imag(target_fft))
        
        return loss * self.loss_weight
    
# class GradientLoss(nn.Module):
#     def __init__(self):
#         super(GradientLoss, self).__init__()
#         self.l1 = CharbonnierLoss()
#
#     def forward(self, pred, target):
#         # 计算 X 方向梯度: I(x+1, y) - I(x, y)
#         grad_pred_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
#         grad_target_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
#
#         # 计算 Y 方向梯度: I(x, y+1) - I(x, y)
#         grad_pred_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
#         grad_target_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
#
#         loss_x = self.l1(grad_pred_x, grad_target_x)
#         loss_y = self.l1(grad_pred_y, grad_target_y)
#
#         return loss_x + loss_y

class GradientLoss(nn.Module):
    def __init__(self, bone_weight=10.0):
        super(GradientLoss, self).__init__()
        self.bone_weight = bone_weight
        # self.l1 = nn.L1Loss(reduction='none') # 不求平均，返回 map

    def forward(self, pred, target, bone_mask):
        # 计算 X 方向梯度: I(x+1, y) - I(x, y)
        grad_pred_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        grad_target_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])

        # 计算 Y 方向梯度: I(x, y+1) - I(x, y)
        grad_pred_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        grad_target_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])

        loss_x = torch.abs(grad_pred_x - grad_target_x)  # (B, C, H, W-1)
        loss_y = torch.abs(grad_pred_y - grad_target_y)  # (B, C, H-1, W)

        loss_x = F.pad(loss_x, (0, 1, 0, 0))
        loss_y = F.pad(loss_y, (0, 0, 0, 1))
        loss_map = loss_x + loss_y

        # 骨头区域 x 10，其他区域 x 1
        weights = 1.0 + (bone_mask * (self.bone_weight - 1.0))

        # 加权平均
        weighted_loss = (loss_map * weights).mean()
        return weighted_loss

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features       
        # 冻结参数
        for param in vgg.parameters():
            param.requires_grad = False

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        
        for x in range(4):  self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9): self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 18): self.slice3.add_module(str(x), vgg[x])
        for x in range(18, 27): self.slice4.add_module(str(x), vgg[x])
        for x in range(27, 36): self.slice5.add_module(str(x), vgg[x])
        self.eval()
        
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, x):
        x = x * 0.5 + 0.5  # [-1,1] -> [0,1]
        x = torch.clamp(x, 0.0, 1.0)

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        return (x - self.mean) / self.std
    
    def forward(self, input, target):
        input = self.preprocess(input)
        target = self.preprocess(target)
        
        loss = 0.0
        weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        slices = [self.slice1, self.slice2, self.slice3, self.slice4, self.slice5]

        x = input
        y = target
        for i, slice_net in enumerate(slices):
            x = slice_net(x)
            y = slice_net(y)
            loss += F.l1_loss(x, y) * weights[i]

        return loss
    
class CompositeLoss(nn.Module):
    def __init__(self, 
                 l1_weight=1.0, 
                 grad_weight=0.5, 
                 perceptual_weight=0.1):
        """
        Args:
            l1_weight: L1 Loss 权重 (控制绝对 HU 精度)
            perceptual_weight: Perceptual Loss 权重 (控制感知质量)
            grad_weight: Gradient Loss 权重 (控制纹理清晰度)
        """
        super(CompositeLoss, self).__init__()
        self.l1_weight = l1_weight
        self.grad_weight = grad_weight
        self.perceptual_weight = perceptual_weight
        
        # L1 Loss
        self.l1 = CharbonnierLoss() if l1_weight > 0 else None
        
        # Gradient Loss
        self.grad = GradientLoss() if grad_weight > 0 else None

        # Perceptual Loss
        self.perceptual = VGGPerceptualLoss() if perceptual_weight > 0 else None

        self.norm_min_hu = -1024.
        self.norm_max_hu = 1500.
        self.threshold = 300

    def denorm(self, norm):
        denorm = norm / 2.0 + 0.5
        norm_range = self.norm_max_hu - self.norm_min_hu
        denorm_hu = denorm * norm_range + self.norm_min_hu
        return denorm_hu

    def forward(self, pred, target, mask=None):
        """
        pred: (B, 1, H, W)
        target: (B, 1, H, W)
        """
        total_loss = 0
        loss_dict = {}

        # L1 Loss
        if self.l1 is not None:
            loss_l1 = self.l1(pred, target)
            total_loss += self.l1_weight * loss_l1
            loss_dict["l1"] = loss_l1.item()

        # Gradient Loss
        if self.grad is not None:
            # loss_grad = self.grad(pred, target)
            # total_loss += self.grad_weight * loss_grad
            # loss_dict["grad"] = loss_grad.item()

            mask = mask if mask is not None else (self.denorm(target) > self.threshold).float()
            loss_grad = self.grad(pred, target, mask)
            total_loss += self.grad_weight * loss_grad
            loss_dict["grad"] = loss_grad.item()


        # Perceptual Loss
        if self.perceptual is not None:
            loss_perceptual = self.perceptual(pred, target)
            total_loss += self.perceptual_weight * loss_perceptual
            loss_dict["perceptual"] = loss_perceptual.item()

        return total_loss, loss_dict
