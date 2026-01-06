import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import SSIM, MS_SSIM


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
    
class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.l1 = CharbonnierLoss()

    def forward(self, pred, target):
        # 计算 X 方向梯度: I(x+1, y) - I(x, y)
        grad_pred_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        grad_target_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        
        # 计算 Y 方向梯度: I(x, y+1) - I(x, y)
        grad_pred_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        grad_target_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        
        loss_x = self.l1(grad_pred_x, grad_target_x)
        loss_y = self.l1(grad_pred_y, grad_target_y)
        
        return loss_x + loss_y

class CompositeLoss(nn.Module):
    def __init__(self, 
                 l1_weight=1.0, 
                 msssim_weight=0.5, 
                 grad_weight=0.5, 
                 device='cuda'):
        """
        Args:
            l1_weight: L1 Loss 权重 (控制绝对 HU 精度)
            msssim_weight: MS-SSIM Loss 权重 (控制整体结构)
            grad_weight: Gradient Loss 权重 (控制纹理清晰度)
        """
        super(CompositeLoss, self).__init__()
        self.l1_weight = l1_weight
        self.msssim_weight = msssim_weight
        self.grad_weight = grad_weight
        
        # 1. L1 Loss
        self.l1 = CharbonnierLoss() if l1_weight > 0 else None
        
        # 2. Gradient Loss
        self.grad = GradientLoss() if grad_weight > 0 else None
        
        # 3. MS-SSIM Loss
        # 【关键计算】
        # 输入范围: [-1.0, 2.18]
        # data_range = Max - Min = 2.18 - (-1.0) = 3.18
        # 我们稍微放大一点点取整，或者直接用 3.18，确保数值稳定性。
        self.data_range = 4
        
        # 假设 sCT 输出是 1 通道 (如果你的网络输出是3通道，这里 channel 改为 3)
        self.msssim_module = MS_SSIM(
            data_range=self.data_range, 
            size_average=True, 
            channel=1
        ).to(device) if msssim_weight > 0 else None

    def forward(self, pred, target):
        """
        pred: (B, 1, H, W)
        target: (B, 1, H, W)
        """
        total_loss = 0
        loss_dict = {}

        # 1. L1 Loss
        if self.l1_weight > 0 and self.l1 is not None:
            loss_l1 = self.l1(pred, target)
            total_loss += self.l1_weight * loss_l1
            loss_dict["l1"] = loss_l1.item()

        # 2. MS-SSIM Loss
        if self.msssim_weight > 0 and self.msssim_module is not None:
            loss_msssim = 1 - self.msssim_module(pred, target)
            total_loss += self.msssim_weight * loss_msssim
            loss_dict["msssim"] = loss_msssim.item()

        # 3. Gradient Loss
        if self.grad_weight > 0 and self.grad is not None:
            loss_grad = self.grad(pred, target)
            total_loss += self.grad_weight * loss_grad
            loss_dict["grad"] = loss_grad.item()

        return total_loss, loss_dict
    
    # perceptual loss