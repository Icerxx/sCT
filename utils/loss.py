import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# class CharbonnierLoss(nn.Module):
#     def __init__(self, eps=1e-6):
#         super(CharbonnierLoss, self).__init__()
#         self.eps = eps
#
#     def forward(self, x, y):
#         diff = x - y
#         loss = torch.sqrt(diff * diff + self.eps * self.eps)
#         return torch.mean(loss)

class CharbonnierLoss(nn.Module):
    """
    Robust L1 Loss: sqrt(x^2 + eps^2)
    """

    def __init__(self, eps=1e-6, reduction='mean'):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

    
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

# class GradientLoss(nn.Module):
#     def __init__(self, bone_weight=10.0):
#         super(GradientLoss, self).__init__()
#         self.bone_weight = bone_weight
#         # self.l1 = nn.L1Loss(reduction='none') # 不求平均，返回 map
#
#     def forward(self, pred, target, bone_mask):
#         # 计算 X 方向梯度: I(x+1, y) - I(x, y)
#         grad_pred_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
#         grad_target_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
#
#         # 计算 Y 方向梯度: I(x, y+1) - I(x, y)
#         grad_pred_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
#         grad_target_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
#
#         loss_x = torch.abs(grad_pred_x - grad_target_x)  # (B, C, H, W-1)
#         loss_y = torch.abs(grad_pred_y - grad_target_y)  # (B, C, H-1, W)
#
#         loss_x = F.pad(loss_x, (0, 1, 0, 0))
#         loss_y = F.pad(loss_y, (0, 0, 0, 1))
#         loss_map = loss_x + loss_y
#
#         # 骨头区域 x 10，其他区域 x 1
#         weights = 1.0 + (bone_mask * (self.bone_weight - 1.0))
#
#         # 加权平均
#         weighted_loss = (loss_map * weights).mean()
#         return weighted_loss

class GradientLoss(nn.Module):
    """
    计算图像梯度的 L1 误差
    """

    def __init__(self, reduction='mean'):
        super(GradientLoss, self).__init__()
        self.criterion = nn.L1Loss(reduction='none')  # 内部始终用 none，最后再 reduce
        self.reduction = reduction

    def forward(self, pred, target):
        # 计算 X 方向梯度: I(x+1) - I(x)
        # Pad 保持尺寸一致 (B, C, H, W)
        grad_pred_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        grad_target_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        loss_x = self.criterion(grad_pred_x, grad_target_x)
        loss_x = F.pad(loss_x, (0, 1, 0, 0))

        # 计算 Y 方向梯度: I(y+1) - I(y)
        grad_pred_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        grad_target_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        loss_y = self.criterion(grad_pred_y, grad_target_y)
        loss_y = F.pad(loss_y, (0, 0, 0, 1))

        loss_map = loss_x + loss_y

        if self.reduction == 'mean':
            return loss_map.mean()
        elif self.reduction == 'sum':
            return loss_map.sum()
        elif self.reduction == 'none':
            return loss_map
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

# class VGGPerceptualLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
#         # 冻结参数
#         for param in vgg.parameters():
#             param.requires_grad = False
#
#         self.slice1 = torch.nn.Sequential()
#         self.slice2 = torch.nn.Sequential()
#         self.slice3 = torch.nn.Sequential()
#         self.slice4 = torch.nn.Sequential()
#         self.slice5 = torch.nn.Sequential()
#
#         for x in range(4):  self.slice1.add_module(str(x), vgg[x])
#         for x in range(4, 9): self.slice2.add_module(str(x), vgg[x])
#         for x in range(9, 18): self.slice3.add_module(str(x), vgg[x])
#         for x in range(18, 27): self.slice4.add_module(str(x), vgg[x])
#         for x in range(27, 36): self.slice5.add_module(str(x), vgg[x])
#         self.eval()
#
#         self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
#         self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
#
#     def preprocess(self, x):
#         x = x * 0.5 + 0.5  # [-1,1] -> [0,1]
#         x = torch.clamp(x, 0.0, 1.0)
#
#         if x.shape[1] == 1:
#             x = x.repeat(1, 3, 1, 1)
#
#         return (x - self.mean) / self.std
#
#     def forward(self, input, target):
#         input = self.preprocess(input)
#         target = self.preprocess(target)
#
#         loss = 0.0
#         weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
#         slices = [self.slice1, self.slice2, self.slice3, self.slice4, self.slice5]
#
#         x = input
#         y = target
#         for i, slice_net in enumerate(slices):
#             x = slice_net(x)
#             y = slice_net(y)
#             loss += F.l1_loss(x, y) * weights[i]
#
#         return loss
#
# class CompositeLoss(nn.Module):
#     def __init__(self,
#                  l1_weight=1.0,
#                  grad_weight=0.5,
#                  perceptual_weight=0.1):
#         """
#         Args:
#             l1_weight: L1 Loss 权重 (控制绝对 HU 精度)
#             perceptual_weight: Perceptual Loss 权重 (控制感知质量)
#             grad_weight: Gradient Loss 权重 (控制纹理清晰度)
#         """
#         super(CompositeLoss, self).__init__()
#         self.l1_weight = l1_weight
#         self.grad_weight = grad_weight
#         self.perceptual_weight = perceptual_weight
#
#         # L1 Loss
#         self.l1 = CharbonnierLoss() if l1_weight > 0 else None
#
#         # Gradient Loss
#         self.grad = GradientLoss() if grad_weight > 0 else None
#
#         # Perceptual Loss
#         self.perceptual = VGGPerceptualLoss() if perceptual_weight > 0 else None
#
#         self.norm_min_hu = -1024.
#         self.norm_max_hu = 1500.
#         self.threshold = 300
#
#     def denorm(self, norm):
#         denorm = norm / 2.0 + 0.5
#         norm_range = self.norm_max_hu - self.norm_min_hu
#         denorm_hu = denorm * norm_range + self.norm_min_hu
#         return denorm_hu
#
#     def forward(self, pred, target, seg_mask=None):
#         """
#         pred: (B, 1, H, W)
#         target: (B, 1, H, W)
#         """
#         total_loss = 0
#         loss_dict = {}
#
#         # L1 Loss
#         if self.l1 is not None:
#             loss_l1 = self.l1(pred, target)
#             total_loss += self.l1_weight * loss_l1
#             loss_dict["l1"] = loss_l1.item()
#
#         # Gradient Loss
#         if self.grad is not None:
#             # loss_grad = self.grad(pred, target)
#             # total_loss += self.grad_weight * loss_grad
#             # loss_dict["grad"] = loss_grad.item()
#
#             mask = seg_mask if seg_mask is not None else (self.denorm(target) > self.threshold).float()
#             loss_grad = self.grad(pred, target, mask)
#             total_loss += self.grad_weight * loss_grad
#             loss_dict["grad"] = loss_grad.item()
#
#
#         # Perceptual Loss
#         if self.perceptual is not None:
#             loss_perceptual = self.perceptual(pred, target)
#             total_loss += self.perceptual_weight * loss_perceptual
#             loss_dict["perceptual"] = loss_perceptual.item()
#
#         return total_loss, loss_dict


# 骨骼 (Bone): 高密度，边缘锐利，主要应用 Gradient Loss
BONE_IDS = [
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, # Vertebrae L/T
    44, 45, 46, 47, 48, 49, 50, # Vertebrae C
    69, 70, # Humerus
    71, 72, # Scapula
    73, 74, # Clavicula
    75, 76, # Femur
    77, 78, # Hip
    91, # Skull
    92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, # Ribs Left
    104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, # Ribs Right
    116, 117 # Sternum, Costal Cartilages
]

# 软组织 (Soft Tissue): 低对比度，易受散射影响，主要应用 L1 Loss
SOFT_IDS = [
    1, # Spleen
    2, 3, # Kidney
    4, # Gallbladder
    5, # Liver
    7, # Pancreas
    8, 9, # Adrenal
    17, # Thyroid
    21, # Bladder
    22, # Prostate
    51, # Heart
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, # Vessels
    79, # Spinal cord
    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, # Muscles
    90 # Brain
]

# (可选) 肺/空腔/肠道: 纹理复杂，暂不特殊处理，归入 Global L1
LUNG_BOWEL_IDS = [6, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20]

class AnatomyAwareLoss(nn.Module):
    def __init__(self,
                 global_l1_weight=1.0,
                 masked_l1_weight=0.0,
                 masked_grad_weight=0.0):
        super(AnatomyAwareLoss, self).__init__()

        self.weights = {
            'global': global_l1_weight,
            'soft': masked_l1_weight,
            'bone': masked_grad_weight
        }

        # Loss Functions (reduction='none' to return maps)
        self.charb = CharbonnierLoss(reduction='none')
        self.grad = GradientLoss(reduction='none')

        # 注册 ID 到 Buffer (自动随模型移动到 device)
        self.register_buffer('bone_ids', torch.tensor(BONE_IDS, dtype=torch.long))
        self.register_buffer('soft_ids', torch.tensor(SOFT_IDS, dtype=torch.long))

    def get_roi_mask(self, seg, target_ids):
        """
        从 seg (B, 1, H, W) 中提取指定 ID 的区域
        """
        if seg is None: return None
        # torch.isin: 检查 seg 中的元素是否在 target_ids 中
        mask = torch.isin(seg, target_ids)
        return mask.float()

    def forward(self, pred, target, seg_mask=None):
        """
        pred, target: (B, 1, H, W)
        seg_mask: (B, 1, H, W) LongTensor, values 0-117
        """
        total_loss = 0
        loss_dict = {}

        # --- 1. Global L1 (Charbonnier) ---
        # 始终计算，保证整体分布
        if self.weights['global'] > 0:
            loss_map_l1 = self.charb(pred, target)
            loss_global = loss_map_l1.mean()
            total_loss += self.weights['global'] * loss_global
            loss_dict['global_l1'] = loss_global.item()
        else:
            # 如果不开 Global，也要算一下 map 给后面用，避免重复计算
            loss_map_l1 = self.charb(pred, target)

        # 如果没有 Seg Mask，直接返回
        if seg_mask is None:
            return total_loss, loss_dict

        # --- 2. Masked L1 (Soft Tissue Focus) ---
        if self.weights['soft'] > 0:
            mask_soft = self.get_roi_mask(seg_mask, self.soft_ids)
            mask_sum = mask_soft.sum()

            if mask_sum > 0:
                # 只计算 Mask 内的平均误差
                # 这样不会因为 Mask 面积小而被稀释
                loss_soft = (loss_map_l1 * mask_soft).sum() / mask_sum
                total_loss += self.weights['soft'] * loss_soft
                loss_dict['soft_l1'] = loss_soft.item()
            else:
                loss_dict['soft_l1'] = 0.0
        else:
            loss_dict['soft_l1'] = 0.0

        # --- 3. Masked Gradient (Bone Focus) ---
        if self.weights['bone'] > 0:
            mask_bone = self.get_roi_mask(seg_mask, self.bone_ids)
            mask_sum = mask_bone.sum()

            if mask_sum > 0:
                # 计算全图梯度误差 Map
                grad_map = self.grad(pred, target)

                # 只计算 Bone Mask 内的梯度误差
                loss_bone = (grad_map * mask_bone).sum() / mask_sum
                total_loss += self.weights['bone'] * loss_bone
                loss_dict['bone_grad'] = loss_bone.item()
            else:
                loss_dict['bone_grad'] = 0.0
        else:
            loss_dict['bone_grad'] = 0.0

        return total_loss, loss_dict