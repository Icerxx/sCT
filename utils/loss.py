import torch
import torch.nn as nn
import torch.nn.functional as F


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
    6,
    7, # Pancreas
    8, 9, # Adrenal
    10, 11, 12, 13, 14, 15, 16,
    17, # Thyroid
    18, 19, 20,
    21, # Bladder
    22, # Prostate
    51, # Heart
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, # Vessels
    79, # Spinal cord
    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, # Muscles
    90 # Brain
]

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