import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
from tqdm import tqdm

class SynthRADDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=256, augment=False):
        self.root = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.clip_min = -1024.
        self.clip_max = 3000.
        self.norm_min_hu = -1024.
        self.norm_max_hu = 1500.
        
        self.data_dir = os.path.join(root_dir, "data", split)
        
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
            
        print(f"Scanning {split} data from {self.data_dir}...")
        self.samples = [] 
        
        case_dirs = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])
        
        for case_name in tqdm(case_dirs, desc=f"Indexing {split}"):
            case_path = os.path.join(self.data_dir, case_name)
            cbct_path = os.path.join(case_path, "cbct.npy")
            ct_path = os.path.join(case_path, "ct.npy")
            
            if os.path.exists(cbct_path) and os.path.exists(ct_path):
                try:
                    shape = np.load(cbct_path, mmap_mode='r').shape
                    num_slices = shape[0]
                    for z in range(num_slices):
                        self.samples.append((cbct_path, ct_path, z))
                except Exception as e:
                    print(f"Error reading {cbct_path}: {e}")

        print(f"Dataset [{split}] ready: {len(self.samples)} slices.")

    def __len__(self):
        return len(self.samples)

    def normalize(self, img):
        """
        Output: torch tensor float32
        Range: Background (-1024) -> -1.0; Bone (1500) -> 1.0; Metal (3000) -> 2.18
        """
        # img = img.astype(np.float32)
        
        # Clip [-1024, 3000]
        clipped_img = np.clip(img, a_min=self.clip_min, a_max=self.clip_max)
        
        # Normalize to [0, 1] relative to [-1024, 1500]
        norm_range_hu = self.norm_max_hu - self.norm_min_hu
        normalized_img = (clipped_img - self.norm_min_hu) / norm_range_hu
        
        # Scale to [-1, 1] (allowing overshoot > 1 for high density)
        scaled_img = (normalized_img - 0.5) * 2.0
        
        return torch.from_numpy(scaled_img)
    
    def __getitem__(self, idx):
        cbct_path, ct_path, slice_idx = self.samples[idx]
        
        # mmap 读取 (int16)
        cbct_vol = np.load(cbct_path, mmap_mode='r')
        ct_vol = np.load(ct_path, mmap_mode='r')
        
        img_cbct = cbct_vol[slice_idx].copy().astype(np.float32)
        img_ct = ct_vol[slice_idx].copy().astype(np.float32)
        
        # 归一化 (使用 SynthRAD 策略)
        tensor_cbct = self.normalize(img_cbct)
        tensor_ct = self.normalize(img_ct)
        
        # 增加维度 (1, 1, H, W)
        tensor_cbct = tensor_cbct.unsqueeze(0).unsqueeze(0)
        tensor_ct = tensor_ct.unsqueeze(0).unsqueeze(0)
        
        # 3. Resize
        tensor_cbct = F.interpolate(tensor_cbct, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        tensor_ct = F.interpolate(tensor_ct, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        tensor_cbct = tensor_cbct.squeeze(0)
        tensor_ct = tensor_ct.squeeze(0)

        # 4. Augmentation
        if self.augment:
            if torch.rand(1) < 0.5:
                tensor_cbct = T.functional.hflip(tensor_cbct)
                tensor_ct = T.functional.hflip(tensor_ct)

        # 5. MedDINO 适配 (1 -> 3 channel)
        tensor_cbct = tensor_cbct.repeat(3, 1, 1)
        
        return {
            'cbct': tensor_cbct, 
            'ct': tensor_ct
        }
