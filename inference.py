import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from tqdm import tqdm

from meddino import MedDINOGenerator


# global config
ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(ROOT, "data", "test")
OUTPUT_DIR = os.path.join(ROOT, "res", "prediction")
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# normalization constants
# CT_CLIP_MIN = -1024.0
# CT_CLIP_MAX = 3000.0
NORM_MIN_HU = -1024.0
NORM_MAX_HU = 1500.0

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for MedDINO model")
    parser.add_argument("--img_size", type=int, default=256, help="Input image size (must match training)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--model", type=str, default="best_model.pth", help="ROOT/checkpoints/meddino_sct")
    parser.add_argument("--gpu", type=int, default=0, help="Specify GPU ID to use (default: 0)")
    return parser.parse_args()

def preprocess_batch(vol_slice_np):
    """
    对原始 HU 切片进行预处理，匹配训练时的逻辑
    Input: (B, H, W) numpy, int16 or float
    Output: (B, 3, 256, 256) tensor
    """
    # 1. 转 float32
    vol = vol_slice_np.astype(np.float32)
    
    # 2. 归一化
    # Clip [-1024, 1500]
    clipped = np.clip(vol, NORM_MIN_HU, NORM_MAX_HU)
    # Normalize based on [-1024, 1500] -> [0, 1]
    norm_range = NORM_MAX_HU - NORM_MIN_HU
    normalized = (clipped - NORM_MIN_HU) / norm_range
    scaled = (normalized - 0.5) * 2.0 
    
    # 3. 转 Tensor: (B, H, W) -> (B, 1, H, W)
    if scaled.ndim == 2:
        scaled = scaled[np.newaxis, ...]
    tensor = torch.from_numpy(scaled).float().unsqueeze(1)
    
    # 4. Resize 到训练尺寸
    # 这一步对应训练时的 F.interpolate
    tensor = F.interpolate(tensor, size=(args.img_size, args.img_size), mode='bilinear', align_corners=False)
    
    # 5. MedDINO 适配: 1通道 -> 3通道
    tensor = tensor.repeat(1, 3, 1, 1)
    
    return tensor

def postprocess_batch(pred_tensor, original_hw):
    """
    后处理：恢复尺寸和 HU 值
    Input: (B, 1, 256, 256) tensor
    Output: (B, OrigH, OrigW) numpy
    """
    # 1. Resize 回原始尺寸 (OrigH, OrigW)
    # pred_tensor: (B, 1, 256, 256)
    # original_hw: (H, W)
    pred_resized = F.interpolate(pred_tensor, size=original_hw, mode='bilinear', align_corners=False)
    
    # 2. 反归一化 (Inverse SynthRAD)
    # y = (x - 0.5) * 2.0  ==>  x = y / 2.0 + 0.5
    pred_denorm = pred_resized / 2.0 + 0.5
    
    # x = (hu - min) / range  ==> hu = x * range + min
    norm_range = NORM_MAX_HU - NORM_MIN_HU
    pred_hu = pred_denorm * norm_range + NORM_MIN_HU
    
    # # 3. 截断回有效 HU 范围
    # pred_hu = torch.clamp(pred_hu, CT_CLIP_MIN, CT_CLIP_MAX)
    
    # 4. 转 Numpy
    return pred_hu.squeeze(1).cpu().numpy()

def inference_case(model, case_name, save_root, device):
    cbct_path = os.path.join(TEST_DATA_DIR, case_name, "cbct.mha")

    if not os.path.exists(cbct_path):
        print(f"Skipping {case_name}: cbct.mha not found.")
        return

    # 1. 读取原始 MHA
    itk_img = sitk.ReadImage(cbct_path)
    origin_vol = sitk.GetArrayFromImage(itk_img) # (Z, Y, X)
    
    mask_path = os.path.join(TEST_DATA_DIR, case_name, "mask.mha")
    if os.path.exists(mask_path):
        mask_vol = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        
        # 将 Mask 外的区域强制设为空气值 (NORM_MIN_HU = -1024.0)
        origin_vol = np.where(mask_vol == 1, origin_vol, NORM_MIN_HU)
    else:
        print(f"Warning: Mask not found for {case_name}, inferring full image.")

    # 获取原始尺寸 (H, W) 用于最后插值回去
    # SimpleITK GetSize 返回 (X, Y, Z) -> 需要反转前两位得到 (Y, X) 即 (H, W)
    raw_size_hw = itk_img.GetSize()[:2][::-1] 
    
    num_slices = origin_vol.shape[0]
    sct_vol_list = []
    
    model.eval()
    
    # 2. 批量推理
    with torch.no_grad():
        for i in range(0, num_slices, args.batch):
            # 取出一个 Batch 的切片: (B, H, W)
            batch_slice_np = origin_vol[i : i + args.batch]
            
            # 预处理 (含归一化 + Resize 256 + 转3通道)
            batch_tensor = preprocess_batch(batch_slice_np)
            batch_tensor = batch_tensor.to(device)
            
            # Forward
            pred = model(batch_tensor) # (B, 1, 256, 256)
            
            # 后处理 (含 Resize 回原尺寸 + 反归一化)
            pred_hu_np = postprocess_batch(pred, raw_size_hw)
            sct_vol_list.append(pred_hu_np)
            
    # 拼接结果 (Z, H, W)
    sct_vol = np.concatenate(sct_vol_list, axis=0)
    
    # 转为 int16 (医学图像标准存储格式)
    sct_vol = sct_vol.astype(np.int16)
    
    # 3. 保存 sCT
    sct_itk = sitk.GetImageFromArray(sct_vol)
    sct_itk.CopyInformation(itk_img) # 关键：复制原始 CBCT 的空间元数据
    
    # 创建输出目录: res/prediction/CaseName/
    case_output_dir = os.path.join(save_root, case_name)
    os.makedirs(case_output_dir, exist_ok=True)
    
    save_path = os.path.join(case_output_dir, "sCT.mha")
    sitk.WriteImage(sct_itk, save_path)


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    MODEL_PATH = os.path.join(ROOT, "checkpoints", "meddino_sct", args.model)

    print(f"Loading model from {MODEL_PATH}...")
    model = MedDINOGenerator(img_size=args.img_size).to(device)
    
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        print("Model loaded successfully.")
    else:
        print(f"Error: Model path {MODEL_PATH} does not exist!")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(TEST_DATA_DIR):
        print(f"Error: Test data directory not found: {TEST_DATA_DIR}")
        return

    test_cases = sorted([d for d in os.listdir(TEST_DATA_DIR) if os.path.isdir(os.path.join(TEST_DATA_DIR, d))])
    print(f"Found {len(test_cases)} test cases in {TEST_DATA_DIR}")

    for case_name in tqdm(test_cases, desc="Inference"):
        inference_case(model, case_name, OUTPUT_DIR, device)

    print(f"\nInference Done! Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    args = parse_args()
    main(args)