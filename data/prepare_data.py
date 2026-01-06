import os
import shutil
import random
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_RAW_DIR = os.path.join(ROOT, "data", "synthRAD2025_Task2_Train", "Task2")
OUTPUT_DIR = os.path.join(ROOT, "data")

ANATOMIES = ['AB', 'HN', 'TH']
RANDOM_SEED = 42

def process_train_val(case_dir, output_root, case_name):
    """
    Train/Val 处理逻辑：读取 -> 转numpy(int16) -> 保存npy
    """
    cbct_path = os.path.join(case_dir, "cbct.mha")
    ct_path = os.path.join(case_dir, "ct.mha")
    mask_path = os.path.join(case_dir, "mask.mha")

    if not (os.path.exists(cbct_path) and os.path.exists(ct_path)):
        return

    # 创建 Case 子文件夹
    case_out_dir = os.path.join(output_root, case_name)
    os.makedirs(case_out_dir, exist_ok=True)

    # 读取数据    
    cbct_vol = sitk.GetArrayFromImage(sitk.ReadImage(cbct_path))
    ct_vol = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))

    if os.path.exists(mask_path):
        mask_vol = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        
        # Mask 外区域填充为 -1024
        cbct_vol = np.where(mask_vol == 1, cbct_vol, -1024)
        ct_vol = np.where(mask_vol == 1, ct_vol, -1024)
    else:
        print(f"Warning: No mask found for {case_name}, using full image.")

    # 保存为 npy
    np.save(os.path.join(case_out_dir, "cbct.npy"), cbct_vol.astype(np.int16))
    np.save(os.path.join(case_out_dir, "ct.npy"), ct_vol.astype(np.int16))

def process_test(case_dir, output_root, case_name):
    """
    Test 处理逻辑：直接复制文件夹及内容
    """
    # 创建 Case 子文件夹
    dest_dir = os.path.join(output_root, case_name)
    
    # 如果目标已存在，先删除（防止旧文件干扰），或者 shutil.copytree 会报错
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
        
    # 复制整个文件夹 (包含 cbct.mha, ct.mha, mask.mha)
    try:
        shutil.copytree(case_dir, dest_dir)
    except Exception as e:
        print(f"Error copying {case_name}: {e}")


if __name__ == "__main__":    
    random.seed(RANDOM_SEED)

    for anatomy in ANATOMIES:
        anatomy_dir = os.path.join(DATASET_RAW_DIR, anatomy)
        if not os.path.exists(anatomy_dir): continue
        
        # 获取该部位下所有病例
        # cases = sorted([d for d in os.listdir(anatomy_dir) if os.path.isdir(os.path.join(anatomy_dir, d))])
        cases = sorted([d for d in os.listdir(anatomy_dir) if os.path.isdir(os.path.join(anatomy_dir, d)) and d != "overviews"])
        random.shuffle(cases)
        
        n_total = len(cases)
        n_train = int(n_total * 0.9)
        n_val = int(n_total * 0.05)
        
        splits = {
            'train': cases[:n_train],
            'val': cases[n_train:n_train+n_val],
            'test': cases[n_train+n_val:]
        }
        
        print(f"\nProcessing {anatomy}...")
        
        for mode, case_list in splits.items():
            mode_root = os.path.join(OUTPUT_DIR, mode)
            os.makedirs(mode_root, exist_ok=True)
            
            for case_name in tqdm(case_list, desc=f"  {mode}"):
                case_dir = os.path.join(anatomy_dir, case_name)
                
                if mode == 'test':
                    process_test(case_dir, mode_root, case_name)
                else:
                    process_train_val(case_dir, mode_root, case_name)

    print("\nData preparation done!")
