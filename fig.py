import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from matplotlib.colors import Normalize

# ==============================================================================
# 配置区域
# ==============================================================================

CASE_AB = "2ABA027"
SLICE_AB = 51
WINDOW_AB = (-1024, 400)

CASE_HN = "2HNC079"
SLICE_HN = 68
WINDOW_HN = (-1024, 730)

CASE_TH = "2THA043"
SLICE_TH = 15
WINDOW_TH = (-1024, 500)

ERR_RANGE = (-300, 300)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "res", "fig")

METHODS = [
    ('ct', 'CBCT'),
    ('unet', 'UNet'),
    ('unetpp', 'UNet++'),
    ('unetr', 'UNETR'),
    ('transunet', 'TransUNet'),
    ('ours_aahl', 'Ours'),
    ('ct', 'CT')
]

FONT_SIZE = 18 # 字体加大，适应大图

# ==============================================================================
# 核心逻辑
# ==============================================================================

def load_slice(path, slice_idx_cfg):
    if not os.path.exists(path): return None
    img_itk = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img_itk)
    
    if slice_idx_cfg is None:
        idx = arr.shape[0] // 2
    else:
        idx = int(slice_idx_cfg)
        idx = max(0, min(idx, arr.shape[0] - 1))
    
    return arr[idx, :, :]

def main():
    rows_config = [
        ('Abdomen', CASE_AB, SLICE_AB, WINDOW_AB),
        ('Head & Neck', CASE_HN, SLICE_HN, WINDOW_HN),
        ('Thorax', CASE_TH, SLICE_TH, WINDOW_TH)
    ]

    # --- 1. 动态计算画布尺寸 ---
    # 为了严丝合缝，我们需要知道图片的原始尺寸
    # 假设所有图片尺寸一致 (通常 SynthRAD 都是 512x512)
    sample_path = os.path.join(DATA_DIR, "ct", CASE_AB, "ct.mha")
    if os.path.exists(sample_path):
        sample_img = load_slice(sample_path, SLICE_AB)
        img_h, img_w = sample_img.shape
    else:
        img_h, img_w = 512, 512 # Fallback
    
    n_rows = 6
    n_cols = 7
    
    # 计算画布尺寸 (单位: 英寸)
    # 假设每张小图宽 2 英寸，高度按比例自动算出
    fig_width = n_cols * 3 
    fig_height = n_rows * (img_h / img_w) * 3
    
    print(f"Image Size: {img_h}x{img_w}")
    print(f"Calculated FigSize: {fig_width:.1f}x{fig_height:.1f} inches")

    # 创建画布
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    
    # 极窄边距
    plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0.05, right=0.99, top=0.95, bottom=0.05)
    
    for i, (anat_name, case_name, slice_cfg, win_cfg) in enumerate(rows_config):
        print(f"Plotting {anat_name}: {case_name}")
        
        row_img = i * 2
        row_err = i * 2 + 1
        
        ct_path = os.path.join(DATA_DIR, "ct", case_name, "ct.mha")
        mask_path = os.path.join(DATA_DIR, "ct", case_name, "mask.mha")
        
        gt_slice = load_slice(ct_path, slice_cfg)
        mask_slice = load_slice(mask_path, slice_cfg)
        
        if gt_slice is None: continue
        if mask_slice is None: mask_slice = np.ones_like(gt_slice)
        else: mask_slice = (mask_slice > 0).astype(float)

        for j, (folder, label) in enumerate(METHODS):
            ax_img = axes[row_img, j]
            ax_err = axes[row_err, j]
            
            # 关闭坐标轴
            ax_img.set_axis_off()
            ax_err.set_axis_off()
            
            # 路径
            if label == 'CBCT': path = os.path.join(DATA_DIR, "ct", case_name, "cbct.mha")
            elif label == 'CT': path = ct_path
            else: path = os.path.join(DATA_DIR, folder, case_name, "sCT.mha")
            
            img_slice = load_slice(path, slice_cfg)
            if img_slice is None: img_slice = np.zeros_like(gt_slice) - 1024

            # --- 绘制图像 ---
            # [关键] aspect='equal' 保证不拉伸
            ax_img.imshow(img_slice, cmap='gray', vmin=win_cfg[0], vmax=win_cfg[1], aspect='equal')
            
            # 标题
            if i == 0:
                ax_img.set_title(label, fontsize=FONT_SIZE, pad=12) # fontweight='bold'
                
            # 左侧标签 (Anatomy)
            if j == 0:
                ax_img.text(-0.05, 0.5, anat_name, fontsize=FONT_SIZE, rotation=90, 
                            va='center', ha='right', transform=ax_img.transAxes) # fontweight='bold'

            # --- 绘制误差图 ---
            if j == 0:
                ax_err.text(-0.05, 0.5, "Error Map", fontsize=FONT_SIZE, rotation=90, 
                            va='center', ha='right', transform=ax_err.transAxes) # fontweight='bold'

            if label == 'CT':
                # CT 列: Colorbar (左移，贴近 Ours)
                # 因为 aspect=equal，格子可能会比图片大，inset_axes 定位需要更精准
                # 我们把 bar 画在格子左侧的空白处
                cax = ax_err.inset_axes([0.025, 0.05, 0.08, 0.9]) 
                
                norm = Normalize(vmin=ERR_RANGE[0], vmax=ERR_RANGE[1])
                sm = plt.cm.ScalarMappable(cmap='bwr', norm=norm)
                sm.set_array([])
                
                cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=12)
            else:
                diff = (img_slice - gt_slice) * mask_slice
                ax_err.imshow(diff, cmap='bwr', vmin=ERR_RANGE[0], vmax=ERR_RANGE[1], aspect='equal')

    save_path = os.path.join(ROOT_DIR, "1.png")
    # bbox_inches='tight' 会自动裁掉多余白边
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"\nSaved to: {save_path}")
    # plt.show()

if __name__ == "__main__":
    main()


# import os
# import numpy as np
# import SimpleITK as sitk

# # ==============================
# # 配置
# # ==============================

# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(ROOT_DIR, "res", "fig")

# REGIONS = {
#     "Abdomen": "AB",
#     "Head & Neck": "HN",
#     "Thorax": "TH"
# }

# METHODS = [
#     "unet",
#     "unetpp",
#     "unetr",
#     "transunet"
# ]

# OURS = "ours_aahl"

# ERR_RANGE = (-300, 300)
# TOP_K = 5   # 每个部位挑几个


# # ==============================
# # 工具函数
# # ==============================

# def load_volume(path):
#     img = sitk.ReadImage(path)
#     return sitk.GetArrayFromImage(img)


# def compute_mae(pred, gt):
#     diff = np.abs(pred - gt)
#     return diff.mean()


# def find_cases_by_region():
#     """
#     根据case名字判断部位
#     2ABxxx -> AB
#     2HNxxx -> HN
#     2THxxx -> TH
#     """
#     ct_dir = os.path.join(DATA_DIR, "ct")
#     cases = os.listdir(ct_dir)

#     region_cases = {k: [] for k in REGIONS.keys()}

#     for case in cases:
#         if case.startswith("2AB"):
#             region_cases["Abdomen"].append(case)
#         elif case.startswith("2HN"):
#             region_cases["Head & Neck"].append(case)
#         elif case.startswith("2TH"):
#             region_cases["Thorax"].append(case)

#     return region_cases


# # ==============================
# # 主逻辑
# # ==============================

# def main():

#     region_cases = find_cases_by_region()

#     for region, cases in region_cases.items():
#         print(f"\n==============================")
#         print(f"{region}")
#         print(f"==============================")

#         improvements = []

#         for case in cases:

#             try:
#                 gt = load_volume(os.path.join(DATA_DIR, "ct", case, "ct.mha"))
#                 ours = load_volume(os.path.join(DATA_DIR, OURS, case, "sCT.mha"))
#             except:
#                 continue

#             n_slices = gt.shape[0]

#             for z in range(n_slices):

#                 gt_slice = gt[z]
#                 ours_slice = ours[z]

#                 ours_mae = compute_mae(ours_slice, gt_slice)

#                 baseline_maes = []

#                 for m in METHODS:
#                     try:
#                         pred = load_volume(
#                             os.path.join(DATA_DIR, m, case, "sCT.mha")
#                         )
#                         baseline_maes.append(
#                             compute_mae(pred[z], gt_slice)
#                         )
#                     except:
#                         continue

#                 if len(baseline_maes) == 0:
#                     continue

#                 best_baseline = min(baseline_maes)

#                 improvement = best_baseline - ours_mae

#                 # 只记录 ours 确实更好 且 margin > 5 HU
#                 if improvement > 5:
#                     improvements.append(
#                         (improvement, case, z, ours_mae, best_baseline)
#                     )

#             print(f"Scanned {case}")

#         # 排序：提升最大的在前
#         improvements.sort(reverse=True, key=lambda x: x[0])

#         print(f"\nTop {TOP_K} slices where Ours clearly better:\n")

#         for item in improvements[:TOP_K]:
#             imp, case, z, ours_mae, best_base = item
#             print(f"Case: {case}  Slice: {z}")
#             print(f"  Ours MAE: {ours_mae:.2f}")
#             print(f"  Best Baseline MAE: {best_base:.2f}")
#             print(f"  Improvement: {imp:.2f}\n")


# if __name__ == "__main__":
#     main()
