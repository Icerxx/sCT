import os
import SimpleITK as sitk
import nibabel as nib
from totalsegmentator.python_api import totalsegmentator
from pathlib import Path


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_RAW_DIR = os.path.join(ROOT, "data", "synthRAD2025_Task2_Train", "Task2")

def totalseg_segmentation(input, output_path, **kwargs):
    """
    input: Union[str, Path, sitk.Image, nib.Nifti1Image]
        - 如果是文件路径，检查是否为.mha格式，如果是则用sitk读取并转化为nib.Nifti1Image格式
        - 如果是sitk.Image格式，转化为nib.Nifti1Image格式
        - 如果是nib.Nifti1Image格式，直接使用
    output_path: Union[str, Path]
        - 如果是文件路径，检查文件夹是否存在，不存在则创建，并保存分割结果到该路径
    **kwargs: 传递给totalsegmentator的其他参数，参考：
    ml = True, task = "total", device = "gpu:1"
    """

    if isinstance(output_path, (str, Path)):
        output_path = Path(output_path)
        # 创建文件夹（如果不存在）
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError("output_path should be a string or Path object.")

    if isinstance(input, (str, Path)):
        input = str(input)
        print("Reading input file:", input)
        if input.endswith('.mha'):
            sitk_image = sitk.ReadImage(input)
            sitk.WriteImage(sitk_image, output_path)
            input = nib.load(str(output_path))
        elif input.endswith('.nii') or input.endswith('.nii.gz'):
            input = nib.load(input)
        else:
            raise ValueError("Unsupported file format. Please provide a .mha, .nii, or .nii.gz file.")
    elif isinstance(input, sitk.Image):
        sitk.WriteImage(input, output_path)
        input = nib.load(output_path)
    elif not isinstance(input, nib.Nifti1Image):
        raise ValueError("Unsupported input type. Please provide a file path, sitk.Image, or nib.Nifti1Image.")

    totalsegmentator(input=input, output=output_path, **kwargs)

def batch_totalseg(data_path, **kwargs):
    dataset_root = Path(data_path)

    for part in ["AB", "HN", "TH"]:
        part_dir = dataset_root / part
        if not part_dir.exists():
            print(f"[WARN] {part_dir} not found, skip.")
            continue

        print(f"\n=== Processing {part} ===")

        for case_dir in part_dir.iterdir():
            if not case_dir.is_dir():
                continue

            if case_dir.name.lower() == "overviews":
                continue

            ct_path = case_dir / "ct.mha"
            out_path = case_dir / "segmentation.nii.gz"

            if not ct_path.exists():
                print(f"[SKIP] No ct.mha in {case_dir}")
                continue

            if out_path.exists():
                print(f"[SKIP] Already done: {case_dir.name}")
                continue

            print(f"[RUN] {part}/{case_dir.name}")

            try:
                totalseg_segmentation(
                    input=ct_path,
                    output_path=out_path,
                    **kwargs
                )
            except Exception as e:
                print(f"[ERROR] {case_dir.name}: {e}")


if __name__ == '__main__':
    batch_totalseg(DATASET_RAW_DIR, ml=True, task="total", device="gpu:0")