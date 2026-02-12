import os
import sys
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(ROOT)

import numpy as np
from utils.image_metrics import ImageMetrics
import SimpleITK as sitk
import csv

DATA_ROOT_DIR = os.path.join(ROOT, "data", "test")

def evaluate_case(case_name):
    metrics_hu = ImageMetrics(hu=True, visualize=False)

    label_dir = os.path.join(DATA_ROOT_DIR, case_name)
    test_dir = os.path.join(PREDICTION_DIR, case_name)

    cbct_path = os.path.join(label_dir, "cbct.mha")
    ct_path = os.path.join(label_dir, "ct.mha")
    mask_path = os.path.join(label_dir, "mask.mha")
    sct_path = os.path.join(test_dir, "sCT.mha")

    print(f"\nEvaluating case: {case_name}")
    cbct_hu = sitk.GetArrayFromImage(sitk.ReadImage(cbct_path))
    ct_hu = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))
    mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    # cleaned_cbct_mu = nib.load(cleaned_cbct_path).get_fdata(dtype=np.float32).transpose(2, 1, 0)
    # cleaned_cbct_mu[cleaned_cbct_mu < 0] = 0
    # cleaned_cbct_hu = nib.load(cleaned_cbct_path).get_fdata(dtype=np.float32).transpose(2, 1, 0)
    sct_hu = sitk.GetArrayFromImage(sitk.ReadImage(sct_path))

    print(f'{" HU-values ":-^100}')
    raw_metrics = metrics_hu.score_patient(ct_hu, cbct_hu, mask)
    print(f'raw CBCT(HU) & CT(HU): {raw_metrics}')
    
    # cleaned_cbct_hu = mu_to_hu(cleaned_cbct_mu)
    sct_hu = np.round(sct_hu).astype(ct_hu.dtype)
    sct_metrics = metrics_hu.score_patient(ct_hu, sct_hu, mask)
    print(f'sCT(HU) & CT(HU): {sct_metrics}')

    # Calculate differences
    diff_metrics = {
        key: sct_metrics[key] - raw_metrics[key]
        for key in raw_metrics
    }
    print(f'differences: {diff_metrics}')

    # Return results for CSV writing
    return {
        "case": case_name,
        **{f"raw_{key}": value for key, value in raw_metrics.items()},
        **{f"sct_{key}": value for key, value in sct_metrics.items()},
        **{f"diff_{key}": value for key, value in diff_metrics.items()},
    }

def write_results_to_csv(results, csv_path):
    # Check if the CSV file exists
    file_exists = os.path.isfile(csv_path)

    # Define the header
    header = [
        "case",
        "raw_mae", "raw_psnr", "raw_ms_ssim",
        "sct_mae", "sct_psnr", "sct_ms_ssim",
        "diff_mae", "diff_psnr", "diff_ms_ssim",
    ]

    # Write to the CSV file
    with open(csv_path, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)

        # Write the header if the file does not exist
        if not file_exists:
            writer.writeheader()

        # Write the results
        writer.writerow(results)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate synthetic CT results.")
    parser.add_argument("--dir", type=str, default="prediction", help="Specify PREDICTION_DIR. Default is 'prediction'.")
    parser.add_argument("--case", type=str, help="Specify a case name to evaluate. If not provided, all cases in prediction will be evaluated.")
    parser.add_argument("--csv", type=str, default="evaluation_results", help="Specify the name of the CSV file (default: 'evaluation_results').")
    args = parser.parse_args()

    PREDICTION_DIR = os.path.join(ROOT, "res", args.dir)
    if not os.path.exists(PREDICTION_DIR):
        print(f"Error: The specified prediction directory '{PREDICTION_DIR}' does not exist.")
        sys.exit(1)

    if args.case:
        # Evaluate a single case
        results = evaluate_case(args.case)
        # results = evaluate_case_per_slice(args.case)
        # write_results_to_csv(results, PREDICTION_DIR + "/evaluation_results.csv")
    else:
        # Evaluate all cases in prediction folder
        all_cases = [d for d in os.listdir(PREDICTION_DIR) if os.path.isdir(os.path.join(PREDICTION_DIR, d))]
        for case in all_cases:
            results = evaluate_case(case)
            csv_path = os.path.join(PREDICTION_DIR, f"{args.csv}.csv")
            write_results_to_csv(results, csv_path)
