"""
Run V1 inference on IDD dashcam images.
IDD does not provide lane-direction labels, so we measure:
  - Detection rate (how often does our CV pipeline find any lane pixels?)
  - Confidence distribution
  - Failure mode examples (save lowest-confidence images for the report)

NOTE: IDD dataset must be downloaded separately.
  Registration: https://idd.insaan.iiit.ac.in  (free, requires institute email)
  Download IDD20k Part II (~5GB), then extract:
    tar -xzf IDD20k_II.tar.gz -C data/idd/
  Expected structure after extraction:
    data/idd/leftImg8bit/val/<city>/*.png
"""

import os
import glob
import cv2
import shutil
import pandas as pd
from tqdm import tqdm
from src.infer_image import infer_single_image

IDD_IMAGE_DIR = "data/idd/leftImg8bit/val"   # Use val split
MAX_IMAGES = 200
FAILURE_OUTPUT_DIR = "results/idd_failures"
os.makedirs(FAILURE_OUTPUT_DIR, exist_ok=True)


def run_idd_evaluation(output_csv: str = "results/idd_results.csv"):
    """
    Evaluate V1 on IDD validation images.
    Saves the 10 worst-confidence images to results/idd_failures/ for
    manual analysis of failure modes.
    """
    all_images = sorted(glob.glob(os.path.join(IDD_IMAGE_DIR, "**", "*.png"),
                                   recursive=True))[:MAX_IMAGES]

    if not all_images:
        print(f"No images found in {IDD_IMAGE_DIR}")
        print("Check your IDD extraction path.")
        return None

    records = []
    print(f"[IDD Evaluation] Processing {len(all_images)} images...\n")

    for img_path in tqdm(all_images):
        verdict, confidence, meta = infer_single_image(
            image_path=img_path,
            save_output=False,
            headless=True
        )
        records.append({
            "image": img_path,
            "verdict": verdict,
            "confidence": confidence,
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

    # Save failure examples (bottom 10 by confidence, excluding uncertain)
    detected = df[df["verdict"] != "uncertain"].sort_values("confidence")
    failures = detected.head(10)
    for _, row in failures.iterrows():
        dest = os.path.join(FAILURE_OUTPUT_DIR, os.path.basename(row["image"]))
        shutil.copy(row["image"], dest)

    # Print summary
    total = len(df)
    uncertain = (df["verdict"] == "uncertain").sum()
    print(f"\n--- IDD Evaluation Summary ---")
    print(f"Total Images       : {total}")
    print(f"Detection Rate     : {100*(total-uncertain)/total:.1f}%")
    print(f"Uncertain/Failed   : {uncertain} ({100*uncertain/total:.1f}%)")
    print(f"Mean Confidence    : {df[df['verdict'] != 'uncertain']['confidence'].mean():.1f}%")
    print(f"Direction Split    : {df['verdict'].value_counts().to_dict()}")
    print(f"Failure images saved to: {FAILURE_OUTPUT_DIR}/")

    return df


if __name__ == "__main__":
    run_idd_evaluation()

