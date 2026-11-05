"""
Evaluates V1 classical ML pipeline on CULane dataset images.
Outputs per-scenario accuracy breakdown to quantify real-world generalization.

NOTE: CULane dataset must be downloaded separately.
  Registration: https://xingangpan.github.io/projects/CULane.html
  After download, extract all parts into data/culane/:
    tar -xzf driver_23_30frame.tar.gz    -C data/culane/
    tar -xzf driver_161_90frame.tar.gz   -C data/culane/
    tar -xzf laneseg_label_w16.tar.gz    -C data/culane/
    tar -xzf list.tar.gz                 -C data/culane/
"""

import os
import cv2
import pandas as pd
from tqdm import tqdm
from src.infer_image import infer_single_image

# CULane scenario categories from the list files
SCENARIO_LIST_FILES = {
    "normal":      "data/culane/list/test_split/test0_normal.txt",
    "crowded":     "data/culane/list/test_split/test1_crowd.txt",
    "night":       "data/culane/list/test_split/test2_night.txt",
    "no_line":     "data/culane/list/test_split/test3_noline.txt",
    "shadow":      "data/culane/list/test_split/test4_shadow.txt",
    "arrow":       "data/culane/list/test_split/test5_arrow.txt",
    "dazzle":      "data/culane/list/test_split/test6_dazzle.txt",
    "curve":       "data/culane/list/test_split/test7_curve.txt",
    "crossroad":   "data/culane/list/test_split/test8_crossroad.txt",
}

CULANE_ROOT = "data/culane"
MAX_IMAGES_PER_SCENARIO = 100   # Limit for speed; increase for full eval


def run_culane_evaluation(output_csv: str = "results/culane_scenario_results.csv"):
    """
    Runs V1 inference on CULane test images, grouped by scenario.
    Records prediction, confidence, pixel reliability flag per image.
    Does NOT use CULane's direction labels (they don't exist in the same
    format) — instead records detection success rate and confidence distribution.
    """
    os.makedirs("results", exist_ok=True)
    records = []

    for scenario, list_path in SCENARIO_LIST_FILES.items():
        if not os.path.exists(list_path):
            print(f"[SKIP] List file not found: {list_path}")
            continue

        with open(list_path, "r") as f:
            image_paths = [line.strip() for line in f.readlines()]

        image_paths = image_paths[:MAX_IMAGES_PER_SCENARIO]
        success_count = 0
        uncertain_count = 0

        print(f"\n[CULane Scenario: {scenario.upper()}]  ({len(image_paths)} images)")

        for rel_path in tqdm(image_paths, desc=scenario):
            full_path = os.path.join(CULANE_ROOT, rel_path.lstrip("/"))
            if not os.path.exists(full_path):
                continue

            verdict, confidence, meta = infer_single_image(
                image_path=full_path,
                save_output=False,
                headless=True
            )

            if verdict == "uncertain":
                uncertain_count += 1
                record_status = "uncertain"
            else:
                success_count += 1
                record_status = verdict

            records.append({
                "scenario": scenario,
                "image": os.path.basename(full_path),
                "verdict": record_status,
                "confidence": confidence,
            })

        total = len(image_paths)
        print(f"  Detection Success : {success_count}/{total} ({100*success_count/max(total,1):.1f}%)")
        print(f"  Uncertain/Failed  : {uncertain_count}/{total} ({100*uncertain_count/max(total,1):.1f}%)")

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\n[DONE] Results saved to {output_csv}")

    # Print summary table
    summary = df.groupby("scenario").apply(
        lambda g: pd.Series({
            "total": len(g),
            "uncertain_pct": 100 * (g["verdict"] == "uncertain").sum() / len(g),
            "mean_confidence": g[g["verdict"] != "uncertain"]["confidence"].mean(),
        })
    ).round(2)
    print("\n--- CULane Scenario Summary ---")
    print(summary.to_string())
    return df


if __name__ == "__main__":
    run_culane_evaluation()

