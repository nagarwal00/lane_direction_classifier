import os
import sys
import cv2
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cv_pipeline import process_frame_cv

FEATURE_NAMES = [
    "A_fit",
    "B_fit",
    "delta_x_top_bottom",
    "mean_slope",
    "left_pixel_count",
    "right_pixel_count",
    "pixel_count_asymmetry",
    "lane_center_offset",
    "std_x_per_row"
]

LABELS_CSV = "data/labels/labels_with_direction.csv"
OUTPUT_FEATURES_CSV = "data/labels/features.csv"
FAILURES_TXT = "results/feature_extraction_failures.txt"

def extract_features(image: np.ndarray):
    """
    Extract 9 hand-engineered classical CV features from a road image.
    Returns (feature_vector: np.ndarray, feature_names: list[str]) or (None, None) if detection fails.
    """
    height, width = image.shape[:2]
    binary_mask, masked_roi, fit_left, fit_right, fit_center, px_data = process_frame_cv(image)
    
    if fit_center is None:
        return None, FEATURE_NAMES
        
    A_fit, B_fit, C_fit = fit_center[0], fit_center[1], fit_center[2]
    
    # 1. Delta x between top and bottom of ROI
    y_top = float(height * 0.45)
    y_bottom = float(height - 1)
    
    x_top = A_fit * (y_top ** 2) + B_fit * y_top + C_fit
    x_bottom = A_fit * (y_bottom ** 2) + B_fit * y_bottom + C_fit
    delta_x = x_top - x_bottom
    
    # 2. Mean derivative dx/dy = 2*A*y + B across y-range
    y_mid = (y_top + y_bottom) / 2.0
    mean_slope = 2.0 * A_fit * y_mid + B_fit
    
    # 3. Pixel counts
    lefty, leftx = px_data["lefty"], px_data["leftx"]
    righty, rightx = px_data["righty"], px_data["rightx"]
    
    left_count = float(len(leftx))
    right_count = float(len(rightx))
    
    # 4. Pixel count asymmetry
    eps = 1e-5
    asymmetry = (right_count - left_count) / (right_count + left_count + eps)
    
    # 5. Lane center offset from image center (width / 2)
    lane_center_bottom = x_bottom
    image_center_x = width / 2.0
    lane_center_offset = lane_center_bottom - image_center_x
    
    # 6. Scatter measure: std of detected pixel x-positions
    all_x = np.concatenate([leftx, rightx]) if (len(leftx) > 0 or len(rightx) > 0) else np.array([0.0])
    std_x = float(np.std(all_x)) if len(all_x) > 0 else 0.0
    
    feature_vector = np.array([
        A_fit,
        B_fit,
        delta_x,
        mean_slope,
        left_count,
        right_count,
        asymmetry,
        lane_center_offset,
        std_x
    ], dtype=float)
    
    return feature_vector, FEATURE_NAMES

def build_dataset_features():
    """
    Extracts features for all images in data/images/ and saves data/labels/features.csv.
    """
    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError(f"Labels CSV not found at {LABELS_CSV}. Run src/labeling.py first.")
        
    df_labels = pd.read_csv(LABELS_CSV)
    print(f"Extracting features for {len(df_labels)} labeled images...")
    
    # Filter out insufficient annotations
    valid_labels = df_labels[df_labels["direction"].isin(["left", "straight", "right"])].copy()
    
    feature_rows = []
    failed_files = []
    
    for idx, row in valid_labels.iterrows():
        img_filename = row["image_filename"]
        img_path = os.path.join("data/images", img_filename)
        
        if not os.path.exists(img_path):
            failed_files.append((img_filename, "Image file not found"))
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            failed_files.append((img_filename, "Failed to read image"))
            continue
            
        feat_vec, feat_names = extract_features(img)
        if feat_vec is None:
            failed_files.append((img_filename, "CV center fit failure"))
            continue
            
        feat_dict = {
            "image_filename": img_filename,
            "direction": row["direction"]
        }
        for name, val in zip(feat_names, feat_vec):
            feat_dict[name] = val
            
        feature_rows.append(feat_dict)
        
    df_features = pd.DataFrame(feature_rows)
    df_features.to_csv(OUTPUT_FEATURES_CSV, index=False)
    
    # Save failures
    os.makedirs(os.path.dirname(FAILURES_TXT), exist_ok=True)
    with open(FAILURES_TXT, "w") as f:
        for fname, reason in failed_files:
            f.write(f"{fname}\t{reason}\n")
            
    total_valid = len(valid_labels)
    success_num = len(feature_rows)
    fail_num = len(failed_files)
    fail_rate = (fail_num / total_valid) * 100 if total_valid > 0 else 0.0
    
    print(f"\n--- Milestone 4 Summary ---")
    print(f"Total input images: {total_valid}")
    print(f"Successfully extracted feature vectors: {success_num} ({100.0 - fail_rate:.1f}%)")
    print(f"Feature extraction failures: {fail_num} ({fail_rate:.2f}%)")
    print(f"Saved feature dataset to: {OUTPUT_FEATURES_CSV}")
    print(f"Logged failures to: {FAILURES_TXT}")
    
    assert fail_rate <= 5.0, f"Failure rate {fail_rate:.2f}% exceeds 5% threshold!"
    return df_features

if __name__ == "__main__":
    build_dataset_features()
