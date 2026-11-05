import os
import sys
import json
import cv2
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import ensure_directories, overlay_verdict_and_curve

PARSED_PARQUET = "data/labels/parsed_annotations.parquet"
PARSED_CSV = "data/labels/parsed_annotations.csv"
OUTPUT_LABELS_CSV = "data/labels/labels_with_direction.csv"
THRESHOLDS_JSON = "models/label_thresholds.json"
GT_CHECK_DIR = "results/sample_outputs/gt_check"

def identify_ego_lanes(lanes: list, h_samples: list, image_width: int = 1280):
    """
    Select the left and right boundary lanes of the ego vehicle.
    Determined by finding the pair of lanes at the bottom-most valid h_sample
    that straddle the horizontal center (image_width / 2).
    """
    center_x = image_width / 2.0
    
    # Find bottom-most row index with at least two valid lane annotations
    left_candidate = None
    right_candidate = None
    
    # Iterate from bottom row upwards
    for row_idx in range(len(h_samples) - 1, -1, -1):
        valid_lanes_at_row = []
        for lane_idx, lane in enumerate(lanes):
            x_val = lane[row_idx]
            if x_val != -2:
                valid_lanes_at_row.append((x_val, lane_idx))
                
        if len(valid_lanes_at_row) >= 2:
            # Sort by x coordinate
            valid_lanes_at_row.sort(key=lambda item: item[0])
            
            # Find the two lanes that straddle center_x (one < center_x, one > center_x)
            lefts = [item for item in valid_lanes_at_row if item[0] <= center_x]
            rights = [item for item in valid_lanes_at_row if item[0] > center_x]
            
            if lefts and rights:
                left_candidate = lanes[lefts[-1][1]]   # closest left to center
                right_candidate = lanes[rights[0][1]]  # closest right to center
                break
            elif len(valid_lanes_at_row) >= 2:
                # Fallback: take the two middle-most lanes
                # e.g., if vehicle is slightly offset
                valid_lanes_at_row.sort(key=lambda item: abs(item[0] - center_x))
                lane_a = lanes[valid_lanes_at_row[0][1]]
                lane_b = lanes[valid_lanes_at_row[1][1]]
                # Assign left/right based on x at bottom
                if lane_a[row_idx] < lane_b[row_idx]:
                    left_candidate, right_candidate = lane_a, lane_b
                else:
                    left_candidate, right_candidate = lane_b, lane_a
                break
                
    return left_candidate, right_candidate

def compute_ground_truth_labeling():
    """
    Generate ground-truth direction labels (left, straight, right) for each image.
    Uses tertile percentiles on quadratic polynomial coefficient A.
    """
    ensure_directories()
    
    if os.path.exists(PARSED_PARQUET):
        df_annotations = pd.read_parquet(PARSED_PARQUET)
    elif os.path.exists(PARSED_CSV):
        df_annotations = pd.read_csv(PARSED_CSV)
    else:
        raise FileNotFoundError(f"Parsed annotations file not found at {PARSED_PARQUET} or {PARSED_CSV}")
        
    print(f"Loaded {len(df_annotations)} parsed annotations.")
    
    results = []
    a_coefficients = []
    valid_indices = []
    
    for idx, row in df_annotations.iterrows():
        image_filename = row["image_filename"]
        lanes = json.loads(row["lanes"])
        h_samples = json.loads(row["h_samples"])
        
        left_lane, right_lane = identify_ego_lanes(lanes, h_samples)
        
        if left_lane is None or right_lane is None:
            results.append({
                "image_filename": image_filename,
                "A_gt": None,
                "direction": "insufficient_annotation"
            })
            continue
            
        # Collect (y, x_center) points where both boundaries are valid (!= -2)
        center_points = []
        for i in range(len(h_samples)):
            y_val = h_samples[i]
            xl = left_lane[i]
            xr = right_lane[i]
            if xl != -2 and xr != -2:
                x_c = (xl + xr) / 2.0
                center_points.append((y_val, x_c))
                
        if len(center_points) < 3:
            results.append({
                "image_filename": image_filename,
                "A_gt": None,
                "direction": "insufficient_annotation"
            })
            continue
            
        # Fit 2nd-degree polynomial: x(y) = A*y^2 + B*y + C
        center_points = np.array(center_points)
        y_arr = center_points[:, 0]
        x_arr = center_points[:, 1]
        
        poly = np.polyfit(y_arr, x_arr, 2)
        A = poly[0]
        
        a_coefficients.append(A)
        valid_indices.append(idx)
        
        results.append({
            "image_filename": image_filename,
            "A_gt": A,
            "poly_A": poly[0],
            "poly_B": poly[1],
            "poly_C": poly[2],
            "direction": None  # to be assigned via tertiles
        })
        
    # Calculate tertile thresholds (33rd and 67th percentiles)
    a_arr = np.array(a_coefficients)
    lower_threshold = float(np.percentile(a_arr, 33.33))
    upper_threshold = float(np.percentile(a_arr, 66.67))
    
    print(f"\nEmpirical Polynomial Coefficient A Tertiles:")
    print(f"  33.3rd percentile (Lower Threshold): {lower_threshold:.8f}")
    print(f"  66.7th percentile (Upper Threshold): {upper_threshold:.8f}")
    
    # Save thresholds to JSON
    threshold_data = {
        "lower_threshold": lower_threshold,
        "upper_threshold": upper_threshold,
        "num_valid_samples": len(a_coefficients)
    }
    with open(THRESHOLDS_JSON, "w") as f:
        json.dump(threshold_data, f, indent=4)
    print(f"Saved threshold parameters to {THRESHOLDS_JSON}")
    
    # Assign labels based on A tertiles
    # Image coordinates: y increases downwards from top (0) to bottom (720).
    # Moving up the road (towards top of image, decreasing y):
    # A < lower_threshold => x decreases as y decreases => road curves left
    # A > upper_threshold => x increases as y decreases => road curves right
    for item in results:
        if item["A_gt"] is not None:
            A_val = item["A_gt"]
            if A_val < lower_threshold:
                item["direction"] = "left"
            elif A_val > upper_threshold:
                item["direction"] = "right"
            else:
                item["direction"] = "straight"
                
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_LABELS_CSV, index=False)
    
    # Print class distribution summary
    dist = df_results["direction"].value_counts(normalize=True) * 100
    counts = df_results["direction"].value_counts()
    
    print(f"\n--- Ground-Truth Class Distribution ---")
    for cls_name, count in counts.items():
        pct = dist[cls_name]
        print(f"  {cls_name.upper():<25}: {count:4d} samples ({pct:.1f}%)")
        
    # Visual ground-truth sanity check for 10 sample images
    save_ground_truth_check_overlays(df_results)
    
    return df_results

def save_ground_truth_check_overlays(df_results: pd.DataFrame, num_samples: int = 10):
    """
    Saves visual sanity check overlays for sample left, right, and straight images.
    """
    os.makedirs(GT_CHECK_DIR, exist_ok=True)
    
    valid_df = df_results[df_results["direction"].isin(["left", "straight", "right"])]
    if len(valid_df) == 0:
        return
        
    # Sample balanced set of images
    sampled = []
    for d in ["left", "straight", "right"]:
        sub = valid_df[valid_df["direction"] == d]
        n_to_pick = min(4, len(sub))
        if n_to_pick > 0:
            sampled.append(sub.sample(n=n_to_pick, random_state=42))
            
    if sampled:
        sampled_df = pd.concat(sampled).head(num_samples)
    else:
        sampled_df = valid_df.head(num_samples)
        
    print(f"\nSaving {len(sampled_df)} visual GT check overlays to {GT_CHECK_DIR}...")
    
    for _, row in sampled_df.iterrows():
        img_filename = row["image_filename"]
        img_path = os.path.join("data/images", img_filename)
        direction = row["direction"]
        
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Draw GT polynomial curve
        poly_A = row["poly_A"]
        poly_B = row["poly_B"]
        poly_C = row["poly_C"]
        
        ploty = np.linspace(280, 710, 100)
        fitx = poly_A * (ploty ** 2) + poly_B * ploty + poly_C
        
        overlay = overlay_verdict_and_curve(
            img, ploty, fitx, verdict=f"GT: {direction}", color=(0, 255, 0), thickness=5
        )
        
        save_path = os.path.join(GT_CHECK_DIR, f"gt_check_{direction}_{img_filename}")
        cv2.imwrite(save_path, overlay)
        
    print(f"GT Check overlays saved successfully.")

if __name__ == "__main__":
    compute_ground_truth_labeling()
