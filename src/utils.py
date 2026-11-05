import os
import cv2
import numpy as np
from collections import Counter

def ensure_directories():
    """Ensure all required project directories exist."""
    dirs = [
        "data/raw",
        "data/images",
        "data/labels",
        "src",
        "models",
        "results/sample_outputs/gt_check",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def get_dir_size_mb(directory_path: str) -> float:
    """Calculate total size of a directory in megabytes."""
    total_size = 0
    for dirpath, _, filenames in os.walk(directory_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

def overlay_verdict_and_curve(
    image: np.ndarray,
    ploty: np.ndarray,
    fit_x: np.ndarray,
    verdict: str,
    color: tuple = (0, 255, 0),
    thickness: int = 4
) -> np.ndarray:
    """
    Draw fitted curve line and text verdict overlay on top of original image.
    """
    out_img = image.copy()
    if fit_x is not None and len(fit_x) > 0 and ploty is not None and len(ploty) > 0:
        pts = np.vstack((fit_x, ploty)).T.astype(np.int32)
        cv2.polylines(out_img, [pts], isClosed=False, color=color, thickness=thickness)
    
    # Verdict text box banner at top left
    banner_height = 60
    cv2.rectangle(out_img, (0, 0), (450, banner_height), (0, 0, 0), -1)
    
    label_text = f"LANE: {verdict.upper()}"
    cv2.putText(
        out_img,
        label_text,
        (15, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    return out_img

def smooth_predictions(history_buffer) -> str:
    """
    Return majority vote prediction from a rolling buffer/deque.
    """
    if not history_buffer:
        return "straight"
    counts = Counter(history_buffer)
    return counts.most_common(1)[0][0]


def compute_radius_meters(A_fit: float,
                          B_fit: float,
                          y_eval_px: int = 360,
                          ym_per_px: float = 30.0 / 720.0,
                          xm_per_px: float = 3.7 / 640.0) -> float:
    """
    Convert pixel-space polynomial coefficients to real-world curvature radius.

    Uses the standard formula:
        R = (1 + (2Ay + B)^2)^1.5 / |2A|
    applied in meter-space (after unit conversion).

    Args:
        A_fit:       Quadratic coefficient from pixel-space polynomial fit.
        B_fit:       Linear coefficient from pixel-space polynomial fit.
        y_eval_px:   y-pixel at which to evaluate radius (default: mid-ROI = 360px).
        ym_per_px:   Meters per pixel in y-direction.
                     TuSimple: 30m visible road / 720px ≈ 0.0417 m/px
        xm_per_px:   Meters per pixel in x-direction.
                     TuSimple: 3.7m lane / 640px ≈ 0.00578 m/px

    Returns:
        Radius in meters. Returns float('inf') for near-straight lines.
    """
    if abs(A_fit) < 1e-10:
        return float('inf')   # Perfectly straight

    # Rescale coefficients to meter space
    # x_m(y_m) = A_m * y_m^2 + B_m * y_m + C_m
    # A_m = A_px * xm_per_px / (ym_per_px^2)
    # B_m = B_px * xm_per_px / ym_per_px
    A_m = A_fit * (xm_per_px / (ym_per_px ** 2))
    B_m = B_fit * (xm_per_px / ym_per_px)
    y_m = y_eval_px * ym_per_px

    numerator = (1 + (2 * A_m * y_m + B_m) ** 2) ** 1.5
    denominator = abs(2 * A_m)

    return numerator / denominator
