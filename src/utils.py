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
