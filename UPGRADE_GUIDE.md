# Lane Direction Classifier — Complete Upgrade & Real Dataset Testing Guide

**Purpose:** This document tells you exactly what to add, in what order, with full code, to transform your project from a "basic classifier" into a research-grade ADAS module that will impress any professor.

---

## Table of Contents

1. [What You Already Have (Strengths to Reference)](#1-what-you-already-have)
2. [The Research Narrative (How to Frame It)](#2-the-research-narrative)
3. [Upgrade 1 — Kalman Filter on Lane Curvature](#3-upgrade-1--kalman-filter-on-lane-curvature)
4. [Upgrade 2 — Lane Departure Warning Module](#4-upgrade-2--lane-departure-warning-module)
5. [Upgrade 3 — Curvature Radius in Meters](#5-upgrade-3--curvature-radius-in-meters)
6. [Upgrade 4 — Uncertainty / Fallback Gate](#6-upgrade-4--uncertainty--fallback-gate)
7. [Upgrade 5 — Optical Flow for Ego-Motion Disambiguation](#7-upgrade-5--optical-flow-for-ego-motion-disambiguation)
8. [Real Dataset Testing — CULane](#8-real-dataset-testing--culane)
9. [Real Dataset Testing — IDD (Indian Driving Dataset)](#9-real-dataset-testing--idd-indian-driving-dataset)
10. [Real Dataset Testing — TuSimple Full Set](#10-real-dataset-testing--tusimple-full-set)
11. [Robustness Report — Edge Cases](#11-robustness-report--edge-cases)
12. [FPS Benchmarking Table](#12-fps-benchmarking-table)
13. [Final Folder Structure After All Upgrades](#13-final-folder-structure-after-all-upgrades)
14. [Professor Demo Script (Verbal)](#14-professor-demo-script-verbal)

---

## 1. What You Already Have

Before adding anything, be clear on what you can claim:

| Component | What It Proves |
|---|---|
| HLS + Sobel-X + Sliding Window (V1) | You understand classical CV, not just library calls |
| 9 hand-engineered features | You translated geometry into ML-friendly numbers |
| 5-fold CV + GridSearchCV (V1) | Rigorous model selection, not just `model.fit()` |
| 94.4% accuracy, macro F1 = 0.9436 | Quantified performance on a held-out test set |
| MobileNetV2 transfer learning (V2) | You know when deep learning is appropriate |
| MC Dropout uncertainty (V2) | Probabilistic reasoning, not just argmax |
| Grad-CAM heatmaps (V2) | Spatial explainability beyond feature importances |
| ONNX export + latency benchmark (V2) | You thought about deployment, not just accuracy |

**What is still missing for a "real system":** temporal consistency, physical output units, failure detection, and validation on diverse real-world data. The upgrades below add all of that.

---

## 2. The Research Narrative

Frame your project in exactly this structure when talking to your professor:

> **V1:** Physics-inspired features + supervised ML. Fast, interpretable, no GPU needed.
>
> **V2:** Transfer learning + uncertainty quantification + deployment optimization. Better generalization, knows when it's uncertain.
>
> **V3 (what you are adding now):** Temporal filtering, physical output units, safety logic, and real-world validation on CULane and IDD. This is where it becomes an ADAS module, not just a classifier.

The jump from V2 to V3 is exactly what a professor cannot call "basic."

---

## 3. Upgrade 1 — Kalman Filter on Lane Curvature

### Why This Matters

Your current video pipeline uses a majority vote over a rolling buffer. That is statistically valid but physically wrong — it treats each frame as independent. In reality, `A_fit` (the curvature coefficient) changes smoothly as a car drives. A Kalman filter models this physical continuity.

**What it shows your professor:** You understand Bayesian state estimation, not just buffering.

### Create: `src/kalman_smoother.py`

```python
"""
Kalman Filter for temporal smoothing of lane curvature coefficient A_fit.

State vector: [A_fit, dA_fit/dt]
Observation:  [A_fit]  (we observe curvature each frame, not its derivative)
"""

import numpy as np


class LaneKalmanFilter:
    """
    1D Kalman filter tracking A_fit (quadratic lane curvature coefficient)
    and its first time derivative.

    Usage:
        kf = LaneKalmanFilter()
        smoothed_A = kf.update(raw_A_fit)   # call once per frame
        is_confident = kf.innovation_ok()   # True if measurement is consistent
    """

    def __init__(self, dt: float = 1.0 / 30.0):
        """
        Args:
            dt: Time between frames in seconds. Default assumes 30 FPS video.
        """
        # --- State transition matrix F  (constant-velocity model) ---
        # [A_new]       [1  dt] [A_old      ]
        # [dA_new]  =   [0   1] [dA_old     ]
        self.F = np.array([[1.0, dt],
                           [0.0, 1.0]])

        # --- Observation matrix H  (we observe A directly) ---
        # observed = [1, 0] * [A, dA]^T
        self.H = np.array([[1.0, 0.0]])

        # --- Process noise covariance Q ---
        # Tune q: larger → filter trusts measurements more (more responsive)
        #         smaller → filter trusts model more (smoother but lagging)
        q = 1e-5
        self.Q = np.array([[q, 0.0],
                           [0.0, q]])

        # --- Measurement noise covariance R ---
        # Tune r: larger → measurements are noisy (smoother output)
        #         smaller → measurements are trusted (more reactive)
        self.R = np.array([[1e-4]])

        # --- Initial state and covariance ---
        self.x = np.zeros((2, 1))          # [A_fit=0, dA_fit=0]
        self.P = np.eye(2) * 1.0           # large initial uncertainty
        self.initialized = False
        self._innovation = 0.0

    def update(self, measurement: float) -> float:
        """
        Feed one A_fit measurement, return the Kalman-smoothed A_fit.

        Args:
            measurement: Raw A_fit from polynomial fit this frame.

        Returns:
            Smoothed A_fit value.
        """
        z = np.array([[measurement]])

        if not self.initialized:
            # Seed state with first observation
            self.x[0, 0] = measurement
            self.initialized = True
            return measurement

        # --- Predict ---
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # --- Update (Kalman gain) ---
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        self._innovation = float(z[0, 0] - (self.H @ x_pred)[0, 0])
        self.x = x_pred + K * self._innovation
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        return float(self.x[0, 0])

    def innovation_ok(self, threshold: float = 0.005) -> bool:
        """
        Returns True if the last measurement was consistent with prediction.
        A large innovation (surprise) indicates an unreliable frame.

        Args:
            threshold: Maximum acceptable |innovation| for A_fit.
        """
        return abs(self._innovation) < threshold

    def reset(self):
        """Reset state — call when a new video clip starts."""
        self.x = np.zeros((2, 1))
        self.P = np.eye(2) * 1.0
        self.initialized = False
        self._innovation = 0.0
```

### How to Plug Into `src/infer_video.py`

Find the frame processing loop and replace your rolling-buffer majority vote with:

```python
# At the top of infer_video.py, add:
from src.kalman_smoother import LaneKalmanFilter

# Before the loop:
kf = LaneKalmanFilter(dt=1.0 / fps)

# Inside the loop, after getting raw A_fit from infer_single_image:
raw_A = feature_vector["A_fit"]          # your existing feature extraction
smoothed_A = kf.update(raw_A)            # Kalman-filtered curvature
is_reliable = kf.innovation_ok()

# Use smoothed_A to classify direction instead of raw_A
# If not is_reliable, overlay "UNCERTAIN" instead of a direction label
```

### What to Show the Professor

Display two plots side by side: raw `A_fit` over 100 frames vs. Kalman-smoothed `A_fit`. The smoothed version will clearly track the road curve without noise spikes. This is a one-graph argument that your system is production-ready.

---

## 4. Upgrade 2 — Lane Departure Warning Module

### Why This Matters

Classification is diagnostic. Warning is actionable. This is the actual downstream purpose of lane detection in a car. Adding LDW converts your academic classifier into a safety system.

### Create: `src/ldw.py`

```python
"""
Lane Departure Warning (LDW) Module.

Uses lane_center_offset (already computed by features.py) to trigger
a visual and logged alert when the vehicle drifts toward a lane boundary.
"""

from dataclasses import dataclass
from enum import Enum


class DepartureStatus(Enum):
    SAFE = "SAFE"
    WARNING_LEFT = "WARNING LEFT"
    WARNING_RIGHT = "WARNING RIGHT"
    CRITICAL_LEFT = "CRITICAL LEFT — DEPART"
    CRITICAL_RIGHT = "CRITICAL RIGHT — DEPART"


@dataclass
class LDWResult:
    status: DepartureStatus
    offset_px: float          # Signed pixel offset from lane center (- = left, + = right)
    offset_meters: float      # Same in real-world meters
    alert: bool               # True if driver intervention is recommended


# Physical calibration constants
# Derived from TuSimple: lane width ≈ 3.7m, image width = 1280px
LANE_WIDTH_PX = 640.0           # Half-lane in pixels (image center = lane center assumption)
LANE_WIDTH_METERS = 3.7         # Standard US/global lane width
PIXELS_PER_METER = LANE_WIDTH_PX / (LANE_WIDTH_METERS / 2.0)   # px/m

# Thresholds (tune these)
WARNING_THRESHOLD_M = 0.4       # 40cm from center → warning
CRITICAL_THRESHOLD_M = 0.8      # 80cm from center → critical (nearly at marking)


def compute_ldw(lane_center_offset_px: float,
                direction_prediction: str) -> LDWResult:
    """
    Compute lane departure status from lane_center_offset.

    Args:
        lane_center_offset_px: Output of features.py — signed horizontal
                               offset of detected lane center from image
                               center column (640px). Negative = left drift.
        direction_prediction:  Current classifier verdict ("left"/"straight"/"right")

    Returns:
        LDWResult with status, offset in meters, and alert flag.
    """
    offset_m = lane_center_offset_px / PIXELS_PER_METER
    abs_offset = abs(offset_m)

    # Only warn on straight roads — turning intentionally is not departure
    if direction_prediction != "straight":
        return LDWResult(
            status=DepartureStatus.SAFE,
            offset_px=lane_center_offset_px,
            offset_meters=offset_m,
            alert=False
        )

    if abs_offset < WARNING_THRESHOLD_M:
        status = DepartureStatus.SAFE
        alert = False

    elif abs_offset < CRITICAL_THRESHOLD_M:
        status = DepartureStatus.WARNING_LEFT if offset_m < 0 else DepartureStatus.WARNING_RIGHT
        alert = True

    else:
        status = DepartureStatus.CRITICAL_LEFT if offset_m < 0 else DepartureStatus.CRITICAL_RIGHT
        alert = True

    return LDWResult(
        status=status,
        offset_px=lane_center_offset_px,
        offset_meters=offset_m,
        alert=alert
    )
```

### Display on Video Overlay

In your `infer_video.py` OpenCV overlay code, add:

```python
from src.ldw import compute_ldw

ldw_result = compute_ldw(feature_vector["lane_center_offset"], predicted_direction)

if ldw_result.alert:
    # Draw a red border around the entire frame
    cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1),
                  (0, 0, 255), 8)
    # Add text
    cv2.putText(frame, ldw_result.status.value,
                (20, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
else:
    cv2.putText(frame, "LDW: SAFE",
                (20, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
```

---

## 5. Upgrade 3 — Curvature Radius in Meters

### Why This Matters

Outputting "LEFT" is a class label. Outputting "Turning LEFT — radius ≈ 185m" is a physical measurement. The second one proves you understand the relationship between your math and the real world.

### Add to: `src/utils.py`

```python
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
```

### How to Use

In your overlay or inference output:

```python
from src.utils import compute_radius_meters

radius = compute_radius_meters(A_fit, B_fit)

if radius == float('inf') or radius > 2000:
    radius_text = "STRAIGHT (R > 2000m)"
elif radius > 500:
    radius_text = f"GENTLE CURVE — R ≈ {radius:.0f}m"
else:
    radius_text = f"SHARP CURVE — R ≈ {radius:.0f}m"
```

---

## 6. Upgrade 4 — Uncertainty / Fallback Gate

### Why This Matters

A classifier that always outputs a class is dangerous in a safety system. When pixel detection fails (rain, night, occlusion), your system should refuse to classify rather than guess. This is a core principle of safety-critical software.

### Add to: `src/infer_image.py`

```python
# Minimum pixel thresholds for a reliable detection
MIN_LANE_PIXELS_TOTAL = 150    # Total left + right pixels detected
MIN_ASYMMETRY_RATIO = 0.98     # If 98%+ of pixels are on one side, something is wrong

def is_detection_reliable(feature_vector: dict) -> tuple[bool, str]:
    """
    Gate: check if lane pixel detection is sufficient to trust the classifier.

    Returns:
        (reliable: bool, reason: str)
    """
    left_px = feature_vector.get("left_pixel_count", 0)
    right_px = feature_vector.get("right_pixel_count", 0)
    total = left_px + right_px

    if total < MIN_LANE_PIXELS_TOTAL:
        return False, f"INSUFFICIENT PIXELS ({total} < {MIN_LANE_PIXELS_TOTAL})"

    if total > 0:
        asymmetry = abs(left_px - right_px) / total
        if asymmetry > MIN_ASYMMETRY_RATIO:
            return False, f"EXTREME ASYMMETRY ({asymmetry:.2f}) — partial occlusion"

    return True, "OK"
```

In your main inference function, add before the classifier call:

```python
reliable, reason = is_detection_reliable(features)
if not reliable:
    return "uncertain", 0.0, {"reason": reason}
```

---

## 7. Upgrade 5 — Optical Flow for Ego-Motion Disambiguation

### Why This Matters

If the driver turns their head sharply while driving straight, the lane markings briefly appear to curve in the image — but the road is still straight. Optical flow measures how the entire scene is moving between frames. If global motion is large, the apparent curvature is due to ego-motion, not road geometry.

### Add to: `src/infer_video.py`

```python
import cv2
import numpy as np

class EgoMotionDetector:
    """
    Uses sparse Lucas-Kanade optical flow on background (sky/road) region
    to detect rapid camera motion that could cause false curvature readings.
    """

    def __init__(self, threshold_px: float = 15.0):
        """
        Args:
            threshold_px: Mean optical flow magnitude above which ego-motion
                          is considered significant (in pixels/frame).
        """
        self.prev_gray = None
        self.threshold = threshold_px
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

    def detect_motion(self, frame_bgr: np.ndarray) -> tuple[bool, float]:
        """
        Args:
            frame_bgr: Current BGR frame from video.

        Returns:
            (high_motion: bool, mean_flow_magnitude: float)
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = gray[:200, :]  # Use only sky/horizon region (top 200 rows)

        if self.prev_gray is None:
            self.prev_gray = gray
            return False, 0.0

        # Detect feature points in previous frame
        pts = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=50,
                                      qualityLevel=0.3, minDistance=10)

        if pts is None or len(pts) < 5:
            self.prev_gray = gray
            return False, 0.0

        # Track them in current frame
        pts_new, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, pts, None, **self.lk_params
        )

        good_old = pts[status.ravel() == 1]
        good_new = pts_new[status.ravel() == 1]

        if len(good_old) < 5:
            self.prev_gray = gray
            return False, 0.0

        flow = good_new - good_old
        magnitudes = np.linalg.norm(flow, axis=2).ravel()
        mean_mag = float(np.mean(magnitudes))

        self.prev_gray = gray
        return mean_mag > self.threshold, mean_mag
```

**How to use:** If `high_motion=True`, append "(MOTION BLUR — LOW CONFIDENCE)" to the overlay text and reduce the LDW alert sensitivity for that frame.

---

## 8. Real Dataset Testing — CULane

### What CULane Is

CULane is the most widely used real-world lane detection benchmark. It has 133,235 frames across 9 driving scenarios including night driving, rain, crowded urban roads, and highway. This is the right dataset to prove your pipeline works (or identify where it fails) on diverse real conditions.

### Download Instructions

```bash
# Step 1: Register at the CULane project page
# URL: https://xingangpan.github.io/projects/CULane.html
# Click "Download" — requires a brief registration form (free, academic use)

# Step 2: You will get these parts (download all):
#   driver_23_30frame.tar.gz    (normal highway driving)
#   driver_161_90frame.tar.gz   (urban roads)
#   driver_182_30frame.tar.gz   (urban roads)
#   driver_193_90frame.tar.gz   (mixed)
#   driver_100_30frame.tar.gz   (challenging)
#   laneseg_label_w16.tar.gz    (ground-truth lane segmentation masks)
#   list.tar.gz                 (train/val/test splits + scenario labels)

# Step 3: Extract all into data/culane/
mkdir -p data/culane
tar -xzf driver_23_30frame.tar.gz    -C data/culane/
tar -xzf driver_161_90frame.tar.gz   -C data/culane/
tar -xzf laneseg_label_w16.tar.gz    -C data/culane/
tar -xzf list.tar.gz                 -C data/culane/
```

### Create: `src/culane_eval.py`

```python
"""
Evaluates V1 classical ML pipeline on CULane dataset images.
Outputs per-scenario accuracy breakdown to quantify real-world generalization.
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
```

### What to Report

After running, your summary table will look like:

```
Scenario     | Total | Uncertain% | Mean Confidence
-------------|-------|------------|----------------
normal       |  100  |    8.0%    |    91.2%
night        |  100  |   47.0%    |    61.4%     ← gap = research finding
rain/crowded |  100  |   35.0%    |    68.3%     ← gap = research finding
curve        |  100  |   22.0%    |    79.1%
```

This degradation on night and rain images is not a failure — it is a **research finding** that justifies future work (retrain on augmented data, switch to V2 for hard cases). Your professor will respect this more than a system that always outputs a confident but wrong answer.

---

## 9. Real Dataset Testing — IDD (Indian Driving Dataset)

### Why IDD Specifically

IDD is shot on Indian roads: unstructured lanes, faded markings, mixed traffic, potholes, construction zones. These are the exact conditions your system will face if deployed in India. Testing on IDD and showing degradation is a strong argument for why the problem is hard, and why your work matters locally.

### Download Instructions

```bash
# Step 1: Register at IDD portal (free, requires institute email)
# URL: https://idd.insaan.iiit.ac.in
# Click "IDD Detection" or "IDD Segmentation" and register

# Step 2: Download IDD20k Part I or II (dashcam images, ~5GB each)
#   IDD20k_II.tar.gz   — has better lane annotation coverage

# Step 3: Extract
mkdir -p data/idd
tar -xzf IDD20k_II.tar.gz -C data/idd/

# IDD structure after extraction:
# data/idd/
#   ├── leftImg8bit/
#   │   ├── train/
#   │   ├── val/
#   │   └── test/
#   └── gtFine/   (segmentation labels — optional for us)
```

### Create: `src/idd_eval.py`

```python
"""
Run V1 inference on IDD dashcam images.
IDD does not provide lane-direction labels, so we measure:
  - Detection rate (how often does our CV pipeline find any lane pixels?)
  - Confidence distribution
  - Failure mode examples (save lowest-confidence images for the report)
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
```

---

## 10. Real Dataset Testing — TuSimple Full Set

You are already using a 500-image TuSimple subset. The full training set has 3,626 labeled clips (~60,000 frames). Testing on the full set quantifies how much your 500-image model generalizes.

### Download

```bash
# Full TuSimple training set (requires registration)
# URL: https://github.com/TuSimple/tusimple-benchmark/issues/3
# Download: train_set.zip (~965MB)

unzip train_set.zip -d data/tusimple_full/

# Structure:
# data/tusimple_full/
#   ├── clips/
#   │   └── <clip_dirs>/
#   └── label_data_0313.json  (and other JSON files)
```

### Run Existing Pipeline on Full Set

Just update the path in `src/data_prep.py`:

```python
# Change this constant:
TUSIMPLE_DATA_ROOT = "data/tusimple_full"   # was "data/raw"
MAX_IMAGES = 1000   # Sample 1000 from the full set for a fair extended test
```

Then re-run the pipeline from Milestone 1 onwards. Compare the new test accuracy to your 94.4% baseline.

---

## 11. Robustness Report — Edge Cases

### Create: `results/robustness_report.md`

After running CULane and IDD evaluations, fill in this template:

```markdown
# Robustness Analysis — Lane Classifier V1

## Test Conditions and Detection Rates

| Condition | Dataset | Total Images | Detection Rate | Mean Confidence |
|---|---|---|---|---|
| Normal highway (day) | TuSimple 500 | 500 | 100% | 94.4% |
| Normal highway (day) | CULane Normal | 100 | XX% | XX% |
| Night driving | CULane Night | 100 | XX% | XX% |
| Crowded urban | CULane Crowded | 100 | XX% | XX% |
| Faded/no markings | CULane No-Line | 100 | XX% | XX% |
| Indian unstructured roads | IDD Val | 200 | XX% | XX% |

## Root-Cause Analysis of Failures

### Failure Type 1: Night Driving
**Symptom:** HLS thresholding detects near-zero lane pixels.
**Root Cause:** White/yellow lane markings in HLS space lose saturation 
under artificial lighting with inconsistent colour temperature.
**Proposed Fix (V3):** Add adaptive histogram equalization (CLAHE) as a 
pre-processing step before HLS thresholding.

### Failure Type 2: Faded Markings
**Symptom:** Sliding window finds insufficient pixels on one side.
**Root Cause:** Gradient threshold too high for low-contrast markings.
**Proposed Fix (V3):** Dynamic threshold selection based on local ROI 
contrast (Otsu's method on the ROI binary mask).

### Failure Type 3: Indian Road Conditions
**Symptom:** High uncertain rate on IDD images.
**Root Cause:** IDD roads frequently have no lane markings, random paint, 
or dual-use surfaces (pedestrians + vehicles) that confuse the ROI assumption.
**Proposed Fix (V3):** Retrain labeling and features on IDD-specific clips 
using weakly supervised labeling.

## Conclusion
V1 achieves 94.4% on clean highway data but degrades significantly under
night (-XX%), rain (-XX%), and Indian road conditions (-XX%). This performance
gap motivates the V2 deep learning upgrade and future work on domain adaptation.
```

---

## 12. FPS Benchmarking Table

Create this in `results/fps_benchmark.md` after running all three pipelines:

```bash
# To measure FPS, add this to any infer script:
import time
start = time.perf_counter()
# ... your inference call ...
elapsed = time.perf_counter() - start
fps = 1.0 / elapsed
print(f"FPS: {fps:.1f}")
```

Target table (fill in your actual numbers):

```markdown
# Inference Latency Benchmark

Hardware: [Your CPU — e.g., Intel i5-11th gen, 8GB RAM]
Image Size: 1280×720

| Pipeline | Mean Latency (ms) | FPS | GPU Required? | Notes |
|---|---|---|---|---|
| V1 Scikit (CV + GBM) | XX ms | XX fps | No | Full pipeline |
| V2 PyTorch MobileNetV2 | XX ms | XX fps | Optional | GPU speeds up 3× |
| V2 ONNX Runtime | XX ms | XX fps | No | Deployable on edge device |
| V1 + Kalman + LDW | XX ms | XX fps | No | Full safety module |

**Reference:** ADAS minimum requirement ≈ 10 FPS at 30 km/h (3m/frame).
V1 at [XX] FPS satisfies this on a standard laptop CPU.
```

---

## 13. Final Folder Structure After All Upgrades

```
lane-direction-classifier/
├── src/
│   ├── kalman_smoother.py          ← NEW (Upgrade 1)
│   ├── ldw.py                      ← NEW (Upgrade 2)
│   ├── culane_eval.py              ← NEW (Real Dataset)
│   ├── idd_eval.py                 ← NEW (Real Dataset)
│   └── ... (all existing scripts)
│
├── data/
│   ├── images/                     (existing TuSimple 500)
│   ├── culane/                     ← NEW (CULane dataset)
│   └── idd/                        ← NEW (IDD dataset)
│
├── results/
│   ├── culane_scenario_results.csv ← NEW
│   ├── idd_results.csv             ← NEW
│   ├── idd_failures/               ← NEW (10 worst images for report)
│   ├── robustness_report.md        ← NEW
│   ├── fps_benchmark.md            ← NEW
│   └── ... (all existing results)
│
└── UPGRADE_GUIDE.md                ← This file
```

---

## 14. Professor Demo Script (Verbal)

Use this sequence when presenting live:

**Opening (30 seconds):**
> "This started as a lane direction classifier. V1 uses classical computer vision — HLS thresholding, edge detection, and sliding windows — to extract 9 geometric features from lane markings, then classifies direction with 94.4% accuracy. No GPU, no pretrained weights, fully interpretable."

**V2 Introduction (30 seconds):**
> "V2 replaces hand-engineered features with MobileNetV2 transfer learning. We added Monte Carlo Dropout so the system can express uncertainty — if it predicts LEFT with 65% confidence across 20 stochastic passes, that's a meaningfully weaker prediction than 95% confidence. Grad-CAM shows *where* in the image the model is looking."

**Safety Module (1 minute):**
> "But classification alone isn't a safety system. We added three things: a Kalman filter that tracks curvature as a physical state across frames instead of treating each frame independently; a Lane Departure Warning that uses the lane center offset we already compute to trigger a visual alert when the vehicle drifts; and an uncertainty gate that refuses to classify when lane pixel count drops below a minimum threshold — because outputting a confident wrong answer on a night road is worse than saying 'uncertain'."

**Real Data Results (1 minute):**
> "We tested on CULane, which has 133,000 real dashcam frames across 9 scenarios. On normal highway it performs at [X%] detection rate. On night and rain it degrades to [Y%] — which we expected, because HLS thresholding is sensitive to lighting. On IDD, the Indian road dataset from IIIT Hyderabad, we see further degradation because many Indian roads have no visible lane markings at all. These failure modes are documented in the robustness report and directly motivate V3: domain-adaptive retraining on IDD."

**Closing (20 seconds):**
> "So the trajectory is: interpretable baseline → explainable deep learning → safety-critical temporal reasoning → real-world validation. That's a complete ADAS research arc, not just a classifier."

---

*End of UPGRADE_GUIDE.md*
*Last updated: November 2026*
