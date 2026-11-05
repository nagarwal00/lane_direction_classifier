# src/infer_video_v2.py
"""
V2 video inference pipeline.
Uses the fine-tuned MobileNetV2 with MC Dropout confidence estimation.
Each frame shows:
  - Lane curve overlay (from the classical CV pipeline)
  - Direction verdict label
  - Confidence percentage bar (green = high confidence, red = low confidence)
  - ⚠ UNCERTAIN flag when confidence < 0.80
"""

import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cv_pipeline import process_frame_cv
from cnn_train import build_model, CLASS_NAMES, IDX_TO_CLASS, IMAGE_SIZE, CHECKPOINT_PATH
from augment import get_val_transforms
from uncertainty import run_mc_dropout, enable_dropout
from utils import overlay_verdict_and_curve, smooth_predictions

RESULTS_DIR       = "results"
CONFIDENCE_THRESH = 0.80
MC_PASSES         = 10   # Fewer passes for real-time speed (vs 20 in analysis)


def process_video_v2(
    source: str = "results/sample_driving_input.mp4",
    output_path: str = "results/video_v2_output.mp4",
    headless: bool = True
):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = build_model(num_classes=len(CLASS_NAMES), freeze_backbone=False)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model     = model.to(device)
    transform = get_val_transforms(IMAGE_SIZE)

    if not os.path.exists(source):
        from infer_video import generate_sample_video
        source = generate_sample_video(source)

    cap = cv2.VideoCapture(source)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    history    = deque(maxlen=7)
    frame_count = 0
    t_start     = time.time()

    print(f"Processing V2 video: {source}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 1. Classical CV overlay
        _, _, fit_left, fit_right, fit_center, px_data = process_frame_cv(frame)

        # 2. CNN inference with MC Dropout
        img_np  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor  = transform(image=img_np)["image"].unsqueeze(0)
        mean_probs, std_probs, pred_cls, confidence = run_mc_dropout(
            model, tensor, n_passes=MC_PASSES, device=device
        )

        pred_label = IDX_TO_CLASS[pred_cls]
        history.append(pred_label)
        smoothed   = smooth_predictions(history)

        # 3. Draw lane overlay
        ploty = px_data.get("ploty", np.linspace(280, 710, 100))
        fitx  = None
        if fit_center is not None:
            fitx = fit_center[0] * (ploty ** 2) + fit_center[1] * ploty + fit_center[2]

        verdict_text = f"TURNING {smoothed.upper()}" if smoothed != "straight" else "STRAIGHT"
        out_frame    = overlay_verdict_and_curve(
            frame, ploty, fitx, verdict=verdict_text, color=(0, 255, 0), thickness=4
        )

        # 4. Confidence bar overlay
        bar_w     = 200
        bar_h     = 18
        bar_x     = w - bar_w - 20
        bar_y     = 60
        fill_w    = int(bar_w * confidence)
        bar_color = (0, 200, 0) if confidence >= CONFIDENCE_THRESH else (0, 50, 220)

        cv2.rectangle(out_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        cv2.rectangle(out_frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(out_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
        cv2.putText(
            out_frame,
            f"Confidence: {confidence:.0%}",
            (bar_x, bar_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
        )

        if confidence < CONFIDENCE_THRESH:
            cv2.putText(
                out_frame, "! UNCERTAIN",
                (bar_x, bar_y + bar_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2
            )

        out_writer.write(out_frame)
        if not headless:
            cv2.imshow("V2 Lane Classifier", out_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out_writer.release()
    if not headless:
        cv2.destroyAllWindows()

    fps_achieved = frame_count / (time.time() - t_start)
    print(f"V2 video complete. Frames: {frame_count}, FPS: {fps_achieved:.1f}")
    print(f"Saved: {output_path}")
    return fps_achieved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="results/sample_driving_input.mp4")
    parser.add_argument("--output", default="results/video_v2_output.mp4")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    process_video_v2(source=args.source, output_path=args.output, headless=args.headless)

