import os
import sys
import time
import argparse
import joblib
import cv2
import numpy as np
from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cv_pipeline import process_frame_cv
from features import extract_features
from utils import overlay_verdict_and_curve, smooth_predictions

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "lane_direction_classifier.joblib")
SCALER_FILE = os.path.join(MODEL_DIR, "feature_scaler.joblib")
ENCODER_FILE = os.path.join(MODEL_DIR, "label_encoder.joblib")

def generate_sample_video(output_video_path: str = "results/sample_driving_input.mp4", num_frames: int = 150):
    """
    Generates a synthetic MP4 driving video sequence transitioning through curves
    (Straight -> Left Curve -> Straight -> Right Curve) to test real-time video processing.
    """
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (1280, 720))
    
    h_samples = list(range(160, 720, 10))
    np.random.seed(100)
    
    for f_idx in range(num_frames):
        # Vary curvature over time to simulate driving along a curving road
        if f_idx < 40:
            A = 0.0000
        elif f_idx < 80:
            progress = (f_idx - 40) / 40.0
            A = -0.0006 * progress  # Left curve
        elif f_idx < 110:
            progress = (f_idx - 80) / 30.0
            A = -0.0006 * (1.0 - progress)  # Straightening out
        else:
            progress = (f_idx - 110) / 40.0
            A = 0.0006 * progress  # Right curve
            
        center_x = 640 + 20 * np.sin(f_idx / 10.0)
        lane_width = 380
        
        y_pts = np.array(h_samples, dtype=float)
        y_rel = y_pts - 720
        x_c = A * (y_rel ** 2) + center_x
        
        scale = 0.3 + 0.7 * (y_pts - 160) / (720 - 160)
        w_y = lane_width * scale
        
        xl_pts = x_c - w_y / 2.0
        xr_pts = x_c + w_y / 2.0
        
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        img[300:, :] = (50, 50, 50)
        img[:300, :] = (120, 90, 70)
        cv2.line(img, (0, 300), (1280, 300), (100, 80, 60), 2)
        
        noise = np.random.normal(0, 6, (420, 1280, 3)).astype(np.int16)
        road = np.clip(img[300:, :].astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img[300:, :] = road
        
        pts_l = np.array([(int(xl), int(y)) for xl, y in zip(xl_pts, h_samples) if 10 <= xl <= 1270 and y >= 280], dtype=np.int32)
        pts_r = np.array([(int(xr), int(y)) for xr, y in zip(xr_pts, h_samples) if 10 <= xr <= 1270 and y >= 280], dtype=np.int32)
        
        if len(pts_l) > 1:
            cv2.polylines(img, [pts_l], isClosed=False, color=(0, 255, 255), thickness=10)
        if len(pts_r) > 1:
            cv2.polylines(img, [pts_r], isClosed=False, color=(255, 255, 255), thickness=10)
            
        out.write(img)
        
    out.release()
    print(f"Generated test driving video at: {output_video_path}")
    return output_video_path

def process_video_stream(
    source: str = "results/sample_driving_input.mp4",
    output_path: str = "results/video_verdict_output.mp4",
    headless: bool = False,
    window_size: int = 10
):
    """
    Frame-by-frame real-time video lane direction classification with temporal smoothing.
    """
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file not found at {MODEL_FILE}. Run src/train.py first.")
        
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    encoder = joblib.load(ENCODER_FILE)
    
    # Handle camera ID vs file path
    if source.isdigit():
        cap_source = int(source)
    else:
        if not os.path.exists(source):
            print(f"Video source '{source}' not found. Generating sample driving video...")
            source = generate_sample_video(source)
        cap_source = source
        
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0:
        fps_in = 30.0
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))
    
    history_buffer = deque(maxlen=window_size)
    
    frame_count = 0
    start_time = time.time()
    
    print(f"\n--- Processing Video Stream: {source} ---")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # 1. Classical CV Pipeline
        binary_mask, masked_roi, fit_left, fit_right, fit_center, px_data = process_frame_cv(frame)
        
        # 2. Extract Features & Predict
        feat_vec, _ = extract_features(frame)
        if feat_vec is not None:
            feat_scaled = scaler.transform(feat_vec.reshape(1, -1))
            pred_idx = model.predict(feat_scaled)[0]
            raw_pred = str(encoder.inverse_transform([pred_idx])[0])
        else:
            raw_pred = "straight"
            
        # 3. Temporal Smoothing via Rolling Buffer Majority Vote
        history_buffer.append(raw_pred)
        smoothed_verdict = smooth_predictions(history_buffer)
        
        # 4. Draw Verdict Overlay
        ploty = px_data.get("ploty", np.linspace(280, 710, 100))
        fitx = None
        if fit_center is not None:
            fitx = fit_center[0] * (ploty ** 2) + fit_center[1] * ploty + fit_center[2]
            
        display_text = f"TURNING {smoothed_verdict.upper()}" if smoothed_verdict != "straight" else "STRAIGHT"
        verdict_frame = overlay_verdict_and_curve(
            frame, ploty, fitx, verdict=display_text, color=(0, 255, 0), thickness=4
        )
        
        out_writer.write(verdict_frame)
        
        if not headless:
            cv2.imshow("Real-Time Lane Curvature Classifier", verdict_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
    cap.release()
    out_writer.release()
    if not headless:
        cv2.destroyAllWindows()
        
    elapsed = time.time() - start_time
    achieved_fps = frame_count / elapsed if elapsed > 0 else 0.0
    
    print(f"\n--- Milestone 8 Summary ---")
    print(f"Processed Frames   : {frame_count}")
    print(f"Total Time Elapsed : {elapsed:.2f} seconds")
    print(f"Achieved FPS       : {achieved_fps:.2f} frames/sec")
    print(f"Output Video Saved : {output_path}")
    
    return achieved_fps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Video/Webcam Lane Curvature Classifier")
    parser.add_argument("--source", type=str, default="results/sample_driving_input.mp4", help="Video file path or camera ID (0)")
    parser.add_argument("--output", type=str, default="results/video_verdict_output.mp4", help="Output video file path")
    parser.add_argument("--headless", action="store_true", help="Run without GUI pop-up windows")
    args = parser.parse_args()
    
    process_video_stream(source=args.source, output_path=args.output, headless=args.headless)
