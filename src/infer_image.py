import os
import sys
import argparse
import joblib
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cv_pipeline import process_frame_cv
from features import extract_features
from utils import overlay_verdict_and_curve

# Minimum pixel thresholds for a reliable detection
MIN_LANE_PIXELS_TOTAL = 150    # Total left + right pixels detected
MIN_ASYMMETRY_RATIO = 0.98     # If 98%+ of pixels are on one side, something is wrong


def is_detection_reliable(feature_vector: dict) -> tuple:
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


MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "lane_direction_classifier.joblib")
SCALER_FILE = os.path.join(MODEL_DIR, "feature_scaler.joblib")
ENCODER_FILE = os.path.join(MODEL_DIR, "label_encoder.joblib")

def infer_single_image(image_path: str, save_output: bool = True, headless: bool = False):
    """
    Run 3-window inference demo on a single image.
    Displays:
    Window 1: Original Image
    Window 2: CV Processed Binary Mask
    Window 3: Fitted Polyline Curve + Verdict Text Overlay
    """
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Trained model not found at {MODEL_FILE}. Run src/train.py first.")
        
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    encoder = joblib.load(ENCODER_FILE)
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image file not found at {image_path}")
        
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"Failed to read image at {image_path}")
        
    # 1. Classical CV Pipeline
    binary_mask, masked_roi, fit_left, fit_right, fit_center, px_data = process_frame_cv(original_img)
    
    # Convert binary mask to 3-channel BGR for consistent display
    cv_processed_bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    
    # 2. Feature Extraction & Machine Learning Classification
    feat_vec, feat_names = extract_features(original_img)
    
    if feat_vec is None:
        verdict = "UNKNOWN"
        confidence = 0.0
    else:
        # Build feature dict for reliability gate
        feat_names_list = feat_names if feat_names is not None else []
        feat_dict = dict(zip(feat_names_list, feat_vec.tolist()))

        # V3 reliability gate — refuse to classify on bad detections
        reliable, reason = is_detection_reliable(feat_dict)
        if not reliable:
            return "uncertain", 0.0, {"reason": reason}

        feat_vec_scaled = scaler.transform(feat_vec.reshape(1, -1))
        pred_class_idx = model.predict(feat_vec_scaled)[0]
        verdict = str(encoder.inverse_transform([pred_class_idx])[0])
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(feat_vec_scaled)[0]
            confidence = float(np.max(probs)) * 100.0
        else:
            confidence = 100.0
            
    # 3. Create Verdict & Fitted Curve Overlay Image
    ploty = px_data.get("ploty", np.linspace(280, 710, 100))
    fitx = None
    if fit_center is not None:
        fitx = fit_center[0] * (ploty ** 2) + fit_center[1] * ploty + fit_center[2]
        
    verdict_display = f"TURNING {verdict.upper()}" if verdict != "straight" else "STRAIGHT"
    verdict_img = overlay_verdict_and_curve(
        original_img, ploty, fitx, verdict=verdict_display, color=(0, 255, 0), thickness=4
    )
    
    # Draw individual left and right detected polylines for detail
    if px_data.get("leftx") is not None and len(px_data["leftx"]) > 0:
        pts_l = np.vstack((px_data["leftx"], px_data["lefty"])).T.astype(np.int32)
        cv2.polylines(verdict_img, [pts_l], isClosed=False, color=(0, 255, 255), thickness=2)
    if px_data.get("rightx") is not None and len(px_data["rightx"]) > 0:
        pts_r = np.vstack((px_data["rightx"], px_data["righty"])).T.astype(np.int32)
        cv2.polylines(verdict_img, [pts_r], isClosed=False, color=(255, 0, 255), thickness=2)
        
    print(f"\nSingle Image Inference Result:")
    print(f"  Input File  : {image_path}")
    print(f"  Prediction  : {verdict.upper()}")
    print(f"  Confidence  : {confidence:.1f}%")
    
    # Save combined 3-window layout collage image
    if save_output:
        os.makedirs("results/sample_outputs", exist_ok=True)
        
        # Resize images for clean horizontal 3-window collage
        h, w = original_img.shape[:2]
        target_w, target_h = 640, 360
        
        w1 = cv2.resize(original_img, (target_w, target_h))
        w2 = cv2.resize(cv_processed_bgr, (target_w, target_h))
        w3 = cv2.resize(verdict_img, (target_w, target_h))
        
        # Add title headers to each collage window
        cv2.putText(w1, "1. ORIGINAL IMAGE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(w2, "2. CV BINARY MASK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(w3, "3. DETECTED CURVE & VERDICT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        collage = np.hstack((w1, w2, w3))
        out_collage_path = f"results/sample_outputs/demo_3window_{os.path.basename(image_path)}"
        cv2.imwrite(out_collage_path, collage)
        print(f"  Collage saved to: {out_collage_path}")
        
    # Interactive GUI 3 Windows Display
    if not headless:
        try:
            cv2.namedWindow("Window 1 -- Original Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Window 2 -- CV Processed Binary Mask", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Window 3 -- Detected Curve & Verdict", cv2.WINDOW_NORMAL)
            
            cv2.imshow("Window 1 -- Original Image", original_img)
            cv2.imshow("Window 2 -- CV Processed Binary Mask", cv_processed_bgr)
            cv2.imshow("Window 3 -- Detected Curve & Verdict", verdict_img)
            
            print("Press any key in OpenCV windows to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"GUI Display note: {e} (headless mode saved collage image to disk)")
            
    return verdict, confidence, verdict_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-Image 3-Window Lane Curvature Inference Demo")
    parser.add_argument("--image", type=str, default=None, help="Path to input road image")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI windows")
    args = parser.parse_args()
    
    if args.image is None:
        import glob
        images = glob.glob("data/images/*.jpg")
        if images:
            args.image = images[0]
            print(f"No --image supplied. Defaulting to sample: {args.image}")
        else:
            raise FileNotFoundError("No images found in data/images/")
            
    infer_single_image(args.image, headless=args.headless)
