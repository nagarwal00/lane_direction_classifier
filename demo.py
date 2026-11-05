import os
import sys
import glob
import cv2
import argparse
import numpy as np
import pandas as pd
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.utils import ensure_directories
from src.data_prep import process_and_prune_dataset
from src.labeling import compute_ground_truth_labeling
from src.features import build_dataset_features
from src.train import train_and_select_model
from src.evaluate import evaluate_model
from src.infer_image import infer_single_image


def run_professor_live_demo(num_demo_images: int = 6):
    """
    All-in-one live demonstration script for professor presentation (V1 Classical ML).
    1. Prepares dataset & extracts classical CV features (if missing).
    2. Trains the machine learning model live from scratch.
    3. Evaluates the newly trained model on the test set.
    4. Displays 3 interactive OpenCV windows sequentially for a fixed number of sample images.
    """
    print("=" * 70)
    print("      LANE CURVATURE DIRECTION CLASSIFIER -- LIVE DEMONSTRATION (V1) ")
    print("=" * 70)
    
    ensure_directories()
    
    # Step 1: Ensure dataset and features exist
    if not os.path.exists("data/images") or len(glob.glob("data/images/*.jpg")) == 0:
        print("\n[STEP 1/4] Preparing dataset and pruning clips...")
        process_and_prune_dataset()
        
    if not os.path.exists("data/labels/labels_with_direction.csv"):
        print("\n[STEP 2/4] Generating ground-truth curvature direction labels...")
        compute_ground_truth_labeling()
        
    if not os.path.exists("data/labels/features.csv"):
        print("\n[STEP 3/4] Extracting 9-dimensional classical CV feature vectors...")
        build_dataset_features()
        
    # Step 2: TRAIN FROM SCRATCH LIVE
    print("\n" + "=" * 70)
    print(" [LIVE TRAINING] Training Machine Learning Classifier From Scratch...")
    print("=" * 70)
    model, scaler, encoder, model_card = train_and_select_model()
    
    # Step 3: EVALUATE NEWLY TRAINED MODEL
    print("\n" + "=" * 70)
    print(" [EVALUATION] Evaluating Newly Trained Model on Held-Out Test Set...")
    print("=" * 70)
    acc, macro_f1 = evaluate_model()
    
    # Step 4: INTERACTIVE 3-WINDOW DEMO ON TEST PICTURES
    print("\n" + "=" * 70)
    print(f" [VISUAL DEMO] Presenting 3-Window Visual Demo on {num_demo_images} Test Images...")
    print("=" * 70)
    
    df_labels = pd.read_csv("data/labels/labels_with_direction.csv")
    valid_df = df_labels[df_labels["direction"].isin(["left", "straight", "right"])]
    
    # Sample balanced set of test images (Left, Straight, Right)
    samples = []
    for d in ["left", "straight", "right"]:
        sub = valid_df[valid_df["direction"] == d]
        n_pick = min(num_demo_images // 3, len(sub))
        if n_pick > 0:
            samples.append(sub.sample(n=n_pick, random_state=42))
            
    if samples:
        demo_df = pd.concat(samples).reset_index(drop=True)
    else:
        demo_df = valid_df.head(num_demo_images)
        
    print("\nINSTRUCTIONS FOR DEMO:")
    print("  - Three windows will pop up for each image:")
    print("      1. Original Image")
    print("      2. CV Processed Binary Mask")
    print("      3. Fitted Polyline Curve + Verdict Overlay")
    print("  - Press ANY KEY on your keyboard to advance to the next sample picture.")
    print("  - Press Ctrl+C in the terminal to exit early.\n")
    
    for idx, row in demo_df.iterrows():
        img_name = row["image_filename"]
        img_path = os.path.join("data/images", img_name)
        gt_direction = row["direction"]
        
        print(f"\n--- Demo Image [{idx+1}/{len(demo_df)}]: {img_name} ---")
        print(f"  Ground-Truth Direction : {gt_direction.upper()}")
        
        verdict, confidence, _ = infer_single_image(
            image_path=img_path,
            save_output=True,
            headless=False  # Pops up interactive 3 windows
        )
        
        print(f"  Model Predicted Verdict : {verdict.upper()} (Confidence: {confidence:.1f}%)")
        print("  --> Window displayed. Press ANY KEY to move to next image...")

    print("\n" + "=" * 70)
    print("           V1 DEMONSTRATION COMPLETED SUCCESSFULLY!              ")
    print("=" * 70)


def run_v2_deep_learning_demo(num_demo_images: int = 6):
    """
    Demonstration script for V2 Deep Learning (MobileNetV2 + MC Dropout + Grad-CAM).
    """
    print("\n" + "=" * 70)
    print("   LANE DIRECTION CLASSIFIER V2 -- DEEP LEARNING & UNCERTAINTY DEMO  ")
    print("=" * 70)

    import torch
    from src.cnn_train import build_model, CLASS_NAMES, IDX_TO_CLASS, IMAGE_SIZE, CHECKPOINT_PATH
    from src.augment import get_val_transforms
    from src.uncertainty import run_mc_dropout
    from src.cv_pipeline import process_frame_cv
    from src.utils import overlay_verdict_and_curve

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: V2 model checkpoint '{CHECKPOINT_PATH}' not found.")
        print("Please train the CNN first: python src/cnn_train.py")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(CLASS_NAMES), freeze_backbone=False)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model = model.to(device)
    transform = get_val_transforms(IMAGE_SIZE)

    df_labels = pd.read_csv("data/labels/labels_with_direction.csv")
    valid_df = df_labels[df_labels["direction"].isin(CLASS_NAMES)]

    samples = []
    for d in CLASS_NAMES:
        sub = valid_df[valid_df["direction"] == d]
        n_pick = min(num_demo_images // 3, len(sub))
        if n_pick > 0:
            samples.append(sub.sample(n=n_pick, random_state=42))

    demo_df = pd.concat(samples).reset_index(drop=True) if samples else valid_df.head(num_demo_images)

    print(f"\nEvaluating MobileNetV2 with MC Dropout on {len(demo_df)} sample images:\n")

    for idx, row in demo_df.iterrows():
        img_name = row["image_filename"]
        img_path = os.path.join("data/images", img_name)
        gt_direction = row["direction"]

        frame = cv2.imread(img_path)
        img_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(image=img_np)["image"].unsqueeze(0)

        mean_probs, std_probs, pred_cls, confidence = run_mc_dropout(
            model, tensor, n_passes=10, device=device
        )
        pred_label = IDX_TO_CLASS[pred_cls]
        std_val = float(std_probs[pred_cls])

        print(f"[{idx+1}/{len(demo_df)}] {img_name}")
        print(f"  Ground Truth: {gt_direction.upper()}")
        print(f"  V2 Predicted: {pred_label.upper()} | Confidence: {confidence:.1%} (std: {std_val:.3f})")
        print(f"  Status: {'✓ HIGH CONFIDENCE' if confidence >= 0.80 else '⚠ UNCERTAIN'}")

    print("\n" + "=" * 70)
    print("           V2 DEMONSTRATION COMPLETED SUCCESSFULLY!              ")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lane Curvature Direction Classifier Demo")
    parser.add_argument("--version", choices=["v1", "v2", "all"], default="all",
                        help="Choose demo version: v1 (Classical ML), v2 (Deep Learning), or all")
    parser.add_argument("--num_images", type=int, default=6, help="Number of demo images to process")
    args = parser.parse_args()

    if args.version in ["v1", "all"]:
        run_professor_live_demo(num_demo_images=args.num_images)

    if args.version in ["v2", "all"]:
        if os.path.exists("models/mobilenet_v2_finetuned.pth"):
            run_v2_deep_learning_demo(num_demo_images=args.num_images)
        else:
            print("\nV2 model not yet trained. Train with: python src/cnn_train.py")
