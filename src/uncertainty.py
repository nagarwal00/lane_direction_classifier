# src/uncertainty.py
"""
Monte Carlo Dropout (MC Dropout) uncertainty estimation for the CNN.

How it works:
  During normal inference, Dropout is disabled (model.eval() turns it off).
  In MC Dropout, we force Dropout to stay active and run N forward passes
  per image. Each pass gives a slightly different output. The mean gives
  the expected prediction; the standard deviation gives uncertainty.

  Frames where std > threshold should be flagged for human review —
  the model is not confident enough to act on autonomously.

Generates:
  - results/uncertainty_histogram.png  : Distribution of confidence scores
                                         across the test set
  - Printed summary: % of frames below confidence threshold
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cnn_train import (
    LaneDataset, build_model,
    CLASS_NAMES, IMAGE_SIZE, BATCH_SIZE, RANDOM_STATE,
    CHECKPOINT_PATH
)
from augment import get_val_transforms

LABELS_CSV   = "data/labels/labels_with_direction.csv"
IMAGES_DIR   = "data/images"
RESULTS_DIR  = "results"

MC_PASSES         = 20      # Number of stochastic forward passes per image
CONFIDENCE_THRESH = 0.80    # Below this, the frame is flagged as uncertain


def enable_dropout(model: nn.Module):
    """
    Sets all Dropout layers to training mode while keeping BatchNorm in eval mode.
    This is the key step that makes MC Dropout work.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def run_mc_dropout(model, input_tensor, n_passes: int = MC_PASSES, device="cpu"):
    """
    Runs n_passes stochastic forward passes on a single input tensor.

    Returns:
        mean_probs (np.ndarray):  Shape [num_classes]. Mean softmax probability per class.
        std_probs  (np.ndarray):  Shape [num_classes]. Std across passes per class.
        predicted_class (int):   The class with the highest mean probability.
        confidence (float):      Max probability of the predicted class.
    """
    model.eval()
    enable_dropout(model)   # Re-enable dropout for MC sampling

    pass_probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            output = model(input_tensor.to(device))
            prob   = F.softmax(output, dim=1).cpu().numpy()[0]
            pass_probs.append(prob)

    pass_probs = np.array(pass_probs)          # [n_passes, num_classes]
    mean_probs = pass_probs.mean(axis=0)       # [num_classes]
    std_probs  = pass_probs.std(axis=0)        # [num_classes]

    predicted_class = int(np.argmax(mean_probs))
    confidence      = float(mean_probs[predicted_class])

    return mean_probs, std_probs, predicted_class, confidence


def run_uncertainty_analysis():
    """
    Runs MC Dropout on the full test set.
    Saves a histogram of confidence scores and prints a summary.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=len(CLASS_NAMES), freeze_backbone=False)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model = model.to(device)

    df = pd.read_csv(LABELS_CSV)
    df = df[df["direction"].isin(CLASS_NAMES)].reset_index(drop=True)

    image_paths = [os.path.join(IMAGES_DIR, fn) for fn in df["image_filename"]]
    labels      = df["direction"].tolist()

    _, X_test_paths, _, y_test_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=RANDOM_STATE
    )

    transform = get_val_transforms(IMAGE_SIZE)

    confidences       = []
    uncertainties     = []
    correct_high_conf = 0
    correct_low_conf  = 0
    total_high_conf   = 0
    total_low_conf    = 0

    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

    print(f"Running MC Dropout ({MC_PASSES} passes per image) on {len(X_test_paths)} test images...")

    for img_path, true_label in zip(X_test_paths, y_test_labels):
        img_np = np.array(Image.open(img_path).convert("RGB"))
        tensor = transform(image=img_np)["image"].unsqueeze(0)

        mean_probs, std_probs, pred_class, confidence = run_mc_dropout(
            model, tensor, n_passes=MC_PASSES, device=device
        )

        true_idx   = CLASS_TO_IDX[true_label]
        is_correct = (pred_class == true_idx)

        confidences.append(confidence)
        uncertainties.append(float(std_probs[pred_class]))

        if confidence >= CONFIDENCE_THRESH:
            total_high_conf += 1
            if is_correct:
                correct_high_conf += 1
        else:
            total_low_conf += 1
            if is_correct:
                correct_low_conf += 1

    confidences   = np.array(confidences)
    uncertainties = np.array(uncertainties)

    # ── Summary ────────────────────────────────────────────────────────────
    flagged_pct = (total_low_conf / len(X_test_paths)) * 100

    print(f"\n{'='*55}")
    print(f"MC DROPOUT UNCERTAINTY ANALYSIS")
    print(f"{'='*55}")
    print(f"Total test samples        : {len(X_test_paths)}")
    print(f"Confidence threshold      : {CONFIDENCE_THRESH:.0%}")
    print(f"High-confidence frames    : {total_high_conf} ({100 - flagged_pct:.1f}%)")
    print(f"  → Accuracy on high-conf : {correct_high_conf / max(total_high_conf, 1):.4f}")
    print(f"Low-confidence frames     : {total_low_conf}  ({flagged_pct:.1f}%) [FLAGGED]")
    print(f"  → Accuracy on low-conf  : {correct_low_conf / max(total_low_conf, 1):.4f}")
    print(f"Mean confidence           : {confidences.mean():.4f}")
    print(f"Mean uncertainty (std)    : {uncertainties.mean():.4f}")

    # ── Histogram ──────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.hist(confidences, bins=25, color="#2b5c8f", edgecolor="white", alpha=0.9)
    plt.axvline(
        CONFIDENCE_THRESH, color="red", linestyle="--", linewidth=2,
        label=f"Confidence threshold ({CONFIDENCE_THRESH:.0%})"
    )
    plt.xlabel("MC Dropout Confidence Score (Max Class Probability)", fontsize=11)
    plt.ylabel("Number of Test Frames", fontsize=11)
    plt.title(
        f"MC Dropout Uncertainty Distribution\n"
        f"{flagged_pct:.1f}% of frames flagged as low-confidence",
        fontsize=13, pad=12
    )
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "uncertainty_histogram.png"), dpi=300)
    plt.close()
    print("\nSaved: results/uncertainty_histogram.png")

    return confidences, uncertainties, flagged_pct


if __name__ == "__main__":
    run_uncertainty_analysis()

