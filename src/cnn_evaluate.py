# src/cnn_evaluate.py
"""
Evaluates the fine-tuned MobileNetV2 on the held-out test set.
Generates:
  - Classification report printed to console
  - results/gradcam_left.png
  - results/gradcam_right.png
  - results/gradcam_straight.png
  - results/calibration_curve.png
  - updates models/cnn_model_card.json with test metrics

Grad-CAM (Gradient-weighted Class Activation Mapping):
  Uses the gradient of the class score with respect to the last convolutional
  feature map. High-gradient regions correspond to pixels the network uses
  to make its decision. Overlaid as a colour heatmap on the original image.
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
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    confusion_matrix
)
from sklearn.calibration import calibration_curve
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cnn_train import (
    LaneDataset, build_model,
    CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS,
    IMAGE_SIZE, BATCH_SIZE, RANDOM_STATE,
    CHECKPOINT_PATH, CNN_CARD_PATH
)
from augment import get_val_transforms

LABELS_CSV   = "data/labels/labels_with_direction.csv"
IMAGES_DIR   = "data/images"
RESULTS_DIR  = "results"


# ── Grad-CAM Implementation ────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM hook for MobileNetV2.

    Registers forward and backward hooks on the last convolutional layer
    (model.features[-1]) to capture activations and gradients.

    Usage:
        cam = GradCAM(model)
        heatmap = cam.generate(input_tensor, target_class_idx)
        cam.remove_hooks()
    """

    def __init__(self, model: nn.Module):
        self.model       = model
        self.activations = None
        self.gradients   = None

        # Register hooks on the last feature block of MobileNetV2
        target_layer = model.features[-1]
        self.forward_hook  = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Generates a Grad-CAM heatmap for the given input and target class.

        Args:
            input_tensor (torch.Tensor): Shape [1, 3, H, W], normalised.
            target_class (int):          Index of the class to visualise.

        Returns:
            np.ndarray: Heatmap of shape [H, W], values in [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)

        self.model.zero_grad()
        # Backpropagate the score for the target class
        class_score = output[0, target_class]
        class_score.backward()

        # Global average pool the gradients over spatial dimensions
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted sum of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)                                              # Keep positive contributions only

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def overlay_gradcam(original_image_path: str, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlays a Grad-CAM heatmap on the original image.

    Args:
        original_image_path (str): Path to the original (un-normalised) image.
        heatmap (np.ndarray):      Normalised heatmap [0,1], any spatial size.
        alpha (float):             Blend weight for heatmap overlay.

    Returns:
        np.ndarray: BGR image with heatmap overlaid (for cv2.imwrite).
    """
    img   = cv2.imread(original_image_path)
    h, w  = img.shape[:2]

    # Resize heatmap to original image size
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    colormap        = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Blend
    overlaid = cv2.addWeighted(img, 1 - alpha, colormap, alpha, 0)
    return overlaid


def evaluate_cnn():
    """
    Full evaluation of the CNN on the test split.
    Generates Grad-CAM images, calibration curve, and updates the model card.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ─────────────────────────────────────────────────────────
    model = build_model(num_classes=len(CLASS_NAMES), freeze_backbone=False)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # ── Prepare test set ───────────────────────────────────────────────────
    df = pd.read_csv(LABELS_CSV)
    df = df[df["direction"].isin(CLASS_NAMES)].reset_index(drop=True)

    image_paths = [os.path.join(IMAGES_DIR, fn) for fn in df["image_filename"]]
    labels      = df["direction"].tolist()

    _, X_test_paths, _, y_test_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=RANDOM_STATE
    )

    test_dataset = LaneDataset(
        X_test_paths, y_test_labels,
        transform=get_val_transforms(IMAGE_SIZE),
        is_train=False
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ── Run inference ──────────────────────────────────────────────────────
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels_batch in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs   = F.softmax(outputs, dim=1).cpu().numpy()
            preds   = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels_batch.numpy())
            all_probs.extend(probs)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    # ── Metrics ────────────────────────────────────────────────────────────
    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report   = classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES, digits=4
    )

    print("\n" + "=" * 60)
    print("CNN TEST EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test Accuracy : {acc * 100:.2f}%")
    print(f"Macro F1      : {macro_f1:.4f}")
    print("\nClassification Report:")
    print(report)

    # ── Calibration Curve ──────────────────────────────────────────────────
    # Checks if confidence scores are trustworthy.
    # A perfectly calibrated model has the diagonal line y=x.
    plt.figure(figsize=(8, 6))
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        binary_labels = (all_labels == cls_idx).astype(int)
        prob_true, prob_pred = calibration_curve(
            binary_labels, all_probs[:, cls_idx], n_bins=10, strategy="quantile"
        )
        plt.plot(prob_pred, prob_true, marker="o", label=f"{cls_name.upper()}")

    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Fraction of Positives", fontsize=11)
    plt.title("Calibration Curve — MobileNetV2 Lane Classifier", fontsize=13, pad=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "calibration_curve.png"), dpi=300)
    plt.close()
    print("Saved: results/calibration_curve.png")

    # ── Grad-CAM ───────────────────────────────────────────────────────────
    print("\nGenerating Grad-CAM visualisations...")

    cam = GradCAM(model)
    transform = get_val_transforms(IMAGE_SIZE)

    for class_idx, class_name in enumerate(CLASS_NAMES):
        # Find a correctly classified test sample for this class
        matching = [i for i, (pred, true) in enumerate(zip(all_preds, all_labels))
                    if true == class_idx and pred == class_idx]

        if not matching:
            print(f"  No correctly classified {class_name} samples — skipping Grad-CAM.")
            continue

        sample_path = X_test_paths[matching[0]]

        # Load and preprocess
        img_np    = np.array(Image.open(sample_path).convert("RGB"))
        tensor    = transform(image=img_np)["image"].unsqueeze(0).to(device)
        tensor.requires_grad = True

        # Generate heatmap
        heatmap = cam.generate(tensor, target_class=class_idx)

        # Overlay on original image
        overlaid = overlay_gradcam(sample_path, heatmap, alpha=0.45)

        # Add text label
        cv2.putText(
            overlaid,
            f"Grad-CAM: {class_name.upper()} (correct)",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
        )

        save_path = os.path.join(RESULTS_DIR, f"gradcam_{class_name}.png")
        cv2.imwrite(save_path, overlaid)
        print(f"  Saved: results/gradcam_{class_name}.png")

    cam.remove_hooks()

    # ── Update model card ──────────────────────────────────────────────────
    with open(CNN_CARD_PATH, "r") as f:
        card = json.load(f)

    card["test_accuracy"] = acc
    card["test_macro_f1"] = macro_f1
    card["test_samples"]  = len(all_labels)

    with open(CNN_CARD_PATH, "w") as f:
        json.dump(card, f, indent=4)

    print("\nCNN evaluation complete.")
    return acc, macro_f1


if __name__ == "__main__":
    evaluate_cnn()

