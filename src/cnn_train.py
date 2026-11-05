# src/cnn_train.py
"""
Fine-tunes MobileNetV2 (ImageNet pretrained) on the lane direction dataset.

Strategy:
  Phase 1 (epochs 1-5):  Freeze all backbone layers. Train only the classifier head.
  Phase 2 (epochs 6-15): Unfreeze the last 3 convolutional blocks. Train end-to-end
                          with a lower learning rate.

This two-phase approach prevents catastrophic forgetting of ImageNet features
while allowing the network to specialise for lane imagery.
"""

import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from augment import get_train_transforms, get_val_transforms

# ── Paths ────────────────────────────────────────────────────────────────────
LABELS_CSV        = "data/labels/labels_with_direction.csv"
IMAGES_DIR        = "data/images"
MODEL_DIR         = "models"
CHECKPOINT_PATH   = os.path.join(MODEL_DIR, "mobilenet_v2_finetuned.pth")
CNN_CARD_PATH     = os.path.join(MODEL_DIR, "cnn_model_card.json")

# ── Hyperparameters ───────────────────────────────────────────────────────────
IMAGE_SIZE        = 224
BATCH_SIZE        = 16
PHASE1_EPOCHS     = 5      # Head-only training
PHASE2_EPOCHS     = 10     # End-to-end fine-tuning
PHASE1_LR         = 1e-3
PHASE2_LR         = 1e-4
WEIGHT_DECAY      = 1e-4
RANDOM_STATE      = 42

# ── Label Setup ───────────────────────────────────────────────────────────────
CLASS_NAMES       = ["left", "right", "straight"]
CLASS_TO_IDX      = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS      = {i: c for c, i in CLASS_TO_IDX.items()}
FLIP_MAP          = {"left": "right", "right": "left", "straight": "straight"}


class LaneDataset(Dataset):
    """
    PyTorch Dataset for lane direction images.

    Handles horizontal flip label swapping: when an image is flipped
    horizontally (left<->right mirror), the label is swapped from LEFT
    to RIGHT or RIGHT to LEFT. STRAIGHT remains STRAIGHT.
    """

    def __init__(self, image_paths, labels, transform=None, is_train=False):
        """
        Args:
            image_paths (list[str]): Absolute paths to image files.
            labels (list[str]):      String labels: 'left', 'right', 'straight'.
            transform:               Albumentations Compose transform.
            is_train (bool):         If True, apply flip-label swapping.
        """
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform
        self.is_train    = is_train
        if self.is_train and self.transform is not None:
            for t in getattr(self.transform, "transforms", []):
                if t.__class__.__name__ == "HorizontalFlip":
                    t.p = 0.0

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path    = self.image_paths[idx]
        label_str   = self.labels[idx]
        label_idx   = CLASS_TO_IDX[label_str]

        # Load image as numpy array (Albumentations requires HWC numpy)
        img = np.array(Image.open(img_path).convert("RGB"))

        if self.transform is not None:
            if self.is_train:
                will_flip = random.random() < 0.5
                if will_flip:
                    img = np.ascontiguousarray(img[:, ::-1, :])
                    label_str = FLIP_MAP[label_str]
                    label_idx = CLASS_TO_IDX[label_str]
                transformed = self.transform(image=img)
                augmented_img = transformed["image"]
            else:
                transformed    = self.transform(image=img)
                augmented_img  = transformed["image"]
        else:
            augmented_img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        return augmented_img, label_idx


def build_model(num_classes: int = 3, freeze_backbone: bool = True) -> nn.Module:
    """
    Loads MobileNetV2 with ImageNet weights and replaces the final classifier.

    Architecture:
      MobileNetV2 backbone (1280 feature channels)
      → Dropout(0.3)                [reduces overfitting on small dataset]
      → Linear(1280, 256)
      → ReLU
      → Dropout(0.3)
      → Linear(256, num_classes)    [3 output logits]

    Args:
        num_classes (int):      Number of output classes.
        freeze_backbone (bool): If True, freeze all backbone parameters.

    Returns:
        nn.Module: Configured model.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Replace the default classifier
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.last_channel, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes)
    )

    if freeze_backbone:
        # Freeze all feature layers
        for param in model.features.parameters():
            param.requires_grad = False
        # Only classifier is trainable
        for param in model.classifier.parameters():
            param.requires_grad = True

    return model


def unfreeze_last_n_blocks(model: nn.Module, n: int = 3):
    """
    Unfreezes the last n convolutional blocks of MobileNetV2 features
    for Phase 2 fine-tuning. All classifier parameters remain trainable.

    MobileNetV2 has 19 feature blocks (indices 0-18).
    Unfreezing the last 3 (indices 16, 17, 18) gives a good trade-off
    between adaptation and preserving ImageNet representations.
    """
    total_blocks = len(model.features)
    unfreeze_from = total_blocks - n

    for i, layer in enumerate(model.features):
        if i >= unfreeze_from:
            for param in layer.parameters():
                param.requires_grad = True

    print(f"Unfrozen last {n} backbone blocks (indices {unfreeze_from}–{total_blocks - 1})")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Runs one full training epoch. Returns mean loss and accuracy."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


def evaluate_epoch(model, loader, criterion, device):
    """Runs one validation pass. Returns mean loss, accuracy, and macro F1."""
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Validating", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs    = model(images)
            loss       = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    mean_loss = total_loss / len(all_labels)

    return mean_loss, acc, macro_f1


def fine_tune_mobilenet():
    """
    Main training function.

    Phase 1: Freeze backbone, train classifier head for PHASE1_EPOCHS epochs.
    Phase 2: Unfreeze last 3 blocks, train end-to-end for PHASE2_EPOCHS epochs.

    Saves:
      - models/mobilenet_v2_finetuned.pth   (best checkpoint by val accuracy)
      - models/cnn_model_card.json           (metadata)
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Load and split dataset ──────────────────────────────────────────────
    df = pd.read_csv(LABELS_CSV)
    df = df[df["direction"].isin(CLASS_NAMES)].reset_index(drop=True)

    image_paths = [os.path.join(IMAGES_DIR, fn) for fn in df["image_filename"]]
    labels      = df["direction"].tolist()

    X_train_paths, X_val_paths, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=RANDOM_STATE
    )

    print(f"Dataset split: {len(X_train_paths)} train / {len(X_val_paths)} validation")
    print(f"Classes: {CLASS_NAMES}")

    # ── Datasets and Loaders ────────────────────────────────────────────────
    train_dataset = LaneDataset(
        X_train_paths, y_train,
        transform=get_train_transforms(IMAGE_SIZE),
        is_train=True
    )
    val_dataset = LaneDataset(
        X_val_paths, y_val,
        transform=get_val_transforms(IMAGE_SIZE),
        is_train=False
    )

    num_workers = 0  # 0 is safe and optimal on Windows to avoid spawn overhead
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    # ── Model and device ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model     = build_model(num_classes=len(CLASS_NAMES), freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()

    history = {"phase1": [], "phase2": []}
    best_val_acc  = 0.0
    best_epoch    = 0
    start_time    = time.time()

    # ── Phase 1: Head Only ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 1: Training classifier head only ({PHASE1_EPOCHS} epochs)")
    print(f"{'='*60}")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE1_LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS)

    for epoch in range(1, PHASE1_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Epoch {epoch:02d}/{PHASE1_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        history["phase1"].append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"    ✓ New best model saved (val_acc={best_val_acc:.4f})")

    # ── Phase 2: End-to-End Fine-Tuning ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 2: End-to-end fine-tuning ({PHASE2_EPOCHS} epochs)")
    print(f"{'='*60}")

    unfreeze_last_n_blocks(model, n=3)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE2_LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS)

    for epoch in range(1, PHASE2_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Epoch {epoch:02d}/{PHASE2_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        history["phase2"].append({
            "epoch": epoch + PHASE1_EPOCHS,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch + PHASE1_EPOCHS
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"    ✓ New best model saved (val_acc={best_val_acc:.4f})")

    elapsed = time.time() - start_time

    # ── Save model card ─────────────────────────────────────────────────────
    card = {
        "model_type": "MobileNetV2 (fine-tuned)",
        "pretrained_weights": "ImageNet IMAGENET1K_V1",
        "image_size": IMAGE_SIZE,
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "batch_size": BATCH_SIZE,
        "phase1_epochs": PHASE1_EPOCHS,
        "phase2_epochs": PHASE2_EPOCHS,
        "phase1_lr": PHASE1_LR,
        "phase2_lr": PHASE2_LR,
        "best_val_accuracy": best_val_acc,
        "best_epoch": best_epoch,
        "total_training_seconds": elapsed,
        "train_samples": len(X_train_paths),
        "val_samples": len(X_val_paths),
        "augmentation": "Albumentations: HFlip, BrightnessContrast, GaussNoise, Rotate5, CLAHE",
        "checkpoint": CHECKPOINT_PATH,
        "training_history": history
    }

    with open(CNN_CARD_PATH, "w") as f:
        json.dump(card, f, indent=4)

    print(f"\n{'='*60}")
    print(f"Training complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Total training time: {elapsed:.1f}s")
    print(f"Model checkpoint: {CHECKPOINT_PATH}")
    print(f"Model card: {CNN_CARD_PATH}")

    return model, best_val_acc


if __name__ == "__main__":
    fine_tune_mobilenet()

