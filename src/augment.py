# src/augment.py
"""
Albumentations augmentation pipeline for training the MobileNetV2 lane classifier.
Augmentations simulate real-world dashcam variations: lighting changes,
camera noise, slight rotations, and horizontal flips (with mirrored label).
"""

import inspect
import albumentations as A
from albumentations.pytorch import ToTensorV2


def _get_gauss_noise(p: float = 0.4):
    """Instantiate GaussNoise compatible with both Albumentations 1.x and 2.x."""
    sig = inspect.signature(A.GaussNoise.__init__)
    if "var_limit" in sig.parameters:
        return A.GaussNoise(var_limit=(10.0, 50.0), p=p)
    return A.GaussNoise(p=p)


def get_train_transforms(image_size: int = 224):
    """
    Returns augmentation pipeline for training images.
    Applied randomly per image on every epoch.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),                   # Mirrors left<->right (must swap label too)
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.6
        ),
        _get_gauss_noise(p=0.4),
        A.Rotate(limit=5, p=0.3),                  # Small rotation only — large rotations are unrealistic
        A.CLAHE(clip_limit=2.0, p=0.3),            # Simulates low-light normalisation
        A.Normalize(
            mean=[0.485, 0.456, 0.406],            # ImageNet normalisation
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_val_transforms(image_size: int = 224):
    """
    Returns deterministic transforms for validation and test images.
    No augmentation — only resize and normalise.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
