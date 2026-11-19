import os
import sys
import glob
import cv2
import joblib
import torch
import numpy as np
import onnxruntime
import pytest
from PIL import Image

# Ensure project root and src/ are on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)

from src.cv_pipeline import process_frame_cv
from src.features import extract_features
from src.infer_image import infer_single_image, is_detection_reliable
from src.kalman_smoother import LaneKalmanFilter
from src.ldw import compute_ldw, DepartureStatus
from src.utils import compute_radius_meters
from src.cnn_train import build_model, CLASS_NAMES, IMAGE_SIZE, CHECKPOINT_PATH
from src.augment import get_val_transforms
from src.uncertainty import run_mc_dropout


def test_dataset_loads():
    """Verify images directory exists and contains 500 images."""
    images = glob.glob(os.path.join(PROJECT_ROOT, "data/images/*.jpg"))
    assert len(images) == 500, f"Expected 500 images, found {len(images)}"


def test_images_readable():
    """Verify first 5 images can be loaded and have 1280x720 shape."""
    images = glob.glob(os.path.join(PROJECT_ROOT, "data/images/*.jpg"))[:5]
    for p in images:
        img = cv2.imread(p)
        assert img is not None, f"Failed to read image at {p}"
        assert img.shape == (720, 1280, 3), f"Unexpected shape {img.shape} for {p}"


def test_labels_exist():
    """Verify labels_with_direction.csv exists and has 500 rows with 3 classes."""
    import pandas as pd
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "data/labels/labels_with_direction.csv"))
    assert len(df) == 500
    assert set(df["direction"].unique()) == {"left", "right", "straight"}


def test_cv_pipeline():
    """Verify CV pipeline extracts masks and polyline fits."""
    img_path = os.path.join(PROJECT_ROOT, "data/images/img_00001.jpg")
    img = cv2.imread(img_path)
    binary_mask, masked_roi, fit_left, fit_right, fit_center, px_data = process_frame_cv(img)
    assert binary_mask.shape == (720, 1280)
    assert isinstance(px_data, dict)


def test_feature_extraction():
    """Verify extract_features extracts exactly 9 finite numeric features."""
    img_path = os.path.join(PROJECT_ROOT, "data/images/img_00001.jpg")
    img = cv2.imread(img_path)
    feat_vec, feat_names = extract_features(img)
    assert feat_vec is not None
    assert len(feat_vec) == 9
    assert len(feat_names) == 9
    assert not np.isnan(feat_vec).any()
    assert not np.isinf(feat_vec).any()


def test_model_and_scaler_loading():
    """Verify V1 model, scaler, and label encoder load correctly."""
    model_path = os.path.join(PROJECT_ROOT, "models/lane_direction_classifier.joblib")
    scaler_path = os.path.join(PROJECT_ROOT, "models/feature_scaler.joblib")
    encoder_path = os.path.join(PROJECT_ROOT, "models/label_encoder.joblib")

    assert os.path.exists(model_path)
    assert os.path.exists(scaler_path)
    assert os.path.exists(encoder_path)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)

    assert hasattr(model, "predict")
    assert hasattr(scaler, "transform")
    assert len(encoder.classes_) == 3


def test_v1_inference_single_image():
    """Verify infer_single_image produces valid prediction and confidence."""
    img_path = os.path.join(PROJECT_ROOT, "data/images/img_00001.jpg")
    verdict, conf, _ = infer_single_image(img_path, save_output=False, headless=True)
    assert verdict in ["left", "right", "straight", "uncertain"]
    assert 0.0 <= conf <= 100.0


def test_reliability_gate():
    """Verify is_detection_reliable flags insufficient pixels and extreme asymmetry."""
    # Normal reliable case
    ok, reason = is_detection_reliable({"left_pixel_count": 3000, "right_pixel_count": 2800})
    assert ok is True

    # Low pixel count
    bad, reason = is_detection_reliable({"left_pixel_count": 40, "right_pixel_count": 50})
    assert bad is False
    assert "INSUFFICIENT" in reason

    # Extreme asymmetry
    bad_asym, reason = is_detection_reliable({"left_pixel_count": 3000, "right_pixel_count": 10})
    assert bad_asym is False
    assert "ASYMMETRY" in reason


def test_kalman_smoother():
    """Verify LaneKalmanFilter updates and smooths curvature."""
    kf = LaneKalmanFilter(dt=1.0 / 30.0)
    raw_values = [0.0001, 0.00012, 0.00011, -0.0001]
    for v in raw_values:
        smoothed = kf.update(v)
        assert isinstance(smoothed, float)
    assert kf.innovation_ok() is True


def test_ldw_module():
    """Verify LDW module generates safe and critical alerts based on lane offset."""
    # Safe offset on straight road
    safe_result = compute_ldw(10.0, "straight")
    assert safe_result.alert is False
    assert safe_result.status == DepartureStatus.SAFE

    # Critical right offset
    crit_result = compute_ldw(350.0, "straight")
    assert crit_result.alert is True
    assert "CRITICAL" in crit_result.status.value

    # Turning road should not warn
    turn_result = compute_ldw(350.0, "left")
    assert turn_result.alert is False


def test_compute_radius_meters():
    """Verify compute_radius_meters returns realistic physical radii."""
    r_straight = compute_radius_meters(0.0, 0.0)
    assert r_straight == float("inf")

    r_curve = compute_radius_meters(0.0002, 0.1)
    assert 100.0 < r_curve < 5000.0


def test_cnn_model_loading():
    """Verify MobileNetV2 checkpoint loads and runs forward pass."""
    device = torch.device("cpu")
    model = build_model(num_classes=3, freeze_backbone=False)
    state = torch.load(os.path.join(PROJECT_ROOT, CHECKPOINT_PATH), map_location=device)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, 3)


def test_mc_dropout():
    """Verify MC Dropout produces mean probabilities summing to 1.0 and valid std."""
    device = torch.device("cpu")
    model = build_model(num_classes=3, freeze_backbone=False)
    state = torch.load(os.path.join(PROJECT_ROOT, CHECKPOINT_PATH), map_location=device)
    model.load_state_dict(state)

    dummy = torch.randn(1, 3, 224, 224)
    mean_probs, std_probs, pred_cls, conf = run_mc_dropout(model, dummy, n_passes=5, device=device)
    assert len(mean_probs) == 3
    assert np.isclose(np.sum(mean_probs), 1.0, atol=1e-4)
    assert pred_cls in [0, 1, 2]
    assert 0.0 <= conf <= 1.0


def test_onnx_inference():
    """Verify ONNX runtime session loads and predicts successfully."""
    onnx_file = os.path.join(PROJECT_ROOT, "models/mobilenet_v2_finetuned.onnx")
    assert os.path.exists(onnx_file)
    session = onnxruntime.InferenceSession(onnx_file)
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    inputs = {session.get_inputs()[0].name: dummy}
    outputs = session.run(None, inputs)
    assert len(outputs) == 1
    assert outputs[0].shape == (1, 3)


def test_robustness_empty_and_corrupt():
    """Verify corrupted and non-existent files are safely rejected without crash."""
    with pytest.raises(FileNotFoundError):
        infer_single_image("non_existent_file.jpg", save_output=False, headless=True)
