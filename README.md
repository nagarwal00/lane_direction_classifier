# Lane Direction Classifier (V1 & V2)

A real-time dashcam lane direction classification system. Given a road image or a continuous video stream, the system detects lane lines using classical Computer Vision and classifies the ego-vehicle lane direction as **LEFT**, **STRAIGHT**, or **RIGHT**.

This repository contains both:
1. **V1 Baseline**: Classical Computer Vision (OpenCV color/gradient thresholding + sliding windows) + 9 hand-crafted geometric features + GradientBoosting classifier.
2. **V2 Upgraded Deep Learning**: Transfer learning with MobileNetV2, Albumentations data augmentation, SHAP explainability, Grad-CAM attention visualisations, Monte Carlo Dropout uncertainty estimation, and ONNX Runtime deployment benchmarking.

---

## 🌟 Architecture & Upgrades (V1 vs V2)

| Feature | V1 (Classical ML) | V2 (Deep Learning) |
|---|---|---|
| **Feature Representation** | 9 hand-crafted geometric features | Learned deep feature representations (ImageNet transfer learning) |
| **Model** | GradientBoosting (Scikit-Learn) | Fine-Tuned MobileNetV2 (PyTorch) |
| **Data Augmentation** | None | Albumentations (Brightness, Noise, Rotation, CLAHE, Flip) |
| **Explainability** | Feature Importances & SHAP Beeswarm/Waterfall | Grad-CAM Attention Heatmaps |
| **Uncertainty Quantification** | None (Deterministic) | Monte Carlo (MC) Dropout (20 forward passes per frame) |
| **Deployment & Latency** | Python / Scikit-Learn | Exported ONNX Model + ONNX Runtime CPU Benchmark |
| **Video Overlay** | Lane curve + verdict | Lane curve + verdict + dynamic confidence bar + uncertainty flag |
| **Reporting** | Text classification report & static PNGs | Self-contained, portable HTML report (`v1_vs_v2_report.html`) |

---

## 📁 Repository Structure

```
lane-direction-classifier/
├── data/
│   ├── images/                        # 500 dashcam images (img_00001.jpg ... img_00500.jpg)
│   ├── labels/
│   │   ├── features.csv               # V1: 9 geometric features per image
│   │   ├── image_index.csv            # Mapping from image filename to clip index
│   │   ├── labels_with_direction.csv  # Ground-truth labels (left/straight/right)
│   │   ├── parsed_annotations.csv     # Raw TuSimple lane point annotations
│   │   ├── parsed_annotations.parquet # Same as above, Parquet format
│   │   └── test_label.json            # Raw TuSimple JSON annotations
│   └── raw/
│       └── clips/                     # Clip directories
│
├── models/
│   ├── feature_scaler.joblib          # V1: Fitted StandardScaler
│   ├── label_encoder.joblib           # V1: LabelEncoder (left=0, right=1, straight=2)
│   ├── label_thresholds.json          # V1: Tertile thresholds
│   ├── lane_direction_classifier.joblib # V1: Trained GradientBoosting model
│   ├── model_card.json                # V1: Metadata & metrics
│   ├── test_split.npz                 # V1: Held-out test set
│   ├── mobilenet_v2_finetuned.pth     # V2: PyTorch checkpoint
│   ├── mobilenet_v2_finetuned.onnx    # V2: Deployable ONNX model
│   └── cnn_model_card.json            # V2: Deep learning metadata & metrics
│
├── results/
│   ├── classification_report.txt      # V1
│   ├── confusion_matrix.png           # V1
│   ├── feature_importance.png         # V1
│   ├── shap_summary.png               # V1 / Explainability
│   ├── shap_waterfall_left.png        # V1 / Explainability
│   ├── shap_waterfall_right.png       # V1 / Explainability
│   ├── shap_waterfall_straight.png    # V1 / Explainability
│   ├── gradcam_left.png               # V2 / Grad-CAM
│   ├── gradcam_right.png              # V2 / Grad-CAM
│   ├── gradcam_straight.png           # V2 / Grad-CAM
│   ├── uncertainty_histogram.png      # V2 / MC Dropout
│   ├── calibration_curve.png          # V2 / Reliability Diagram
│   ├── latency_benchmark.json         # V2 / ONNX vs PyTorch Latency
│   ├── v1_vs_v2_report.html           # Full Side-by-Side Comparison Report
│   ├── sample_driving_input.mp4       # Synthetic input driving sequence
│   ├── video_verdict_output.mp4       # V1 Video inference
│   └── video_v2_output.mp4            # V2 Video inference with confidence bar
│
├── src/
│   ├── augment.py                     # V2: Albumentations training & validation transforms
│   ├── cnn_train.py                   # V2: MobileNetV2 2-phase fine-tuning
│   ├── cnn_evaluate.py                # V2: CNN test evaluation & Grad-CAM generation
│   ├── cv_pipeline.py                 # V1/V2: Color/gradient thresholding & sliding windows
│   ├── data_prep.py                   # V1: Dataset extraction & clip pruning
│   ├── evaluate.py                    # V1: Scikit-learn model evaluation
│   ├── features.py                    # V1: Extract 9 classical geometric CV features
│   ├── infer_image.py                 # V1: 3-window interactive image inference
│   ├── infer_video.py                 # V1: Video inference with temporal smoothing
│   ├── infer_video_v2.py              # V2: Video inference with CNN + MC Dropout confidence
│   ├── labeling.py                    # V1: Geometric curve fitting & tertile labeling
│   ├── onnx_export.py                 # V2: Export to ONNX & benchmark inference latency
│   ├── report_gen.py                  # V2: Generate standalone HTML report
│   ├── shap_explain.py                # V2: SHAP analysis for V1 model
│   ├── uncertainty.py                 # V2: Monte Carlo Dropout inference & analysis
│   └── utils.py                       # Helpers, polyline overlay & voting
│
├── demo.py                            # Interactive multi-version live demonstration
├── LANE_CLASSIFIER_BUILD_SPEC.md      # Full engineering build specification
├── explanation.md                     # Deep technical documentation
├── requirements.txt                   # Complete dependencies
└── README.md                          # Project documentation
```

---

## 🚀 Quick Start & Installation

### 1. Environment Setup

Clone the repository and install all dependencies:
```bash
pip install -r requirements.txt
```

> **Note**: Python 3.10 or 3.11 is recommended.

---

## 🏃 Execution Order (End-to-End Run Guide)

Run all steps sequentially from the project root:

```bash
# Step 1: Prepare dataset & clips
python src/data_prep.py

# Step 2: Derive ground-truth labels
python src/labeling.py

# Step 3: Run V1 Pipeline (Features, Train, Evaluate)
python src/features.py
python src/train.py
python src/evaluate.py

# Step 4: Train V2 MobileNetV2 CNN (2-phase transfer learning)
python src/cnn_train.py

# Step 5: Evaluate CNN, generate Grad-CAM heatmaps & calibration curve
python src/cnn_evaluate.py

# Step 6: Generate SHAP beeswarm and waterfall explainability plots
python src/shap_explain.py

# Step 7: Run Monte Carlo Dropout uncertainty analysis
python src/uncertainty.py

# Step 8: Export fine-tuned CNN to ONNX and benchmark latency
python src/onnx_export.py

# Step 9: Generate unified HTML comparison report (V1 vs V2)
python src/report_gen.py

# Step 10: Run V2 video inference with dynamic confidence bar
python src/infer_video_v2.py --headless
```

---

## 🎓 Interactive Live Demo (`demo.py`)

Run the complete interactive demo:
```bash
# Run both V1 and V2 demos
python demo.py --version all

# Or run V2 Deep Learning demo specifically
python demo.py --version v2
```

---

## 📊 V1 vs V2 Performance Comparison

| Metric | V1 (Classical ML) | V2 (MobileNetV2 CNN) |
|---|---|---|
| **Test Accuracy** | ~94.40% | ≥ 90.00% |
| **Macro F1 Score** | 0.9436 | ≥ 0.8800 |
| **Interpretability** | SHAP Feature Importance | Grad-CAM Spatial Heatmaps |
| **Uncertainty** | None | MC Dropout Standard Deviation |
| **Inference Engine** | Scikit-learn CPU | ONNX Runtime CPU (1.5x+ speedup) |
| **Edge Deployment** | Requires full Python stack | Self-contained ONNX runtime |

Open `results/v1_vs_v2_report.html` in any browser to inspect the full interactive report with embedded visualizations.
