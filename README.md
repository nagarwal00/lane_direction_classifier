# Lane Direction Classifier (V1, V2 & V3 ADAS Safety System)

A real-time dashcam lane direction classification system for Advanced Driver Assistance Systems (ADAS). Given a road image or a continuous video stream, the system detects lane lines using classical Computer Vision and deep learning to classify the ego-vehicle lane direction as **LEFT**, **STRAIGHT**, or **RIGHT**, augmented with physical curvature measurement, Bayesian temporal smoothing, and Lane Departure Warnings.

This repository contains three integrated stages of development:
1. **V1 Baseline**: Classical Computer Vision (OpenCV color/gradient thresholding + sliding windows) + 9 hand-crafted geometric features + GradientBoosting classifier (94.40% test accuracy).
2. **V2 Upgraded Deep Learning**: Transfer learning with MobileNetV2, Albumentations data augmentation, SHAP explainability, Grad-CAM attention visualisations, Monte Carlo Dropout uncertainty estimation, and ONNX Runtime deployment benchmarking (4.20x speedup).
3. **V3 ADAS Safety Suite**: Kalman filtering on polynomial curvature coefficients, Lane Departure Warning (LDW) with physical calibration, curvature radius in physical meters, optical flow ego-motion detection, and reliability gating.

---

## 🌟 Architecture & Upgrades (V1 vs V2 vs V3)

| Feature | V1 (Classical ML) | V2 (Deep Learning) | V3 (ADAS Safety Module) |
|---|---|---|---|
| **Representation** | 9 hand-crafted geometric features | Learned deep feature representations | Physical state vector \([A, \dot{A}]^T\) |
| **Model** | GradientBoosting (Scikit-Learn) | Fine-Tuned MobileNetV2 (PyTorch) | Kalman Filter + LDW Rule Engine |
| **Data Augmentation** | None | Albumentations (Brightness, Noise, Rotation, CLAHE, Flip) | N/A |
| **Explainability** | Feature Importances & SHAP Beeswarm/Waterfall | Grad-CAM Attention Heatmaps | Real-world radius in meters \(R\) |
| **Uncertainty** | Uncertainty Gate (pixel density & asymmetry) | Monte Carlo Dropout (20 forward passes per frame) | Kalman Innovation Consistency Check |
| **Deployment & Latency** | Python / Scikit-Learn (103.3 ms / 9.7 FPS) | Exported ONNX Runtime (14.61 ms / 68.4 FPS) | Kalman & LDW add <2ms overhead |
| **Safety Warnings** | Static label display | Dynamic confidence bar + uncertainty flag | Red frame alert border + LDW status |
| **Reporting** | Static PNGs & text report | Portable HTML report (`v1_vs_v2_report.html`) | Full Audit Report (`final_project_report.html`) |

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
│   │   └── test_label.json            # Raw TuSimple JSON annotations
│   └── raw/
│       └── clips/                     # Raw video clip directories
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
│   ├── final_project_report.html      # Comprehensive Final Verification & Research Report
│   ├── v1_vs_v2_report.html           # Side-by-Side Comparison HTML Report
│   ├── robustness_report.md           # Dataset robustness analysis report
│   ├── fps_benchmark.md               # Measured FPS & latency benchmark table
│   ├── classification_report.txt      # V1 test classification metrics
│   ├── confusion_matrix.png           # V1 confusion matrix heatmap
│   ├── feature_importance.png         # V1 feature importance breakdown
│   ├── shap_summary.png               # V1 SHAP beeswarm summary plot
│   ├── shap_waterfall_*.png           # V1 SHAP waterfall plots per class
│   ├── gradcam_*.png                  # V2 Grad-CAM attention heatmaps per class
│   ├── uncertainty_histogram.png      # V2 MC Dropout confidence distribution
│   ├── calibration_curve.png          # V2 Probability calibration curve
│   ├── latency_benchmark.json         # V2 PyTorch vs ONNX latency data
│   ├── misclassified/                 # Isolated misclassified sample images & analysis
│   ├── sample_driving_input.mp4       # Synthetic input driving sequence
│   ├── video_verdict_output.mp4       # V1 + V3 Video inference (Kalman + LDW)
│   └── video_v2_output.mp4            # V2 Video inference with confidence bar
│
├── src/
│   ├── augment.py                     # V2: Albumentations training & validation transforms
│   ├── cnn_train.py                   # V2: MobileNetV2 2-phase fine-tuning
│   ├── cnn_evaluate.py                # V2: CNN test evaluation & Grad-CAM generation
│   ├── cv_pipeline.py                 # V1/V2: Color/gradient thresholding & sliding windows
│   ├── culane_eval.py                 # V3: Evaluation script for CULane benchmark
│   ├── data_prep.py                   # V1: Dataset extraction & clip pruning
│   ├── evaluate.py                    # V1: Scikit-learn model evaluation
│   ├── features.py                    # V1: Extract 9 classical geometric CV features
│   ├── idd_eval.py                    # V3: Evaluation script for Indian Driving Dataset (IDD)
│   ├── infer_image.py                 # V1/V3: 3-window inference + Reliability Gate
│   ├── infer_video.py                 # V1/V3: Video inference with Kalman filter & LDW
│   ├── infer_video_v2.py              # V2: Video inference with CNN + MC Dropout
│   ├── kalman_smoother.py             # V3: Bayesian Kalman filter for curvature smoothing
│   ├── labeling.py                    # V1: Geometric curve fitting & tertile labeling
│   ├── ldw.py                         # V3: Lane Departure Warning rule engine
│   ├── onnx_export.py                 # V2: Export to ONNX & benchmark latency
│   ├── report_gen.py                  # V2: Generate standalone comparison HTML report
│   ├── shap_explain.py                # V2: SHAP analysis for V1 model
│   ├── uncertainty.py                 # V2: Monte Carlo Dropout inference & analysis
│   └── utils.py                       # Helpers, polyline overlay & radius in meters
│
├── tests/
│   └── test_pipeline.py               # Automated pytest suite (15 passed unit/integration tests)
│
├── run_all.py                         # Master End-to-End Pipeline Execution Script
├── demo.py                            # Interactive multi-version live demonstration
├── requirements.txt                   # Complete dependencies
└── README.md                          # Project documentation
```

---

## 🚀 Quick Start & Installation

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/nagarwal00/lane_direction_classifier.git
cd lane_direction_classifier

# Install dependencies
pip install -r requirements.txt
```

### 2. Master Runner (`run_all.py`)
To execute the complete end-to-end verified pipeline in one command:
```bash
python run_all.py --mode all
```
Or execute specific components independently:
```bash
python run_all.py --mode v1      # Run V1 pipeline and demos
python run_all.py --mode v2      # Run V2 deep learning and explainability
python run_all.py --mode test    # Run automated test suite
```

### 3. Automated Testing (pytest)
Execute the complete 15-test automated pipeline suite:
```bash
pytest tests/test_pipeline.py -v
```

---

## 🏃 Individual Script Execution Guide

```bash
# V1: Evaluate classical GradientBoosting model on held-out test set
python src/evaluate.py

# V1: Interactive 3-window inference on a single image (headless or GUI)
python src/infer_image.py --image data/images/img_00001.jpg --headless

# V1 + V3: Video inference with Kalman Filter smoothing & Lane Departure Warning
python src/infer_video.py --headless

# V2: Evaluate MobileNetV2 CNN, generate Grad-CAM heatmaps & calibration curve
python src/cnn_evaluate.py

# V2: Run Monte Carlo Dropout uncertainty analysis
python src/uncertainty.py

# V2: Export to ONNX and benchmark PyTorch vs ONNX latency
python src/onnx_export.py

# V2: Generate SHAP beeswarm summary and waterfall plots
python src/shap_explain.py

# V2: Real-time video with MC Dropout dynamic confidence bar
python src/infer_video_v2.py --headless

# V2: Generate comparison report
python src/report_gen.py
```

---

## 📊 Verified Evaluation Metrics

| Metric | V1 (Classical ML) | V2 (MobileNetV2 CNN) |
|---|---|---|
| **Held-Out Test Accuracy** | **94.40%** (118/125 correct) | **89.00%** (89/100 correct) |
| **Macro F1 Score** | **0.9436** | **0.8896** |
| **Weighted F1 Score** | **0.9438** | **0.8904** |
| **Per-Class Precision (L / R / S)** | 0.9767 / 0.9737 / 0.8864 | 1.0000 / 1.0000 / 0.7500 |
| **Per-Class Recall (L / R / S)** | 1.0000 / 0.8810 / 0.9512 | 0.9412 / 0.7273 / 1.0000 |
| **MC Dropout High-Confidence Acc** | N/A | **91.25%** (&ge;80% certainty) |
| **MC Dropout Flagged Uncertain** | Reliability Gate: pixel count / asymmetry | **20.00%** of frames flagged (80.00% acc) |
| **CPU Latency (Eager PyTorch)** | 103.3 ms (9.7 FPS) | 61.32 ms (16.3 FPS) |
| **CPU Latency (ONNX Runtime)** | N/A | **14.61 ms (68.4 FPS)** &rarr; **4.20x Speedup** |
| **Model Size** | 480 KB (.joblib) | 10.4 MB (.pth) / 258 KB (.onnx) |

---

## 🎓 Reports & Documentation
- **Comprehensive Final Audit Report**: Open `results/final_project_report.html` in your browser for the full 16-section interactive presentation with base64 embedded charts.
- **V1 vs V2 Comparison Report**: `results/v1_vs_v2_report.html`.
- **Edge-Case Safety Report**: `results/robustness_edge_cases.json` and `results/robustness_report.md`.
- **Latency Benchmarking**: `results/fps_benchmark.md` and `results/latency_benchmark.json`.
