# Project Task Tracking — Lane Curvature Direction Classifier

## Milestone Status Overview

- [x] **Milestone 0 — Environment Setup**
  - [x] Write `requirements.txt`
  - [x] Install dependencies & verify importability
- [x] **Milestone 1 — Dataset Acquisition & Pruning (`src/data_prep.py`)**
  - [x] Obtain TuSimple test dataset & `test_label.json`
  - [x] Extract and prune clips (retaining 20th frame per clip)
  - [x] Build mapping table & parse annotations

- [x] **Milestone 2 — Ground-Truth Direction Label Generation (`src/labeling.py`)**
  - [x] Extract ego-lane boundary lines & compute lane-center curve
  - [x] Fit 2nd-degree polynomial \(x(y) = A y^2 + B y + C\)
  - [x] Compute empirical tertile thresholds for coefficient \(A\)
  - [x] Generate `{left, straight, right}` labels & visual verification overlays

- [x] **Milestone 3 — Classical CV Lane-Pixel Detection Pipeline (`src/cv_pipeline.py`)**
  - [x] Color (HLS) & gradient (Sobel-X) thresholding binary mask
  - [x] Region-of-Interest (ROI) trapezoid masking
  - [x] Histogram peak & sliding window pixel detection
  - [x] 2nd-degree polynomial curve fitting on detected pixels
  - [x] Visual smoke testing on sample images

- [x] **Milestone 4 — Feature Extraction (`src/features.py`)**
  - [x] Implement 8 hand-engineered CV features (\(A_{fit}, B_{fit}, \Delta x_{top\_bottom}, \text{mean\_slope}\), pixel counts, asymmetry, lane center offset, std of x)
  - [x] Extract features across entire image dataset to `data/labels/features.csv`
  - [x] Log failed detections

- [x] **Milestone 5 — Model Training (`src/train.py`)**
  - [x] Stratified train/test split (75/25)
  - [x] Feature scaling with `StandardScaler`
  - [x] Evaluate multiple candidate classifiers (Logistic Regression, RBF SVM, Random Forest, Gradient Boosting) via 5-fold cross-validation
  - [x] Hyperparameter tuning (`GridSearchCV`) on top model
  - [x] Save trained model, scaler, label encoder, and model card JSON

- [x] **Milestone 6 — Model Evaluation (`src/evaluate.py`)**
  - [x] Evaluate on held-out test split
  - [x] Generate classification report (precision, recall, F1, accuracy)
  - [x] Render & save confusion matrix heatmap
  - [x] Extract & plot feature importances

- [x] **Milestone 7 — Single-Image Inference Demo (`src/infer_image.py`)**
  - [x] Build CLI for single image inference
  - [x] Render 3 simultaneous OpenCV display windows (Original, CV Binary Mask, Detected Curve + Verdict)
  - [x] Validate on sample test images

- [x] **Milestone 8 — Real-Time Video Processing (`src/infer_video.py`)**
  - [x] Video file and camera stream inference pipeline
  - [x] Temporal smoothing via rolling buffer majority vote
  - [x] Measure and log FPS

- [x] **Milestone 9 — Documentation & Wrap-Up**
  - [x] Create/Update `README.md`
  - [x] Complete `explanation.md` technical documentation
  - [x] Final verification against Definition of Done

