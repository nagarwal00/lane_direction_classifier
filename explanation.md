# Lane Curvature Direction Classifier — Technical Project Explanation & Guide

---

## 1. Executive Summary & Project Goal

The **Lane Curvature Direction Classifier** is an autonomous vehicle vision system designed to classify whether the ego-lane on a road is turning **Left**, staying **Straight**, or turning **Right**.

### Core Philosophy
Rather than relying on heavy deep-learning neural networks (which act as black boxes, require GPUs, and demand thousands of annotated images), this project uses a **hybrid classical computer vision + supervised machine learning pipeline**:
1. **Classical CV Pipeline (OpenCV)**: Extracts physical lane markings using HLS color space filtering, Sobel-X edge detection, trapezoidal ROI masking, sliding-window histogram tracking, and 2nd-degree polynomial curve fitting.
2. **Hand-Engineered Features**: Converts raw curve fits into a 9-dimensional physical feature vector (\(A_{fit}, B_{fit}, \Delta x_{top\_bottom}, \text{mean\_slope}\), pixel counts, asymmetry, lane center offset, std of x).
3. **Supervised ML Classifier (scikit-learn)**: Uses a Gradient Boosting classifier trained on empirical tertile ground-truth labels to make high-confidence turn decisions (**94.40% Test Accuracy**).

---

## 2. System Architecture & Data Flow

```
+-----------------------------------------------------------------------------------+
|                                 INPUT IMAGE/FRAME                                 |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                         STAGE 1: CLASSICAL CV FEATURE EXTRACTION                 |
|                                                                                   |
|  1. HLS Color Space Thresholding (White & Yellow Markings)                        |
|  2. Sobel-X Directional Gradient Thresholding (Edge Detection)                    |
|  3. Trapezoidal Region of Interest (ROI) Masking                                  |
|  4. Histogram Peak & Sliding Window Pixel Detection                               |
|  5. 2nd-Degree Polynomial Fitting: x(y) = A*y^2 + B*y + C                        |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        STAGE 2: FEATURE VECTOR CONSTRUCTION                       |
|                                                                                   |
|  - A_fit (Quadratic curvature coefficient)                                        |
|  - B_fit (Linear slope coefficient)                                              |
|  - delta_x_top_bottom (Lateral displacement top vs. bottom ROI)                   |
|  - mean_slope (Average derivative across y-range)                                |
|  - left_pixel_count & right_pixel_count                                           |
|  - pixel_count_asymmetry                                                          |
|  - lane_center_offset (Lateral offset from image center)                           |
|  - std_x_per_row (Scatter measure)                                               |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     STAGE 3: SUPERVISED ML CLASSIFICATION                         |
|                                                                                   |
|  - Feature Scaling via StandardScaler                                             |
|  - Classification via Trained Model (Gradient Boosting Classifier)               |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     STAGE 4: VISUAL DEMO & OUTPUT VERDICT                         |
|                                                                                   |
|  Window 1: Original Image                                                         |
|  Window 2: CV Processed Binary Mask                                               |
|  Window 3: Fitted Polyline Overlay + Verdict Text ("LANE: TURNING LEFT/RIGHT")    |
+-----------------------------------------------------------------------------------+
```

---

## 3. Computer Vision & Mathematical Concepts

### 3.1 2nd-Degree Polynomial Lane Model
Lane lines in 2D camera perspective are modeled as quadratic curves:
\[ x(y) = A \cdot y^2 + B \cdot y + C \]

- **\(A\)**: Quadratic curvature coefficient. Controls how sharply the lane curves left or right (second derivative \(\frac{d^2x}{dy^2} = 2A\)).
- **\(B\)**: Linear slope coefficient at \(y=0\).
- **\(C\)**: Horizontal intercept position.

### 3.2 Image Coordinate Sign Convention
In computer vision, image coordinates use the top-left corner as \((0,0)\):
- \(x\) increases from left (0) to right (1280).
- \(y\) increases **downwards** from top (0) to bottom (720).

When looking ahead at a road (moving towards the horizon at decreasing \(y\)):
- **Left Curve**: As \(y\) decreases (moving up the image), \(x(y)\) shifts to the left (smaller \(x\)). This corresponds to \(A < \text{Threshold}_{\text{lower}}\).
- **Right Curve**: As \(y\) decreases, \(x(y)\) shifts to the right (larger \(x\)). This corresponds to \(A > \text{Threshold}_{\text{upper}}\).
- **Straight**: \(A\) lies between the lower and upper thresholds.

### 3.3 HLS Color Thresholding
In standard RGB space, shadows, road asphalt variations, and lighting change pixel values dramatically. Converting to **HLS (Hue, Lightness, Saturation)** decouples color from illumination:
- **White Markings**: High Lightness (\(L \ge 180\)).
- **Yellow Markings**: Moderate Saturation (\(S \ge 100\)) and yellow Hue range.

### 3.4 Sobel-X Edge Detection
Calculates the horizontal derivative of pixel intensity \(\frac{\partial I}{\partial x}\) using a \(3 \times 3\) Sobel operator. This highlights near-vertical lane boundaries while ignoring horizontal road shadows.

### 3.5 Sliding Window Histogram Pixel Search
1. Computes column-wise pixel histogram on the bottom half of the ROI binary mask.
2. Identifies the two highest peaks in the left and right halves as initial lane centers.
3. Steps vertically upward in 9 horizontal bands, centering a sliding window of width \(2 \times 80\) pixels on the mean position of pixels found in the previous band.

---

## 4. Derived Features & Machine Learning Pipeline

### 4.1 Feature Vector Formulation (`src/features.py`)
Each processed frame yields a 9-dimensional feature vector:

| # | Feature Name | Formula / Description |
|---|---|---|
| 1 | `A_fit` | Quadratic coefficient \(A\) from fitted lane center curve |
| 2 | `B_fit` | Linear slope coefficient \(B\) from fitted lane center curve |
| 3 | `delta_x_top_bottom` | \(x(y_{\text{top}}) - x(y_{\text{bottom}})\) lateral displacement across ROI |
| 4 | `mean_slope` | Average derivative \(2A y_{\text{mid}} + B\) across ROI |
| 5 | `left_pixel_count` | Total binary pixels detected on left boundary |
| 6 | `right_pixel_count` | Total binary pixels detected on right boundary |
| 7 | `pixel_count_asymmetry` | \(\frac{\text{right\_count} - \text{left\_count}}{\text{right\_count} + \text{left\_count} + \varepsilon}\) |
| 8 | `lane_center_offset` | Distance between fitted lane center at bottom row and image center (640) |
| 9 | `std_x_per_row` | Standard deviation of detected pixel x-coordinates |

---

## 5. Model Selection & Empirical Results

### 5.1 Stratified 5-Fold Cross-Validation Comparison
Evaluated on 500 TuSimple benchmark images (75% train, 25% test split):

| Candidate Algorithm | Mean CV Accuracy | Std CV Accuracy | Mean CV F1 (Macro) | Status |
|---|---|---|---|---|
| **Logistic Regression** | 92.27% | 0.0320 | 0.9222 | Baseline |
| **SVC (RBF Kernel)** | 92.53% | 0.0333 | 0.9252 | Candidate |
| **Random Forest** | 97.07% | 0.0131 | 0.9704 | Strong |
| **Gradient Boosting** | **97.60%** | **0.0053** | **0.9759** | **WINNER** |

### 5.2 Hyperparameter Tuning (`GridSearchCV`)
Grid search on Gradient Boosting yields:
- `n_estimators`: `150`
- `learning_rate`: `0.05`
- `max_depth`: `3`
- **Tuned CV Accuracy**: **97.87%**

### 5.3 Held-Out Test Evaluation Results (`src/evaluate.py`)

- **Overall Test Accuracy**: **94.40%**
- **Macro F1-Score**: **0.9436**

#### Per-Class Breakdown:
| Class | Precision | Recall | F1-Score | Test Support |
|---|---|---|---|---|
| **LEFT** | 97.67% | 100.00% | **0.9882** | 42 |
| **RIGHT** | 97.37% | 88.10% | **0.9250** | 42 |
| **STRAIGHT** | 88.64% | 95.12% | **0.9176** | 41 |

#### Feature Importance Ranking:
1. `A_fit` (Quadratic curvature coefficient): **95.82%**
2. `pixel_count_asymmetry`: **1.26%**
3. `right_pixel_count`: **1.16%**

---

## 6. Script Breakdown & Usage Guide

| Script | Command | Purpose |
|---|---|---|
| `src/data_prep.py` | `python src/data_prep.py` | Download/generate dataset, prune 20th frames, generate index & annotations |
| `src/labeling.py` | `python src/labeling.py` | Calculate ground-truth curvature labels via polynomial tertiles & visual check |
| `src/cv_pipeline.py` | `python src/cv_pipeline.py` | Run HLS/Sobel-X thresholding, ROI, sliding windows & curve fitting |
| `src/features.py` | `python src/features.py` | Extract 9-dimensional feature vectors across dataset to `features.csv` |
| `src/train.py` | `python src/train.py` | 5-fold cross validation model selection, grid search tuning, save `.joblib` models |
| `src/evaluate.py` | `python src/evaluate.py` | Test set evaluation, generate confusion matrix plot & classification report |
| `src/infer_image.py` | `python src/infer_image.py --image <path>` | 3-window interactive single-image demo CLI |
| `src/infer_video.py` | `python src/infer_video.py --source <video>` | Real-time video processing with temporal rolling buffer smoothing |

---

## 7. Verification & Summary

All project milestones (Milestones 0 through 9) have been executed, tested, and validated against their strict Definitions of Done.

- [x] All core Python scripts written and verified without errors.
- [x] Ground-truth labels, feature matrix, trained model artifacts, confusion matrix plots, and video output generated.
- [x] Complete task tracking recorded in `task.md`.
- [x] Full technical documentation finalized in `explanation.md`.
