# Implementation Plan — Lane Curvature Direction Classifier

**Project:** Detect whether the ego-lane in a road image (and later, a video stream) is turning **Left**, **Right**, or is **Straight**, using classical computer vision for feature extraction and `scikit-learn` for classification.

**Audience for this document:** an autonomous coding agent (or a developer) executing this project end-to-end, milestone by milestone, without needing further clarification. Each milestone has explicit tasks and a **Definition of Done (DoD)** that must be satisfied before moving to the next milestone.

---

## 0. Objective

> Given a single road image, output a predicted class in `{left, straight, right}` describing which way the ego-lane is curving, display the result visually, and evaluate the system as a supervised classification problem (train/test split, accuracy, confusion matrix). Phase 2 extends the same pipeline to run frame-by-frame on video/webcam input in real time.

This is explicitly **not** a pixel-level segmentation project and **not** a deep-learning project. The entire pipeline is classical CV (OpenCV) for feature extraction + `scikit-learn` for the classification decision.

---

## 1. Scope

**In scope**
- Single-image lane curvature direction classification (Left / Straight / Right).
- A classical CV pipeline that detects lane-line pixels and fits a curve to them.
- A hand-engineered feature vector derived from that curve/pixel data.
- A `scikit-learn` classifier trained and evaluated as a standard supervised classification problem.
- A 3-window visual output: (1) original image, (2) CV-processed (binary/edge) image, (3) original image with the fitted curve drawn and a verdict text overlay.
- Phase 2 (stretch): the same pipeline applied per-frame to a video file or webcam feed in real time, with temporal smoothing.

**Out of scope**
- Deep learning / CNN-based segmentation (explicitly not used — this project uses classical CV + scikit-learn only).
- GPS/map-based turn-by-turn navigation.
- Real vehicle control / steering actuation.
- Multi-lane (beyond ego-lane) reasoning.

---

## 2. Tech stack

| Component | Choice | Purpose |
|---|---|---|
| Language | Python 3.10+ | — |
| `opencv-python` | Classical CV pipeline, color/gradient thresholding, drawing, the 3 display windows | Core CV |
| `numpy` | Array math, polynomial fitting (`np.polyfit`) | Core math |
| `scikit-learn` | Train/test split, `StandardScaler`, classifiers, cross-validation, metrics | Core ML |
| `pandas` | Feature table management, label bookkeeping | Data wrangling |
| `matplotlib` | Confusion matrix plot, saved evaluation figures (not used in the live 3-window demo, which uses `cv2.imshow`) | Reporting |
| `joblib` | Persisting the trained model + scaler | Model serialization |

`requirements.txt` (pin approximate major versions when implementing):
```
opencv-python
numpy
scikit-learn
pandas
matplotlib
joblib
```

---

## 3. Dataset

### 3.1 Selection rationale

The project needs a dataset that is (a) free, (b) has precise lane-geometry ground truth (so curvature direction labels can be generated automatically instead of manually annotated), and (c) is **not exceptionally large**, since the whole point of this pipeline is to be lightweight and runnable on a personal laptop.

### 3.2 Chosen dataset: TuSimple Lane Detection Benchmark — **test split only**

The TuSimple benchmark is a widely-used, free, research-licensed lane detection dataset recorded on US highways. Critically for this project:

- TuSimple released `test_label.json`, the official ground-truth annotation for its **test split**, after the original competition closed. This means the test split alone is a **complete, self-contained, labeled dataset** — we do not need the (much larger) training split at all.
- The test split consists of **2,782 clips**, each 20 frames long, of which only the **last (20th) frame** is annotated.
- Compressed download size for `test_set.zip` is on the order of **1.5–2 GB**. After extracting only the 20th frame of each clip (discarding the other 19 unannotated frames per clip) the retained image set is roughly **600–700 MB** on disk — a manageable, one-time footprint.
- Image resolution is 1280×720; annotations are stored as `lanes` (a list of per-lane x-coordinate lists) and `h_samples` (the shared list of y-row samples those x-coordinates correspond to).

This gives **2,782 labeled images** — plenty for a classical-feature classification problem (which needs far fewer samples than a deep segmentation network would), while keeping storage and processing time modest.

> **Optional extension (not required for v1):** if more training data is later wanted, the TuSimple `train_set.zip` (3,626 additional labeled clips, ~5 GB compressed, ~1 GB after pruning to labeled frames only) can be added using the identical pruning procedure. Treat this as a stretch goal, not a v1 requirement.

### 3.3 Download sources
- Official: https://github.com/TuSimple/tusimple-benchmark/issues/3 (links to `test_set.zip`, `test_label.json`, `train_set.zip`)
- Kaggle mirror (convenient for scripted download via the Kaggle API): search "TuSimple" on kaggle.com/datasets.

### 3.4 License
Free for academic/research use. Cite TuSimple in any report accompanying this project; do not redistribute the raw dataset commercially.

---

## 4. Repository structure

```
lane-direction-classifier/
├── data/
│   ├── raw/                     # downloaded zip(s) — safe to delete after extraction
│   ├── images/                  # pruned: only the labeled (20th) frame per clip
│   └── labels/                  # test_label.json and any derived label tables
├── src/
│   ├── data_prep.py             # download + prune + organize dataset
│   ├── labeling.py              # derive ground-truth Left/Straight/Right labels
│   ├── cv_pipeline.py           # classical CV: threshold, ROI, pixel detection, curve fit
│   ├── features.py              # feature vector extraction from cv_pipeline output
│   ├── train.py                 # build feature table, train/test split, train classifiers
│   ├── evaluate.py              # metrics, confusion matrix, classification report
│   ├── infer_image.py           # single-image demo — the 3-window display
│   ├── infer_video.py           # Phase 2 — real-time video/webcam processing
│   └── utils.py                 # shared helpers (drawing, smoothing, I/O)
├── models/
│   ├── lane_direction_classifier.joblib
│   ├── feature_scaler.joblib
│   └── model_card.json          # chosen algorithm, hyperparameters, CV scores
├── results/
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── sample_outputs/          # saved example visualizations for sanity-checking
├── requirements.txt
├── README.md
├── implementation_plan.md       # this file
└── explanation.md               # full technical write-up
```

---

## 5. Milestones

Execute these **in order**. Do not start a milestone until the previous one's DoD is satisfied.

### Milestone 0 — Environment setup
**Tasks**
1. Create a virtual environment.
2. Write `requirements.txt` (see Section 2) and install it.

**Definition of Done**
- `pip install -r requirements.txt` completes with no errors.
- `python -c "import cv2, numpy, sklearn, pandas, matplotlib, joblib"` runs with no errors.

---

### Milestone 1 — Dataset acquisition & pruning (`src/data_prep.py`)
**Tasks**
1. Download `test_set.zip` and `test_label.json` (see Section 3.3 for sources; document the exact URL(s) used, e.g. via Kaggle API `kaggle datasets download`).
2. Extract the zip into `data/raw/`.
3. Walk every clip subfolder; copy **only** the 20th frame image (the one referenced by `raw_file` in `test_label.json`) into `data/images/`, renamed to a flat, unique filename (e.g. `img_00001.jpg`).
4. Build a mapping table (`data/labels/image_index.csv`) with columns: `image_filename, raw_file, annotation_index`.
5. Parse `test_label.json` (one JSON object per line) into a `pandas.DataFrame` with columns `raw_file, lanes, h_samples`, joined against the mapping table.
6. Optionally delete `data/raw/` after extraction to reclaim disk space.

**Definition of Done**
- `data/images/` contains exactly 2,782 `.jpg` files (or fewer, minus any corrupt/unreadable files, which must be logged).
- `data/labels/parsed_annotations.parquet` (or `.csv`) exists with one row per image.
- Script prints total size of `data/` and confirms it is under ~1 GB.

---

### Milestone 2 — Ground-truth direction label generation (`src/labeling.py`)

This milestone converts TuSimple's raw per-point annotation into a single `{left, straight, right}` label per image, which becomes the **training target** for the classifier.

**Tasks**
1. `identify_ego_lanes(lanes, h_samples, image_width)`: from the list of annotated lane lines, select the two whose x-position (at the bottom-most valid `h_sample`) straddle the image's horizontal center — these are the left and right boundaries of the ego-lane. Lanes marked invalid at a given row use the sentinel value `-2`; skip those rows.
2. Build the set of `(y, x_center)` points where `x_center = (x_left + x_right) / 2`, for every `h_sample` row where both boundaries are valid.
3. Require at least 3 valid points; otherwise flag the image as `insufficient_annotation` and exclude it from training.
4. Fit a 2nd-degree polynomial `x(y) = A·y² + B·y + C` via `numpy.polyfit(y, x_center, 2)`.
5. Collect the coefficient `A` for every image into an array.
6. Compute two thresholds from the **empirical distribution** of `A` using tertiles (`numpy.percentile` at the 33rd and 67th percentile) rather than a hand-picked constant — this keeps the three classes approximately balanced regardless of the dataset's natural curvature distribution. Persist these thresholds to `models/label_thresholds.json` for reproducibility.
7. Assign `direction = "left"` if `A` is below the lower threshold, `"right"` if above the upper threshold, else `"straight"`. (Note: confirm the left/right sign convention empirically against a handful of visually-obvious sample images in image coordinates — document the confirmed convention in `explanation.md` once verified, since image-coordinate y increases downward.)

**Definition of Done**
- `data/labels/labels_with_direction.csv` produced with a `direction` column.
- Printed class distribution shows all three classes present and roughly balanced (no class below ~20% of the total).
- For 10 manually-chosen sample images (a mix of visibly left-curving, right-curving, and straight roads), the script saves an overlay image (ground-truth curve drawn in a distinct color + the assigned label as text) to `results/sample_outputs/gt_check/` for manual visual confirmation that the sign convention (left vs. right) is correct.

---

### Milestone 3 — Classical CV lane-pixel detection pipeline (`src/cv_pipeline.py`)

**Tasks** — implement each as an independently testable function:
1. `to_binary_mask(image) -> binary_image`: convert to HLS color space; threshold the S/L channels to isolate white and yellow lane-marking colors; combine with a Sobel-x gradient threshold on grayscale to catch high-contrast edges; combine (logical OR) into a single binary mask. **This binary mask is the "CV processed image" shown in display window 2.**
2. `apply_roi_mask(binary_image) -> masked_image`: apply a fixed trapezoidal region-of-interest mask covering the road area ahead of the camera, zeroing out sky/roadside clutter.
3. *(Recommended, not strictly required)* `warp_to_birdseye(image, src_pts, dst_pts) -> warped_image` and its inverse, using `cv2.getPerspectiveTransform` / `cv2.warpPerspective`, with a fixed, hand-tuned set of source/destination points appropriate for TuSimple's consistent camera framing. A bird's-eye view makes curvature estimation more geometrically meaningful, but a direct fit in image space is an acceptable simpler fallback if time is limited — document which was used.
4. `detect_lane_pixels(masked_binary) -> (pixel_ys, pixel_xs_left, pixel_xs_right)`: use a histogram-peak + sliding-window approach (scan column-sum histogram of the bottom half of the image to find left/right lane base x-positions, then step upward in horizontal bands re-centering the window on the mean x of pixels found in the previous band) to collect the pixel coordinates belonging to the left and right lane lines.
5. `fit_polynomial(pixel_ys, pixel_xs) -> (A, B, C)`: wraps `numpy.polyfit`, degree 2, with a guard clause returning `None` if fewer than 3 points are available.

**Definition of Done**
- Running `cv_pipeline.py` directly on 5 sample images (as a `__main__` smoke test) displays/saves: the binary mask, the ROI-masked result, the detected pixel scatter, and the fitted curve overlay — without exceptions.
- No crash on at least 50 randomly sampled images from `data/images/`.

---

### Milestone 4 — Feature extraction (`src/features.py`)

**Tasks**
1. Implement `extract_features(image) -> (feature_vector: np.ndarray, feature_names: list[str])` that runs the Milestone 3 pipeline internally and computes the following features (minimum set — extend if useful):

   | # | Feature | Description |
   |---|---|---|
   | 1 | `A_fit` | Quadratic coefficient of the fitted ego-lane center curve (from **detected**, not ground-truth, pixels) |
   | 2 | `B_fit` | Linear coefficient of the same fit |
   | 3 | `delta_x_top_bottom` | Fitted curve's x-position at the top of the ROI minus its x-position at the bottom |
   | 4 | `mean_slope` | Average derivative of the fitted curve across the ROI's y-range |
   | 5 | `left_pixel_count` / `right_pixel_count` | Number of detected binary-mask pixels assigned to each lane boundary |
   | 6 | `pixel_count_asymmetry` | `(right_count − left_count) / (right_count + left_count + ε)` |
   | 7 | `lane_center_offset` | Fitted lane-center x-position at the bottom row minus image-center x |
   | 8 | `std_x_per_row` | Standard deviation of detected pixel x-position across rows (a rough proxy for how "curvy" vs. "noisy" the detection is) |

2. Ensure a fixed, documented feature order (needed so training and inference use identical column ordering).
3. Handle failure gracefully: if the CV pipeline cannot produce a valid fit (e.g., no lane pixels detected), return `None` and log the filename for exclusion.

**Definition of Done**
- Running feature extraction across all of `data/images/` produces `data/labels/features.csv` (one row per successfully processed image, joined with its Milestone-2 `direction` label) with a failure rate under ~2%.
- All failed images are logged to `results/feature_extraction_failures.txt`.

---

### Milestone 5 — Model training (`src/train.py`)

**Tasks**
1. Load `features.csv`; drop rows with missing features; encode `direction` labels with `sklearn.preprocessing.LabelEncoder`.
2. Stratified split: `train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)`.
3. Standardize features with `sklearn.preprocessing.StandardScaler` (fit on train only, apply to both splits).
4. Train and compare, via 5-fold stratified cross-validation on the training split:
   - `LogisticRegression` (linear baseline)
   - `SVC(kernel="rbf")`
   - `RandomForestClassifier`
   - *(optional additional candidate)* `GradientBoostingClassifier` or `KNeighborsClassifier`
5. Select the candidate with the best mean cross-validated accuracy (report F1-macro alongside accuracy, since class sizes may not be perfectly equal). Do a small hyperparameter sweep (e.g., `GridSearchCV`) around the winning model's key hyperparameters (e.g., `C`/`gamma` for SVC, `n_estimators`/`max_depth` for RandomForest).
6. Refit the chosen, tuned model on the full training split.
7. Persist: `models/lane_direction_classifier.joblib` (the fitted model), `models/feature_scaler.joblib` (the fitted scaler), `models/label_encoder.joblib`, and `models/model_card.json` documenting the chosen algorithm, final hyperparameters, and cross-validation scores.

**Definition of Done**
- All three model artifacts exist under `models/`.
- Console/log output shows a comparison table of CV accuracy (and F1-macro) across all candidate algorithms.
- `model_card.json` records the winning configuration.

---

### Milestone 6 — Evaluation (`src/evaluate.py`)

**Tasks**
1. Load the held-out test split (from Milestone 5's split) and the trained model/scaler.
2. Predict on the test split; compute:
   - Overall accuracy
   - Per-class precision, recall, F1 (`sklearn.metrics.classification_report`)
   - Confusion matrix (`sklearn.metrics.confusion_matrix`), rendered as a heatmap with `matplotlib` and saved to `results/confusion_matrix.png`
3. Save the full text report to `results/classification_report.txt`.
4. *(Optional but recommended)* If `RandomForestClassifier` was selected, report `feature_importances_` to show which engineered features actually drive the left/right/straight decision — save as `results/feature_importance.png`. This directly answers "how many did it detect correctly and why" for the grading rubric.

**Definition of Done**
- `results/confusion_matrix.png` and `results/classification_report.txt` exist and are non-empty.
- The overall test accuracy is printed to console and recorded in `explanation.md`'s Results section.

---

### Milestone 7 — Single-image inference demo with 3 windows (`src/infer_image.py`)

**Tasks**
1. CLI: `python src/infer_image.py --image path/to/image.jpg`
2. Load the image; run the Milestone 3 CV pipeline to obtain the binary mask (for window 2) and the detected-pixel polynomial fit.
3. Run Milestone 4's `extract_features` + the trained model/scaler to get a predicted label in `{left, straight, right}`.
4. Draw the fitted curve as a polyline on a copy of the original image; overlay verdict text with `cv2.putText`, e.g. `"LANE: TURNING LEFT"`, `"LANE: TURNING RIGHT"`, or `"LANE: STRAIGHT"`.   
5. Display exactly three `cv2` windows simultaneously:
   - **Window 1 — "Original"**: the unmodified input image.
   - **Window 2 — "CV Processed"**: the binary/edge mask used for lane pixel detection.
   - **Window 3 — "Detected Curve + Verdict"**: the original image with the fitted curve drawn and the verdict text overlay.
6. `cv2.waitKey(0)` then `cv2.destroyAllWindows()` to close on any keypress.

**Definition of Done**
- Running the script on at least 5 test images (a mix of clearly-left, clearly-right, and straight roads, ideally cross-checked against Milestone 2's ground truth) pops up all three windows each time with the correct-looking verdict.

---

### Milestone 8 — Real-time video processing (`src/infer_video.py`) — *Phase 2 / stretch goal*

**Tasks**
1. CLI: `python src/infer_video.py --source path/to/video.mp4` (support `--source 0` for a live webcam).
2. Loop over frames; apply the identical per-frame pipeline (Milestones 3–4) and model inference.
3. Maintain a rolling buffer (`collections.deque(maxlen=10)`) of the last N frames' predicted class (or raw `A_fit` values); display the **majority vote** (or the mean-smoothed curvature's resulting class) rather than the raw single-frame prediction, to prevent the verdict text from flickering between classes on noisy frames.
4. Update all three display windows every frame; exit the loop cleanly on pressing `q`.
5. Log/print the achieved frames-per-second.

**Definition of Done**
- The script runs end-to-end on a sample driving video without crashing.
- The verdict text updates smoothly (no rapid flicker) thanks to temporal smoothing.
- Achieved FPS is reported.

---

### Milestone 9 — Documentation & wrap-up

**Tasks**
1. Write `README.md`: setup instructions, how to run each script, expected outputs.
2. Update `explanation.md`'s "Results" section with the actual numbers produced in Milestone 6 (replace any placeholder values).
3. Confirm every milestone's DoD in this file is satisfied.

**Definition of Done**
- A fresh environment following only `README.md` can reproduce: dataset preparation → label generation → training → evaluation → the single-image 3-window demo.

---

## 6. Testing / validation strategy summary

This project is evaluated as a **supervised classification problem**, exactly as requested:

1. **Ground truth** comes from TuSimple's precise polyline annotations (Milestone 2), not manual labeling.
2. **Features** come from the project's own classical CV detection pipeline run on the raw image (Milestones 3–4) — i.e., the model must learn to predict the correct direction from imperfect, noisy, real-world-detected pixels, not from the ground truth directly.
3. **Train/test split** is stratified (Milestone 5) so all three classes are represented proportionally in both splits.
4. **Model selection** uses 5-fold cross-validation on the training split only, to avoid tuning against the test set.
5. **Final evaluation** is a single, held-out test set pass reporting accuracy, per-class precision/recall/F1, and a confusion matrix (Milestone 6) — this is the "train on these images, then feed test images and see how many it detects correctly" outcome the project explicitly needs.
6. **Qualitative validation** is layered on top via saved sample overlay images at both the ground-truth stage (Milestone 2) and the live-demo stage (Milestone 7), so results can be sanity-checked visually, not just numerically.

---

## 7. Risk register

| Risk | Mitigation |
|---|---|
| Sign convention for "left" vs. "right" curvature is easy to get backwards in image coordinates (y increases downward) | Milestone 2 explicitly requires visual confirmation against sample images before locking in the convention |
| Classical CV pixel detection fails on shadowed/faded-marking frames, corrupting features | Log and exclude failed frames (target <2% failure rate); consider excluding TuSimple frames tagged as low-visibility if this rate is exceeded |
| Class imbalance (most highway driving is "straight") | Tertile-based thresholding in Milestone 2 forces roughly balanced classes by construction; also use `stratify=` in the train/test split and report F1-macro, not just accuracy |
| Video mode verdict flickers frame-to-frame | Rolling-buffer majority vote / smoothing in Milestone 8 |
| Environment/device constraints | Entire pipeline uses classical CV + shallow `scikit-learn` classifiers — no GPU required, training completes in seconds to a few minutes on CPU only |

---

## 8. Execution order (for the coding agent)

Proceed strictly in this order, validating each Definition of Done before continuing:

`Milestone 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → (8, optional) → 9`

Do not skip ahead. If a milestone's DoD cannot be met, stop and report the specific blocker rather than proceeding with an unvalidated pipeline stage.
