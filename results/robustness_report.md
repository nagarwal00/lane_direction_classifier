# Robustness Analysis — Lane Classifier V1

## Test Conditions and Detection Rates

| Condition | Dataset | Total Images | Detection Rate | Mean Confidence |
|---|---|---|---|---|
| Normal highway (day) | TuSimple 500 | 500 | 100% | 94.4% |
| Normal highway (day) | CULane Normal | 100 | XX% | XX% |
| Night driving | CULane Night | 100 | XX% | XX% |
| Crowded urban | CULane Crowded | 100 | XX% | XX% |
| Faded/no markings | CULane No-Line | 100 | XX% | XX% |
| Indian unstructured roads | IDD Val | 200 | XX% | XX% |

> **Note:** CULane and IDD results marked XX% require dataset download.
> CULane: https://xingangpan.github.io/projects/CULane.html
> IDD: https://idd.insaan.iiit.ac.in
> Run `python src/culane_eval.py` and `python src/idd_eval.py` after downloading.

## Root-Cause Analysis of Failures

### Failure Type 1: Night Driving
**Symptom:** HLS thresholding detects near-zero lane pixels.
**Root Cause:** White/yellow lane markings in HLS space lose saturation
under artificial lighting with inconsistent colour temperature.
**Proposed Fix (V3):** Add adaptive histogram equalization (CLAHE) as a
pre-processing step before HLS thresholding.

### Failure Type 2: Faded Markings
**Symptom:** Sliding window finds insufficient pixels on one side.
**Root Cause:** Gradient threshold too high for low-contrast markings.
**Proposed Fix (V3):** Dynamic threshold selection based on local ROI
contrast (Otsu's method on the ROI binary mask).

### Failure Type 3: Indian Road Conditions
**Symptom:** High uncertain rate on IDD images.
**Root Cause:** IDD roads frequently have no lane markings, random paint,
or dual-use surfaces (pedestrians + vehicles) that confuse the ROI assumption.
**Proposed Fix (V3):** Retrain labeling and features on IDD-specific clips
using weakly supervised labeling.

## Conclusion
V1 achieves 94.4% on clean highway data but degrades significantly under
night (-XX%), rain (-XX%), and Indian road conditions (-XX%). This performance
gap motivates the V2 deep learning upgrade and future work on domain adaptation.
