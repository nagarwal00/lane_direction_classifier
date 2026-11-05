# src/shap_explain.py
"""
SHAP (SHapley Additive exPlanations) analysis for the V1 GradientBoosting classifier.

Generates:
  - results/shap_summary.png          : Beeswarm plot of all features across test samples
  - results/shap_waterfall_left.png   : Waterfall plot for one LEFT prediction
  - results/shap_waterfall_right.png  : Waterfall plot for one RIGHT prediction
  - results/shap_waterfall_straight.png: Waterfall plot for one STRAIGHT prediction

SHAP explains WHY the model made a specific prediction by attributing
the prediction score to each input feature. A positive SHAP value pushes
the prediction towards the predicted class; a negative value pushes away.
"""

import os
import sys
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

MODEL_FILE   = "models/lane_direction_classifier.joblib"
SCALER_FILE  = "models/feature_scaler.joblib"
ENCODER_FILE = "models/label_encoder.joblib"
TEST_SPLIT   = "models/test_split.npz"
RESULTS_DIR  = "results"


def generate_shap_plots():
    """
    Loads V1 model and test set, computes SHAP values, and saves visualisations.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model   = joblib.load(MODEL_FILE)
    scaler  = joblib.load(SCALER_FILE)
    encoder = joblib.load(ENCODER_FILE)

    data        = np.load(TEST_SPLIT)
    X_test      = scaler.transform(data["X_test"])
    y_test      = data["y_test"]
    feat_names  = list(data["feature_names"])
    class_names = list(encoder.classes_)

    print("Setting up SHAP Explainer on test set...")
    # Attempt TreeExplainer first as specified; fallback to model-agnostic Explainer if multiclass tree ensemble is unsupported
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        has_tree = True
    except Exception as e:
        print(f"TreeExplainer note: {e}")
        print("Using SHAP model Explainer with background dataset...")
        background = shap.sample(X_test, 40, random_state=42)
        explainer  = shap.Explainer(model.predict_proba, background)
        has_tree = False

    # ── Beeswarm Summary Plot ───────────────────────────────────────────────
    print("Generating SHAP beeswarm summary plot...")
    plt.figure(figsize=(12, 6))

    if has_tree:
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=feat_names,
            class_names=class_names,
            show=False
        )
    else:
        # Use first 40 test samples for fast, clean summary
        summary_subset = X_test[:40]
        exp_summary = explainer(summary_subset)
        shap_values_list = [exp_summary.values[:, :, i] for i in range(len(class_names))]
        shap.summary_plot(
            shap_values_list,
            summary_subset,
            feature_names=feat_names,
            class_names=class_names,
            show=False
        )

    plt.title("SHAP Feature Importance — V1 GradientBoosting Classifier", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "shap_summary.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: results/shap_summary.png")

    # ── Per-Class Waterfall Plots ───────────────────────────────────────────
    for class_idx, class_name in enumerate(class_names):
        matching_indices = np.where(y_test == class_idx)[0]
        if len(matching_indices) == 0:
            print(f"  No test samples for class '{class_name}' — skipping waterfall.")
            continue

        sample_idx = matching_indices[0]
        print(f"Generating SHAP waterfall for class '{class_name}' (sample {sample_idx})...")

        if has_tree:
            shap_exp = shap.Explanation(
                values=shap_values[class_idx][sample_idx],
                base_values=explainer.expected_value[class_idx],
                data=X_test[sample_idx],
                feature_names=feat_names
            )
        else:
            single_exp = explainer(X_test[sample_idx:sample_idx+1])
            # single_exp shape: [1, n_features, n_classes]
            shap_exp = single_exp[0, :, class_idx]
            shap_exp.feature_names = feat_names

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_exp, show=False)
        plt.title(
            f"SHAP Waterfall — True Label: {class_name.upper()} (Sample {sample_idx})",
            fontsize=12, pad=12
        )
        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, f"shap_waterfall_{class_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: results/shap_waterfall_{class_name}.png")

    print("\nSHAP analysis complete.")


if __name__ == "__main__":
    generate_shap_plots()
