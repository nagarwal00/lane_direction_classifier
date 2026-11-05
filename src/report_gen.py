# src/report_gen.py
"""
Generates a self-contained HTML report comparing V1 (GradientBoosting)
and V2 (MobileNetV2) side by side.

All images are embedded as base64 so the HTML file is portable.

Output: results/v1_vs_v2_report.html
"""

import os
import json
import base64
from datetime import datetime

RESULTS_DIR    = "results"
MODEL_DIR      = "models"
REPORT_PATH    = os.path.join(RESULTS_DIR, "v1_vs_v2_report.html")

V1_REPORT_TXT  = os.path.join(RESULTS_DIR, "classification_report.txt")
V1_CM_PNG      = os.path.join(RESULTS_DIR, "confusion_matrix.png")
V1_FI_PNG      = os.path.join(RESULTS_DIR, "feature_importance.png")
V1_MODEL_CARD  = os.path.join(MODEL_DIR,   "model_card.json")

V2_MODEL_CARD  = os.path.join(MODEL_DIR,   "cnn_model_card.json")
V2_GRADCAM_L   = os.path.join(RESULTS_DIR, "gradcam_left.png")
V2_GRADCAM_R   = os.path.join(RESULTS_DIR, "gradcam_right.png")
V2_GRADCAM_S   = os.path.join(RESULTS_DIR, "gradcam_straight.png")
V2_SHAP        = os.path.join(RESULTS_DIR, "shap_summary.png")
V2_UNCERT      = os.path.join(RESULTS_DIR, "uncertainty_histogram.png")
V2_CALIB       = os.path.join(RESULTS_DIR, "calibration_curve.png")
V2_LATENCY     = os.path.join(RESULTS_DIR, "latency_benchmark.json")


def img_to_base64(path: str) -> str:
    """Reads an image file and returns a base64-encoded data URI."""
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    ext = path.rsplit(".", 1)[-1].lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")
    return f"data:{mime};base64,{data}"


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def load_txt(path: str) -> str:
    if not os.path.exists(path):
        return "File not found."
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def generate_report():
    """Assembles and writes the HTML report."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    v1_card    = load_json(V1_MODEL_CARD)
    v2_card    = load_json(V2_MODEL_CARD)
    latency    = load_json(V2_LATENCY)
    v1_report  = load_txt(V1_REPORT_TXT)

    v1_acc = v1_card.get("tuned_cv_accuracy", "N/A")
    v1_acc_str = f"{float(v1_acc)*100:.2f}%" if v1_acc != "N/A" else "N/A"
    v2_acc = v2_card.get("test_accuracy", "N/A")
    v2_acc_str = f"{float(v2_acc)*100:.2f}%" if v2_acc != "N/A" else "N/A"
    v2_f1  = v2_card.get("test_macro_f1", "N/A")
    v2_f1_str  = f"{float(v2_f1):.4f}" if v2_f1 != "N/A" else "N/A"

    pt_ms  = latency.get("pytorch", {}).get("mean_latency_ms", "N/A")
    ort_ms = latency.get("onnx", {}).get("mean_latency_ms", "N/A")
    spdup  = latency.get("speedup_factor", "N/A")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Lane Direction Classifier — V1 vs V2 Report</title>
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #f4f6f9; color: #1a1a2e; }}
    header {{ background: #1a1a2e; color: white; padding: 28px 40px; }}
    header h1 {{ margin: 0; font-size: 1.8em; }}
    header p  {{ margin: 4px 0 0; opacity: 0.7; font-size: 0.95em; }}
    .container {{ max-width: 1200px; margin: 30px auto; padding: 0 24px; }}
    h2 {{ color: #1a1a2e; border-left: 5px solid #4a6fa5; padding-left: 14px; margin-top: 40px; }}
    h3 {{ color: #4a6fa5; }}
    .card {{ background: white; border-radius: 10px; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 24px; }}
    .compare-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
    .v1 {{ border-top: 4px solid #e07b39; }}
    .v2 {{ border-top: 4px solid #3ca65a; }}
    .tag {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.8em; font-weight: bold; margin-bottom: 8px; }}
    .tag-v1 {{ background: #fdebd0; color: #e07b39; }}
    .tag-v2 {{ background: #d5f0dc; color: #3ca65a; }}
    .metric-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0; }}
    .metric {{ background: #f0f4ff; border-radius: 8px; padding: 12px 20px; text-align: center; min-width: 120px; }}
    .metric .value {{ font-size: 1.6em; font-weight: bold; color: #1a1a2e; }}
    .metric .label {{ font-size: 0.8em; color: #666; margin-top: 4px; }}
    img.result {{ max-width: 100%; border-radius: 6px; border: 1px solid #ddd; margin: 10px 0; }}
    pre {{ background: #f0f4ff; border-radius: 6px; padding: 16px; font-size: 0.85em; overflow-x: auto; white-space: pre-wrap; }}
    .gradcam-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
    .gradcam-grid img {{ width: 100%; border-radius: 6px; }}
    .gradcam-grid p {{ text-align: center; font-size: 0.85em; color: #555; margin: 4px 0 0; }}
    footer {{ text-align: center; padding: 24px; color: #888; font-size: 0.85em; }}
  </style>
</head>
<body>

<header>
  <h1>Lane Direction Classifier — V1 vs V2 Evaluation Report</h1>
  <p>Generated: {timestamp} | Project: lane-direction-classifier | Author: Nistha Agarwal, TIET</p>
</header>

<div class="container">

  <h2>1. Model Comparison Summary</h2>
  <div class="compare-grid">

    <div class="card v1">
      <span class="tag tag-v1">V1 — Classical ML</span>
      <h3>GradientBoosting + Hand-Crafted Features</h3>
      <div class="metric-row">
        <div class="metric"><div class="value">{v1_acc_str}</div><div class="label">CV Accuracy</div></div>
        <div class="metric"><div class="value">{v1_card.get('chosen_model', 'GradientBoosting')}</div><div class="label">Best Model</div></div>
        <div class="metric"><div class="value">9</div><div class="label">Features</div></div>
        <div class="metric"><div class="value">{v1_card.get('num_train_samples', 375)}</div><div class="label">Train Samples</div></div>
      </div>
      <p><strong>Approach:</strong> OpenCV lane detection → 9 geometric features → GridSearchCV model selection</p>
    </div>

    <div class="card v2">
      <span class="tag tag-v2">V2 — Deep Learning</span>
      <h3>MobileNetV2 Fine-Tuned + Augmentation</h3>
      <div class="metric-row">
        <div class="metric"><div class="value">{v2_acc_str}</div><div class="label">Test Accuracy</div></div>
        <div class="metric"><div class="value">{v2_f1_str}</div><div class="label">Macro F1</div></div>
        <div class="metric"><div class="value">MobileNetV2</div><div class="label">Architecture</div></div>
        <div class="metric"><div class="value">{v2_card.get('train_samples', 'N/A')}</div><div class="label">Train Samples</div></div>
      </div>
      <p><strong>Approach:</strong> ImageNet pretrained → 2-phase fine-tuning → Albumentations augmentation</p>
    </div>
  </div>

  <h2>2. V1 — Classification Report</h2>
  <div class="card">
    <pre>{v1_report}</pre>
  </div>

  <div class="compare-grid">
    <div class="card">
      <h3>V1 — Confusion Matrix</h3>
      <img class="result" src="{img_to_base64(V1_CM_PNG)}" alt="V1 Confusion Matrix">
    </div>
    <div class="card">
      <h3>V1 — Feature Importance</h3>
      <img class="result" src="{img_to_base64(V1_FI_PNG)}" alt="V1 Feature Importance">
    </div>
  </div>

  <h2>3. V1 — SHAP Explainability</h2>
  <div class="card">
    <p>SHAP (SHapley Additive exPlanations) shows how each geometric feature pushes the
    model's prediction. Red bars increase the predicted class score; blue bars decrease it.</p>
    <img class="result" src="{img_to_base64(V2_SHAP)}" alt="SHAP Summary">
  </div>

  <h2>4. V2 — Grad-CAM Visualisations</h2>
  <div class="card">
    <p>Grad-CAM heatmaps show which image regions the MobileNetV2 CNN attends to
    when predicting each class. Warm colours (red/yellow) = high attention.</p>
    <div class="gradcam-grid">
      <div>
        <img src="{img_to_base64(V2_GRADCAM_L)}" alt="Grad-CAM LEFT">
        <p>LEFT prediction</p>
      </div>
      <div>
        <img src="{img_to_base64(V2_GRADCAM_S)}" alt="Grad-CAM STRAIGHT">
        <p>STRAIGHT prediction</p>
      </div>
      <div>
        <img src="{img_to_base64(V2_GRADCAM_R)}" alt="Grad-CAM RIGHT">
        <p>RIGHT prediction</p>
      </div>
    </div>
  </div>

  <h2>5. V2 — Uncertainty Quantification (MC Dropout)</h2>
  <div class="card">
    <p>Monte Carlo Dropout runs 20 stochastic forward passes per frame.
    Frames below the 80% confidence threshold are flagged for human review.</p>
    <img class="result" src="{img_to_base64(V2_UNCERT)}" alt="Uncertainty Histogram">
  </div>

  <h2>6. V2 — Calibration Curve</h2>
  <div class="card">
    <p>A well-calibrated model has points close to the diagonal. This checks whether
    the CNN's confidence scores are trustworthy (e.g., 80% confident → correct ~80% of the time).</p>
    <img class="result" src="{img_to_base64(V2_CALIB)}" alt="Calibration Curve">
  </div>

  <h2>7. ONNX Deployment Benchmark</h2>
  <div class="card">
    <div class="metric-row">
      <div class="metric">
        <div class="value">{pt_ms} ms</div>
        <div class="label">PyTorch CPU</div>
      </div>
      <div class="metric">
        <div class="value">{ort_ms} ms</div>
        <div class="label">ONNX Runtime CPU</div>
      </div>
      <div class="metric">
        <div class="value">{spdup}×</div>
        <div class="label">Speedup</div>
      </div>
      <div class="metric">
        <div class="value">{latency.get('model_size_mb', 'N/A')} MB</div>
        <div class="label">ONNX Model Size</div>
      </div>
    </div>
    <p>ONNX export allows the model to run in production systems (edge devices, C++ runtimes)
    without requiring Python or PyTorch.</p>
  </div>

</div>

<footer>
  <p>Lane Direction Classifier — V1 vs V2 | Nistha Agarwal | TIET 2024–2028 | Generated {timestamp}</p>
</footer>

</body>
</html>"""

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report generated: {REPORT_PATH}")
    return REPORT_PATH


if __name__ == "__main__":
    generate_report()

