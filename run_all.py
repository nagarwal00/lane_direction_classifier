#!/usr/bin/env python3
"""
Master Pipeline Runner for Lane Curvature Direction Classifier (V1 + V2 + V3).
Supports full execution, validation, testing, and report generation.

Usage:
    python run_all.py --mode all     # Runs complete end-to-end pipeline (V1 + V2 + Analysis + Tests)
    python run_all.py --mode v1      # Runs V1 classical ML pipeline + evaluation + single/video demo
    python run_all.py --mode v2      # Runs V2 deep learning evaluation, Grad-CAM, MC Dropout, ONNX
    python run_all.py --mode test    # Runs pytest automated test suite only
"""

import os
import sys
import time
import argparse
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Auto-Detect and Re-exec with Project Virtual Environment ───────────────
venv_python = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
if not os.path.exists(venv_python):
    venv_python = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")

if os.path.exists(venv_python):
    current_exe = os.path.abspath(sys.executable).lower()
    target_exe = os.path.abspath(venv_python).lower()
    if current_exe != target_exe:
        try:
            import joblib
            import torch
            import cv2
        except ImportError:
            print(f"[AUTO-ENV] Detected virtual environment at: {venv_python}")
            print(f"[AUTO-ENV] Current python ({sys.executable}) lacks required packages.")
            print(f"[AUTO-ENV] Re-launching under project virtual environment...\n")
            res = subprocess.run([venv_python] + sys.argv, cwd=PROJECT_ROOT)
            sys.exit(res.returncode)
    PYTHON_EXE = venv_python
else:
    PYTHON_EXE = sys.executable


def print_banner(text: str):
    print("\n" + "=" * 75)
    print(f"  {text}")
    print("=" * 75)


def run_command(cmd_args, step_name: str, check: bool = True):
    print(f"\n>>> [STEP] {step_name}")
    print(f"    Command: {' '.join(cmd_args)}")
    t0 = time.time()
    result = subprocess.run(cmd_args, cwd=PROJECT_ROOT)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"    FAILED (Exit code {result.returncode}) after {elapsed:.1f}s")
        if check:
            print(f"\n[CRITICAL ERROR] Pipeline aborted at step: {step_name}")
            sys.exit(result.returncode)
        return False
    print(f"    PASSED in {elapsed:.1f}s")
    return True


def run_v1_pipeline():
    print_banner("PHASE 1: V1 Classical Machine Learning Pipeline")
    # 1. Evaluate existing trained model on test set
    run_command([PYTHON_EXE, "src/evaluate.py"], "Evaluate V1 Model on Test Set")
    # 2. Single image inference
    run_command([PYTHON_EXE, "src/infer_image.py", "--headless"], "Single Image Inference with Reliability Gate")
    # 3. Video inference with Kalman & LDW
    run_command([PYTHON_EXE, "src/infer_video.py", "--headless"], "Video Stream Inference (Kalman Filter + LDW)")


def run_v2_pipeline():
    print_banner("PHASE 2: V2 Deep Learning & Explainability Pipeline")
    # 1. Augmentation validation
    run_command([PYTHON_EXE, "src/augment.py"], "Verify Albumentations Transformations")
    # 2. CNN evaluation & Grad-CAM
    run_command([PYTHON_EXE, "src/cnn_evaluate.py"], "Evaluate MobileNetV2 & Generate Grad-CAM Heatmaps")
    # 3. MC Dropout uncertainty
    run_command([PYTHON_EXE, "src/uncertainty.py"], "Monte Carlo Dropout Uncertainty Analysis")
    # 4. ONNX Export and Latency Benchmark
    run_command([PYTHON_EXE, "src/onnx_export.py"], "ONNX Model Export & Benchmark")
    # 5. Video V2 inference
    run_command([PYTHON_EXE, "src/infer_video_v2.py", "--headless"], "V2 Video Inference with Confidence Overlays")
    # 6. SHAP explainability
    run_command([PYTHON_EXE, "src/shap_explain.py"], "SHAP Explainability Analysis (Summary & Waterfall Plots)")


def run_analysis_and_reports():
    print_banner("PHASE 3: Edge-Case Robustness & Final Report Generation")
    # 1. Misclassification analysis
    run_command([PYTHON_EXE, "scratch/analyze_misclassified.py"], "Misclassification Analysis & Image Isolation")
    # 2. Robustness edge cases
    run_command([PYTHON_EXE, "scratch/test_robustness_edge_cases.py"], "14-Scenario Robustness Testing")
    # 3. HTML Report generation
    run_command([PYTHON_EXE, "src/report_gen.py"], "Generate Comparison HTML Report")
    run_command([PYTHON_EXE, "scratch/generate_final_comprehensive_report.py"], "Generate Comprehensive Final Project Report")


def run_tests():
    print_banner("PHASE 4: Automated Testing Suite")
    run_command([PYTHON_EXE, "-m", "pytest", "tests/test_pipeline.py", "-v"], "Pytest Automated Pipeline Test Suite")


def main():
    # Normalize common CLI argument formats (e.g., PowerShell '-mode', bare 'all', etc.)
    raw_args = sys.argv[1:]
    normalized_args = []
    i = 0
    while i < len(raw_args):
        arg = raw_args[i]
        if arg in ["-mode", "/mode"]:
            normalized_args.append("--mode")
        elif arg in ["all", "v1", "v2", "test", "report"]:
            if not normalized_args or normalized_args[-1] != "--mode":
                normalized_args.extend(["--mode", arg])
            else:
                normalized_args.append(arg)
        else:
            normalized_args.append(arg)
        i += 1

    parser = argparse.ArgumentParser(description="Lane Direction Classifier Master Runner")
    parser.add_argument(
        "--mode", "-m",
        choices=["all", "v1", "v2", "test", "report"],
        default="all",
        help="Pipeline execution mode (default: all)"
    )
    args = parser.parse_args(normalized_args)

    start_total = time.time()
    print_banner(f"STARTING LANE CLASSIFIER MASTER PIPELINE (Mode: {args.mode.upper()})")

    if args.mode == "test":
        run_tests()
    elif args.mode == "v1":
        run_v1_pipeline()
        run_tests()
    elif args.mode == "v2":
        run_v2_pipeline()
        run_tests()
    elif args.mode == "report":
        run_analysis_and_reports()
    else:  # all
        run_v1_pipeline()
        run_v2_pipeline()
        run_analysis_and_reports()
        run_tests()

    total_time = time.time() - start_total
    print_banner(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.1f}s")
    print("All models, visualizations, reports, and tests are verified and up to date.")
    print("Reports available at:")
    print("  - results/final_project_report.html")
    print("  - results/v1_vs_v2_report.html")
    print("  - results/robustness_report.md")
    print("  - results/fps_benchmark.md\n")


if __name__ == "__main__":
    main()

