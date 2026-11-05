# src/onnx_export.py
"""
Exports the fine-tuned MobileNetV2 to ONNX format and benchmarks inference speed.

ONNX (Open Neural Network Exchange) is a deployment format that allows
models to run in production environments without Python or PyTorch.
ONNX Runtime is typically 1.5x–3x faster than PyTorch on CPU for inference.

Generates:
  - models/mobilenet_v2_finetuned.onnx  : Deployable ONNX model
  - results/latency_benchmark.json       : Speed comparison results
"""

import os
import sys
import json
import time
import glob
import numpy as np
import torch
import torch.nn.functional as F
import onnx
import onnxruntime as ort
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cnn_train import build_model, CLASS_NAMES, IMAGE_SIZE, CHECKPOINT_PATH
from augment import get_val_transforms

MODEL_DIR   = "models"
RESULTS_DIR = "results"
ONNX_PATH   = os.path.join(MODEL_DIR, "mobilenet_v2_finetuned.onnx")
BENCH_PATH  = os.path.join(RESULTS_DIR, "latency_benchmark.json")

N_WARMUP    = 10    # Warmup runs (not counted in benchmark)
N_BENCH     = 100   # Benchmark runs per model


def export_to_onnx():
    """
    Exports the PyTorch model to ONNX with opset 17.

    Uses a dummy input of shape [1, 3, 224, 224] to trace the computation graph.
    Dynamic axes allow variable batch sizes at runtime.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    device = torch.device("cpu")  # ONNX export must be on CPU

    model = build_model(num_classes=len(CLASS_NAMES), freeze_backbone=False)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,    # Optimises constant expressions
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image":  {0: "batch_size"},
            "logits": {0: "batch_size"}
        }
    )

    # Verify the ONNX model
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)

    model_size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print(f"ONNX model exported successfully: {ONNX_PATH}")
    print(f"ONNX model size: {model_size_mb:.2f} MB")

    return ONNX_PATH


def benchmark_inference(sample_image_path: str = None):
    """
    Benchmarks PyTorch vs ONNX Runtime inference on a single image.

    Args:
        sample_image_path (str): Path to a test image. If None, uses a random synthetic image.

    Returns:
        dict: Benchmark results with mean latency and speedup factor.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device    = torch.device("cpu")
    transform = get_val_transforms(IMAGE_SIZE)

    # ── Prepare sample input ───────────────────────────────────────────────
    if sample_image_path and os.path.exists(sample_image_path):
        img_np = np.array(Image.open(sample_image_path).convert("RGB"))
    else:
        img_np = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    tensor = transform(image=img_np)["image"].unsqueeze(0)  # [1, 3, 224, 224]

    # ── PyTorch Inference ──────────────────────────────────────────────────
    model = build_model(num_classes=len(CLASS_NAMES), freeze_backbone=False)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    print(f"Warming up PyTorch ({N_WARMUP} runs)...")
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = model(tensor)

    print(f"Benchmarking PyTorch ({N_BENCH} runs)...")
    torch_times = []
    with torch.no_grad():
        for _ in range(N_BENCH):
            t0 = time.perf_counter()
            _  = model(tensor)
            torch_times.append((time.perf_counter() - t0) * 1000)  # ms

    torch_mean = float(np.mean(torch_times))
    torch_std  = float(np.std(torch_times))

    # ── ONNX Runtime Inference ─────────────────────────────────────────────
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    tensor_np  = tensor.numpy()

    print(f"Warming up ONNX Runtime ({N_WARMUP} runs)...")
    for _ in range(N_WARMUP):
        _ = session.run(None, {input_name: tensor_np})

    print(f"Benchmarking ONNX Runtime ({N_BENCH} runs)...")
    onnx_times = []
    for _ in range(N_BENCH):
        t0 = time.perf_counter()
        _  = session.run(None, {input_name: tensor_np})
        onnx_times.append((time.perf_counter() - t0) * 1000)

    onnx_mean  = float(np.mean(onnx_times))
    onnx_std   = float(np.std(onnx_times))
    speedup    = torch_mean / onnx_mean

    # ── Results ────────────────────────────────────────────────────────────
    results = {
        "pytorch": {
            "mean_latency_ms": round(torch_mean, 3),
            "std_latency_ms":  round(torch_std, 3),
            "backend": "PyTorch CPU"
        },
        "onnx": {
            "mean_latency_ms": round(onnx_mean, 3),
            "std_latency_ms":  round(onnx_std, 3),
            "backend": "ONNX Runtime CPU"
        },
        "speedup_factor":   round(speedup, 2),
        "benchmark_runs":   N_BENCH,
        "image_size":       IMAGE_SIZE,
        "model_size_mb":    round(os.path.getsize(ONNX_PATH) / (1024 * 1024), 2)
    }

    with open(BENCH_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n{'='*50}")
    print(f"LATENCY BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"PyTorch CPU  : {torch_mean:.2f} ± {torch_std:.2f} ms/frame")
    print(f"ONNX Runtime : {onnx_mean:.2f} ± {onnx_std:.2f} ms/frame")
    print(f"Speedup      : {speedup:.2f}x")
    print(f"Saved: {BENCH_PATH}")

    return results


if __name__ == "__main__":
    export_to_onnx()

    images = sorted(glob.glob("data/images/*.jpg"))
    sample = images[0] if images else None
    benchmark_inference(sample_image_path=sample)

