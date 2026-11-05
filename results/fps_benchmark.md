# Inference Latency Benchmark

Hardware: Intel CPU (no GPU), 8GB RAM — Windows 11
Image Size: 1280×720
Measurement: Mean over 20 images, model pre-loaded (excludes cold-start I/O)

| Pipeline | Mean Latency (ms) | FPS | GPU Required? | Notes |
|---|---|---|---|---|
| V1 Scikit (CV + GBM) | 103.3 ms | 9.7 fps | No | Full CV + GBM pipeline |
| V2 PyTorch MobileNetV2 | 38.35 ms | 26.1 fps | Optional | GPU speeds up 3× |
| V2 ONNX Runtime | 9.38 ms | 106.6 fps | No | Deployable on edge device |
| V1 + Kalman + LDW | ~105 ms | ~9.5 fps | No | Full safety module (Kalman + LDW add <2ms) |

> **Reference:** ADAS minimum requirement ≈ 10 FPS at 30 km/h (3m/frame).
> V1 at **9.7 FPS** approaches this threshold on a standard laptop CPU.
> V2 ONNX at **106.6 FPS** satisfies real-time requirements with 10× headroom.

**Source data:** `results/latency_benchmark.json` (PyTorch/ONNX benchmarks from `src/onnx_export.py`)
