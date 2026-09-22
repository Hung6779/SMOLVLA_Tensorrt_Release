#!/usr/bin/env python3
"""Run an ONNX model through ONNX Runtime and compare it to the FP32 baseline.

Middle stage between compare_pytorch_dtypes.py (PyTorch only, no ONNX/TensorRT)
and validate_tensorrt.py (TensorRT engine vs FP32): this checks the exported
ONNX graph itself, run by ONNX Runtime, before spending a TensorRT build on
it. Used for both the FP16 graph (export_onnx.py --dtype float16 output) and
the INT8 weight-only graph (int8_weight_only_quantize_onnx.py output).

Usage:
    ./work/venv/bin/python scripts/compare_onnx_vs_fp32.py \
        --fp32-model work/onnx_base/fp32/model.onnx \
        --other-model work/onnx_base/fp16/model.onnx \
        --reference-dir work/onnx_base/fp32/reference \
        --output work/onnx_base/fp16_ort_vs_fp32_comparison.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-model", type=Path, required=True)
    parser.add_argument("--other-model", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    inputs = {path.stem: np.load(path) for path in sorted(args.reference_dir.glob("*.npy"))}

    sess32 = ort.InferenceSession(str(args.fp32_model), providers=["CPUExecutionProvider"])
    out32 = sess32.run(["actions"], inputs)[0]

    print(f"loading {args.other_model} (may take a bit for a large graph)...", flush=True)
    sess_other = ort.InferenceSession(str(args.other_model), providers=["CPUExecutionProvider"])
    out_other = sess_other.run(["actions"], inputs)[0]

    diff = np.abs(out32.astype(np.float64) - out_other.astype(np.float64))
    report = {
        "fp32_model": str(args.fp32_model),
        "other_model": str(args.other_model),
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "fp32_range": [float(out32.min()), float(out32.max())],
        "other_range": [float(out_other.min()), float(out_other.max())],
    }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
