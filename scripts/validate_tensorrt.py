#!/usr/bin/env python3
"""Validate a TensorRT engine's `actions` output against the FP32 ONNX Runtime baseline.

Converts the learning path's reference .npy inputs to raw binary, runs the engine
via trtexec, and compares against ONNX Runtime CPU execution of the original FP32
ONNX model on the same inputs.

Usage:
    ./work/venv/bin/python scripts/validate_tensorrt.py --engine work/tensorrt/smolvla-fp32.engine
    ./work/venv/bin/python scripts/validate_tensorrt.py --engine work/tensorrt/smolvla-fp16.engine
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnxruntime as ort

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


def load_reference_inputs(reference_dir: Path) -> dict[str, np.ndarray]:
    return {path.stem: np.load(path) for path in sorted(reference_dir.glob("*.npy"))}


def write_raw_inputs(inputs: dict[str, np.ndarray], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, arr in inputs.items():
        path = out_dir / f"{name}.bin"
        arr.tofile(path)
        paths[name] = path
    return paths


def run_ort_baseline(onnx_path: Path, inputs: dict[str, np.ndarray]) -> np.ndarray:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return sess.run(["actions"], inputs)[0]


def run_trt_engine(
    engine_path: Path,
    raw_inputs: dict[str, Path],
    output_json: Path,
    output_shape: tuple[int, ...],
) -> np.ndarray:
    load_inputs = ",".join(f"{name}:{path}" for name, path in raw_inputs.items())
    cmd = [
        TRTEXEC,
        f"--loadEngine={engine_path}",
        f"--loadInputs={load_inputs}",
        "--iterations=1",
        f"--exportOutput={output_json}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or "&&&& PASSED" not in result.stdout:
        print(result.stdout[-4000:], file=sys.stderr)
        print(result.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"trtexec failed for {engine_path}")

    with open(output_json) as f:
        data = json.load(f)
    values = next(item["values"] for item in data if item["name"] == "actions")
    return np.array(values, dtype=np.float32).reshape(output_shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True, help="TensorRT engine file")
    parser.add_argument(
        "--fp32-model", type=Path, default=Path("work/onnx/fp32/model.onnx"),
        help="Baseline FP32 ONNX model for ONNX Runtime comparison",
    )
    parser.add_argument(
        "--reference-dir", type=Path, default=Path("work/onnx/fp32/reference"),
        help="Directory with reference .npy inputs (from export_onnx.py)",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=Path("work/tensorrt"),
        help="Directory for raw input/output scratch files",
    )
    args = parser.parse_args()

    inputs = load_reference_inputs(args.reference_dir)
    raw_inputs = write_raw_inputs(inputs, args.work_dir / "raw_inputs")

    print(f"Running ONNX Runtime FP32 baseline on {args.fp32_model} ...")
    ort_out = run_ort_baseline(args.fp32_model, inputs)

    print(f"Running TensorRT engine {args.engine} ...")
    output_json = args.work_dir / f"{args.engine.stem}_validation_output.json"
    trt_out = run_trt_engine(args.engine, raw_inputs, output_json, tuple(ort_out.shape))

    diff = np.abs(ort_out - trt_out)
    result = {
        "engine": str(args.engine),
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "rmse": float(np.sqrt(np.mean((ort_out - trt_out) ** 2))),
        "ort_fp32_range": [float(ort_out.min()), float(ort_out.max())],
        "trt_range": [float(trt_out.min()), float(trt_out.max())],
    }

    comparison_path = args.work_dir / f"{args.engine.stem}_comparison.json"
    comparison_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"Saved {comparison_path}")


if __name__ == "__main__":
    main()
