#!/usr/bin/env python3
"""Check how accurate a SmolVLA checkpoint is when computed in reduced precision.

Runs the model in PyTorch (CPU) once in FP32 and once with all weights and
the VLM+expert cast to the given dtype (the projection layers and denoising
state stay FP32, as in the model's own design), on the reference inputs, and reports the
difference. If reduced precision is already inaccurate here, no TensorRT
build of that dtype can be accurate, so this is a cheap first check before
exporting.

Usage:
    ./work/venv/bin/python scripts/compare_pytorch_dtypes.py \
        --checkpoint smolvla_base --dtypes float16 bfloat16
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from export_onnx import ExportableSmolVLA, cast_policy, make_inputs, policy_dimensions
from workspace import load_smolvla_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dtypes", nargs="+", choices=["float16", "bfloat16"], default=["float16", "bfloat16"]
    )
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output", type=Path, default=None, help="optional JSON report path")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    policy = load_smolvla_policy(args.checkpoint).to(device="cpu", dtype=torch.float32).eval()
    dimensions = policy_dimensions(policy)
    inputs = make_inputs(policy, dimensions, args.seed)

    start = time.perf_counter()
    with torch.inference_mode():
        reference = ExportableSmolVLA(policy, dimensions).eval()(*inputs).numpy()
    print(f"float32 forward: {time.perf_counter() - start:.1f}s")

    reports = []
    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        reduced_policy = cast_policy(copy.deepcopy(policy), dtype).eval()
        start = time.perf_counter()
        with torch.inference_mode():
            reduced = ExportableSmolVLA(reduced_policy, dimensions).eval()(*inputs).numpy()
        print(f"{dtype_name} forward: {time.perf_counter() - start:.1f}s")
        if not np.isfinite(reduced).all():
            print(f"WARNING: {dtype_name} output contains non-finite values")
        difference = np.abs(reference.astype(np.float64) - reduced.astype(np.float64))
        reports.append(
            {
                "checkpoint": str(args.checkpoint),
                "dtype": dtype_name,
                "max_abs_error": float(np.nanmax(difference)),
                "mean_abs_error": float(np.nanmean(difference)),
                "float32_range": [float(reference.min()), float(reference.max())],
                "reduced_range": [float(np.nanmin(reduced)), float(np.nanmax(reduced))],
            }
        )
        del reduced_policy
    report = reports
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
