#!/usr/bin/env python3
"""Check how accurate SmolVLA is under simulated INT8 quantization, in PyTorch.

Fake-quantizes every nn.Linear in `vlm_with_expert` (the VLM + action expert
only — the denoising-loop projections and Euler state stay FP32, same
boundary as compare_pytorch_dtypes.py/cast_policy for FP16) to INT8, in two
modes:
  - weight-only: per-output-channel symmetric INT8 round-trip on weights,
    activations stay FP32. This is what quantize_onnx_torchao.py's INT4
    weight-only quantization does (which WAS accurate for smolvla_libero,
    just slow on this CPU) — the INT8 analogue.
  - full: weight-only quantization plus per-tensor symmetric INT8 activation
    fake-quant on each Linear's input, calibrated from one forward pass'
    worth of activations. This mirrors what the section-N TensorRT INT8
    calibration build actually quantizes (weights + activations).

No TensorRT/ONNX involved — this is a fast CPU-only sanity check to run
before spending hours on a real INT8 TensorRT build (see build_int8_engine.py).

Usage:
    ./work/venv/bin/python scripts/compare_pytorch_int8.py --checkpoint smolvla_base
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from export_onnx import ExportableSmolVLA, make_inputs, policy_dimensions
from workspace import load_smolvla_policy


def quantize_weight_per_channel(weight: torch.Tensor) -> torch.Tensor:
    """Round-trip `weight` through symmetric per-output-channel INT8."""

    amax = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = amax / 127.0
    quantized = torch.clamp(torch.round(weight / scale), -127, 127)
    return quantized * scale


def apply_weight_only_int8(module: nn.Module) -> None:
    for linear in module.modules():
        if isinstance(linear, nn.Linear):
            with torch.no_grad():
                linear.weight.copy_(quantize_weight_per_channel(linear.weight))


def add_activation_fake_quant(module: nn.Module) -> list:
    """Register forward pre-hooks that fake-quantize each Linear's input to INT8.

    Per-tensor symmetric, scale computed on the fly per call (dynamic
    quantization) so no separate calibration pass is needed.
    """

    handles = []

    def make_hook():
        def hook(mod, args):
            x = args[0]
            amax = x.abs().amax().clamp_min(1e-8)
            scale = amax / 127.0
            x_q = torch.clamp(torch.round(x / scale), -127, 127) * scale
            return (x_q,) + args[1:]

        return hook

    for linear in module.modules():
        if isinstance(linear, nn.Linear):
            handles.append(linear.register_forward_pre_hook(make_hook()))
    return handles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    policy = load_smolvla_policy(args.checkpoint).to(device="cpu", dtype=torch.float32).eval()
    dimensions = policy_dimensions(policy)
    inputs = make_inputs(policy, dimensions, args.seed)

    with torch.inference_mode():
        reference = ExportableSmolVLA(policy, dimensions).eval()(*inputs).numpy()
    print("float32 reference computed")

    reports = []

    weight_only_policy = copy.deepcopy(policy)
    apply_weight_only_int8(weight_only_policy.model.vlm_with_expert)
    with torch.inference_mode():
        weight_only_out = ExportableSmolVLA(weight_only_policy, dimensions).eval()(*inputs).numpy()
    del weight_only_policy
    diff = np.abs(reference.astype(np.float64) - weight_only_out.astype(np.float64))
    reports.append(
        {
            "mode": "int8_weight_only",
            "max_abs_error": float(diff.max()),
            "mean_abs_error": float(diff.mean()),
            "reduced_range": [float(weight_only_out.min()), float(weight_only_out.max())],
        }
    )
    print("int8_weight_only:", json.dumps(reports[-1], indent=2))

    full_policy = copy.deepcopy(policy)
    apply_weight_only_int8(full_policy.model.vlm_with_expert)
    handles = add_activation_fake_quant(full_policy.model.vlm_with_expert)
    with torch.inference_mode():
        full_out = ExportableSmolVLA(full_policy, dimensions).eval()(*inputs).numpy()
    for h in handles:
        h.remove()
    del full_policy
    diff = np.abs(reference.astype(np.float64) - full_out.astype(np.float64))
    reports.append(
        {
            "mode": "int8_weight_and_activation",
            "max_abs_error": float(diff.max()),
            "mean_abs_error": float(diff.mean()),
            "reduced_range": [float(full_out.min()), float(full_out.max())],
        }
    )
    print("int8_weight_and_activation:", json.dumps(reports[-1], indent=2))

    for r in reports:
        r["float32_range"] = [float(reference.min()), float(reference.max())]
        r["checkpoint"] = str(args.checkpoint)

    print(json.dumps(reports, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
