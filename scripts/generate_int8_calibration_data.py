#!/usr/bin/env python3
"""Generate a representative calibration dataset for TensorRT INT8 quantization.

Reuses export_onnx.py's make_inputs() (random images/state/noise, tokenized
instruction; camera count and sizes come from the checkpoint config), producing N diverse
samples across different seeds and instructions instead of one fixed sample,
since TensorRT INT8 calibration needs a representative dataset to compute
per-tensor dynamic ranges.

Usage:
    ./work/venv/bin/python scripts/generate_int8_calibration_data.py \
        --checkpoint work/artifacts/smolvla_libero --num-samples 50 \
        --output-dir work/tensorrt/calib_data
"""

from __future__ import annotations

import argparse
from pathlib import Path

from export_onnx import input_names, make_inputs, policy_dimensions
from workspace import configure_workspace, load_smolvla_policy

WORK_ROOT = configure_workspace()

INSTRUCTIONS = [
    "pick up the black bowl and place it on the plate",
    "open the top drawer and put the bowl inside",
    "move the pot to the stove",
    "close the microwave door",
    "pick up the alphabet soup and put it in the basket",
    "turn on the stove",
    "put the cream cheese box in the drawer",
    "pick up the ketchup bottle",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    policy = load_smolvla_policy(args.checkpoint)
    policy.eval()
    dimensions = policy_dimensions(policy)
    names = input_names(dimensions["num_cameras"])

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.num_samples):
        seed = args.seed + i
        instruction = INSTRUCTIONS[i % len(INSTRUCTIONS)]
        inputs = make_inputs(policy, dimensions, seed, instruction)

        sample_dir = args.output_dir / f"sample_{i:04d}"
        sample_dir.mkdir(exist_ok=True)
        for name, tensor in zip(names, inputs, strict=True):
            tensor.numpy().tofile(sample_dir / f"{name}.bin")

    print(f"Wrote {args.num_samples} calibration samples to {args.output_dir}")


if __name__ == "__main__":
    main()
