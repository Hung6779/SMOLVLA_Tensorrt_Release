#!/usr/bin/env python3
"""Rewrite vlm_with_expert's Linear weights as real INT8 + DequantizeLinear.

Scoped exactly like cast_policy() in export_onnx.py: every MatMul whose
second input is a constant weight AND whose node name is NOT one of the
denoising-loop projections (state_proj, action_in_proj, action_out_proj,
action_time_mlp_in/out) gets its weight replaced with a per-output-channel
symmetric INT8 initializer + a DequantizeLinear node. This is the ONNX-level
version of compare_pytorch_int8.py's `int8_weight_only` mode (same math), but
as real int8-typed tensors so TensorRT's builder can see and optionally
exploit them, instead of a fake-quantized FP32 constant.

Run this on the *FP32* export (after fix_bool_constants_for_trt.py), so
activations stay FP32 everywhere except the weights this script touches.

Usage:
    ./work/venv/bin/python scripts/int8_weight_only_quantize_onnx.py \
        --input work/onnx_base/fp32/model.trt.onnx \
        --output work/onnx_base/int8w/model.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

EXCLUDED_NAME_PARTS = (
    "state_proj",
    "action_in_proj",
    "action_out_proj",
    "action_time_mlp_in",
    "action_time_mlp_out",
)


def is_denoising_projection(node_name: str) -> bool:
    return any(part in node_name for part in EXCLUDED_NAME_PARTS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(args.input), load_external_data=True)
    initializer_names = {i.name for i in model.graph.initializer}
    initializer_by_name = {i.name: i for i in model.graph.initializer}

    # Pass 1: find every weight initializer consumed by an eligible MatMul.
    # Weights are shared across the unrolled denoising-loop iterations (the same
    # nn.Linear is called 10x), so multiple MatMul nodes reference the same
    # initializer — quantize each unique weight exactly once.
    eligible_names = []
    seen = set()
    skipped_dynamic = 0
    for node in model.graph.node:
        if node.op_type != "MatMul" or len(node.input) != 2:
            continue
        if node.input[1] not in initializer_names:
            skipped_dynamic += 1
            continue
        if is_denoising_projection(node.name):
            continue
        if node.input[1] not in seen:
            seen.add(node.input[1])
            eligible_names.append(node.input[1])

    # Pass 2: quantize each unique weight once (int8 initializer + per-channel
    # scale + one DequantizeLinear node).
    new_initializers = []
    dq_nodes = []
    dq_output_by_original = {}
    skipped_non_matrix = 0
    for name in eligible_names:
        weight_init = initializer_by_name[name]
        weight = numpy_helper.to_array(weight_init)
        if weight.ndim != 2:
            skipped_non_matrix += 1
            continue
        # weight is (in_features, out_features); quantize per output channel (axis 1)
        amax = np.abs(weight).max(axis=0, keepdims=True)
        amax = np.maximum(amax, 1e-8)
        scale = (amax / 127.0).astype(np.float32)
        quantized = np.clip(np.round(weight / scale), -127, 127).astype(np.int8)

        int8_name = name + "__int8"
        scale_name = name + "__scale"
        dq_output = name + "__dequant"

        new_initializers.append(numpy_helper.from_array(quantized, name=int8_name))
        new_initializers.append(numpy_helper.from_array(scale.reshape(-1), name=scale_name))
        dq_nodes.append(
            helper.make_node(
                "DequantizeLinear",
                [int8_name, scale_name],
                [dq_output],
                name=name + "_weight_dequant",
                axis=1,
            )
        )
        dq_output_by_original[name] = dq_output

    # Pass 3: rewire *every* node that references an original weight name
    # (not just the directly-consuming MatMul) to the dequantized output.
    # The exporter shares one initializer across the unrolled denoising-loop
    # iterations via `Identity` passthrough nodes for the other 9 copies —
    # those must be rewired too, or they're left pointing at a deleted
    # initializer. Insert the DequantizeLinear nodes before first use.
    new_nodes = []
    inserted = False
    for node in model.graph.node:
        touches_target = any(inp in dq_output_by_original for inp in node.input)
        if not inserted and touches_target:
            new_nodes.extend(dq_nodes)
            inserted = True
        if touches_target:
            new_inputs = [dq_output_by_original.get(inp, inp) for inp in node.input]
            del node.input[:]
            node.input.extend(new_inputs)
        new_nodes.append(node)
    if not inserted:
        new_nodes = dq_nodes + new_nodes

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(new_initializers)
    kept = [i for i in model.graph.initializer if i.name not in dq_output_by_original]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    print(f"Converted {len(dq_output_by_original)} unique vlm_with_expert weights to INT8 (per-output-channel).")
    print(f"Skipped {skipped_non_matrix} non-2D eligible weights, {skipped_dynamic} dynamic MatMuls untouched.")

    onnx.save_model(
        model,
        str(args.output),
        save_as_external_data=True,
        all_tensors_to_one_file=False,
        size_threshold=1024,
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
