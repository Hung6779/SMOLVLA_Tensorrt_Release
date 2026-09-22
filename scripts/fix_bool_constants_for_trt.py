#!/usr/bin/env python3
"""Rewrite BOOL Constant nodes as INT32 Constant + Cast(to=BOOL) for TensorRT.

TensorRT's builder fails with `Constant does not support output type Bool`
on the boolean mask constants that PyTorch tracing bakes into the exported
graph. Only small in-graph Constant nodes are rewritten; large weights stay in
their existing external-data files (the output must be saved next to the input).

Usage:
    ./work/venv/bin/python scripts/fix_bool_constants_for_trt.py \
        --input work/onnx_base/fp32/model.onnx --output work/onnx_base/fp32/model.trt.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.input.parent.resolve() != args.output.parent.resolve():
        raise ValueError("--output must be in the same directory as --input (shared external data)")

    model = onnx.load(str(args.input), load_external_data=False)

    new_nodes = []
    converted = 0
    for node in model.graph.node:
        value_attr = next(
            (a for a in node.attribute if a.name == "value" and a.t.data_type == TensorProto.BOOL),
            None,
        ) if node.op_type == "Constant" else None
        if value_attr is None:
            new_nodes.append(node)
            continue
        array = numpy_helper.to_array(value_attr.t).astype(np.int32)
        int_output = node.output[0] + "__bool_as_int32"
        new_nodes.append(
            helper.make_node(
                "Constant", [], [int_output], name=node.name,
                value=numpy_helper.from_array(array, name=value_attr.t.name),
            )
        )
        new_nodes.append(
            helper.make_node(
                "Cast", [int_output], [node.output[0]],
                name=node.name + "_cast_to_bool", to=TensorProto.BOOL,
            )
        )
        converted += 1

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    onnx.save_model(model, str(args.output), save_as_external_data=False)
    print(f"Converted {converted} BOOL Constant nodes -> {args.output}")


if __name__ == "__main__":
    main()
