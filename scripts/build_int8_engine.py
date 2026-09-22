#!/usr/bin/env python3
"""Build a TensorRT INT8 engine for SmolVLA using real calibration data.

Unlike `trtexec --int8` with no --calib (which calibrates on random noise),
this feeds a representative calibration dataset (see
generate_int8_calibration_data.py) through TensorRT's IInt8EntropyCalibrator2
so per-tensor dynamic ranges reflect realistic activations.

Must run with a Python that has the `tensorrt` bindings installed — on this
Jetson that's the system python3 (apt package python3-libnvinfer), not the
project's venv. Usage:

    python3 scripts/build_int8_engine.py \
        --onnx work/onnx/fp32/model.trt.onnx \
        --calib-dir work/tensorrt/calib_data \
        --output work/tensorrt/smolvla-int8.engine \
        --cache work/tensorrt/int8_calibration.cache
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorrt as trt


def network_input_specs(network) -> dict:
    """Map each network input name to (shape, numpy dtype), read from the parsed ONNX."""
    specs = {}
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        specs[tensor.name] = (tuple(tensor.shape), np.dtype(trt.nptype(tensor.dtype)))
    return specs


TRT_LOGGER = trt.Logger(trt.Logger.INFO)


class SmolVLACalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calib_dir: Path, cache_path: Path, input_specs: dict):
        super().__init__()
        self.cache_path = cache_path
        self.input_specs = input_specs
        self.sample_dirs = sorted(p for p in calib_dir.iterdir() if p.is_dir())
        if not self.sample_dirs:
            raise FileNotFoundError(f"No calibration samples found under {calib_dir}")
        self.index = 0

        from cuda import cudart

        self.cudart = cudart
        self.device_buffers = {}
        for name, (shape, dtype) in self.input_specs.items():
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            err, ptr = cudart.cudaMalloc(nbytes)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMalloc failed for {name}: {err}")
            self.device_buffers[name] = ptr

    def get_batch_size(self) -> int:
        return 1

    def get_batch(self, names):
        if self.index >= len(self.sample_dirs):
            return None
        sample_dir = self.sample_dirs[self.index]
        self.index += 1

        pointers = []
        for name in names:
            shape, dtype = self.input_specs[name]
            data = np.ascontiguousarray(
                np.fromfile(sample_dir / f"{name}.bin", dtype=dtype).reshape(shape)
            )
            ptr = self.device_buffers[name]
            err, = self.cudart.cudaMemcpy(
                ptr, data.ctypes.data, data.nbytes,
                self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
            if err != self.cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMemcpy failed for {name}: {err}")
            pointers.append(int(ptr))
        return pointers

    def read_calibration_cache(self):
        if self.cache_path.is_file():
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache: memoryview) -> None:
        self.cache_path.write_bytes(cache)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--calib-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--timing-cache", type=Path, default=None)
    parser.add_argument("--workspace-mib", type=int, default=8192)
    args = parser.parse_args()

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser_ = trt.OnnxParser(network, TRT_LOGGER)

    with open(args.onnx, "rb") as f:
        if not parser_.parse(f.read(), path=str(args.onnx)):
            for i in range(parser_.num_errors):
                print(parser_.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib * 1024 * 1024)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)  # allow FP16/FP32 fallback for INT8-unfriendly layers
    config.int8_calibrator = SmolVLACalibrator(
        args.calib_dir, args.cache, network_input_specs(network)
    )

    if args.timing_cache is not None:
        cache_data = args.timing_cache.read_bytes() if args.timing_cache.is_file() else b""
        timing_cache = config.create_timing_cache(cache_data)
        config.set_timing_cache(timing_cache, ignore_mismatch=False)

    print("Building INT8 engine (calibrating on {} samples)...".format(
        len(list(args.calib_dir.iterdir()))
    ))
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed")

    args.output.write_bytes(serialized_engine)
    print(f"Saved {args.output} ({args.output.stat().st_size / (1024**2):.1f} MiB)")

    if args.timing_cache is not None:
        updated = config.get_timing_cache()
        args.timing_cache.write_bytes(updated.serialize())


if __name__ == "__main__":
    main()
