# SmolVLA TensorRT

TensorRT deployment and optimization of SmolVLA, including:

- PyTorch → ONNX export
- ONNX validation
- TensorRT FP32
- TensorRT FP16 with source-side casting and `--stronglyTyped`
- Numerical accuracy comparison
- Runtime and engine-size benchmarks

This repository contains support for:

- **SmolVLA_libero** — 2 cameras, 7D action
- **SmolVLA_base** — 3 cameras, 6D action

---

## 1. Environment Setup

Clone the repository:

```bash
git clone https://github.com/Hung6779/SMOLVLA_Tensorrt_Release
cd SMOLVLA_Tensorrt_Release
```

Run the setup script:

```bash
bash scripts/setup.sh
```

The setup script is patched for **Python 3.10**, but may still fail partway through the installation.

If this happens, manually patch:

```text
work/lerobot/pyproject.toml
```

so that:

```toml
requires-python = ">=3.10"
```

Then finish the installation manually:

```bash
./work/venv/bin/python -m pip install -e "./work/lerobot[smolvla]"
```

---

## 2. Download Model Checkpoints

### 2.1 SmolVLA_libero

Download the SmolVLA Libero checkpoint:

```bash
./work/venv/bin/hf download HuggingFaceVLA/smolvla_libero \
  --revision 6721902bc4d61e50a3bfdb11dfb4cb626f05d102 \
  --exclude 'onnx/**' \
  --local-dir "$WORK_ROOT/artifacts/smolvla_libero"
```

### 2.2 SmolVLM2 Base

Download the SmolVLM2-500M-Instruct backbone:

```bash
./work/venv/bin/hf download HuggingFaceTB/SmolVLM2-500M-Instruct \
  --revision 7b375e1b73b11138ff12fe22c8f2822d8fe03467 \
  --exclude 'onnx/**' \
  --local-dir "$WORK_ROOT/artifacts/smolvlm_base"
```

Activate the environment:

```bash
source work/venv/bin/activate
```

Verify the downloaded assets:

```bash
python scripts/check_assets.py
```

---

# 3. SmolVLA_libero

**Configuration**

- Cameras: 2
- Action dimension: 7
- CPU FP32 baseline: **3663.9 ms**

---

## 3.1 Export SmolVLA to ONNX

Export the model to FP32 ONNX:

```bash
./work/venv/bin/python scripts/export_onnx.py \
  --checkpoint work/artifacts/smolvla_libero \
  --output work/onnx/fp32/model.onnx \
  --reference-dir work/onnx/fp32/reference
```

Validation result:

```text
max error: 2.6e-6
```

---

## 3.2 TensorRT FP32

TensorRT requires a small ONNX preprocessing step to handle boolean constants.

### Fix boolean constants

```bash
./work/venv/bin/python scripts/fix_bool_constants_for_trt.py \
  --input work/onnx/fp32/model.onnx \
  --output work/onnx/fp32/model.trt.onnx
```

### Build TensorRT engine

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=work/onnx/fp32/model.trt.onnx \
  --saveEngine=work/tensorrt/smolvla-fp32.engine \
  --memPoolSize=workspace:8192 \
  --timingCacheFile=work/tensorrt/timing.cache
```

### Validate TensorRT engine

```bash
./work/venv/bin/python scripts/validate_tensorrt.py \
  --engine work/tensorrt/smolvla-fp32.engine
```

Result:

```text
max_abs_error: 0.00107
latency: ~175 ms
speedup vs CPU ORT: 20.9x
```

---

## 3.3 TensorRT FP16

Before exporting the FP16 ONNX model, an optional PyTorch dtype comparison can be performed:

```bash
./work/venv/bin/python scripts/compare_pytorch_dtypes.py \
  --checkpoint work/artifacts/smolvla_libero \
  --dtypes float16
```

Approximate PyTorch FP16 error:

```text
max error: ~0.00256
```

### Export FP16 ONNX

```bash
./work/venv/bin/python scripts/export_onnx.py \
  --checkpoint work/artifacts/smolvla_libero \
  --dtype float16 \
  --output work/onnx/fp16/model.onnx \
  --reference-dir work/onnx/fp16/reference
```

### Build TensorRT FP16 engine

Use the typed TensorRT pipeline:

```bash
bash scripts/run_typed_trt_pipeline.sh \
  work/onnx/fp16 \
  work/onnx/fp32 \
  work/tensorrt \
  smolvla-fp16-typed
```

Result:

```text
max_abs_error: 0.00556
mean_abs_error: 0.00137
latency: ~100.9 ms
speedup vs CPU ORT: 36.3x
speedup vs TensorRT FP32: 1.73x
engine size: 1.26 GB
```

The typed FP16 pipeline is significantly more accurate than the naive TensorRT `--fp16` approach:

```text
Typed FP16 max error: 0.00556
Naive --fp16 max error: 3.108
```

---

# 4. SmolVLA_base

**Configuration**

- Cameras: 3
- Action dimension: 6
- TensorRT FP32 latency: **174.7 ms**

---

## 4.1 Export SmolVLA_base to ONNX

Export the FP32 model:

```bash
./work/venv/bin/python scripts/export_onnx.py \
  --checkpoint smolvla_base \
  --output work/onnx_base/fp32/model.onnx \
  --reference-dir work/onnx_base/fp32/reference
```

---

## 4.2 TensorRT FP32

Fix boolean constants:

```bash
./work/venv/bin/python scripts/fix_bool_constants_for_trt.py \
  --input work/onnx_base/fp32/model.onnx \
  --output work/onnx_base/fp32/model.trt.onnx
```

Build the TensorRT engine:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=work/onnx_base/fp32/model.trt.onnx \
  --saveEngine=work/tensorrt_base/smolvla_base-fp32.engine \
  --memPoolSize=workspace:8192 \
  --timingCacheFile=work/tensorrt_base/timing.cache
```

Result:

```text
max_abs_error: 0.00177
latency: 174.7 ms
```

---

## 4.3 TensorRT FP16

Export the FP16 ONNX model:

```bash
./work/venv/bin/python scripts/export_onnx.py \
  --checkpoint smolvla_base \
  --dtype float16 \
  --output work/onnx_base/fp16/model.onnx \
  --reference-dir work/onnx_base/fp16/reference
```

Build the strongly typed TensorRT FP16 engine:

```bash
bash scripts/run_typed_trt_pipeline.sh \
  work/onnx_base/fp16 \
  work/onnx_base/fp32 \
  work/tensorrt_base \
  smolvla_base-fp16-typed
```

Result:

```text
max_abs_error: 0.00874
mean_abs_error: 0.00232
latency: 86.3 ms
speedup vs TensorRT FP32: ~2.0x
engine size: 861 MB
```

---

# 5. Benchmark Results

## 5.1 SmolVLA_libero

**Configuration:** 2 cameras, 7D action

**CPU FP32 baseline: 3663.9 ms**

| Backend | Precision | Max Abs Error | Mean Abs Error | Latency | Speedup | Engine / Model Size |
|---|---|---:|---:|---:|---:|---:|
| ONNX Runtime CPU | FP32 | — | — | 3663.9 ms | 1.0x | 2.22 GB |
| ONNX Runtime CUDA EP | FP32 | 0.00119 | 0.00026 | 326.4 ms | 11.2x | 2.22 GB |
| **TensorRT** | **FP32** | **0.00107** | **0.00029** | **~175 ms** | **20.9x** | **2.33 GB** |
| TensorRT | FP16, source-cast + stronglyTyped | 0.00556 | 0.00137 | ~100.9 ms | 36.3x | 1.26 GB |

### Summary

Compared with the CPU FP32 baseline:

```text
TensorRT FP32:
  3663.9 ms → ~175 ms
  20.9x speedup

TensorRT FP16:
  3663.9 ms → ~100.9 ms
  36.3x speedup
```

The source-cast + strongly typed FP16 pipeline provides approximately:

```text
1.73x speedup vs TensorRT FP32
```

while maintaining substantially better numerical accuracy than the naive TensorRT `--fp16` approach.

---

## 5.2 SmolVLA_base

**Configuration:** 3 cameras, 6D action

| Backend | Precision | Max Abs Error | Mean Abs Error | Latency | Speedup vs TRT FP32 | Engine Size |
|---|---|---:|---:|---:|---:|---:|
| **TensorRT** | **FP32** | **0.00177** | **0.00039** | **174.7 ms** | **1.0x** | **1.64 GB** |
| TensorRT | FP16, source-cast + stronglyTyped | 0.00874 | 0.00232 | 86.3 ms | ~2.0x | 861 MB |

### Summary

```text
TensorRT FP32:
  174.7 ms

TensorRT FP16:
  86.3 ms
  ~2.0x faster
```

The FP16 engine also reduces the engine size:

```text
FP32: 1.64 GB
FP16: 861 MB
```

---

# 6. Overall Results

| Model | Cameras | Action Dim | Precision | Latency | Speedup | Max Error | Engine Size |
|---|---:|---:|---|---:|---:|---:|---:|
| SmolVLA_libero | 2 | 7 | TensorRT FP32 | ~175 ms | 20.9x vs CPU | 0.00107 | 2.33 GB |
| SmolVLA_libero | 2 | 7 | TensorRT FP16 | ~100.9 ms | 36.3x vs CPU | 0.00556 | 1.26 GB |
| SmolVLA_base | 3 | 6 | TensorRT FP32 | 174.7 ms | 1.0x | 0.00177 | 1.64 GB |
| SmolVLA_base | 3 | 6 | TensorRT FP16 | 86.3 ms | ~2.0x vs TRT FP32 | 0.00874 | 861 MB |

---

# 7. Notes on FP16 Accuracy

For SmolVLA_libero, directly enabling TensorRT FP16 with the naive approach can produce a very large numerical error:

```text
Naive TensorRT --fp16:
  max error ≈ 3.108
```

Using source-side FP16 conversion together with TensorRT `--stronglyTyped` gives:

```text
Typed FP16:
  max error = 0.00556
  mean error = 0.00137
```

This reduces the maximum error by approximately **559x** compared with the naive FP16 pipeline.

Therefore, the recommended FP16 deployment flow is:

```text
PyTorch
   │
   ├── source-side FP16 casting
   ↓
ONNX FP16
   │
   ↓
TensorRT stronglyTyped
   │
   ↓
FP16 TensorRT Engine
```

rather than simply relying on TensorRT's automatic `--fp16` layer conversion.

---

# 8. Directory Structure

After completing the export and TensorRT build process, the expected structure is approximately:

```text
SMOLVLA_Tensorrt_Release/
├── scripts/
│   ├── setup.sh
│   ├── check_assets.py
│   ├── export_onnx.py
│   ├── fix_bool_constants_for_trt.py
│   ├── validate_tensorrt.py
│   ├── compare_pytorch_dtypes.py
│   └── run_typed_trt_pipeline.sh
│
├── work/
│   ├── venv/
│   │
│   ├── artifacts/
│   │   ├── smolvla_libero/
│   │   └── smolvlm_base/
│   │
│   ├── onnx/
│   │   ├── fp32/
│   │   │   ├── model.onnx
│   │   │   ├── model.trt.onnx
│   │   │   └── reference/
│   │   │
│   │   └── fp16/
│   │       ├── model.onnx
│   │       └── reference/
│   │
│   ├── onnx_base/
│   │   ├── fp32/
│   │   │   ├── model.onnx
│   │   │   ├── model.trt.onnx
│   │   │   └── reference/
│   │   │
│   │   └── fp16/
│   │       ├── model.onnx
│   │       └── reference/
│   │
│   ├── tensorrt/
│   │   ├── smolvla-fp32.engine
│   │   ├── smolvla-fp16-typed.engine
│   │   └── timing.cache
│   │
│   └── tensorrt_base/
│       ├── smolvla_base-fp32.engine
│       ├── smolvla_base-fp16-typed.engine
│       └── timing.cache
│
└── README.md
```

---

# 9. Quick Start

For a complete SmolVLA_libero TensorRT FP16 deployment:

```bash
# 1. Clone
git clone https://github.com/Hung6779/SMOLVLA_Tensorrt_Release
cd SMOLVLA_Tensorrt_Release

# 2. Setup
bash scripts/setup.sh

# 3. Install LeRobot
./work/venv/bin/python -m pip install -e "./work/lerobot[smolvla]"

# 4. Download checkpoint
./work/venv/bin/hf download HuggingFaceVLA/smolvla_libero \
  --revision 6721902bc4d61e50a3bfdb11dfb4cb626f05d102 \
  --exclude 'onnx/**' \
  --local-dir "$WORK_ROOT/artifacts/smolvla_libero"

# 5. Export FP16 ONNX
./work/venv/bin/python scripts/export_onnx.py \
  --checkpoint work/artifacts/smolvla_libero \
  --dtype float16 \
  --output work/onnx/fp16/model.onnx \
  --reference-dir work/onnx/fp16/reference

# 6. Build TensorRT FP16
bash scripts/run_typed_trt_pipeline.sh \
  work/onnx/fp16 \
  work/onnx/fp32 \
  work/tensorrt \
  smolvla-fp16-typed
```

Expected performance:

```text
Latency:      ~100.9 ms
Speedup:      36.3x vs CPU ORT
Max error:    0.00556
Mean error:   0.00137
Engine size:  1.26 GB
```

---

# 10. References

- [SmolVLA](https://huggingface.co/HuggingFaceVLA/smolvla_libero)
- [SmolVLM2-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Instruct)
- [LeRobot](https://github.com/huggingface/lerobot)
- [TensorRT](https://developer.nvidia.com/tensorrt)
