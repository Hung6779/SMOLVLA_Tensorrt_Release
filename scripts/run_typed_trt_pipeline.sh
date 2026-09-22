#!/usr/bin/env bash
# Chain: (wait for a running ONNX export) -> fix Bool constants -> build a
# strongly typed TensorRT engine (keeps the dtypes of the exported graph, unlike
# `--fp16` which lets TensorRT pick precisions) -> validate against FP32.
#
# Meant to be launched detached so it survives an SSH disconnect:
#   setsid nohup scripts/run_typed_trt_pipeline.sh <onnx_dir> <fp32_onnx_dir> <trt_dir> <name> [wait_pid] \
#       > pipeline.log 2>&1 < /dev/null &
#
# Result summary: <trt_dir>/<name>_summary.txt (last line is PIPELINE_OK or PIPELINE_FAILED).
set -uo pipefail

ONNX_DIR=${1:?dir of the exported reduced-dtype model.onnx}
FP32_DIR=${2:?dir of the FP32 model.onnx + reference/ used as ground truth}
TRT_DIR=${3:?output dir for the engine}
NAME=${4:?engine name}
WAIT_PID=${5:-}

cd "$(dirname "$0")/.."
PY=./work/venv/bin/python
TRTEXEC=/usr/src/tensorrt/bin/trtexec
SUMMARY="$TRT_DIR/${NAME}_summary.txt"
mkdir -p "$TRT_DIR"
: > "$SUMMARY"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY"; }
fail() { log "FAILED at: $*"; echo PIPELINE_FAILED >> "$SUMMARY"; exit 1; }

if [[ -n "$WAIT_PID" ]]; then
  log "waiting for export process $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
fi

[[ -f "$ONNX_DIR/model.onnx" ]] || fail "export produced no $ONNX_DIR/model.onnx"
log "export done: $(du -sh "$ONNX_DIR" | cut -f1)"

$PY scripts/fix_bool_constants_for_trt.py --input "$ONNX_DIR/model.onnx" \
  --output "$ONNX_DIR/model.trt.onnx" >> "$SUMMARY" 2>&1 || fail "fix_bool_constants"

log "building strongly typed engine (this takes a while)"
$TRTEXEC --onnx="$ONNX_DIR/model.trt.onnx" --saveEngine="$TRT_DIR/$NAME.engine" \
  --stronglyTyped --memPoolSize=workspace:8192 --timingCacheFile="$TRT_DIR/timing.cache" \
  --verbose > "$TRT_DIR/${NAME}_build.log" 2>&1
grep -q "&&&& PASSED" "$TRT_DIR/${NAME}_build.log" || fail "trtexec build (see $TRT_DIR/${NAME}_build.log)"
grep -E "\[I\] (Throughput|Latency)" "$TRT_DIR/${NAME}_build.log" | cut -c1-200 >> "$SUMMARY"

log "validating against FP32 ONNX Runtime"
$PY scripts/validate_tensorrt.py --engine "$TRT_DIR/$NAME.engine" \
  --fp32-model "$FP32_DIR/model.onnx" --reference-dir "$FP32_DIR/reference" \
  --work-dir "$TRT_DIR" >> "$SUMMARY" 2>&1 || fail "validate_tensorrt"

echo PIPELINE_OK >> "$SUMMARY"
