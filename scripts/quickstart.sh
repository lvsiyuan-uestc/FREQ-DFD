#!/usr/bin/env bash
set -euo pipefail

# ====== 可按需修改 ======
REAL_ROOT="${REAL_ROOT:-/path/to/data/real}"
FAKE_ROOT="${FAKE_ROOT:-/path/to/data/fake}"
INPUT_DIR="${INPUT_DIR:-/path/to/images_or_frames}"
OUTPUT_DIR="${OUTPUT_DIR:-./output_freq}"

# 训练超参（可通过环境变量覆盖）
IMG_SIZE="${IMG_SIZE:-224}"
EPOCHS="${EPOCHS:-3}"
BATCH_TRAIN="${BATCH_TRAIN:-128}"
BATCH_INFER="${BATCH_INFER:-256}"

# ====== 训练 ======
echo "[quickstart] training..."
python train_freq_head.py \
  --real-root "${REAL_ROOT}" \
  --fake-root "${FAKE_ROOT}" \
  --img-size "${IMG_SIZE}" \
  --epochs "${EPOCHS}" \
  --batch-train "${BATCH_TRAIN}"

# ====== 有标签评估 ======
echo "[quickstart] labeled evaluation..."
python infer_freq_only.py \
  --real-root "${REAL_ROOT}" \
  --fake-root "${FAKE_ROOT}" \
  --img-size "${IMG_SIZE}" \
  --batch "${BATCH_INFER}" \
  --output "${OUTPUT_DIR}"

# ====== 无标签推理 ======
echo "[quickstart] unlabeled inference..."
python infer_freq_only.py \
  --input "${INPUT_DIR}" \
  --img-size "${IMG_SIZE}" \
  --batch "${BATCH_INFER}" \
  --output "${OUTPUT_DIR}"

echo "[quickstart] done. outputs in ${OUTPUT_DIR}"
