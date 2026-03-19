#!/usr/bin/env bash

set -euo pipefail

export EXP_NAME="${EXP_NAME:-grpo}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Set VLM_R1_ENV_BIN to the bin directory of your conda/venv environment, e.g.:
# export VLM_R1_ENV_BIN="/path/to/your/envs/vlm-r1/bin"
VLM_R1_ENV_BIN="${VLM_R1_ENV_BIN:-}"

if [ -d "${VLM_R1_ENV_BIN}" ]; then
    export PATH="${VLM_R1_ENV_BIN}:${PATH}"
fi

TORCHRUN_BIN="${TORCHRUN_BIN:-}"
if [ -z "${TORCHRUN_BIN}" ]; then
    TORCHRUN_BIN="$(command -v torchrun || true)"
fi

if [ -z "${TORCHRUN_BIN}" ]; then
    echo "torchrun not found. Set TORCHRUN_BIN or VLM_R1_ENV_BIN to a valid environment." >&2
    exit 127
fi

# Required: path to your image root directory
if [ -z "${IMAGE_ROOT:-}" ]; then
    echo "Error: IMAGE_ROOT is not set. Please export IMAGE_ROOT=/path/to/images" >&2
    exit 1
fi

cd "${REPO_ROOT}"

RUN_NAME="${RUN_NAME:-Qwen2.5-VL-7B-GRPO-REC}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-12346}"

SAVE_PATH="./output/${EXP_NAME}"
mkdir -p "${SAVE_PATH}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Uncomment to enable debug logging:
# export DEBUG_MODE="true"
# export LOG_PATH="./debug_log_${RUN_NAME}.txt"

"${TORCHRUN_BIN}" --nproc_per_node="${NPROC}" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="${MASTER_PORT}" \
    src/open_r1/grpo_rec.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir "${SAVE_PATH}" \
    --model_name_or_path "${MODEL_NAME}" \
    --dataset_name data_config/rec.yaml \
    --image_root "${IMAGE_ROOT}" \
    --max_prompt_length 1024 \
    --num_generations 8 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to "${REPORT_TO:-tensorboard}" \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name "${RUN_NAME}" \
    --save_steps 100 \
    --save_only_model true

echo "Training completed for ${EXP_NAME}"
