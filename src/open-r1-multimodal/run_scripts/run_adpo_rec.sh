#!/usr/bin/env bash

set -euo pipefail

export EXP_NAME="${EXP_NAME:-adpo}"
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

cd "${REPO_ROOT}"

# Required: colon-separated list of training JSONL files, one per dataset.
# e.g. export DATA_PATHS="/data/refcoco_train.jsonl:/data/refcocop_train.jsonl:/data/refcocog_train.jsonl"
if [ -z "${DATA_PATHS:-}" ]; then
    echo "Error: DATA_PATHS is not set. Please export DATA_PATHS=path1.jsonl:path2.jsonl:..." >&2
    exit 1
fi

# Required: colon-separated list of image root directories, one per dataset (same order as DATA_PATHS).
# e.g. export IMAGE_FOLDERS="/data/coco:/data/coco:/data/coco"
if [ -z "${IMAGE_FOLDERS:-}" ]; then
    echo "Error: IMAGE_FOLDERS is not set. Please export IMAGE_FOLDERS=dir1:dir2:..." >&2
    exit 1
fi

# Required: path to the base model (e.g. a local Qwen2.5-VL checkpoint or HF model ID).
# e.g. export MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
if [ -z "${MODEL_PATH:-}" ]; then
    echo "Error: MODEL_PATH is not set. Please export MODEL_PATH=/path/to/model" >&2
    exit 1
fi

data_paths="${DATA_PATHS}"
image_folders="${IMAGE_FOLDERS}"
model_path="${MODEL_PATH}"
is_reward_customized_from_vlm_module=True
echo "data_paths: $data_paths"
echo "image_folders: $image_folders"

SAVE_PATH="./output/${EXP_NAME}"
mkdir -p "${SAVE_PATH}"

# GPU_COUNT=1
# pip install json_repair
# pip install math_verify
TASK_TYPE="rec"
# cd ${REPO_HOME}/src/open-r1-multimodal

# export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
# create the run directory and log file
# mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log
# export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"
MAX_STEPS=1200 
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-12349}"

# export WANDB_DISABLED=true
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
"${TORCHRUN_BIN}" --nproc_per_node="${NPROC}" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="${MASTER_PORT}" \
  src/open_r1/adpo_jsonl.py \
    --use_vllm False \
    --output_dir "${SAVE_PATH}" \
    --resume_from_checkpoint False \
    --model_name_or_path "${model_path}" \
    --data_file_paths "${data_paths}" \
    --image_folders "${image_folders}" \
    --is_reward_customized_from_vlm_module "${is_reward_customized_from_vlm_module}" \
    --task_type "${TASK_TYPE}" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --save_only_model true \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs 2 \
    --max_steps "${MAX_STEPS}" \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --ds3_gather_for_generation true \
    --run_name "${EXP_NAME}" \
    --data_seed 42 \
    --save_steps 300 \
    --num_generations 8 \
    --max_completion_length 2048 \
    --reward_funcs accuracy format scoreformat \
    --beta 0.04 \
    --report_to tensorboard \
    --dataset-name this_is_not_used \
    --deepspeed ./local_scripts/zero3.json

echo "Training completed for ${EXP_NAME}"
