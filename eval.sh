#!/bin/bash

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)

temperature=0.2
topk=-1
topp=0.99
export N_SAMPLE="${N_SAMPLE:-8}"
export TEST_DATASET="${TEST_DATASET:-lisa_test}"

# Required: set DATA_ROOT to the directory containing your dataset JSON files.
# e.g. export DATA_ROOT=/path/to/data
if [ -z "${DATA_ROOT:-}" ]; then
    echo "Error: DATA_ROOT is not set. Please export DATA_ROOT=/path/to/data" >&2
    exit 1
fi

# Required: set MODEL_PATH to your trained model checkpoint directory.
# e.g. export MODEL_PATH=/path/to/checkpoint
if [ -z "${MODEL_PATH:-}" ]; then
    echo "Error: MODEL_PATH is not set. Please export MODEL_PATH=/path/to/model" >&2
    exit 1
fi

# Required: set IMAGE_ROOT to the directory containing the evaluation images.
# e.g. export IMAGE_ROOT=/path/to/images
if [ -z "${IMAGE_ROOT:-}" ]; then
    echo "Error: IMAGE_ROOT is not set. Please export IMAGE_ROOT=/path/to/images" >&2
    exit 1
fi

export IMAGE_ROOT

# Required: set PREDICTIONS_BASE_DIR to the directory where per-GPU output files will be written.
# e.g. export PREDICTIONS_BASE_DIR=/path/to/predictions/run_name
if [ -z "${PREDICTIONS_BASE_DIR:-}" ]; then
    PREDICTIONS_BASE_DIR="./predictions/adpo_${TEST_DATASET}_t${temperature}_topp${topp}_topk${topk}_nsample${N_SAMPLE}"
    echo "PREDICTIONS_BASE_DIR not set, defaulting to: ${PREDICTIONS_BASE_DIR}"
fi

export MODEL_PATH

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_script="vllm_inference.py"
python_merge_script="${REPO_ROOT}/src/eval/merge_json.py"
LOG_DIR="${REPO_ROOT}/logs"

# Compute total samples dynamically from the dataset file
ds_file="${DATA_ROOT}/${TEST_DATASET}.json"
if [ ! -f "${ds_file}" ]; then
    echo "Error: Dataset file not found: ${ds_file}" >&2
    exit 1
fi
TOTAL_SAMPLES=$(python3 -c "import json; data=json.load(open('${ds_file}')); print(len(data))")

export VLLM_WORKER_MULTIPROC_METHOD=spawn
SAMPLES_PER_GPU=$((TOTAL_SAMPLES / GPU_COUNT))
REMAINDER=$((TOTAL_SAMPLES % GPU_COUNT))

echo "Total samples: $TOTAL_SAMPLES"
echo "GPU count: $GPU_COUNT"
echo "Samples per GPU: $SAMPLES_PER_GPU"
echo "Remainder: $REMAINDER"

mkdir -p "$LOG_DIR"
mkdir -p "$PREDICTIONS_BASE_DIR"
cd "${REPO_ROOT}/src/eval"

for ((i=0; i<GPU_COUNT; i++)); do
    START_IDX=$((i * SAMPLES_PER_GPU))

    if [ $i -lt $REMAINDER ]; then
        END_IDX=$(((i + 1) * SAMPLES_PER_GPU + i + 1))
        START_IDX=$((START_IDX + i))
    else
        END_IDX=$(((i + 1) * SAMPLES_PER_GPU + REMAINDER))
        START_IDX=$((START_IDX + REMAINDER))
    fi

    echo "GPU $i: samples $START_IDX to $((END_IDX-1))"
    formatted_i=$(printf "%03d" $i)
    current_output_path="${PREDICTIONS_BASE_DIR}/part_${formatted_i}.json"
    current_log_path="${LOG_DIR}/vllm_part_${formatted_i}.log"
    CUDA_VISIBLE_DEVICES=$i python $python_script  \
        --start_idx $START_IDX \
        --end_idx $END_IDX \
        --temperature $temperature \
        --topp $topp \
        --topk $topk \
        --output_path $current_output_path > "$current_log_path" 2>&1 &
    sleep 2
done

wait

echo "All processes complete."
last_part=$(basename "$PREDICTIONS_BASE_DIR")
echo "$last_part"
python "$python_merge_script" \
    --directory_path "$PREDICTIONS_BASE_DIR" > "${LOG_DIR}/${last_part}.log"
