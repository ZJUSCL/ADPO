<div align="center">

# Unified Generation and Self-Verification for Vision-Language Models via Advantage Decoupled Preference Optimization

[![Project Website](https://img.shields.io/badge/🌐-Project%20Website-deepgray)](https://ZJUSCL.github.io/ADPO/)[![arXiv](https://img.shields.io/badge/arXiv-2601.01483-b31b1b.svg)](https://arxiv.org/abs/2601.01483)
</div>

## 📖 Overview
Official repo for the paper **"Unified Generation and Self-Verification for Vision-Language Models via Advantage Decoupled Preference Optimization"**

ADPO is a unified reinforcement learning framework that jointly optimizes answer generation and self-verification within a single policy. This code release focuses on the **Referring Expression Comprehension (REC)** / visual grounding setting: we train on **RefCOCO / RefCOCO+ / RefCOCOg** and evaluate on **LISA-Grounding**.

<!-- The data setup follows the REC release in [VLM-R1](https://github.com/om-ai-lab/VLM-R1). -->

The repository includes:

- ADPO training code for grounding
- GRPO grounding baseline code
- LISA-Grounding evaluation scripts

<p align="center">
  <img src="assets/pipeline.png" alt="Teaser" width="900"/>
</p>

## 📰 News

- **[2026-03-07]** We release the official grounding experiment code for ADPO.

---

<a name="quick-start"></a>
## 🚀 Quick Start

### 💻 Installation

```bash
conda create -n adpo python=3.11 -y
conda activate adpo

bash setup.sh
```

---

<a name="dataset"></a>
## 📁 Dataset
The data setup follows the [VLM-R1](https://github.com/om-ai-lab/VLM-R1). We use the same REC data setup as the **Referring Expression Comprehension (REC)** section of VLM-R1:

- **Training**: RefCOCO / RefCOCO+ / RefCOCOg
- **Evaluation**: LISA-Grounding


```bash
mkdir -p data/vlm-r1

huggingface-cli download omlab/VLM-R1 --repo-type dataset --include "train2014.zip" --local-dir data/vlm-r1
huggingface-cli download omlab/VLM-R1 --repo-type dataset --include "rec_jsons_processed.zip" --local-dir data/vlm-r1
huggingface-cli download omlab/VLM-R1 --repo-type dataset --include "lisa-test.zip" --local-dir data/vlm-r1
```

Unzip the downloaded files and organize your local paths as needed. For ADPO training with `adpo_jsonl.py`, the three REC training annotation files are:

```bash
/path/to/rec_jsons_processed/refcoco_train.jsonl
/path/to/rec_jsons_processed/refcocop_train.jsonl
/path/to/rec_jsons_processed/refcocog_train.jsonl
```

The corresponding environment variables for training are:

```bash
export DATA_PATHS="/path/to/refcoco_train.jsonl:/path/to/refcocop_train.jsonl:/path/to/refcocog_train.jsonl"
export IMAGE_FOLDERS="/path/to/coco:/path/to/coco:/path/to/coco"
```


---

<a name="training"></a>
## 🚀 Training

Our main grounding experiments use `src/open-r1-multimodal/src/open_r1/adpo_jsonl.py`, launched via the top-level `train.sh` (which calls `src/open-r1-multimodal/run_scripts/run_adpo_rec.sh`).

> We recommend at least **8 × 80 GB GPUs** (e.g. A100 / H100) for training.

Set the following environment variables before running:

| Variable | Required | Description |
|---|---|---|
| `DATA_PATHS` | ✅ | Colon-separated list of training JSONL files (one per dataset) |
| `IMAGE_FOLDERS` | ✅ | Colon-separated list of image root directories (same order as `DATA_PATHS`) |
| `MODEL_PATH` | ✅ | Path to the base model (local checkpoint or HF model ID) |
| `EXP_NAME` | optional | Experiment name, used for output directory (default: `adpo`) |
| `NPROC` | optional | Number of GPUs per node (default: `4`) |
| `VLM_R1_ENV_BIN` | optional | Path to conda/venv `bin` directory if `torchrun` is not on `PATH` |
| `MASTER_PORT` | optional | Master port for distributed training (default: `12349`) |

Then launch training:

```bash
export DATA_PATHS="/path/to/refcoco_train.jsonl:/path/to/refcocop_train.jsonl:/path/to/refcocog_train.jsonl"
export IMAGE_FOLDERS="/path/to/coco:/path/to/coco:/path/to/coco"
export MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
export NPROC=8  # number of GPUs

bash train.sh
```

Key training hyperparameters (configured in `run_adpo_rec.sh`):

- `--reward_funcs accuracy format scoreformat`
- `--per_device_train_batch_size 8`, `--gradient_accumulation_steps 4`
- `--num_generations 8`, `--max_completion_length 2048`
- `--max_steps 1200`, `--save_steps 300`, `--beta 0.04`
- DeepSpeed ZeRO-3 (`./local_scripts/zero3.json`), `flash_attention_2`, `bf16`

---

<a name="evaluation"></a>
## 📊 Evaluation

Evaluation is done via `eval.sh`, which runs vLLM inference in parallel across all available GPUs and then merges the per-GPU output files.

Set the following environment variables before running:

| Variable | Required | Description |
|---|---|---|
| `DATA_ROOT` | ✅ | Directory containing the dataset JSON files |
| `IMAGE_ROOT` | ✅ | Directory containing the evaluation images |
| `MODEL_PATH` | ✅ | Path to the trained model checkpoint |
| `TEST_DATASET` | optional | Dataset name without extension (default: `lisa_test`) |
| `N_SAMPLE` | optional | Number of samples per question for best-of-N evaluation (default: `8`) |
| `PREDICTIONS_BASE_DIR` | optional | Output directory for per-GPU prediction files (auto-generated if unset) |

Then run:

```bash
export DATA_ROOT="/path/to/lisa"       # directory containing lisa_test.json
export IMAGE_ROOT="/path/to/lisa"      # directory containing the LISA images (x['image'] is relative to this)
export MODEL_PATH="/path/to/checkpoint"
export TEST_DATASET="lisa_test"   # evaluates DATA_ROOT/lisa_test.json
export N_SAMPLE=8

bash eval.sh
```

`eval.sh` automatically:
1. Detects all available GPUs and distributes samples evenly across them
2. Runs `src/eval/vllm_inference.py` on each GPU in parallel (temperature `0.2`, top-p `0.99`)
3. Merges per-GPU outputs with `src/eval/merge_json.py`

Merged results are written to `${PREDICTIONS_BASE_DIR}/` and a summary log is saved under `logs/`.

---

<a name="citation"></a>
## 📚 Citation

Please kindly cite our paper if you use our code, data, or results:

```bibtex
@misc{qiu2026unifiedgenerationselfverificationvisionlanguage,
      title={Unified Generation and Self-Verification for Vision-Language Models via Advantage Decoupled Preference Optimization},
      author={Xinyu Qiu and Heng Jia and Zhengwen Zeng and Shuheng Shen and Changhua Meng and Yi Yang and Linchao Zhu},
      year={2026},
      eprint={2601.01483},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.01483},
}
```

---

<a name="acknowledgements"></a>
## 🙏 Acknowledgements

This repository builds on and benefits from several excellent open-source projects and resources, including [VLM-R1](https://github.com/om-ai-lab/VLM-R1), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [RefCOCO](https://github.com/lichengunc/refer), and [LISA](https://github.com/dvlab-research/LISA).
