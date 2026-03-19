import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import json
import random
import re
from dataclasses import asdict
import argparse
import warnings

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, EngineArgs, SamplingParams

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
def seed_all(seed=42):
    """Fix random seeds for all relevant libraries to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    import transformers
    transformers.set_seed(seed)
def extract_bbox_answer(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\{.*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]\s*.*\}'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        bbox_match = re.search(bbox_pattern, content_answer, re.DOTALL)
        if bbox_match:
            bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
            return bbox
    return [0, 0, 0, 0]
def extract_score_solution_exp03(solution_str: str):
    """Extract the last <score>...</score> value from the model response."""
    processed_str = solution_str
    answer_pattern = r"<score>(.*?)</score>"
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))

    if not matches:
        return None, processed_str

    final_answer = matches[-1].group(1).strip()

    return final_answer, processed_str
def iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2]-1, box2[2]-1)
    inter_y2 = min(box1[3]-1, box2[3]-1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return float(inter)/union

def main(rank_data, model, processor,sampling_params, n_sample, DATA_ROOT, TEST_DATASETS, IMAGE_ROOT, OUTPUT_PATH):
    
    for ds in TEST_DATASETS:
        
        QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."

        messages = []
        inputs = []
        max_len = -1
        for x in tqdm(rank_data, desc="processing data"):
            image_path = os.path.join(IMAGE_ROOT, x['image'])
            message = [
                # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {
                        "type": "text",
                        "text": QUESTION_TEMPLATE.format(Question=x['problem'])+ "After outputting the answer, assign a score between 0-1 to this answer to express whether this answer provides accurate bounding box coordinates that correctly satisfy the user query. For example: <score>0.5</score>"
                    }
                ]
            }]
            # message = [msg.model_dump() for msg in message]
            # Apply chat template
            text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            if len(text) > max_len:
                max_len = len(text)
            image = Image.open(image_path)
            input = {
                "prompt": text,
                "multi_modal_data": {"image": image},
            }
            inputs.append(input)

        chunk_size = 512 // n_sample
        inputs_chunk_list = [inputs[i:i + chunk_size] for i in range(0, len(inputs), chunk_size)]
        
        outputs = []
        for input in inputs_chunk_list:
            outputs.extend(model.generate(input, sampling_params=sampling_params)) 

        rank_outputs = [] # List to store answers for this rank
        all_outputs = []  # List to store all answers
        final_output = []
        for j in range(len(inputs)):
            sample_output = []
            for o in outputs[j].outputs:
                output_text = o.text
                sample_output.append(output_text)
            input_example = rank_data[j]
            result = {
                'image': input_example['image'],
                'question': input_example['problem'],
                'ground_truth': input_example['solution'],
                "promblem": input_example['problem'],
                'model_output': sample_output
            }
            final_output.append(result)
        output_path = OUTPUT_PATH
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_path, "w") as f:
            json.dump({
                'results': final_output
            }, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_idx', type=int, required=True, help='Start index of data slice.')
    parser.add_argument('--end_idx', type=int, required=True, help='End index of data slice.')
    parser.add_argument('--output_path', type=str, required=True, help='Output file path.')
    parser.add_argument('--temperature', type=float, required=True, help='Sampling temperature.')
    parser.add_argument('--topp', type=float, required=True, help='Top-p (nucleus) sampling threshold.')
    parser.add_argument('--topk', type=int, required=True, help='Top-k sampling threshold.')
    args = parser.parse_args()

    # seed_all()

    MODEL_PATH = os.environ.get("MODEL_PATH")
    if not MODEL_PATH:
        print("Error: MODEL_PATH environment variable is not set.", flush=True)
        import sys; sys.exit(1)

    OUTPUT_PATH = args.output_path
    n_sample = int(os.environ.get("N_SAMPLE", 1))

    DATA_ROOT = os.environ.get("DATA_ROOT")
    if not DATA_ROOT:
        print("Error: DATA_ROOT environment variable is not set.", flush=True)
        import sys; sys.exit(1)

    IMAGE_ROOT = os.environ.get("IMAGE_ROOT")
    if not IMAGE_ROOT:
        print("Error: IMAGE_ROOT environment variable is not set.", flush=True)
        import sys; sys.exit(1)

    TEST_DATASETS = ['lisa_test']
    ds = os.environ.get("TEST_DATASET", "lisa_test")
    # ds = TEST_DATASETS[0]
    ds_path = os.path.join(DATA_ROOT, f"{ds}.json")
    data = json.load(open(ds_path, "r"))
    rank_data = data[args.start_idx:args.end_idx]
    del data
    #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    engine_args = EngineArgs(
        model=MODEL_PATH,
        max_model_len=22768,
        # disable_custom_all_reduce=True,
        enforce_eager=True,
        # max_num_seqs=25,
        # max_num_batched_tokens=4096*25,
        gpu_memory_utilization=0.9,
        enable_chunked_prefill=False,
        # mm_processor_kwargs={
        #     "min_pixels": 256 * 28 * 28,
        #     "max_pixels": 1280 * 28 * 28,
        #     "fps": 1,
        # },
        tensor_parallel_size=1
    )
    sampling_params = SamplingParams(n=n_sample, temperature=args.temperature, top_p=args.topp, top_k=args.topk, max_tokens=350)
    engine_args = asdict(engine_args) | {"seed": 1}
    llm = LLM(**engine_args)

    # default processer
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    main(rank_data, llm, processor,sampling_params, n_sample, DATA_ROOT, TEST_DATASETS, IMAGE_ROOT, OUTPUT_PATH)

