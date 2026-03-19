import argparse
import json
import glob
import os
import re

from ciou import ciou
from majority_vote import majority_vote

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

def giou(box1, box2):
    """
    Compute GIoU (Generalized Intersection over Union) between two bounding boxes.

    Args:
        box1, box2: bounding boxes in [x1, y1, x2, y2] format.

    Returns:
        GIoU value.
    """
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2]-1, box2[2]-1)
    inter_y2 = min(box1[3]-1, box2[3]-1)
    
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter_area = (inter_x2-inter_x1+1) * (inter_y2-inter_y1+1)
    else:
        inter_area = 0

    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0
    iou_value = float(inter_area) / union_area

    enclosing_x1 = min(box1[0], box2[0])
    enclosing_y1 = min(box1[1], box2[1])
    enclosing_x2 = max(box1[2]-1, box2[2]-1)
    enclosing_y2 = max(box1[3]-1, box2[3]-1)
    
    enclosing_area = (enclosing_x2-enclosing_x1+1) * (enclosing_y2-enclosing_y1+1)

    if enclosing_area == 0:
        return iou_value
    
    giou_value = iou_value - (enclosing_area - union_area) / enclosing_area
    
    return giou_value

def load_classification_data(classification_file):
    """Load per-image category labels from a classification JSON file."""
    try:
        with open(classification_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        classification_map = {}

        if "is_sentence_true" in data:
            for img_path in data["is_sentence_true"]:
                filename = img_path.replace("test/", "")
                classification_map[filename] = "long_query"

        if "is_sentence_false" in data:
            for img_path in data["is_sentence_false"]:
                filename = img_path.replace("test/", "")
                classification_map[filename] = "short_query"
        
        return classification_map
    except Exception as e:
        print(f"Error loading classification file: {e}")
        return {}

def get_image_category(image_path, classification_map):
    """Return the category label for a given image path."""
    if "test/" in image_path:
        filename = image_path.split("test/")[1]
        return classification_map.get(filename, "unknown")
    return "unknown"

def calculate_metrics(data_list):
    """Compute aggregate accuracy and IoU metrics over a list of result dicts."""
    if not data_list:
        return {
            'count': 0,
            'accuracy': 0,
            'passn_accuracy': 0,
            'bestn_accuracy': 0,
            'mv_accuracy': 0,
            'first_ciou': 0,
            'bestn_ciou': 0,
            'mv_ciou': 0,
            'first_giou': 0,
            'bestn_giou': 0,
            'mv_giou': 0
        }
    
    count = len(data_list)
    correct_number = sum(d['correct'] for d in data_list)
    passn_number = sum(d['passn'] for d in data_list)
    bestn_number = sum(d['bestn'] for d in data_list)
    mv_number = sum(d['mv_correct'] for d in data_list)
    first_ciou_value = sum(d['first_ciou'] for d in data_list)
    bestn_ciou_value = sum(d['bestn_ciou'] for d in data_list)
    mv_ciou_value = sum(d['mv_ciou'] for d in data_list)
    first_giou_value = sum(d['first_giou'] for d in data_list)
    bestn_giou_value = sum(d['bestn_giou'] for d in data_list)
    mv_giou_value = sum(d['mv_giou'] for d in data_list)
    
    return {
        'count': count,
        'accuracy': correct_number / count * 100,
        'passn_accuracy': passn_number / count * 100,
        'bestn_accuracy': bestn_number / count * 100,
        'mv_accuracy': mv_number / count * 100,
        'first_ciou': first_ciou_value / count * 100,
        'bestn_ciou': bestn_ciou_value / count * 100,
        'mv_ciou': mv_ciou_value / count * 100,
        'first_giou': first_giou_value / count * 100,
        'bestn_giou': bestn_giou_value / count * 100,
        'mv_giou': mv_giou_value / count * 100
    }

parser = argparse.ArgumentParser(description="Merge per-GPU inference results and compute metrics.")
parser.add_argument("--directory_path", type=str, required=True, help="Directory containing part_*.json result files.")
parser.add_argument("--classification_file", type=str, default=None,
                   help="Path to the classification JSON file for long/short query breakdown. "
                        "If not provided, all samples are reported as a single group.")

args = parser.parse_args()
directory_path = args.directory_path
classification_file = args.classification_file
output_filename = 'output.json'
file_pattern = 'part_*.json'

if classification_file:
    print(f"Loading classification data from: {classification_file}")
    classification_map = load_classification_data(classification_file)
    print(f"Loaded {len(classification_map)} image classifications")
else:
    classification_map = {}
    print("No classification file provided; long/short query breakdown will be skipped.")

search_path = os.path.join(directory_path, file_pattern)
output_filepath = os.path.join(directory_path, output_filename)

json_files = sorted(glob.glob(search_path))

if not json_files:
    print(f"No files matching '{file_pattern}' found in '{directory_path}'.")
    exit()

print(f"Merging files: {json_files}")

merged_data = []

for file_path in json_files:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data["results"], list):
                    merged_data.extend(data["results"])
                else:
                    print(f"Warning: '{file_path}' does not contain a 'results' list (got {type(data)}), skipping.")

            except json.JSONDecodeError:
                print(f"Error: '{file_path}' is not valid JSON, skipping.")
            except Exception as e:
                print(f"Error reading '{file_path}': {e}")

    except IOError as e:
        print(f"Cannot open '{file_path}': {e}")

try:
    final_output = []
    long_query_results = []
    short_query_results = []
    unknown_results = []
    
    for d in merged_data:
        box_list = []
        original_output = d["model_output"]
        ground_truth = d["ground_truth"]
        image_path = d["image"]
        
        category = get_image_category(image_path, classification_map)
        
        model_answer = extract_bbox_answer(original_output[0])
        
        # Count correct answers
        correct = 0
        first_ciou = 0
        first_giou = 0
        if model_answer is not None:
            first_ciou = ciou(model_answer, ground_truth)
            first_giou = giou(model_answer, ground_truth)
            if iou(model_answer, ground_truth) > 0.5:
                correct = 1
        
        passn = 0
        bestn = 0
        mv_ciou = 0
        mv_giou = 0
        bestn_ciou = 0
        bestn_giou = 0
        tmp_score = -1
        tmp_c = 0
        flag = 1
        for o in original_output:
            model_answer = extract_bbox_answer(o)
            score_answer, _ = extract_score_solution_exp03(o)
            # Count correct answers
            if model_answer is not None:
                box_list.append(model_answer)
                # first_ciou_value += ciou(model_answer, ground_truth)
                if iou(model_answer, ground_truth) > 0.5:
                    tmp_c = 1
                    passn = 1
                else:
                    tmp_c = 0
                if flag:
                    bestn = tmp_c
                    bestn_ciou = ciou(model_answer, ground_truth)
                    bestn_giou = giou(model_answer, ground_truth)
                    flag = 0
                try:
                    if score_answer is not None and float(score_answer) > tmp_score:
                        bestn = tmp_c
                        bestn_ciou = ciou(model_answer, ground_truth)
                        bestn_giou = giou(model_answer, ground_truth)
                        tmp_score = float(score_answer)
                except:
                    pass
        
        # Majority vote
        mv_correct = 0
        if box_list:
            mv_box_result = majority_vote([box_list], iou_threshold=0.5, min_votes=1)
            mv_box = mv_box_result[0] if mv_box_result else box_list[0]
        else:
            mv_box = box_list[0] if box_list else [0, 0, 0, 0]
        
        if mv_box != [0, 0, 0, 0]:
            mv_correct = int(iou(mv_box, ground_truth) > 0.5)
            mv_ciou = ciou(mv_box, ground_truth)
            mv_giou = giou(mv_box, ground_truth)
        else:
            mv_ciou = 0
            mv_giou = 0

        result = {
            'image': d['image'],
            'question': d['question'],
            'category': category,
            'ground_truth': ground_truth,
            'model_output': original_output,
            'extracted_answer': model_answer,
            'correct': correct,
            'passn': passn,
            'bestn': bestn,
            'mv_correct': mv_correct,
            'first_ciou': first_ciou,
            'bestn_ciou': bestn_ciou,
            'mv_ciou': mv_ciou,
            'first_giou': first_giou,
            'bestn_giou': bestn_giou,
            'mv_giou': mv_giou
        }
        final_output.append(result)
        
        if category == "long_query":
            long_query_results.append(result)
        elif category == "short_query":
            short_query_results.append(result)
        else:
            unknown_results.append(result)

    overall_metrics = calculate_metrics(final_output)
    long_query_metrics = calculate_metrics(long_query_results)
    short_query_metrics = calculate_metrics(short_query_results)
    
    ds = "Lisa Test"
    print(f"\n{'='*60}")
    print(f"OVERALL RESULTS ({ds})")
    print(f"{'='*60}")
    print(f"Total samples: {overall_metrics['count']}")
    print(f"Accuracy: {overall_metrics['accuracy']:.2f}%")
    print(f"Pass@N Accuracy: {overall_metrics['passn_accuracy']:.2f}%")
    print(f"Best of N Accuracy: {overall_metrics['bestn_accuracy']:.2f}%")
    print(f"Majority Vote Accuracy: {overall_metrics['mv_accuracy']:.2f}%")
    print(f"First CIOU: {overall_metrics['first_ciou']:.2f}")
    print(f"Best_of_N CIOU: {overall_metrics['bestn_ciou']:.2f}")
    print(f"MV CIOU: {overall_metrics['mv_ciou']:.2f}")
    print(f"First GIOU: {overall_metrics['first_giou']:.2f}")
    print(f"Best_of_N GIOU: {overall_metrics['bestn_giou']:.2f}")
    print(f"MV GIOU: {overall_metrics['mv_giou']:.2f}")
    
    print(f"\n{'='*60}")
    print(f"LONG QUERY RESULTS (is_sentence_true)")
    print(f"{'='*60}")
    print(f"Total samples: {long_query_metrics['count']}")
    print(f"Accuracy: {long_query_metrics['accuracy']:.2f}%")
    print(f"Pass@N Accuracy: {long_query_metrics['passn_accuracy']:.2f}%")
    print(f"Best of N Accuracy: {long_query_metrics['bestn_accuracy']:.2f}%")
    print(f"Majority Vote Accuracy: {long_query_metrics['mv_accuracy']:.2f}%")
    print(f"First CIOU: {long_query_metrics['first_ciou']:.2f}")
    print(f"Best_of_N CIOU: {long_query_metrics['bestn_ciou']:.2f}")
    print(f"MV CIOU: {long_query_metrics['mv_ciou']:.2f}")
    print(f"First GIOU: {long_query_metrics['first_giou']:.2f}")
    print(f"Best_of_N GIOU: {long_query_metrics['bestn_giou']:.2f}")
    print(f"MV GIOU: {long_query_metrics['mv_giou']:.2f}")
    
    print(f"\n{'='*60}")
    print(f"SHORT QUERY RESULTS (is_sentence_false)")
    print(f"{'='*60}")
    print(f"Total samples: {short_query_metrics['count']}")
    print(f"Accuracy: {short_query_metrics['accuracy']:.2f}%")
    print(f"Pass@N Accuracy: {short_query_metrics['passn_accuracy']:.2f}%")
    print(f"Best of N Accuracy: {short_query_metrics['bestn_accuracy']:.2f}%")
    print(f"Majority Vote Accuracy: {short_query_metrics['mv_accuracy']:.2f}%")
    print(f"First CIOU: {short_query_metrics['first_ciou']:.2f}")
    print(f"Best_of_N CIOU: {short_query_metrics['bestn_ciou']:.2f}")
    print(f"MV CIOU: {short_query_metrics['mv_ciou']:.2f}")
    print(f"First GIOU: {short_query_metrics['first_giou']:.2f}")
    print(f"Best_of_N GIOU: {short_query_metrics['bestn_giou']:.2f}")
    print(f"MV GIOU: {short_query_metrics['mv_giou']:.2f}")
    
    if unknown_results:
        print(f"\nWarning: {len(unknown_results)} samples could not be classified")

    output_path = output_filepath
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_data = {
        'overall_metrics': overall_metrics,
        'long_query_metrics': long_query_metrics,
        'short_query_metrics': short_query_metrics,
        'results': final_output,
        'classification_summary': {
            'long_query_count': len(long_query_results),
            'short_query_count': len(short_query_results),
            'unknown_count': len(unknown_results),
            'total_count': len(final_output)
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print("-"*100)

except IOError as e:
    print(f"Cannot write output file '{output_filepath}': {e}")
except Exception as e:
    print(f"Error writing merged data: {e}")

# python merge_json2.py --directory_path <your-predictions-dir>/qwen_lisa_test_test_t0.2_topp0.99_topk-1_nsample1/