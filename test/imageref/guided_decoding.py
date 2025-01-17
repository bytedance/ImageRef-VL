import jsonlines
import shutil
import re
from pathlib import Path
from tqdm import tqdm
import argparse

from outlines.serve.vllm import RegexLogitsProcessor
from vllm import SamplingParams

from inference.model import get_model_cls
from dataset import NewsImage
import requests
from PIL import Image
from io import BytesIO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_model_path", default="/opt/tiger/ImagePositionPrediction/models/InternVL2-8B", type=str)
    parser.add_argument("--model_name", default="", type=str)
    parser.add_argument("--model_path", default="/opt/tiger/ImagePositionPrediction/output/checkpoint-740", type=str)
    parser.add_argument("--test_annotation_file", default="/opt/tiger/ImagePositionPrediction/data/multi_modal/internvl2/opensource2/cmi_test_500.jsonl", type=str)
    parser.add_argument("--output_file", default="/opt/tiger/ImagePositionPrediction/output/test_results/image-8b-740.jsonl", type=str)
    parser.add_argument("--wcap", action="store_true", default=False)
    return parser.parse_args()

args = parse_args()
raw_model_path = args.raw_model_path
model_path = args.model_path
test_annotation_file = args.test_annotation_file
wcap = args.wcap
model_name = args.model_name

def copy_files_if_not_exists(source_dir, target_dir, file_pattern="*.py"):
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    target_path.mkdir(parents=True, exist_ok=True)

    for file in source_path.glob(file_pattern):
        target_file = target_path / file.name
        
        if not target_file.exists():
            shutil.copy(file, target_file)
            print(f"Copied {file} to {target_file}")
        else:
            print(f"File {file} already exists in the target directory")

# %%
model_cls = get_model_cls(args.model_name)

copy_files_if_not_exists(raw_model_path, model_path)
model = model_cls(model_name=model_name, model_path=model_path, load_type="vllm", limit_image_per_prompt=50, add_generation_prompt=False)
model.model.llm_engine.get_model_config().hf_config.max_dynamic_patch = 2


# %%
def conv_to_prompt(conversation):
    prompt = [{"role": c["from"], "content": c["value"]} for c in conversation]
    return prompt

def load_pil_image(image_url):
    while True:
        try:
            response = requests.get(image_url)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content)).convert('RGB')
                return image
            else:
                raise Exception(f"Failed to load image from URL {image_url}. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error loading image from URL {image_url}: {e}")
            continue

def construct_prompt(text):
    lines = [i.strip() for i in text.strip().split('\n') if i.strip() != ""]
    output_lines = []
    num_blanks = 0

    for line in lines:
        if line.startswith("#"):
            output_lines.append(line + "\n\n")
        else:
            output_lines.append(line + f"\n\n[BLANK_{num_blanks}]")
            num_blanks += 1
        
    processed_text = ''.join(output_lines)
    
    return [processed_text.strip(), num_blanks]


def construct_blank_prompt(text, blank_index):
    if f"[BLANK_{blank_index}]" in text:
        blank_place = text.index(f"[BLANK_{blank_index}]")
        return text[:blank_place]
    else:
        return None

# %%
    
inputs, max_num_blanks = [], 0
with jsonlines.open(test_annotation_file, 'r') as reader:
    for obj in tqdm(reader):
        prompt = {
            "prompt": conv_to_prompt([obj["conversations"][0], {"from": "gpt", "value": ""}]),
            "multi_modal_data": {
                "image": [load_pil_image(i) for i in obj["image"]]
            },
        }
        response = construct_prompt(obj["conversations"][1]["value"])
        inputs.append(
            {"id": obj["id"], "prompt": prompt, "response": response}
        )
        max_num_blanks = max(max_num_blanks, response[1])
        
# %%

completed_rslt = []
completed_ids = set()
for blank_index in range(max_num_blanks + 1):
    prompts, sampling_params = [], []
    for i in inputs:
        response, num_blanks = i["response"]

        if blank_index < num_blanks:
            i["prompt"]["prompt"][1]["content"] = construct_blank_prompt(i["response"][0], blank_index)
            prompts.append(i)
            image_placeholders = re.findall(f"IMAGE_\d+", i["prompt"]["prompt"][0]["content"])
            
            if not wcap:
                regex_string =  "|".join(image_placeholders + [r"###", r"\s+"])
            else:
                regex_string =  "|".join([f"!\[[^\]]*\]\({i}\)" for i in image_placeholders] + [r"###", r"\s+"])

            logits_processor = RegexLogitsProcessor(regex_string=regex_string, llm=model.model.llm_engine)
            sampling_params.append(
                SamplingParams(max_tokens=512, temperature=0, logits_processors=[logits_processor])
            )
        else:
            if i["id"] not in completed_ids:
                completed_ids.add(i["id"])
                completed_rslt.append(i)

    if len(prompts) > 0:
        results = model.generate(
            [i["prompt"] for i in prompts],
            **{"sampling_params": sampling_params})
        
        for i, r in zip(prompts, results):
            if "IMAGE" in r:
                i["response"][0] = i["response"][0].replace(f"[BLANK_{blank_index}]", r + "\n\n")
            else:
                i["response"][0] = i["response"][0].replace(f"[BLANK_{blank_index}]", "")
            

# %%
Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

objs = []
for result in completed_rslt:
    objs.append({"id": result["id"], "result": result["response"][0]})

with jsonlines.open(args.output_file, 'w') as writer:
    writer.write_all(objs)
    
# %%
