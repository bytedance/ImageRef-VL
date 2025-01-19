# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jsonlines
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse

from inference.model import get_model_cls
from dataset import NewsImage
import requests
from PIL import Image
from io import BytesIO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_model_path", default="/opt/tiger/ImagePositionPrediction/models/InternVL2-8B", type=str)
    parser.add_argument("--model_path", default="/opt/tiger/ImagePositionPrediction/output/checkpoint-740", type=str)
    parser.add_argument("--test_annotation_file", default="/opt/tiger/ImagePositionPrediction/data/multi_modal/internvl2/opensource2/cmi_test_500.jsonl", type=str)
    parser.add_argument("--output_file", default="/opt/tiger/ImagePositionPrediction/output/test_results/8b-740.jsonl", type=str)
    return parser.parse_args()
    
args = parse_args()

raw_model_path = args.raw_model_path
model_path = args.model_path
test_annotation_file = args.test_annotation_file


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


model_cls = get_model_cls("internvl2")


copy_files_if_not_exists(raw_model_path, model_path)
model = model_cls(model_name="internvl2_8b", model_path=model_path, load_type="vllm", limit_image_per_prompt=50)
model.model.llm_engine.get_model_config().hf_config.max_dynamic_patch = 2


def conv_to_prompt(conversation):
    prompt = [{"role": c["from"], "content": c["value"]} for c in conversation]
    return prompt

def load_pil_image(image_url):
    while True:
        try:
            response = requests.get(image_url)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                return image
            else:
                raise Exception(f"Failed to load image from URL {image_url}. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error loading image from URL {image_url}: {e}")
            continue


# %%
    
prompts = []
ids = []
with jsonlines.open(test_annotation_file, 'r') as reader:
    for obj in tqdm(reader):
        prompt = {
            "prompt": conv_to_prompt([obj["conversations"][0]]),
            "multi_modal_data": {
                "image": [load_pil_image(i) for i in obj["image"]]
            },
        }
        prompts.append(prompt)
        ids.append(obj["id"])
        
results = model.generate(prompts, **{"temperature": 0, "max_tokens": 1024})

objs = []
for i, result in zip(ids, results):
    objs.append({"id": i, "result": result})

with jsonlines.open(args.output_file, 'w') as writer:
    writer.write_all(objs)