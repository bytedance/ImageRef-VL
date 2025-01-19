# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from inference.model import get_model_cls

import argparse
from pathlib import Path
import jsonlines
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
from io import BytesIO

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_file", default="", type=str)
    parser.add_argument("--root_image_path", default="", type=str)
    parser.add_argument("--model_name", default="", type=str)
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--output_file", default="", type=str)
    parser.add_argument("--load_type", default="api", type=str)
    args = parser.parse_args()
    return args

args = parse_args()

annotation_file = args.annotation_file
root_image_path = Path(args.root_image_path)
model_name = args.model_name

annotations = []
with jsonlines.open(annotation_file, 'r') as reader:
    annotations = list(reader)
    

model_csl = get_model_cls(model_name)
model = model_csl(model_name=model_name, model_path=args.model_path, load_type=args.load_type, limit_image_per_prompt=20)

def pil_image(path, max_size=1024, quality=85, target_size=1 * 1024 * 1024):
    try:
        img = Image.open(path).convert("RGB")
        
        # Step 1: Resize the image to max_size (e.g., 1024x1024)
        img.thumbnail((max_size, max_size), Image.LANCZOS)

        # Step 1.5: Ensure both width and height are at least 50 pixels
        if img.width < 50 or img.height < 50:
            scale_factor = max(50 / img.width, 50 / img.height)
            new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
            img = img.resize(new_size, Image.LANCZOS)

        # Step 2: Adjust quality in memory until the file size is within target
        img_bytes = BytesIO()
        img.save(img_bytes, format="JPEG", quality=quality)
        
        # Step 3: Check file size and reduce quality if needed
        while img_bytes.tell() > target_size and quality > 10:
            quality -= 5  # Reduce quality by 5 for each iteration
            img_bytes.seek(0)
            img_bytes.truncate(0)
            img.save(img_bytes, format="JPEG", quality=quality)
        
        # print(f"Final quality: {quality}, Final size: {img_bytes.tell() / 1024:.2f} KB")
        img_bytes.seek(0)  # Reset pointer to start of the buffer
        return Image.open(img_bytes)
    
    except Exception as e:
        raise ValueError(f"Failed to load {path}. {e}")

exists_nids = set()
if Path(args.output_file).exists():
    with jsonlines.open(args.output_file, 'r') as reader:
        for obj in reader:
            exists_nids.add(obj["id"])
else:
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

prompts, ids = [], []
for annotation in tqdm(annotations):
    if annotation["id"] in exists_nids:
        continue

    if len(annotation["image"]) > 20:
        print(f"Skip {annotation['id']}: too many images.")
        continue
    
    instruct = [{"role": "user", "content": annotation["conversations"][0]["value"]}]
    image = [pil_image(root_image_path / image_path) for image_path in annotation["image"]]
    prompt = {"prompt": instruct, "multi_modal_data": {"image": image}}
    prompts.append(prompt)
    ids.append(annotation["id"])

responses = model.generate(inputs=prompts, temperature=0, max_tokens=1024)

with jsonlines.open(args.output_file, 'a') as writer:
    for response, nid, prompt in zip(responses, ids, prompts):
        result = {'id': nid, "result": response, "prompt": prompt["prompt"]}

        writer.write(result)