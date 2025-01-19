# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import json
import re
import jsonlines
import sys
from pathlib import Path
from dotenv import load_dotenv
import argparse

load_dotenv()

from inference.model import get_model_cls


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file", default="test/imageref/prompt.jsonl", type=str)
    parser.add_argument("--prompt_name", default="single-cmi", type=str)
    parser.add_argument("--eval_model", default="", type=str)
    parser.add_argument("--annotation_file", default="", type=str)
    parser.add_argument("--response_file", default="", type=str)
    parser.add_argument("--wcap", action="store_true", help="Whether to use wcap")
    parser.add_argument("--output_file", default="", type=str)
    
    return parser.parse_args()
    
args = parse_args()

prompt_file = args.prompt_file
prompt_name = args.prompt_name
eval_model = args.eval_model

annotation_file = args.annotation_file
response_file = args.response_file
wcap = args.wcap


def load_prompt(prompt_name):
    prompts = {}
    with jsonlines.open(prompt_file, 'r') as f:
        for obj in f:
            prompts[obj["name"]] = obj
    return prompts[prompt_name]

prompt_template = load_prompt(prompt_name)


model_cls = get_model_cls(eval_model)
eval_llm = model_cls(eval_model, load_type="api")

with jsonlines.open(annotation_file, 'r') as reader:
    annotations = {obj["id"]: obj for obj in list(reader)}
    
with jsonlines.open(response_file, 'r') as reader:
    responses = {obj["id"]: obj for obj in list(reader)}


def remove_image(text, regex_pattern=r'IMAGE_\d+'):
    cleaned_text = re.sub(regex_pattern, '', text)
    return cleaned_text

def judge(prompt, annotation, response):
    prompt_template = prompt["prompt_template"]

    question = remove_image(annotation["conversations"][0]["value"], r"IMAGE_\d+: <image>")
    ref_answer = remove_image(annotation["conversations"][1]["value"], r"!\[([^\]]*)\]\((IMAGE_\d+)\)" if wcap else r"IMAGE_\d+")
    answer = remove_image(response["result"], r"!\[([^\]]*)\]\((IMAGE_\d+)\)" if wcap else r"\n\nIMAGE_\d+")
    
    user_prompt = prompt_template.format(
        question=question,
        ref_answer_1=ref_answer,
        answer=answer,
    )
    
    prompt = [
        {"role": "system", "content": prompt["system_prompt"]},
        {"role": "user", "content": user_prompt}
    ]
    return prompt

prompts = []
for nid in annotations:
    if nid in responses and "result" in responses[nid]:
        if responses[nid]["result"]:
            prompt = judge(prompt_template, annotations[nid], responses[nid])
        prompts.append({"prompt": prompt, "nid": nid})

responses = eval_llm.generate(prompts)

rslts = []
for prompt, response in zip(prompts, responses):
    rslt = {"id": prompt["nid"], "prompt": prompt["prompt"], "response": response}
    rslts.append(rslt)
    
    
with jsonlines.open(args.output_file, 'w') as writer:
    writer.write_all(rslts)