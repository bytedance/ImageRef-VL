# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import List, Dict, Union, Tuple
from PIL import Image
import io
import base64

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModel


def get_stop_token_ids(model_name: str, tokenizer: PreTrainedTokenizer):
    if "internvl2" in model_name.lower():
        stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
        stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    elif "qwen" in model_name.lower():
        stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
        stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    elif "phi" in model_name.lower():
        stop_token_ids = None

    return stop_token_ids


def get_optimal_dtype(dtype="str"):
    if not torch.cuda.is_available():
        return "float32" if dtype == "str" else torch.float32

    device = torch.device("cuda")
    gpu_properties = torch.cuda.get_device_properties(device)

    if gpu_properties.major >= 8:
        return "bfloat16" if dtype == "str" else torch.bfloat16
    elif gpu_properties.major >= 7 or (
        gpu_properties.major == 6 and gpu_properties.minor >= 1
    ):
        return "float16" if dtype == "str" else torch.float16
    else:
        return "float32" if dtype == "str" else torch.float32


def image_to_data_url(image: Image.Image) -> str:
    buffered = io.BytesIO()

    image_format = image.format or "PNG"
    image.save(buffered, format=image_format)

    extension = image_format.lower()

    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    data_url = f"data:image/{extension};base64,{img_str}"
    return data_url


def replace_image(prompt: str, base64_images: List[str]) -> Tuple[str, int]:
    content = []
    image_index = 0

    while "<image>" in prompt and image_index < len(base64_images):
        placeholder_index = prompt.find("<image>")
        if placeholder_index == -1:
            break

        pre_message = prompt[:placeholder_index]
        if prompt.strip() != "":
            content.append({"type": "text", "text": pre_message})

        base64_image = base64_images[image_index]
        content.append({"type": "image_url", "image_url": {"url": base64_image}})

        prompt = prompt[placeholder_index + len("<image>") :]
        image_index += 1

    if prompt.strip() != "":
        content.append({"type": "text", "text": prompt})

    return content, image_index  # Return remaining images


def vllm_to_openai_format(
    vllm_input: Dict[str, Union[List[Dict[str, str]], Dict[str, List[Image.Image]]]]
) -> List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]]:
    prompts = vllm_input.get("prompt", [])
    multi_modal_data = vllm_input.get("multi_modal_data", {}).get("image", [])

    if len(multi_modal_data) > 0 and isinstance(multi_modal_data[0], Image.Image):
        base64_images = [image_to_data_url(img) for img in multi_modal_data]
    else:
        base64_images = multi_modal_data

    openai_input = []

    for entry in prompts:
        role = entry["role"]
        content = entry["content"]

        if "<image>" in content:
            replaced_content, num_images = replace_image(content, base64_images)
            openai_input.append({"role": role, "content": replaced_content})
            base64_images = base64_images[num_images:]
        else:
            openai_input.append({"role": role, "content": content})
    
    return openai_input
