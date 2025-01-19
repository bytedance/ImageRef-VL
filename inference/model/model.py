# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union, List, Dict, Any
from pathlib import Path
from dataclasses import dataclass
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

import torch
from transformers import (
    GenerationConfig,
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoTokenizer,
)
from qwen_vl_utils import fetch_image
from deepseek_vl.models import VLChatProcessor
from openai import OpenAI

from .base import BaseModel, VllmModel, HgModel, ApiModel
from .utils import get_optimal_dtype, vllm_to_openai_format

from .internvl.chat_utils import load_image as internvl_load_image


class InternVL2(VllmModel, HgModel, ApiModel):
    @torch.no_grad()
    def hg_gen(self, inputs: List[Dict], **kwargs):
        max_new_tokens = kwargs.pop("max_tokens", 1024)
        temperature = kwargs.pop("temperature", 0)
        do_sample = temperature > 0
        temperature = None if temperature == 0 else temperature

        generation_config = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            **kwargs,
        )

        responses = []
        for i in inputs:
            pixel_values = [
                internvl_load_image(pil_image, max_num=1)
                .to(get_optimal_dtype(dtype="torch"))
                .cuda()
                for pil_image in i["multi_modal_data"]["image"]
            ]
            question = self.tokenizer.apply_chat_template(
                i["prompt"], tokenize=False, add_generation_prompt=True
            )
            num_patches_list = [pixel_value.size(0) for pixel_value in pixel_values]
            pixel_values = torch.cat(pixel_values, dim=0)

            response, history = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
                history=None,
                return_history=True,
                num_patches_list=num_patches_list,
                verbose=False,
            )
            responses.append(response)

        return responses

    def _initialize_api(self):
        self.client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="token-abc123",
        )
        self.thread_pool = ThreadPoolExecutor(max_workers=self.num_api_threads)
        self.request_queue = Queue()
        self._start_request_handler()


@dataclass
class Qwen2VL(VllmModel, HgModel, ApiModel):
    def __post_init__(self, **kwargs):

        if self.load_type == "hg":
            self.model = (
                AutoModelForVision2Seq.from_pretrained(
                    self.model_path,
                    torch_dtype=get_optimal_dtype(dtype="torch"),
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    device_map='auto'
                )
                .eval()
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True, use_fast=False
            )
        else:
            super().__post_init__()
            
        if self.load_type != "api":
            self.processor = AutoProcessor.from_pretrained(self.model_path)

    @torch.no_grad()
    def hg_gen(self, inputs: List[Dict], **kwargs):
        max_new_tokens = kwargs.pop("max_tokens", 1024)
        temperature = kwargs.pop("temperature", 0)
        do_sample = temperature > 0
        temperature = None if temperature == 0 else temperature

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            **kwargs,
        )

        responses = []
        for i in inputs:
            image_inputs = [
                fetch_image({"image": pil_image})
                for pil_image in i["multi_modal_data"]["image"]
            ]

            question = self.tokenizer.apply_chat_template(
                i["prompt"], tokenize=False, add_generation_prompt=True
            )
            question = question.replace("<image>", "<|image_pad|>")

            model_inputs = self.processor(
                text=[question],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )

            model_inputs = model_inputs.to("cuda")

            generated_ids = self.model.generate(
                **model_inputs, generation_config=generation_config
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            responses.append(response)

        return responses

    
    def vllm_preprocess_prompt(self, raw_input):
        messages = vllm_to_openai_format(raw_input)

        messages[0]["content"] = [
            i if i["type"] == "text" else 
            {"type": "image", "image": i["image_url"]['url']} 
            for i in messages[0]["content"]
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt
       

@dataclass
class DeepseekVL(VllmModel, HgModel):
    def __post_init__(self, **kwargs):
        super().__post_init__()
        self.processor = VLChatProcessor.from_pretrained(self.model_path)

    def hg_gen(self, inputs: List[Dict[str, Any]], **kwargs):
        role_map = {}

@dataclass
class PhiModel(VllmModel, HgModel):
    def vllm_preprocess_prompt(self, raw_input):
        def replace_image_tags(text):
            count = 1
            
            def replace(match):
                nonlocal count
                replacement = f"<|image_{count}|>"
                count += 1
                return replacement
            
            replaced_text = re.sub(r'<image>', replace, text)
            
            return replaced_text
            
        prompt = self.tokenizer.apply_chat_template(
            raw_input["prompt"], tokenize=False, add_generation_prompt=self.add_generation_prompt
        )

        prompt = replace_image_tags(prompt)
        return prompt
    
    def _initialize_vllm(self):
        from vllm import LLM

        self.model = LLM(
            self.model_path,
            trust_remote_code=True,
            dtype=get_optimal_dtype(),
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=131072,
            gpu_memory_utilization=0.9,
            distributed_executor_backend="ray",
            enforce_eager=True,
            limit_mm_per_prompt={"image": self.limit_image_per_prompt},
            mm_processor_kwargs={"num_crops": 4}
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_path, 
            trust_remote_code=True, 
            num_crops=4
        )

        self.tokenizer = self.processor.tokenizer
   

class OpenAIModel(ApiModel):
    pass

class GeminiModel(ApiModel):
    def __post_init__(self, **kwargs):
        os.environ["AZURE_OPENAI_ENDPOINT"] = os.environ["AZURE_OPENAI_ENDPOINT_MM"]
        return super().__post_init__()

class ClaudeModel(ApiModel):
    def vllm_to_model_format(self, input_data: Dict[str, Any]):
        openai_input = super().vllm_to_model_format(input_data)

        claude_content = []

        for item in openai_input:
            if isinstance(item["content"], str):
                claude_content.append(item)
            elif isinstance(item["content"], list):
                formatted_item = []
                for sub_item in item["content"]:
                    if sub_item["type"] == "text":
                        formatted_item.append(sub_item)
                    elif sub_item["type"] == "image_url":
                        base64_image = sub_item["image_url"]["url"]
                        media_type = base64_image.split(";")[0][len("data:") :]
                        data = base64_image.split(";")[1][len("base64,") :]

                        formatted_item.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": data,
                                },
                            }
                        )
                claude_content.append({"role": item["role"], "content": formatted_item})
                
        return claude_content
