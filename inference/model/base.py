# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union, List, Dict, Any
from pathlib import Path
import copy
import time
from dataclasses import dataclass
from PIL.Image import Image
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

import torch
from transformers import (
    GenerationConfig,
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoModel,
)
from openai import AzureOpenAI, BadRequestError, RateLimitError, APITimeoutError, APIConnectionError

from .utils import get_stop_token_ids, get_optimal_dtype, vllm_to_openai_format


@dataclass
class BaseModel:
    model_name: str
    model_path: Optional[Union[Path, str]] = None
    load_type: str = "hg"
    load_in_8bits: bool = False
    num_api_threads: int = 10
    limit_image_per_prompt: int = 5
    add_generation_prompt: bool = True

    def __post_init__(self, **kwargs):
        if self.load_type == "hg":
            self._initialize_hg()
        elif self.load_type == "vllm":
            self._initialize_vllm()
        elif self.load_type == "api":
            self._initialize_api()
        else:
            raise ValueError(f"Unsupported load_type: {self.load_type}")

    def _initialize_api(self):
        raise NotImplementedError

    def _initialize_hg(self):
        raise NotImplementedError

    def _initialize_vllm(self):
        raise NotImplementedError

    def api_initialize_api_gen(self):
        raise NotImplementedError

    def generate(self, inputs, **kwargs):
        if self.load_type == "hg":
            responses = self.hg_gen(inputs, **kwargs)
        elif self.load_type == "vllm":
            responses = self.vllm_gen(inputs, **kwargs)
        elif self.load_type == "api":
            responses = self.api_gen(inputs, **kwargs)

        return responses

    def hg_gen(self, inputs: List[Dict], **kwargs):
        raise NotImplementedError

    def vllm_gen(self, inputs: List[Dict], **kwargs):
        raise NotImplementedError

    def api_gen(self, inputs: List[Dict], **kwargs):
        raise NotImplementedError

    def shutdown(self):
        if self.load_type == "api":
            self.api_shutdown()
        elif self.load_type == "vllm":
            self.vllm_shutdown()
        elif self.load_type == "hg":
            self.hg_shutdown()

    def hg_shutdown(self):
        raise NotImplementedError

    def vllm_shutdown(self):
        raise NotImplementedError

    def api_shutdown(self):
        raise NotImplementedError


@dataclass
class VllmModel(BaseModel):
    limit_image_per_prompt: int = 5
    add_generation_prompt: bool = True

    def _initialize_vllm(self):
        from vllm import LLM

        self.model = LLM(
            self.model_path,
            trust_remote_code=True,
            dtype=get_optimal_dtype(),
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=32000,
            gpu_memory_utilization=0.9,
            distributed_executor_backend="ray",
            enforce_eager=True,
            limit_mm_per_prompt={"image": self.limit_image_per_prompt}
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=False
        )
    
    def vllm_preprocess_prompt(self, raw_input):
        return self.tokenizer.apply_chat_template(
            raw_input["prompt"], tokenize=False, add_generation_prompt=self.add_generation_prompt
        )

    def vllm_gen(self, inputs: List[Dict], **kwargs):
        from vllm import SamplingParams

        inputs = copy.deepcopy(inputs)
        for i in inputs:
            i["prompt"] = self.vllm_preprocess_prompt(i)

        stop_token_ids = get_stop_token_ids(self.model_name, self.tokenizer)
        sampling_params = kwargs["sampling_params"] if "sampling_params" in kwargs else SamplingParams(stop_token_ids=stop_token_ids, **kwargs) 

        outputs = self.model.generate(inputs, sampling_params=sampling_params)
        responses = [o.outputs[0].text.strip() for o in outputs]
        return responses

    def vllm_shutdown(self):
        pass


class HgModel(BaseModel):
    def _initialize_hg(self):
        self.model = (
            AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=get_optimal_dtype(dtype="torch"),
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                load_in_8bit=self.load_in_8bits,
            )
            .eval()
            .cuda()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=False
        )

    @torch.no_grad()
    def hg_gen(self, inputs: List[Dict], **kwargs):
        raise NotImplementedError

    def hg_shutdown(self):
        pass


class ApiModel(BaseModel):
    def _initialize_api(self):
        self.client = AzureOpenAI(api_version="2023-03-15-preview")
        self.thread_pool = ThreadPoolExecutor(max_workers=self.num_api_threads)
        self.request_queue = Queue()
        self._start_request_handler()

    def vllm_to_model_format(self, input_data: Dict[str, Any]):
        return vllm_to_openai_format(input_data)

    def api_gen(self, inputs: List[Dict[str, Any]], **kwargs):
        def call_api(input_data, idx):
            while True:
                try:
                    formatted_input = self.vllm_to_model_format(input_data)
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=formatted_input,
                        seed=2024,
                        **kwargs,
                    )
                    return idx, response.choices[0].message.content
                except BadRequestError as e:
                    print(f"BadRequestError: {e}")
                    return idx, None
                except RateLimitError as e:
                    time.sleep(0.5)
                except Exception as e:
                    raise e

        futures = [
            self.thread_pool.submit(call_api, input_data, idx)
            for idx, input_data in enumerate(inputs)
        ]

        responses = [None] * len(inputs)

        for future in futures:
            idx, result = future.result()
            responses[idx] = result

        return responses

    def _start_request_handler(self):
        def handle_requests():
            while True:
                func, args, kwargs = self.request_queue.get()
                if func is None:
                    break
                func(*args, **kwargs)
                self.request_queue.task_done()

        self.request_handler_thread = threading.Thread(
            target=handle_requests, daemon=True
        )
        self.request_handler_thread.start()

    def api_shutdown(self):
        self.request_queue.put((None, (), {}))
        self.request_handler_thread.join()
        self.thread_pool.shutdown(wait=True)
