# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import aiohttp
from tqdm.asyncio import tqdm
import jsonlines
import re
import requests

from inference.model.utils import vllm_to_openai_format


def parse_args():
    parser = argparse.ArgumentParser(description="Process annotations using a model pipeline.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--annotation_file", type=str, required=True, help="Path to the annotation file.")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1", help="Base URL for API requests.")
    parser.add_argument("--api_key", type=str, default="token-abc123", help="API key for authentication.")
    return parser.parse_args()

args = parse_args()

base_url = args.base_url
api_key = args.api_key
model_path = args.model_path
annotation_file = args.annotation_file

# 读取文件
def read_annotations(file_path):
    with jsonlines.open(file_path, 'r') as reader:
        return list(reader)

def remove_image(text, regex_pattern=r"IMAGE_\d+: <image>"):
    cleaned_text = re.sub(regex_pattern, '', text)
    return cleaned_text

async def e2e_gen(session, annotation, retries=3):
    try:
        content = annotation.get("conversations", [{}])[0].get("value", "")
        image_urls = annotation.get("image", [])
        if not content:
            raise ValueError("Annotation content is missing.")

        prompt = {
            "prompt": [{"role": "user", "content": content}],
            "multi_modal_data": {"image": image_urls},
        }
        instruct = vllm_to_openai_format(prompt)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model_path,
            "messages": instruct,
            "stop": ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"],
            "temperature": 0,
            "max_tokens": 2048,
            "skip_special_tokens": False
        }

        for attempt in range(retries):
            try:
                async with session.post(f"{base_url}/chat/completions", json=payload, headers=headers) as response:
                    if response.status == 400:
                        error_details = await response.text()
                        raise ValueError(f"BadRequestError: {error_details}")
                    elif response.status != 200:
                        raise Exception(f"Request failed with status {response.status}: {await response.text()}")
                    return await response.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)  # 指数退避重试
                    continue
                else:
                    raise Exception(f"Request failed after {retries} attempts: {e}")
    except KeyError as e:
        raise ValueError(f"Missing required field in annotation: {e}")

async def gather_with_concurrency(n, tasks):
    semaphore = asyncio.Semaphore(n)

    async def sem_task(task):
        async with semaphore:
            try:
                return await task
            except Exception as e:
                return f"Task failed with error: {e}"

    return await asyncio.gather(*(sem_task(task) for task in tasks), return_exceptions=True)

async def main():
    annotations = read_annotations(annotation_file)

    async with aiohttp.ClientSession() as session:
        tasks = [e2e_gen(session, annotation) for annotation in annotations]
        try:
            results = await gather_with_concurrency(100, tasks)  # 限制并发为10个请求
            for result in results:
                if isinstance(result, Exception):
                    print(f"Task encountered an error: {result}")
                else:
                    print(result)
        except Exception as e:
            print(f"Error during processing: {e}")

if __name__ == "__main__":
    asyncio.run(main())
