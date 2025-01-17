import argparse
import asyncio
import aiohttp
import jsonlines
import re
import os
import json
from pathlib import Path
from tqdm import tqdm
from dataset import News, Event
from inference.caption.method import get_caption_generation_method
from inference.insert.method import get_ipp_method
from inference.model.model import InternVL2
from inference.model.utils import vllm_to_openai_format


def parse_args():
    parser = argparse.ArgumentParser(description="Process annotations using a model pipeline.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--annotation_file", type=str, required=True, help="Path to the annotation file.")
    parser.add_argument("--caption_example_file", type=str, required=True, help="Path to the caption example file.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the images directory.")
    parser.add_argument("--news_path", type=str, required=True, help="Path to the news directory.")
    parser.add_argument("--event_file", type=str, required=True, help="Path to the event file.")
    parser.add_argument("--insert_example_file", type=str, required=True, help="Path to the insert example file.")
    parser.add_argument("--url_prefix", type=str, required=True, help="Url prefix to the images.")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1", help="Base URL for API requests.")
    parser.add_argument("--api_key", type=str, default="token-abc123", help="API key for authentication.")
    return parser.parse_args()

args = parse_args()
model_path = args.model_path
annotation_file = args.annotation_file
caption_example_file = args.caption_example_file
image_path = Path(args.image_path)
news_path = Path(args.news_path)
event_file = args.event_file
insert_example_file = args.insert_example_file
url_prefix = args.url_prefix

model = InternVL2(model_name="InternVL2-26B", model_path=model_path, load_type="api")

origin_prompt_prefix = """给定一系列相关事件的新闻（包含图片和文字），提取关键信息并生成简洁的图文总结，使用 Markdown 格式：
# 标题
概述事件核心内容（不超过100字）。  
![标题相关图片描述](IMAGE_i) （如有相关图片）

### 小标题
相关信息概述（不超过100字）。  
![小标题相关图片描述](IMAGE_j) （如有相关图片）

### 小标题
相关信息概述（不超过100字）。  
![小标题相关图片描述](IMAGE_k) （如有相关图片）
...


要求：
1. 生成 3-5 个小标题段落，灵活调整段落数量。
2. 标题下方可插入一张与整体新闻相关的图片（如概述图、公司logo等），其余段落根据内容插入相关图片（如无相关图片可省略）。
3. 每段不超过 100 字，插图使用 ![图片描述](IMAGE_x) 格式，其中“图片描述”概述该图片内容，IMAGE_x 为原新闻图片标号。
"""

new_prompt_prefix = """给定一系列相关事件的新闻，提取关键信息并生成简洁的图文总结，使用 Markdown 格式：
# 标题
概述事件核心内容（不超过100字）。  

### 小标题
相关信息概述（不超过100字）。  

### 小标题
相关信息概述（不超过100字）。  
...


要求：
1. 生成 3-5 个小标题段落，灵活调整段落数量。
3. 每段不超过 100 字。
"""

method_cls = get_caption_generation_method("caption-in-context")
method = method_cls(llm=model)

insert_method_cls = get_ipp_method("i4p-ic")
insert_method = insert_method_cls(llm=model)

def load_insert_ic_examples(event_file, example_file, image_dir, news_dir):
    with open(example_file, "r") as f:
        examples = json.load(f)

    example_event_ids = []
    for example in examples:
        event_id = example["event_id"]
        example_event_ids.append(event_id)

    example_event_infos = {}
    with open(event_file, "r") as f:
        for line in f:
            event_info = json.loads(line)
            if int(event_info["paasEventID"]) in example_event_ids:
                example_event_infos[int(event_info["paasEventID"])] = event_info

    ic_examples = []
    for example in examples:
        event_id = example["event_id"]
        event_info = example_event_infos[event_id]
        event = Event(
            event_info=event_info, root_news_dir=news_dir, root_image_dir=image_dir
        )

        example_text, num_positions = Event.preprocess_text(
            event.title, event.summary, placeholder_policy="bc"
        )
        example_image_caption_str = "\n".join(example["caption_str"])
        example_result = json.dumps(example["result"], indent=4, ensure_ascii=False)
        ic_example = {
            "example_image_caption_str": example_image_caption_str,
            "example_text": example_text,
            "example_num_positions": num_positions,
            "example_result": example_result,
        }
        ic_examples.append(ic_example)

    return ic_examples

def load_ic_examples(example_file, image_dir, news_dir):
    with open(example_file, "r") as f:
        examples = json.load(f)

    ic_examples = []
    for example in examples:
        image_path = example["image_path"]
        news_url = f"{url_prefix}/{image_path}"
        
        image_index = int(image_path.split("/")[1].split(".")[0])

        news_id = image_path.split("/")[0]
        news_path = news_dir / f"{news_id}.json"
        news = News(news_id=news_id, news_path=news_path, root_image_dir=image_dir)

        example_text = News.preprocess_text(
            news.news_title, news.news_content, {image_index: "<image>"}
        )
        ic_example = {
            "example_text": example_text,
            "example_response_0": example["description"],
            "example_response": example["caption"],
            "image": news_url
        }
        ic_examples.append(ic_example)

    return ic_examples

def preprocess_text(
    text: str, placeholder_policy = None
):
    lines = [
        i.strip() for i in text.strip().split("\n") if i.strip() != ""
    ]
    output_lines = []
    num_positions = 0

    for line in lines:
        output_lines.append(line)
        if placeholder_policy == "bc" and line.startswith("#"):
            output_lines.append(f"<Position {num_positions + 1}>")
            num_positions += 1
        elif placeholder_policy == "ac" and not line.startswith("#"):
            output_lines.append(f"<Position {num_positions + 1}>")
            num_positions += 1

    processed_text = "\n\n".join(output_lines)
    return processed_text, num_positions
        

def construct_input(stage1_rslt, image_captions):
    image_caption_inputs = []
    for index, image_caption in enumerate(image_captions):
        image_caption_inputs.append(f"图像{index}: ![{image_caption}](IMAGE_{index})")

    text, num_positions = preprocess_text(
        stage1_rslt, placeholder_policy="bc"
    )
    inputs = {
        "text": text,
        "image_caption_str": "\n".join(image_caption_inputs),
        "num_positions": num_positions,
    }
    return inputs

ic_examples = load_ic_examples(caption_example_file, image_dir=image_path, news_dir=news_path)[:2]

insert_ic_examples = load_insert_ic_examples(
    event_file, insert_example_file, image_path, news_path
)[:2]


async def async_generate_request(session, instruct, retries=3):
    base_url = "http://localhost:8000/v1"
    api_key = "token-abc123"

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
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)  # 指数退避重试
                continue
            else:
                raise Exception(f"Request failed after {retries} attempts: {e}")

async def stage1_gen(annotation, session):
    content = re.sub(r"IMAGE_\d+: <image>", '', annotation["conversations"][0]["value"])
    content = content.replace(origin_prompt_prefix, new_prompt_prefix)
    instruct = [{"role": "user", "content": content}]

    completion = await async_generate_request(session, instruct)
    return completion["choices"][0]["message"]["content"]

async def stage2_gen(text, image, session):
    inputs = {
        "chat_inputs": {"text": text, "image": image},
        "example_inputs": ic_examples
    }
    
    tasks = []
    for index, (messages, example_messages) in enumerate(zip(method.messages, method.example_messages)):
        prompt = method.format_prompt_w_ic(messages, example_messages, inputs)
        instruct = vllm_to_openai_format(prompt)
        response = await async_generate_request(session, instruct)
        inputs["chat_inputs"][f"response_{index}"] = response["choices"][0]["message"]["content"]

    return response

async def stage3_gen(stage1_rslt, image_captions, session):
    inputs = construct_input(stage1_rslt, image_captions)
    
    inputs = {
        "chat_inputs": inputs,
        "example_inputs": insert_ic_examples
    }

    tasks = []
    for index, (messages, example_messages) in enumerate(zip(insert_method.messages, insert_method.example_messages)):
        prompt = insert_method.format_prompt_w_ic(messages, example_messages, inputs)
        instruct = vllm_to_openai_format(prompt)
        tasks.append(async_generate_request(session, instruct))

    responses = await asyncio.gather(*tasks)

    for index, response in enumerate(responses):
        inputs["chat_inputs"][f"response_{index}"] = response["choices"][0]["message"]["content"]
    return inputs["chat_inputs"]["response_0"]


async def gather_with_concurrency(n, tasks):
    semaphore = asyncio.Semaphore(n)

    async def sem_task(task):
        async with semaphore:
            try:
                return await task
            except Exception as e:
                return f"Task failed with error: {e}"

    return await asyncio.gather(*(sem_task(task) for task in tasks), return_exceptions=True)
 
async def process_annotations():
    async with aiohttp.ClientSession() as session:
        with jsonlines.open(annotation_file, 'r') as reader:
            annotations = list(reader)

        tasks = []

        for annotation in annotations:
            async def process_single_annotation(annotation):
                stage12_tasks = []
                stage12_tasks.append(stage1_gen(annotation, session))

                # Stage 2
                images = annotation["image"]
                for image in images:
                    nid, filename = image.split("/")[-2:]
                    image_index = int(filename.split(".")[0])

                    with open(news_path / f"{nid}.json", 'r') as f:
                        news = json.load(f)

                    text = News.preprocess_text(
                        news["title"], news["article_text"], {image_index: "<image>"}
                    )
                    stage12_tasks.append(stage2_gen(text, image, session))

                stage12_responses = await asyncio.gather(*stage12_tasks)
                stage1_response = stage12_responses[0]
                images_captions = stage12_responses[1:]

                # print(stage1_response)
                # for stage2_response in images_captions:
                #     print(stage2_response)

                # # Stage 3
                stage3_response = await stage3_gen(stage1_response, images_captions, session)
                # print(stage3_response)

            # Add annotation task
            tasks.append(process_single_annotation(annotation))

        # Execute all annotation tasks concurrently
        try:
            results = await gather_with_concurrency(100, tasks)  # 限制并发为10个请求
        except Exception as e:
            print(f"Error during processing: {e}")


if __name__ == "__main__":
    asyncio.run(process_annotations())
