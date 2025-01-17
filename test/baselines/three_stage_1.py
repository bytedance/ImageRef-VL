from inference.model import get_model_cls

import argparse
from pathlib import Path
import jsonlines
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
from io import BytesIO
import re

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_file", default="", type=str)
    parser.add_argument("--root_image_path", default="", type=str)
    parser.add_argument("--model_name", default="", type=str)
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--output_file", default=None, type=str)
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
model = model_csl(model_name=model_name, model_path=args.model_path, load_type=args.load_type, limit_image_per_prompt=50)


exists_nids = set()
if Path(args.output_file).exists():
    with jsonlines.open(args.output_file, 'r') as reader:
        for obj in reader:
            exists_nids.add(obj["id"])
else:
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)


def remove_image(text, regex_pattern=r"IMAGE_\d+: <image>"):
    cleaned_text = re.sub(regex_pattern, '', text)
    return cleaned_text


origin_prompt_prefix = """给定一系列相关事件的新闻（包含图片和文字），提取关键信息并生成简洁的图文总结，使用 Markdown 格式：
```
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
```

要求：
1. 生成 3-5 个小标题段落，灵活调整段落数量。
2. 标题下方可插入一张与整体新闻相关的图片（如概述图、公司logo等），其余段落根据内容插入相关图片（如无相关图片可省略）。
3. 每段不超过 100 字，插图使用 `![图片描述](IMAGE_x)` 格式，其中“图片描述”概述该图片内容，`IMAGE_x` 为原新闻图片标号。
"""

new_prompt_prefix = """给定一系列相关事件的新闻，提取关键信息并生成简洁的图文总结，使用 Markdown 格式：
```
# 标题
概述事件核心内容（不超过100字）。  

### 小标题
相关信息概述（不超过100字）。  

### 小标题
相关信息概述（不超过100字）。  
...
```

要求：
1. 生成 3-5 个小标题段落，灵活调整段落数量。
3. 每段不超过 100 字。
"""

prompts, ids = [], []
for annotation in tqdm(annotations):
    if annotation["id"] in exists_nids:
        continue
    content = remove_image(annotation["conversations"][0]["value"])
    content = content.replace(origin_prompt_prefix, new_prompt_prefix)
    instruct = [{"role": "user", "content": content}]
    prompt = {"prompt": instruct}
    prompts.append(prompt)
    ids.append(annotation["id"])
        
responses = model.generate(inputs=prompts, temperature=0, max_tokens=1024)

with jsonlines.open(args.output_file, 'a') as writer:
    for response, nid, prompt in zip(responses, ids, prompts):
        result = {'id': nid, "result": response, "prompt": prompt["prompt"]}

        writer.write(result)