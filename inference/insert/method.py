from typing import List, Union, Dict, Any, Optional
from dataclasses import dataclass

from transformers import PreTrainedTokenizer
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

from inference.model import BaseModel
from inference.method_utils import BaseGenerator


IMAGE_INSERTATION_METHODS = {}


def register_ipp_method(method_name: str):
    def decorator(cls):
        IMAGE_INSERTATION_METHODS[method_name] = cls
        return cls

    return decorator


def get_ipp_method(method_name: str) -> BaseGenerator:
    return IMAGE_INSERTATION_METHODS[method_name]



@dataclass
@register_ipp_method("i4p-ic")
class I4PICGenerator(BaseGenerator):
    is_in_context: Optional[bool] = True

    def _init_instruct(self):
        self.example_messages = [
            [
                (
                    "human",
                    """图片描述:
[图片的开始]
{example_image_caption_str}
[图片的结束]

新闻文本:
[新闻文本的开始]
{example_text}
[新闻文本的结束]

请直接按照指定JSON格式返回新闻文本中{example_num_positions}个图像插入点的插图结果。""",
                ),
                ("assistant", """{example_result}"""),
            ]
        ]

        self.messages = [
            [
                (
                    "system",
                    """你是一个专业的新闻编辑，擅长于分析给定的新闻文本，并根据段落内容插入合适的图片。\
你会收到一段新闻文本和一些图片的描述。新闻中有一些图像插入点（格式为 <Position i>），这些插入点基于新闻的段落划分，每个段落标题后都预留了一个插入点。请分析每张图片与新闻内容的相关性，并按以下要求决定是否插入：

1. 请插入与插入点所在段落最匹配的图片。分析段落内容时请基于段落标题与段落文字共同分析，而不仅仅基于段落标题。
2. 每段最多插入一张相关图片，如无相关的图片则不插入。
3. 图片中的人物身份不明确时，请勿猜测新闻人物进行插入。
4. 每张图片只能插入到一个段落中，已经插入文章的图片请勿插入到其他位置。
5. 对于第一个插入点(<Position 1>)，除了考虑段落相关性外，还要考虑该图片是否适合作为新闻的头图，可以是新闻的概述或具有代表性的图片，如公司标志、重要人物等。

对于每个插入位置，请分析插入点所在段落主旨，确定适合插入的图片类型和内容。从提供的图片中选择相关图片并说明原因。如无相关图片，请标记为'Not Insert'并解释原因。

请将结果以如下JSON格式返回：
```json
{{
    "predictions": [
        {{
            "position": "<Position 1>",
            "analysis": "分析段落主旨和适合的图片类型与内容",
            "image": "相关图片序号（如 IMAGE_i）或'Not Insert'",
            "explanation": "解释选择该图片或不插入的原因"
        }},
        ...
    ]
}}
```
""",
                ),
                "IN_CONTEXT_PLACEHOLDER",
                (
                    "human",
                    """图片描述:
[图片的开始]
{image_caption_str}
[图片的结束]

新闻文本:
[新闻文本的开始]
{text}
[新闻文本的结束]

请直接按照指定JSON格式返回新闻文本中{num_positions}个图像插入点的插图结果。""",
                ),
            ]
        ]

