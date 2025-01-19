# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Dict, Any, Optional
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate

from inference.method_utils import BaseGenerator


CAPTION_GENERATION_METHODS = {}


def register_caption_generation_method(method_name: str):
    def decorator(cls):
        CAPTION_GENERATION_METHODS[method_name] = cls
        return cls

    return decorator


def get_caption_generation_method(method_name: str) -> BaseGenerator:
    return CAPTION_GENERATION_METHODS[method_name]



@dataclass
@register_caption_generation_method("caption-in-context")
class CaptionInContextGenerator(BaseGenerator):
    is_in_context: Optional[bool] = True

    def _init_instruct(self):
        self.example_messages = [
            [
                (
                    "human",
                    "<image> 请直接返回一段 40 字以内的图片描述，无需添加其他说明。",
                ),
                ("ai", "{example_response_0}"),
            ],
            [
                (
                    "human",
                    """[图片新闻上下文的开始]
{example_text}
[图片新闻上下文的结束]

[原始图片描述的开始]
{example_response_0}
[原始图片描述的结束]

请根据新闻上下文补全图片中的人物事件信息与图片类型。请直接返回一段 40 字以内的图片描述, 无需添加其他说明。""",
                ),
                ("ai", "{example_response}"),
            ],
        ]

        self.messages = [
            [
                (
                    "system",
                    """你是一个专业的新闻图像分析助手。请分析图片内容，按照如下要求描述图片。
1. 图片内容应尽可能清晰准确地描述，突出图片中的核心要素，如人物、地点、场景、活动或其他重要的视觉元素。
2. 尽量简洁，但要保持信息的完整性，避免过于冗长或模糊。
3. 对于人物的描述，如果模型确定人物的身份或姓名，请直接指出。如果模型不确定，请不要猜测或编造，改为描述人物的生物信息（如性别、年龄段、衣着等）。
4. 当图片为金融图表、公告文字、社交媒体截图或论文配图时：
   a. 清楚说明图表或文本的类型（如“金融图表”、“公告”、“Twitter截图”、“照片”、“广告”等）。
   b. 准确提取并总结图表、公告、截图或配图中的关键信息，如数据趋势、主要文字内容、关键论点或结论。
   c. 如果图片中包含具体数据、日期、数字或名称，确保这些信息准确无误。
5. 如果图片中有任何文字、标志或其他具有识别意义的符号，请简要提及它们的内容和含义（如公告中的公司名称、Twitter用户名等）。
6. 无需推测图片背后的事件，仅需描述图片中的可见信息和情境。

请直接返回一段 40 字以内的描述，无需添加其他说明。
""",
                ),
                "IN_CONTEXT_PLACEHOLDER",
                ("user", "<image> 请直接返回一段 40 字以内的描述，无需添加其他说明。"),
            ],
            [
                (
                    "system",
                    """你是一个专业的图像分析和新闻理解助手。请结合图片新闻上下文，补充原始图片描述中与新闻内容相关的关键信息(如新闻人物、事件等），注意不要补充图片本身不包含的信息。\
描述中应明确图片的类型（如海报、照片、图表、肖像等）。

请直接返回一段 40 字以内的图片描述, 无需添加其他说明。""",
                ),
                "IN_CONTEXT_PLACEHOLDER",
                (
                    "user",
                    """[图片新闻上下文的开始]
{text}
[图片新闻上下文的结束]

[原始图片描述的开始]
{response_0}
[原始图片描述的结束]

请根据新闻上下文补全图片中的人物事件信息与图片类型。请直接返回一段 40 字以内的图片描述, 无需添加其他说明。""",
                ),
            ],
        ]
