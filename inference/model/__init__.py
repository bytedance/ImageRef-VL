# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from .base import BaseModel


def get_model_cls(model_name: str) -> BaseModel:
    if "internvl2" in model_name.lower():
        from .model import InternVL2
        return InternVL2
    elif "qwen2-vl" in model_name.lower() or "qwen-vl" in model_name.lower():
        from .model import Qwen2VL
        return Qwen2VL
    elif "deepseek-vl" in model_name.lower():
        from .model import DeepseekVL
        return DeepseekVL
    elif "phi" in model_name.lower():
        from .model import PhiModel
        return PhiModel
    elif "gpt" in model_name.lower():
        from .model import OpenAIModel
        return OpenAIModel
    elif "claude" in model_name.lower():
        from .model import ClaudeModel
        return ClaudeModel
    elif "gemini" in model_name.lower():
        from .model import GeminiModel
        return GeminiModel
