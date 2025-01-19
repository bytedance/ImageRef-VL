# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Dict, Any, Optional
from dataclasses import dataclass

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

from inference.model import BaseModel

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)


def _convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {"role": "function", "content": message.content}
    else:
        raise ValueError(f"Got unknown type {message}")
    return message_dict


def convert_messages(messages: list[BaseMessage]) -> list[dict]:
    return [_convert_message_to_dict(m) for m in messages]


def replace_placeholder(chat_message, few_shot_prompt):
    updated_chat_message = chat_message.copy()

    for index, item in enumerate(updated_chat_message):
        if item == "IN_CONTEXT_PLACEHOLDER":
            updated_chat_message[index] = few_shot_prompt

    return updated_chat_message


def filter_template_input(template_input_variables, inputs):
    filtered_inputs = {k: v for k, v in inputs.items() if k in template_input_variables}
    return filtered_inputs


@dataclass
class BaseGenerator:
    llm: BaseModel
    is_in_context: Optional[bool] = False

    def __post_init__(self):
        self._init_instruct()

    def _init_instruct(self):
        """Add all instructions here"""
        raise NotImplementedError

    def format_prompt_wo_ic(
        self, messages: List[Any], inputs: Dict[str, Any]
    ) -> Union[Dict, str]:
        """Given inputs and instructs, output prompt for LLMs"""
        template = ChatPromptTemplate.from_messages(messages)
        filtered_inputs = filter_template_input(template.input_variables, inputs)

        chat_prompt = convert_messages(template.invoke(input=filtered_inputs).messages)

        if "image" in inputs:
            prompt = {
                "prompt": chat_prompt,
                "multi_modal_data": {"image": [inputs["image"]]},
            }
        else:
            prompt = {"prompt": chat_prompt}
        return prompt

    def format_prompt_w_ic(
        self, messages: List[Any], example_messages: List[Any], inputs: Dict[str, Any]
    ) -> Union[Dict, str]:
        example_inputs = inputs["example_inputs"]
        chat_inputs = inputs["chat_inputs"]

        example_prompt = ChatPromptTemplate.from_messages(example_messages)

        filtered_example_inputs = [
            filter_template_input(example_prompt.input_variables, example_input)
            for example_input in example_inputs
        ]
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=filtered_example_inputs,
        )
        messages = replace_placeholder(messages, few_shot_prompt)
        template = ChatPromptTemplate.from_messages(messages)

        filtered_chat_inputs = filter_template_input(
            template.input_variables, chat_inputs
        )
        chat_prompt = convert_messages(
            template.invoke(input=filtered_chat_inputs).messages
        )
        if "image" in chat_inputs:
            prompt = {
                "prompt": chat_prompt,
                "multi_modal_data": {
                    "image": [i["image"] for i in example_inputs]
                    + [chat_inputs["image"]]
                },
            }
        else:
            prompt = {"prompt": chat_prompt}
        return prompt

    def __call__(
        self,
        inputs: Union[List[Dict[str, Any]], Dict[str, Any]],
        generation_config: Dict[str, Any],
    ):
        if self.is_in_context:
            return self._generate_w_ic(inputs, generation_config)
        else:
            return self._generate_wo_ic(inputs, generation_config)

    def _generate_wo_ic(
        self,
        inputs: Union[List[Dict[str, Any]], Dict[str, Any]],
        generation_config: Dict[str, Any],
    ):
        for index, messages in enumerate(self.messages):
            if isinstance(inputs, list):
                prompts = [self.format_prompt_wo_ic(messages, i) for i in inputs]
            else:
                prompts = [self.format_prompt_wo_ic(messages, inputs)]
                
            responses = self.llm.generate(prompts, **generation_config)

            if isinstance(inputs, list):
                for i in range(len(inputs)):
                    inputs[i][f"response_{index}"] = responses[i]
            else:
                inputs[f"response_{index}"] = responses[0]

        return responses

    def _generate_w_ic(
        self,
        inputs: Union[List[Dict[str, Any]], Dict[str, Any]],
        generation_config: Dict[str, Any],
    ):
        for index, (messages, example_messages) in enumerate(
            zip(self.messages, self.example_messages)
        ):
            if isinstance(inputs, list):
                prompts = [
                    self.format_prompt_w_ic(messages, example_messages, i)
                    for i in inputs
                ]
            else:
                prompts = [self.format_prompt_w_ic(messages, example_messages, inputs)]

            responses = self.llm.generate(prompts, **generation_config)

            if isinstance(inputs, list):
                for i in range(len(inputs)):
                    inputs[i]["chat_inputs"][f"response_{index}"] = responses[i]
            else:
                inputs["chat_inputs"][f"response_{index}"] = responses[0]

        return responses
