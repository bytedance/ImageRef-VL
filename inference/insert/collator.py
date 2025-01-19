# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import List, Any
from collections import defaultdict

from dataset.element import Event


class ImageRefIPPCollator:
    def construct_input(self, event: Event):
        image_captions, image_paths = [], []
        index = 1
        for source in event.sources:
            for image_index in source.images:
                image = source.images[image_index]
                image_captions.append(f"图像{index}: ![{image.caption}](IMAGE_{index})")
                image_paths.append(image.relative_path)
                index += 1

        text, num_positions = Event.preprocess_text(
            event.title, event.summary, placeholder_policy="bc"
        )
        inputs = {
            "text": text,
            "image_caption_str": "\n".join(image_captions),
            "num_positions": num_positions,
        }
        return inputs, image_paths

    def __call__(self, batch: List[Event]):
        batch_inputs, batch_event_ids, batch_image_paths = ([], [], [])

        for event in batch:
            sample_input, image_paths = self.construct_input(event)
            batch_inputs.append(sample_input)
            batch_event_ids.append(event.event_id)
            batch_image_paths.append(image_paths)

        return batch_inputs, batch_event_ids, batch_image_paths


class ICImageRefIPPCollator(ImageRefIPPCollator):
    def __init__(self, in_context_examples: List[Any], num_examples_per_sample):
        # TODO: Randomly or propose methods to select in-context examples (currently select top examples)
        self.in_context_examples = in_context_examples[:num_examples_per_sample]

    def construct_input(self, event: Event):
        chat_inptus, image_paths = super().construct_input(event)

        inputs = {
            "chat_inputs": chat_inptus,
            "example_inputs": self.in_context_examples,
        }
        return inputs, image_paths
