from typing import List, Any
from collections import defaultdict

from dataset.element import News, NewsImage

MAX_NUMS_TOKENS = 4096

class ImageRefCaptionCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def trucate_text(self, text, max_tokens):
        lines = text.splitlines()
        selected_lines = []
        token_count = 0
   
        for line in lines:
            encoded_line = self.tokenizer.encode(line, add_special_tokens=False)
            if token_count + len(encoded_line) > max_tokens:
                break
            selected_lines.append(line)
            token_count += len(encoded_line)
        
        text = "\n".join(selected_lines)
        return text
    
    def trucate_image_text(self, text):
        image_index = text.find("<image>")

        if self.tokenizer:
            before_image = self.trucate_text(text[:image_index], MAX_NUMS_TOKENS)
            after_image = self.trucate_text(text[image_index + len("<image>"):], MAX_NUMS_TOKENS)

            return f"{before_image}\n\n<image>\n\n{after_image}"
        else:
            return text

    def construct_input(self, news: News, image_index: int, image: NewsImage):
        text = News.preprocess_text(
            news.news_title, news.news_content, {image_index: "<image>"}
        )
        text = self.trucate_image_text(text)
        inputs = {
            "text": text,
            "image": image.pil_image,
        }
        return inputs

    def __call__(self, batch: List[News]):
        batch_inputs, batch_image_path, nid2index = (
            [],
            [],
            defaultdict(list),
        )

        index = 0
        for news in batch:
            for image_index in news.images:
                image = news.images[image_index]
                batch_inputs.append(self.construct_input(news, image_index, image))
                batch_image_path.append(image.relative_path)

                nid2index[news.news_id].append(index)
                index += 1

        return batch_inputs, batch_image_path, nid2index


class ICImageRefCaptionCollator(ImageRefCaptionCollator):
    def __init__(self, tokenizer, in_context_examples: List[Any], num_examples_per_sample):
        # TODO: Randomly or propose methods to select in-context examples (currently select top examples)
        super().__init__(tokenizer)
        self.in_context_examples = in_context_examples[:num_examples_per_sample]

    def construct_input(self, news: News, image_index: int, image: NewsImage):
        chat_inputs = super().construct_input(news, image_index, image)
        return {"chat_inputs": chat_inputs, "example_inputs": self.in_context_examples}
