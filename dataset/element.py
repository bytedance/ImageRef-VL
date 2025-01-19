# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, List, Dict, Union, Any
import re
from dataclasses import dataclass
from pathlib import Path
import json
import os
import logging
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

@dataclass
class NewsImage:
    url: Optional[str] = None
    path: Optional[str] = None
    relative_path: Optional[str] = None
    caption: Optional[str] = None
    volc_info: Optional[Dict] = None

    @property
    def extension(self):
        extension = self.path.split(".")[-1]
        return extension

    @property
    def pil_image(self):
        try:
            img = Image.open(self.path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Fail to load {self.path}. {e}")
        return img

    @property
    def html_image(self):
        return f"""<img src="/file={self.path}" style="max-width: 300px;>"""

    def html_image(self, max_width: str = "300px"):
        return f"""<img src="/file={self.path}" style="max-width: {max_width};>"""

    @property
    def markdown_image(self):
        if self.caption:
            return f"![{self.caption}]({self.path})"
        else:
            return f"![]({self.path})"

    def set_caption(self, caption):
        self.caption = caption

    def set_volc_info(self, volc_info):
        self.volc_info = volc_info

    def filter_images_bysize(self):
        resolution_flag, aspect_ratio_flag = True, True
        try:
            with self.pil_image as img:
                width, height = img.size

                # Check if height or width is less than 100
                if height < 100 or width < 100:
                    resolution_flag = False

                # Check if aspect ratio is between 0.5 and 2
                aspect_ratio = height / width
                if 0.5 <= aspect_ratio or aspect_ratio >= 2:
                    aspect_ratio_flag = False

        except IOError:
            print(f"Error opening image file: {self.path}")
        except Exception as e:
            print(f"Error processing image {self.path}: {str(e)}")

        return resolution_flag, aspect_ratio_flag

    def filter_by_extension(self):
        extension_flag = True
        if self.extension.lower() == ".gif":
            extension_flag = False

        return extension_flag

    def filter_by_volc(
        self,
    ):
        if self.volc_info is None:
            logging.warn(f"No volc info for image {self.path}")
            return True, True

        aesthetic_flag, text_flag = True, True

        image_info = self.volc_info
        aesthetics_score = image_info.get("score_aesthetics_v2", 0)
        line_texts = image_info.get("line_texts", [])
        text_len = len("".join(line_texts))

        if aesthetics_score < 0.01:
            aesthetic_flag = False

        if text_len > 100:
            text_flag = False

        return aesthetic_flag, text_flag


@dataclass
class News:
    news_id: str
    news_path: str
    root_image_dir: str

    def __post_init__(self):
        with open(self.news_path, "r") as f:
            content = json.load(f)
            self.news_title = content["title"]
            self.news_content = content["article_text"]
            image_infos = content["image_info"]

        self.images = {}
        image_folder = Path(os.path.join(self.root_image_dir, self.news_id))
        if image_folder.exists() and image_folder.is_dir():
            for image_file in image_folder.iterdir():
                if image_file.suffix.lower() not in ['.mpo']:
                    image_index = int(image_file.name.split(".")[0])

                    try:
                        news_image = NewsImage(
                            url=image_infos[image_index][0],
                            path=str(image_file),
                            relative_path=f"{self.news_id}/{image_file.name}",
                        )
                    except:
                        import pdb; pdb.set_trace()

                    self.images[image_index] = news_image

        self._pure_content = None
        self._content_w_placeholder = None

    @classmethod
    def preprocess_text(
        cls, title, content: str, image_replacements: Dict[int, str] = None
    ):
        lines = content.strip().split("\n")
        output_lines = [f"# {title}"] if title else []
        for line in lines:

            def replace_image(match):
                image_number = int(match.group(1))
                if image_replacements:
                    return image_replacements.get(image_number, "")
                else:
                    return ""

            processed_line = re.sub(r"\[Image (\d+)\]", replace_image, line)
            output_lines.append(processed_line)

        processed_text = "\n".join(output_lines)
        return processed_text

    @property
    def pure_content(self):
        if self._pure_content is None:
            self._pure_content = News.preprocess_text(
                self.news_title, self.news_content, image_replacements=None
            )
        return self._pure_content


@dataclass
class Event:
    event_info: Dict
    root_news_dir: str = os.environ.get("DOUBAO_NEWS_DIR", "")
    root_image_dir: str = os.environ.get("DOUBAO_IMAGE_DIR", "")

    def __post_init__(self):
        self.event_id = self.event_info["paasEventID"]
        self.title = self.event_info["title"]
        self.summary = self.event_info["summary"]
        self.category = json.loads(self.event_info["attrs"])["category"]

        self.sources = []
        for source in self.event_info["sources"]:
            news_id = source["newsID"]

            if os.path.exists(os.path.join(self.root_news_dir, f"{news_id}.json")):
                # Only add exists news to sources
                self.sources.append(
                    News(
                        news_id=news_id,
                        news_path=os.path.join(self.root_news_dir, f"{news_id}.json"),
                        root_image_dir=self.root_image_dir,
                    )
                )

    @classmethod
    def preprocess_text(
        cls, title: str, text: str, placeholder_policy: Optional[str] = None
    ):
        """Return the processed content of ImageRef news event.

        Args:
        - title (str): The title of the news summarization.
        - text (str): The content of the news summarization.
        - placeholder_policy (str): The policy used for inserting placeholders.
            "bc" means inserting placeholders before content,
            "ac" means inserting placeholders after the first header.
            Otherwise, no placeholders will be inserted.
        """

        lines = [f"# {title}"] + [
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
