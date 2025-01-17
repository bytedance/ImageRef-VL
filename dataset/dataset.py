from typing import List, Union, Any, Optional, Set

import jsonlines
from tqdm import tqdm
from pathlib import Path
import json

from torch.utils.data import Dataset
from dataset.element import Event


class BaseImageRefDataset(Dataset):
    def __init__(
        self,
        event_file: Union[Path, str],
        news_dir: Union[Path, str],
        image_dir: Union[Path, str],
        verbose: bool = False,
        volc_info_file: Optional[Union[Path, str]] = None,
        image_caption_file: Optional[Union[Path, str]] = None,
        include_event_ids: Optional[Union[set, list, None]] = None,
        exclude_event_ids: Optional[Union[set, list, None]] = None,
    ):

        self.news_dir = news_dir
        self.image_dir = image_dir
        self.volc_info_file = volc_info_file
        self.image_caption_file = image_caption_file
        self.verbose = verbose
        self.include_event_ids = include_event_ids
        self.exclude_event_ids = exclude_event_ids

        self.events = self.load_event_file(event_file)
        if self.image_caption_file:
            self.events = self.load_caption_info(self.image_caption_file)

        if self.volc_info_file:
            self.events = self.load_volc_info(self.volc_info_file)

    def load_event_file(self, event_file: Union[str, Path]):
        events = []
        with jsonlines.open(event_file, "r") as reader:
            for event_info in tqdm(reader, desc="Loading event file", disable=not self.verbose):
                if (
                    self.include_event_ids
                    and event_info["paasEventID"] not in self.include_event_ids
                ):
                    continue

                if (
                    self.exclude_event_ids
                    and event_info["paasEventID"] in self.exclude_event_ids
                ):
                    continue

                event = Event(
                    event_info=event_info,
                    root_news_dir=self.news_dir,
                    root_image_dir=self.image_dir,
                )
                events.append(event)
        return events

    def load_caption_info(self, caption_file: Union[str, Path]) -> List[Event]:
        caption_info = {}
        with jsonlines.open(caption_file, "r") as reader:
            for obj in tqdm(
                reader, desc="Loading image caption info", disable=not self.verbose
            ):
                caption_info.update(obj["caption"])

        for event in tqdm(
            self.events, desc="Setting volc and caption info", disable=not self.verbose
        ):
            for source in event.sources:
                for image_index, image in source.images.items():
                    caption = caption_info[image.relative_path]
                    if caption is not None:
                        image.set_caption(caption)

        return self.events

    def load_volc_info(self, volc_info_file: Union[str, Path]) -> List[Event]:
        volc_service_info = {}
        with jsonlines.open(volc_info_file, "r") as reader:
            for obj in tqdm(
                reader, desc="Loading volc service info", disable=not self.verbose
            ):
                nid = obj.get("nid")
                image_index = obj.get("image_index")

                image_info = {}
                ocr_data = obj.get("OCRNormal", {}).get("data", {})
                image_info["line_texts"] = ocr_data.get("line_texts", [])

                image_score_data = obj.get("ImageScore", {}).get("data", {})
                image_info["score_aesthetics_v2"] = image_score_data.get(
                    "score_aesthetics_v2", 0
                )
                image_info["score_face"] = image_score_data.get("score_face", 0)
                volc_service_info[(nid, image_index)] = image_info

        for event in tqdm(
            self.events, desc="Setting volc and caption info", disable=not self.verbose
        ):
            for source in event.sources:
                for image_index, image in source.images.items():
                    volc_info = volc_service_info.get((source.news_id, image_index))
                    if volc_info is not None:
                        image.set_volc_info(volc_info)

        return self.events


class ImageRefEventDataset(BaseImageRefDataset):
    def __init__(
        self,
        event_file: Union[Path, str],
        news_dir: Union[Path, str],
        image_dir: Union[Path, str],
        verbose: bool = False,
        volc_info_file: Optional[Union[Path, str]] = None,
        image_caption_file: Optional[Union[Path, str]] = None,
        include_event_ids: Optional[Union[set, list, None]] = None,
        exclude_event_ids: Optional[Union[set, list, None]] = None,
    ):
        super().__init__(
            event_file=event_file,
            news_dir=news_dir,
            image_dir=image_dir,
            verbose=verbose,
            volc_info_file=volc_info_file,
            image_caption_file=image_caption_file,
            include_event_ids=include_event_ids,
            exclude_event_ids=exclude_event_ids,
        )

    def __len__(self):
        return len(self.events)

    def __getitem__(self, index) -> Event:
        return self.events[index]


class ImageRefNewsDataset(BaseImageRefDataset):
    def __init__(
        self,
        event_file: Union[Path, str],
        news_dir: Union[Path, str],
        image_dir: Union[Path, str],
        verbose: bool = False,
        volc_info_file: Optional[Union[Path, str]] = None,
        image_caption_file: Optional[Union[Path, str]] = None,
        include_event_ids: Optional[Union[set, list, None]] = None,
        exclude_event_ids: Optional[Union[set, list, None]] = None,
        exclude_news_ids: Optional[Union[Set[str], List[str], None]] = None,
    ):
        super().__init__(
            event_file=event_file,
            news_dir=news_dir,
            image_dir=image_dir,
            verbose=verbose,
            volc_info_file=volc_info_file,
            image_caption_file=image_caption_file,
            include_event_ids=include_event_ids,
            exclude_event_ids=exclude_event_ids,
        )

        self.news = []
        self.news_ids = set()
        for event in self.events:
            for source in event.sources:
                if exclude_news_ids is None or source.news_id not in exclude_news_ids:
                    if source.news_id not in self.news_ids:
                        self.news.append(source)
                        self.news_ids.add(source.news_id)

        if verbose:
            print(f"Total news after filtering: {len(self.news)}")


    def __len__(self) -> int:
        return len(self.news)

    def __getitem__(self, index) -> Any:
        return self.news[index]


class ImageRefImageDataset(BaseImageRefDataset):
    def __init__(
        self,
        event_file: Union[Path, str],
        news_dir: Union[Path, str],
        image_dir: Union[Path, str],
        verbose: bool = False,
        volc_info_file: Optional[Union[Path, str]] = None,
        image_caption_file: Optional[Union[Path, str]] = None,
        include_event_ids: Optional[Union[set, list, None]] = None,
        exclude_event_ids: Optional[Union[set, list, None]] = None,
    ):
        super().__init__(
            event_file=event_file,
            news_dir=news_dir,
            image_dir=image_dir,
            verbose=verbose,
            volc_info_file=volc_info_file,
            image_caption_file=image_caption_file,
            include_event_ids=include_event_ids,
            exclude_event_ids=exclude_event_ids,
        )

        self.images = []
        for event in self.events:
            for source in event.sources:
                for image in source.images.values():
                    self.images.append(image)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index) -> Any:
        return self.images[index]
