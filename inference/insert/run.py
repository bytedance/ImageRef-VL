# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import json
import argparse
import pandas as pd
import logging
from ast import literal_eval

from dotenv import load_dotenv
from pathlib import Path
import jsonlines

from torch.utils.data import DataLoader

from dataset import ImageRefEventDataset
from dataset.element import Event
from inference.model import get_model_cls
from method import get_ipp_method
from collator import ImageRefIPPCollator, ICImageRefIPPCollator

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="gpt-4o-2024-05-13", type=str)
    parser.add_argument("--method_name", default="i4p", type=str)
    parser.add_argument(
        "--data_dir",
        default="",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default="",
        type=str,
    )
    parser.add_argument(
        "--root_model_path", default="", type=str
    )
    parser.add_argument(
        "--caption_file",
        default="",
        type=str,
    )
    parser.add_argument("--label_file", default=None, type=str)
    parser.add_argument("--output_file", default=None, type=str)
    parser.add_argument("--event_file", default="all_news_list.jsonl", type=str)
    parser.add_argument(
        "--example_file", default="examples/i4p_examples.json", type=str
    )
    parser.add_argument(
        "--load_type", default="api", type=str, choices=["api", "vllm", "hg"]
    )
    parser.add_argument("--batch_event_size", default=1, type=int)
    parser.add_argument("--num_api_threads", default=20, type=int)
    parser.add_argument("--num_ic_examples", default=3, type=int)
    parser.add_argument("--news_w_all_imgs", default=False, action="store_true")
    parser.add_argument("--unavailable_news_file", default="unavailable_news.csv", type=str)
    args = parser.parse_args()
    return args


def load_ic_examples(event_file, example_file, image_dir, news_dir):
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


if __name__ == "__main__":
    args = parse_args()

    logger.info("Starting caption generation with the following parameters:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")

    model_name = args.model_name
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    caption_file = Path(args.caption_file)
    event_file = data_dir / args.event_file
    news_dir = data_dir / "news"
    image_dir = data_dir / "image"

    if args.label_file:
        label_file = data_dir / args.label_file
        label_df = pd.read_csv(
            label_file,
            converters={"image_map": literal_eval, "result": literal_eval},
            dtype={"paasEventID": str},
        )
        include_event_ids = set(label_df["paasEventID"])
    else:
        include_event_ids = None

    if args.news_w_all_imgs:
        # some news expires and cannot be crawled anymore
        unavailable_news_df = pd.read_csv(data_dir / args.unavailable_news_file, header=None, names=["nid", "url"], dtype=str)
        unavailable_news = set(unavailable_news_df['nid'].values)

        all_events_full_news_count = 0
        valid_event_ids = set()

        with jsonlines.open(data_dir / "all_news_list.jsonl", "r") as reader:
            for obj in reader:
                nids = [i["newsID"] for i in obj["sources"]]
                all_crawlable_nids = [nid for nid in nids if nid not in unavailable_news ]

                if all((news_dir / f"{nid}.json").exists() for nid in all_crawlable_nids):
                    all_events_full_news_count += 1
                    valid_event_ids.add(str(obj["paasEventID"]))

        include_event_ids = (
            valid_event_ids.intersection(include_event_ids)
            if include_event_ids
            else valid_event_ids
        )
        logger.info(
            f"Total events with all news existing: {all_events_full_news_count}"
        )

    exclude_event_ids = set()
    if args.output_file:
        output_file_path = output_dir / Path(args.output_file)
        if output_file_path.exists():
            with jsonlines.open(output_file_path, "r") as reader:
                for obj in reader:
                    exclude_event_ids.add(str(obj["event_id"]))

    ds = ImageRefEventDataset(
        event_file=event_file,
        news_dir=news_dir,
        image_dir=image_dir,
        image_caption_file=caption_file,
        verbose=True,
        include_event_ids=include_event_ids,
        exclude_event_ids=exclude_event_ids,
    )

    logger.info(f"Dataset loaded with {len(ds)} items")

    model_path = Path(args.root_model_path) / args.model_name
    logger.info(f"Loading model from {model_path}")

    model_cls = get_model_cls(args.model_name)
    model = model_cls(
        model_name=args.model_name, model_path=model_path, load_type=args.load_type
    )

    method_cls = get_ipp_method(args.method_name)
    method = method_cls(llm=model)
    logger.info(f"Using IPP prediction method: {args.method_name}")

    if method.is_in_context:
        example_file = data_dir / args.example_file
        in_context_examples = load_ic_examples(
            event_file, example_file, image_dir, news_dir
        )

        collate_fn = ICImageRefIPPCollator(in_context_examples, args.num_ic_examples)
    else:
        collate_fn = ImageRefIPPCollator()

    dl = DataLoader(
        ds,
        batch_size=args.batch_event_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    for batch_inputs, batch_event_ids, batch_image_paths in dl:
        batch_response = method(batch_inputs, {"temperature": 0, "max_tokens": 4096, "response_format": {"type": "json_object"}})

        if batch_response is not None:
            write_objs = []
            for response, event_id, image_paths in zip(
                batch_response, batch_event_ids, batch_image_paths
            ):
                write_objs.append(
                    {
                        "event_id": event_id,
                        "image_paths": image_paths,
                        "result": response,
                    }
                )

            with jsonlines.open(output_dir / Path(args.output_file), "a") as writer:
                writer.write_all(write_objs)

    model.shutdown()
