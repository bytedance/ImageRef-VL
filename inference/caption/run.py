from typing import List, Any, Set
import argparse
from pathlib import Path
import jsonlines
import json
import logging
import pandas as pd
from ast import literal_eval
from dotenv import load_dotenv

from torch.utils.data import DataLoader

from dataset import ImageRefNewsDataset
from dataset.element import News
from inference.caption.method import get_caption_generation_method
from inference.caption.collator import ImageRefCaptionCollator, ICImageRefCaptionCollator
from inference.model import get_model_cls

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_model_path", default="", type=str
    )
    parser.add_argument("--model_name", default="InternVL2-26B", type=str)
    parser.add_argument("--method_name", default="caption-v1", type=str)
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
        "--event_file", default="news_list_20241024.jsonl", type=str, help="Event file."
    )
    parser.add_argument(
        "--example_file",
        default="",
        type=str,
        help="Example file.",
    )
    parser.add_argument("--output_caption_file", default="caption.jsonl", type=str)
    parser.add_argument("--label_file", default=None, type=str)
    parser.add_argument("--batch_news_size", default=100, type=int)
    parser.add_argument("--num_ic_examples", default=2, type=int)
    parser.add_argument("--load_type", default="vllm", type=str)
    args = parser.parse_args()
    return args


def load_ic_examples(example_file, image_dir, news_dir):
    with open(example_file, "r") as f:
        examples = json.load(f)

    ic_examples = []
    for example in examples:
        image_path = example["image_path"]
        image_index = int(image_path.split("/")[1].split(".")[0])

        news_id = image_path.split("/")[0]
        news_path = news_dir / f"{news_id}.json"
        news = News(news_id=news_id, news_path=news_path, root_image_dir=image_dir)

        example_text = News.preprocess_text(
            news.news_title, news.news_content, {image_index: "<image>"}
        )
        ic_example = {
            "example_text": example_text,
            "example_response_0": example["description"],
            "example_response": example["caption"],
            "image": news.images[image_index].pil_image,
        }
        ic_examples.append(ic_example)

    return ic_examples


def get_processed_news_ids(output_file: Path) -> Set[str]:
    processed_ids = set()
    if output_file.exists():
        with jsonlines.open(output_file) as reader:
            for obj in reader:
                processed_ids.add(obj["nid"])
    return processed_ids


if __name__ == "__main__":
    args = parse_args()

    logger.info("Starting caption generation with the following parameters:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")

    data_dir = Path(args.data_dir)

    output_dir = Path(args.output_dir) / f"caption-{args.model_name}"
    output_dir.mkdir(exist_ok=True, parents=True)

    output_file = output_dir / args.output_caption_file
    processed_news_ids = get_processed_news_ids(output_file)
    logger.info(f"Found {len(processed_news_ids)} already processed news IDs")

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

    ds = ImageRefNewsDataset(
        event_file=event_file,
        news_dir=news_dir,
        image_dir=image_dir,
        verbose=True,
        include_event_ids=include_event_ids,
        exclude_news_ids=processed_news_ids,
    )

    logger.info(
        f"Dataset loaded with {len(ds)} items (excluding already processed news)"
    )

    model_path = Path(args.root_model_path) / args.model_name

    logger.info(f"Loading model from {model_path}")
    model_cls = get_model_cls(args.model_name)
    model = model_cls(
        model_name=args.model_name, model_path=model_path, load_type=args.load_type
    )
    logger.info("Model loaded successfully")

    method_cls = get_caption_generation_method(args.method_name)
    method = method_cls(llm=model)
    logger.info(f"Using caption generation method: {args.method_name}")

    if method.is_in_context:
        example_file = data_dir / args.example_file
        in_context_examples = load_ic_examples(example_file, image_dir, news_dir)

        collate_fn = ICImageRefCaptionCollator(
            tokenizer=model.tokenizer if args.load_type != "api" else None,
            in_context_examples=in_context_examples,
            num_examples_per_sample=args.num_ic_examples
        )
    else:
        collate_fn = ImageRefCaptionCollator(tokenizer=model.tokenizer)

    dl = DataLoader(
        ds,
        batch_size=args.batch_news_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    for batch_inputs, batch_image_path, nid2index in dl:
        batch_response = method(batch_inputs, {"temperature": 0, "max_tokens": 1024})

        if batch_response is not None:
            write_objs = []
            for nid in nid2index:
                nid_image_rslt = {}
                for index in nid2index[nid]:
                    image_path = batch_image_path[index]
                    response = batch_response[index]
                    nid_image_rslt[image_path] = response

                write_objs.append({"nid": nid, "caption": nid_image_rslt})

            with jsonlines.open(output_dir / args.output_caption_file, "a") as writer:
                writer.write_all(write_objs)

    model.shutdown()
