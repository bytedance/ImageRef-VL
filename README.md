# ImageRef-VL
The code of our work "ImageRef-VL: Enabling Contextual Image Referencing in Vision-Language Models".

## Overview
Vision-Language Models (VLMs) are increasingly used in Retrieval-Augmented Generation (RAG) systems for multimodal conversations. While these models can reference textual sources, they often fail to incorporate contextually relevant images into their responses. This repository addresses this gap by introducing ImageRef-VL, the ability to reference images based on conversation context, along with a comprehensive evaluation framework featuring a curated dataset and metrics. We include implementation of ImageRef-VL in this repository.

## Requirements

Install the following python requirements.
```bash
# Basic Python utilities
pybind11
einops
einops-exts
termcolor
scipy
imageio
opencv-python-headless
pycocoevalcap
yacs
tensorboardX

# Specific versions for compatibility
tqdm==4.66.5
torch==2.4.0
torchvision==0.19.0
torchaudio==2.4.0
bitsandbytes==0.43.3
peft==0.12.0
rich==13.7.1
tiktoken==0.6.0
transformers_stream_generator==0.0.5
pandas==2.2.2
auto_gptq==0.7.1
optimum==1.21.4
vllm==0.6.1.post2
httpx==0.23.3
jsonlines==4.0.0
openai==1.40.6
python-dotenv==1.0.1
datasets==2.21.0
deepspeed==0.14.4
protobuf==3.20.3
accelerate==0.33.0
sentencepiece==0.2.0
scikit-learn==1.5.1
evaluate==0.4.2
timm==1.0.8
byted-wandb
qwen_vl_utils==0.0.4
langchain==0.2.16
langchain-community==0.2.16
flash-attn==2.6.1 --no-build-isolation

# Git-based installations
git+https://github.com/huggingface/transformers.git
git+https://github.com/deepseek-ai/DeepSeek-VL.git
```

We recommend conducting experiments on nodes equipped with 16 A100 GPUs for training the models.

## How to use

### Construct training dataset
1. Put your instruction dataset (with crawled documents) under `data/path`

2. Generate textural response
```bash
python test/baselines/three_stage_1.py --annotation_file training/annotation/file --root_image_path data/image/path --model_name model/name --model_path model/checkpoint/path --output_file response/file/path --load_type [api|vllm|hg]
```

3. Generate image captions
```bash
python inference/caption/run.py --root_model_path root/model/path --model_name model_name --method_name caption-in-context --data_dir data/dir --output_dir output/dir --event_file event/file/path --example_file example/file/path --output_caption_file output/caption/file/path
--label_file label/file/path --batch_news_size 100 --num_ic_examples 2 --load_type [api|vllm|hg]
```

4. Generate responses with contextual image references
```bash
python inference/insert/run.py --model_name model_name --method_name i4p-ic --data_dir data/dir --output_dir output/dir --root_model_path root/model/path --caption_file output/caption/file/path --label_file [label/file/path|None] --output_file output/response/file/path --event_file event/file/path  --example_file example/file/path --load_type [api|vllm|hg] --batch_event_size 200 --num_api_threads 20 --num_ic_examples 2 --news_w_all_imgs --unavailable_news_file unavailable/news/file
```

### Train
Running the following commands to train ImageRef-VL models.

```bash
# train 8B model
cd train
bash shell/imagerefvl/2nd_finetune/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_full.sh shell/data/imageref_finetune_1:4_cap.json output/8b_model true

# train 26B model
cd train
bash shell/imagerefvl/2nd_finetune/internvl2_26b_internlm2_20b_dynamic_res_2nd_finetune_full.sh 
shell/data/imageref_finetune_1:1_cap.json output/26b_model true
```

### Evaluation
#### Text Evaluation
1. Generation
```bash
python test/imageref/decoding.py --raw_model_path internVL/raw/model/path --model_path your/trained/model --test_annotation_file test/annotation/file/path --output_file response/file/path
```

2. LLM-as-judge
```bash 
# Generate llm-as-judge evaluation
python test/imageref/llm_as_judge.py --eval_model eval_llm_name --annotation_file test/annotation/file/path --response_file response/file/path --wcap --output_file evaluation/file/path

# Calculate score
python test/imageref/judge_rslt.py --rslt_file evaluation/file/path
```

#### Image Position Evaluation
1. Guided decoding
```bash
python test/imageref/guided_decoding.py --model_name internvl2 --raw_model_path internVL/raw/model/path --model_path your/trained/model --test_annotation_file test/annotation/file/path --output_file response/file/path --wcap
```

2. Evaluation
```bash
python test/imageref/ipp_eval.py --output_file response/file/path --label_file label/file/path --event_file event/file/path --annotation_file test/annotation/file/path --model_type guided
```

## License
This project is licensed under the license found in the [LICENSE](LICENSE) file in the root directory of this source tree. Portions of the source code are based on the [InternVL](https://github.com/OpenGVLab/InternVL) project.
