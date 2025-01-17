set -x

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: $0 <meta_path> <output_dir> [resume]"
  exit 1
fi

META_PATH="$1"
OUTPUT_DIR="$2"


GPUS=${GPUS:-16}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch


if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

MASTER_ADDR=${WORKER_0_HOST}
MASTER_PORT=${WORKER_0_PORT}
NPROC_PER_NODE=${WORKER_GPU} 
NNODES=${WORKER_NUM}
NODE_RANK=${ID}

torchrun \
  --master_port=${MASTER_PORT} \
  --master_addr=${MASTER_ADDR} \
  --nproc_per_node=${NPROC_PER_NODE} \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "models/InternVL2-26B" \
  --conv_style "internlm2-chat" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path ${META_PATH} \
  $( [ "$#" -eq 3 ] && echo "--overwrite_output_dir True" ) \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.4 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --max_steps 1000 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 50 \
  --learning_rate 2e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 16384 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage3_config.json" \
  --report_to "wandb" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

