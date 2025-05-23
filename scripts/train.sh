#!/bin/bash

export WANDB_PROJECT="CoF-4B"
export MASTER_ADDR=""
echo "MASTER_ADDR=$MASTER_ADDR"
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)

base_model=OpenGVLab/InternVL2_5-4B
output_dir=checkpoints/cl/CoF-InternVL2_5-4B
NNODES=$SLURM_NNODES
GPUS=$SLURM_GPUS_ON_NODE
BATCH_SIZE=2
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACC=1

export LAUNCHER=pytorch

if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir"
fi


torchrun  \
  --nnodes=${NNODES} \
  --node_rank=$SLURM_NODEID \
  --master_addr=$MASTER_ADDR \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  -m internvl.train.internvl_chat_finetune \
  --model_name_or_path $base_model \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir $output_dir \
  --meta_path "cof_meta_file.jsonl" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 1 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 1 \
  --learning_rate 2e-6 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 8192 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  2>&1 | tee -a "$output_dir/training_log.txt"
"
