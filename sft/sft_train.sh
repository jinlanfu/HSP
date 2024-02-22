#!/bin/bash
#SBATCH --partition=llm
#SBATCH --job-name=sft
#SBATCH --quotatype=reserved
#SBATCH --output=sft.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8


head_node_ip=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo Node IP: $head_node_ip nodes_array: $SLURM_NODELIST
export LOGLEVEL=ERROR
export NCCL_DEBUG=ERROR   # INFO #TRACE
srun bash -c 'echo $SLURMD_NODENAME-$SLURM_JOB_GPUS'


model_llemma_7b="EleutherAI/llemma_7b"
epoch_output_dir_llemma_7b="./ckpt/with_hint_epochs_llemma_7b"

srun  \
torchrun --nnodes 1 --nproc_per_node 8 \
--rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29549 \
sft_gsm8k.py \
--deepspeed  "./deepspeed_config.json"  \
--model_name_or_path $model_llemma_7b \
--output_dir $epoch_output_dir_llemma_7b \
--model_max_length 1024 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 4 \
--num_train_epochs 5 \
--data_path "HSPMATH.jsonl" \
--need_hint True \
--bf16  \
--save_strategy epoch \
--learning_rate 2e-5 \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 50 \
--logging_dir "./logging_dir" \
--report_to="tensorboard" \
--gradient_checkpointing True \
--save_total_limit 10
