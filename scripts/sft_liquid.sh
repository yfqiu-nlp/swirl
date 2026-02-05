#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=100:00:00
#SBATCH --gpus-per-node=8
#SBATCH --output=logs/slurm-%j.out
#SBATCH --exclusive
# #SBATCH --account=YOUR_ACCOUNT
# #SBATCH --qos=YOUR_QOS

# Print job information
echo "=================================="
echo "SLURM Job Information"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total tasks: $SLURM_NTASKS"
echo "Node list: $SLURM_NODELIST"
echo "=================================="

# Load environment
# Activate your environment before running, e.g.:
# source /path/to/conda.sh && conda activate swirl
mkdir -p logs

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT:$REPO_ROOT/src"
# Optional: set Hugging Face cache locations
# export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
# export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_TREE_THRESHOLD=0
STORAGE_PATH="${STORAGE_PATH:-$REPO_ROOT}"

# # # # # Multi-node Training with FSDP for Pico-banana
EXPERIMENT_NAME="Liquid_V1_7B_pb-multiturn:1.0,pb-idp:1.0,pb-fdp:1.0,ag-fdp:1.0,something-fdp:1.0,kubric-fdp:1.0,magicbrush-fdp:1.0,ag-idp:1.0,something-idp:1.0,kubric-idp:1.0,magicbrush-idp:1.0_bs128_lr2e5_epoch5"
export WANDB_NAME=$EXPERIMENT_NAME

srun --cpu-bind=none \
    python src/train/sft.py \
    --model_size 7B \
    --model_name_or_path ${STORAGE_PATH}/models/Liquid_V1_7B \
    --torch_dtype bfloat16 \
    --use_flash_attention_2 True \
    --data_mixture "pico-banana-multiturn:1.0,pico-banana-idp:1.0,pico-banana-fdp:1.0,aurora-ag-fdp:1.0,aurora-something-fdp:1.0,aurora-kubric-fdp:1.0,aurora-magicbrush-fdp:1.0,aurora-ag-idp:1.0,aurora-something-idp:1.0,aurora-kubric-idp:1.0,aurora-magicbrush-idp:1.0" \
    --dataset_base_path "$STORAGE_PATH/datasets" \
    --validation_split 0.005 \
    --max_seq_length 10265 \
    --output_dir $STORAGE_PATH/rlwm-checkpoints/${EXPERIMENT_NAME} \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --logging_steps 10 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 5 \
    --eval_strategy steps \
    --save_strategy steps \
    --bf16 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --seed 42 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config src/train/fsdp_config.json \
