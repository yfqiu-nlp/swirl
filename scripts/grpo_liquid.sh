#!/bin/bash
#SBATCH --job-name=liquid-grpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=8
#SBATCH --output=logs/slurm-%j.out
# #SBATCH --account=YOUR_ACCOUNT
# #SBATCH --qos=YOUR_QOS

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT:$REPO_ROOT/src"
export WANDB_PROJECT="${WANDB_PROJECT:-liquid-grpo}"
# export WANDB_API_KEY="YOUR_WANDB_API_KEY"
# export WANDB_HOST="https://api.wandb.ai"

# ---------- Environment Setup ----------
# Activate your environment before running, e.g.:
# source /path/to/conda.sh && conda activate swirl
mkdir -p logs

# ---------- Multi-node Setup ----------
# Number of nodes and GPUs
NUM_NODES=$SLURM_JOB_NUM_NODES
NUM_GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
TOTAL_PROCESSES=$((NUM_NODES * NUM_GPUS_PER_NODE))

# SLURM-provided master and rank info
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
NODE_RANK=$SLURM_NODEID

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NODE_RANK=$NODE_RANK"
echo "NUM_NODES=$NUM_NODES"
echo "NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE"
echo "TOTAL_PROCESSES=$TOTAL_PROCESSES"

# ---------- Paths ----------
STORAGE_PATH="${STORAGE_PATH:-$REPO_ROOT}"
DATASET_ROOT="${DATASET_ROOT:-$STORAGE_PATH/datasets}"
CKPT_ROOT="${CKPT_ROOT:-$STORAGE_PATH/rlwm-checkpoints}"
CKPT_PATH="${CKPT_PATH:-$CKPT_ROOT/Liquid_V1_7B_pb-multiturn:1.0,pb-idp:1.0,pb-fdp:1.0,ag-fdp:1.0,something-fdp:1.0,kubric-fdp:1.0,magicbrush-fdp:1.0,ag-idp:1.0,something-idp:1.0,kubric-idp:1.0,magicbrush-idp:1.0_bs128_lr2e5_epoch5}"
DS_CONFIG="src/grpo/ds_config_zero2.json"

# # # # ---------- FDP Training ----------
RUN_NAME=Liquid_V1_7B-pico-aurora-multiturn-sft-grpo_fdp_kinetics+ucf+mit_roll8_lr1e6_warmup100_lrScheCosine_beta0.04_bsz32_temp0.7_TokenRewardClip10
accelerate launch \
    --num_machines $NUM_NODES \
    --num_processes $TOTAL_PROCESSES \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --use_deepspeed \
    --deepspeed_config_file $DS_CONFIG \
    --mixed_precision bf16 \
    src/grpo/train_grpo.py \
        --model_name_or_path ${CKPT_PATH} \
        --dataset_paths \
            ${DATASET_ROOT}/unsupervised-frames-ucf-kinetics-mit/tokenized_liquid_kinetics_grpo \
            ${DATASET_ROOT}/unsupervised-frames-ucf-kinetics-mit/tokenized_liquid_mit_grpo \
            ${DATASET_ROOT}/unsupervised-frames-ucf-kinetics-mit/tokenized_liquid_ucf_grpo \
        --output_dir ${CKPT_ROOT}/${RUN_NAME} \
        --learning_rate 1e-6 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --num_generations 8 \
        --chunk_size 2 \
        --gradient_checkpointing True \
        --bf16 True \
        --logging_steps 1 \
        --save_steps 100 \
        --report_to wandb \
        --run_name ${RUN_NAME} \
        --warmup_steps 100 \
        --lr_scheduler_type "cosine" \
        --beta 0.04 \


# # ## ---------- IDP ----------
RUN_NAME=Liquid_V1_7B-pico-aurora-multiturn-sft-grpo_idp_kinetics+ucf+mit_roll16_lr1e6_warmup100_lrScheCosine_beta0.1_bsz128_temp0.7_topk50
accelerate launch \
    --num_machines $NUM_NODES \
    --num_processes $TOTAL_PROCESSES \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --use_deepspeed \
    --deepspeed_config_file $DS_CONFIG \
    --mixed_precision bf16 \
    src/grpo_idp/train_grpo.py \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_paths \
        ${DATASET_ROOT}/unsupervised-frames-ucf-kinetics-mit/tokenized_liquid_kinetics_idp_grpo \
        ${DATASET_ROOT}/unsupervised-frames-ucf-kinetics-mit/tokenized_liquid_mit_idp_grpo \
        ${DATASET_ROOT}/unsupervised-frames-ucf-kinetics-mit/tokenized_liquid_ucf_idp_grpo \
    --output_dir ${CKPT_ROOT}/${RUN_NAME} \
    --learning_rate 1e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_generations 16 \
    --chunk_size 8 \
    --gradient_checkpointing True \
    --bf16 True \
    --logging_steps 1 \
    --save_steps 100 \
    --run_name ${RUN_NAME} \
    --report_to wandb \
    --warmup_steps 100 \
    --lr_scheduler_type "cosine" \
    --beta 0.1 \
    --max_grad_norm 0.1 \
    --our_temperature 0.7 \
    --our_top_k 50 \
