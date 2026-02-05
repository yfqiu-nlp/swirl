#!/bin/bash
#SBATCH --job-name=eval-liquid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=150:00:00
#SBATCH --gpus-per-node=8
#SBATCH --output=logs/slurm-%j.out
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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT:$REPO_ROOT/src"
# export HF_HUB_OFFLINE=1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_TREE_THRESHOLD=0

# Activate your environment before running, e.g.:
# source /path/to/conda.sh && conda activate swirl
mkdir -p logs

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
EXPERIMENT_NAME="Liquid_V1_7B-pico-aurora-multiturn-sft-grpo_fdp_kinetics+ucf+mit_roll8_lr1e6_warmup100_lrScheCosine_beta0.1_bsz128_temp0.7_TokenRewardClip10__gradAcc4_4nodes/checkpoint-100"
# EXPERIMENT_NAME="Liquid_V1_7B_pb-multiturn:1.0,pb-idp:1.0,pb-fdp:1.0,ag-fdp:1.0,something-fdp:1.0,kubric-fdp:1.0,magicbrush-fdp:1.0,ag-idp:1.0,something-idp:1.0,kubric-idp:1.0,magicbrush-idp:1.0_bs128_lr2e5_epoch5"
STORAGE_PATH="${STORAGE_PATH:-$REPO_ROOT}"
MODEL_PATH="${MODEL_PATH:-$STORAGE_PATH/rlwm-checkpoints/$EXPERIMENT_NAME}"
DATASET_ROOT="${DATASET_ROOT:-$STORAGE_PATH/datasets/ByteMorph}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/bytemorph/$EXPERIMENT_NAME}"
GEN_DIR="${GEN_DIR:-$OUTPUT_DIR/gen_image}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
SPLIT_FILE="${SPLIT_FILE:-$DATASET_ROOT/data/test-00000-of-00001.parquet}"
VAE_PATH="${VAE_PATH:-$STORAGE_PATH/vision_tokenizers/chameleon/}"
GPT_EVALUATOR="gpt-4o"

TOTAL_NODES=${TOTAL_NODES:-1}
NODE_ID=${NODE_ID:-0}
LOCAL_GPUS=${LOCAL_GPUS:-0,1,2,3,4,5,6,7}
MASTER_NODE=${MASTER_NODE:-0}

mkdir -p "$OUTPUT_DIR" "$GEN_DIR" "$LOG_DIR"

# -----------------------------------------------------------------------------
# Image generation
# -----------------------------------------------------------------------------
python src/evaluation/bytemorph/gen_images_bytemorph.py \
  --model_path "$MODEL_PATH" \
  --dataset_root "$DATASET_ROOT" \
  --output_dir "$GEN_DIR" \
  --devices "$LOCAL_GPUS" \
  --shard_id "$NODE_ID" \
  --total_shards "$TOTAL_NODES" \
  --split_file "$SPLIT_FILE" \
  --vae_path "$VAE_PATH" \
  2>&1 | tee "$LOG_DIR/generation_node${NODE_ID}.log"

echo "ByteMorph generation complete for node $NODE_ID"

# -----------------------------------------------------------------------------
# Only master node performs evaluation
# -----------------------------------------------------------------------------
if [[ "$MASTER_NODE" == "$NODE_ID" ]]; then
  python src/evaluation/bytemorph/evaluate_bytemorph.py \
    --dataset_root "$DATASET_ROOT" \
    --generated_root "$GEN_DIR" \
    --split_file "$SPLIT_FILE" \
    --output_path "$OUTPUT_DIR/bytemorph_eval_metrics.json" \
    2>&1 | tee "$LOG_DIR/eval_clip.log"
  echo "ByteMorph CLIP evaluation completed"

  if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    python src/evaluation/bytemorph/evaluate_bytemorph_gpt.py \
      --dataset_root "$DATASET_ROOT" \
      --generated_root "$GEN_DIR" \
      --split_file "$SPLIT_FILE" \
      --model_name "$GPT_EVALUATOR" \
      --save_path "$OUTPUT_DIR/bytemorph_eval_metrics_gpt.json" \
      2>&1 | tee "$LOG_DIR/eval_gpt.log"
    echo "ByteMorph GPT evaluation completed"
  else
    echo "OPENAI_API_KEY not set; skipping GPT-based evaluation."
  fi
else
  echo "Node $NODE_ID completed generation; evaluation handled by MASTER_NODE=$MASTER_NODE."
fi
