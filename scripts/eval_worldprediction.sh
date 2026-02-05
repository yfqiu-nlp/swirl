#!/bin/bash
#SBATCH --job-name=eval-worldprediction
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=100:00:00
#SBATCH --gpus-per-node=8
#SBATCH --output=logs/slurm-%j.out
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT:$REPO_ROOT/src"
mkdir -p logs

STORAGE_PATH="${STORAGE_PATH:-$REPO_ROOT}"
DATASET_ROOT="${DATASET_ROOT:-$STORAGE_PATH/datasets/worldprediction}"
METADATA_PATH="${METADATA_PATH:-$DATASET_ROOT/metadata.json}"
FRAME_MAP_PATH="${FRAME_MAP_PATH:-}"
FRAME_CACHE_DIR="${FRAME_CACHE_DIR:-$DATASET_ROOT/_frame_cache}"
MODEL_PATH="${MODEL_PATH:-$STORAGE_PATH/rlwm-checkpoints/Liquid_V1_7B}"
VAE_CONFIG_PATH="${VAE_CONFIG_PATH:-$STORAGE_PATH/vision_tokenizers/chameleon/vqgan.yaml}"
VAE_CHECKPOINT_PATH="${VAE_CHECKPOINT_PATH:-$STORAGE_PATH/vision_tokenizers/chameleon/vqgan.ckpt}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-Liquid_V1_7B-worldprediction}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/worldprediction/$EXPERIMENT_NAME}"
GEN_DIR="${GEN_DIR:-$OUTPUT_DIR/gen_rollouts}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"

DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
MODEL_TYPE="${MODEL_TYPE:-liquid}"
CLIP_MODEL="${CLIP_MODEL:-ViT-B/32}"
GPT_EVALUATOR="${GPT_EVALUATOR:-gpt-4o}"

mkdir -p "$OUTPUT_DIR" "$GEN_DIR" "$LOG_DIR"

echo "Starting WorldPrediction generation..."

GEN_ARGS=(
  --metadata_path "$METADATA_PATH"
  --dataset_root "$DATASET_ROOT"
  --frame_cache_dir "$FRAME_CACHE_DIR"
  --output_dir "$GEN_DIR"
  --model_type "$MODEL_TYPE"
  --model_path "$MODEL_PATH"
  --vae_config_path "$VAE_CONFIG_PATH"
  --vae_checkpoint_path "$VAE_CHECKPOINT_PATH"
  --devices "$DEVICES"
)

if [[ -n "${FRAME_MAP_PATH}" ]]; then
  GEN_ARGS+=(--frame_map_path "$FRAME_MAP_PATH")
fi

python src/evaluation/worldprediction/worldprediction_eval.py \
  "${GEN_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/generation.log"

echo "WorldPrediction generation complete."

echo "Starting WorldPrediction CLIP evaluation..."

EVAL_ARGS=(
  --metadata_path "$METADATA_PATH"
  --dataset_root "$DATASET_ROOT"
  --frame_cache_dir "$FRAME_CACHE_DIR"
  --generated_root "$GEN_DIR"
  --output_path "$OUTPUT_DIR/worldprediction_clip_metrics.json"
  --clip_model "$CLIP_MODEL"
)

if [[ -n "${FRAME_MAP_PATH}" ]]; then
  EVAL_ARGS+=(--frame_map_path "$FRAME_MAP_PATH")
fi

python src/evaluation/worldprediction/evaluate_worldprediction.py \
  "${EVAL_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/eval_clip.log"

echo "WorldPrediction CLIP evaluation completed."

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  echo "Starting WorldPrediction GPT evaluation..."
  GPT_ARGS=(
    --metadata_path "$METADATA_PATH"
    --dataset_root "$DATASET_ROOT"
    --frame_cache_dir "$FRAME_CACHE_DIR"
    --generated_root "$GEN_DIR"
    --output_path "$OUTPUT_DIR/worldprediction_gpt_metrics.json"
    --model_name "$GPT_EVALUATOR"
  )
  if [[ -n "${FRAME_MAP_PATH}" ]]; then
    GPT_ARGS+=(--frame_map_path "$FRAME_MAP_PATH")
  fi
  python src/evaluation/worldprediction/evaluate_worldprediction_gpt.py \
    "${GPT_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/eval_gpt.log"
  echo "WorldPrediction GPT evaluation completed."
else
  echo "OPENAI_API_KEY not set; skipping GPT-based evaluation."
fi
