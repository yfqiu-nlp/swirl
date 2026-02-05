#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=100:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/slurm-%j.out
# #SBATCH --account=YOUR_ACCOUNT
# #SBATCH --qos=YOUR_QOS

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT:$REPO_ROOT/src"

# Activate your environment before running, e.g.:
# source /path/to/conda.sh && conda activate swirl
mkdir -p logs

STORAGE_PATH="${STORAGE_PATH:-$REPO_ROOT}"

### Pico banana
python src/liquid/tokenize_dataset.py \
  --base_dir ${STORAGE_PATH}/datasets/pico-banana-400k \
  --model_path ${STORAGE_PATH}/models/Liquid_V1_7B \
  --vae_config_path ${STORAGE_PATH}/vision_tokenizers/chameleon/vqgan.yaml \
  --vae_checkpoint_path ${STORAGE_PATH}/vision_tokenizers/chameleon/vqgan.ckpt \
  --dataset pico-banana \
  --task_type fdp \
  --output_dir_name tokenized_liquid_short_action_fdp \

python src/liquid/tokenize_dataset.py \
  --base_dir ${STORAGE_PATH}/datasets/pico-banana-400k \
  --model_path ${STORAGE_PATH}/models/Liquid_V1_7B \
  --vae_config_path ${STORAGE_PATH}/vision_tokenizers/chameleon/vqgan.yaml \
  --vae_checkpoint_path ${STORAGE_PATH}/vision_tokenizers/chameleon/vqgan.ckpt \
  --dataset pico-banana \
  --task_type idp \
  --output_dir_name tokenized_liquid_short_action_idp \

python src/liquid/tokenize_dataset.py \
  --base_dir ${STORAGE_PATH}/datasets/pico-banana-400k \
  --model_path ${STORAGE_PATH}/models/Liquid_V1_7B \
  --vae_config_path ${STORAGE_PATH}/vision_tokenizers/chameleon/vqgan.yaml \
  --vae_checkpoint_path ${STORAGE_PATH}/vision_tokenizers/chameleon/vqgan.ckpt \
  --dataset pico-banana \
  --task_type multiturn \
  --data_file_name multi-turn.jsonl \
  --output_dir_name tokenized_liquid_multiturn \

# ### AURORA
AURORA_SUBSET=magicbrush
python src/liquid/tokenize_dataset.py \
  --model_path ${STORAGE_PATH}/models/Liquid_V1_7B \
  --vae_config_path ${STORAGE_PATH}/vision_tokenizers/chameleon/vqgan.yaml \
  --vae_checkpoint_path ${STORAGE_PATH}/vision_tokenizers/chameleon/vqgan.ckpt \
  --dataset aurora \
  --base_dir ${STORAGE_PATH}/datasets/AURORA \
  --task_type fdp \
  --subset ${AURORA_SUBSET} \
  --output_dir_name tokenized_liquid_aurora_${AURORA_SUBSET}_fdp \


python src/liquid/tokenize_dataset.py \
  --model_path ${STORAGE_PATH}/models/Liquid_V1_7B \
  --vae_config_path ${STORAGE_PATH}/vision_tokenizers/chameleon/vqgan.yaml \
  --vae_checkpoint_path ${STORAGE_PATH}/vision_tokenizers/chameleon/vqgan.ckpt \
  --dataset aurora \
  --base_dir ${STORAGE_PATH}/datasets/AURORA \
  --task_type idp \
  --subset ${AURORA_SUBSET} \
  --output_dir_name tokenized_liquid_aurora_${AURORA_SUBSET}_idp \


### processing for tokenizing the datasets for GRPO training
AURORA_SUBSET=ag
python src/liquid/tokenize_grpo_dataset.py \
  --model_path ${STORAGE_PATH}/models/Liquid_V1_7B \
  --vae_config_path ${STORAGE_PATH}/vision_tokenizers/chameleon/vqgan.yaml \
  --vae_checkpoint_path ${STORAGE_PATH}/vision_tokenizers/chameleon/vqgan.ckpt \
  --dataset aurora \
  --base_dir ${STORAGE_PATH}/datasets/AURORA \
  --subset ${AURORA_SUBSET} \
  --output_dir_name tokenized_liquid_aurora_${AURORA_SUBSET}_grpo \

## Annotating actions for tokenizing the mit, kinetics, ucf with Liquid
python preprocess/annotate_mit_kinetics_ucf.py --dataset ucf
python preprocess/annotate_mit_kinetics_ucf.py --dataset mit
python preprocess/annotate_mit_kinetics_ucf.py --dataset kinetics

## processing for tokenizing the SFT datasets with ucf, mit, kinetics
DATASET="kinetics"
python src/liquid/tokenize_dataset.py \
  --model_path ${STORAGE_PATH}/models/Liquid_V1_7B \
  --vae_config_path ${STORAGE_PATH}/vision_tokenizers/chameleon/vqgan.yaml \
  --vae_checkpoint_path ${STORAGE_PATH}/vision_tokenizers/chameleon/vqgan.ckpt \
  --dataset ${DATASET} \
  --base_dir ${STORAGE_PATH}/datasets/unsupervised-frames-ucf-kinetics-mit \
  --task_type fdp \
  --output_dir_name tokenized_liquid_${DATASET}_fdp \

DATASET="mit"
python src/liquid/tokenize_grpo_dataset.py \
  --model_path ${STORAGE_PATH}/models/Liquid_V1_7B \
  --vae_config_path ${STORAGE_PATH}/vision_tokenizers/chameleon/vqgan.yaml \
  --vae_checkpoint_path ${STORAGE_PATH}/vision_tokenizers/chameleon/vqgan.ckpt \
  --dataset ${DATASET} \
  --base_dir ${STORAGE_PATH}/datasets/unsupervised-frames-ucf-kinetics-mit \
  --output_dir_name tokenized_liquid_${DATASET}_idp_grpo \
  --mode idp

################################################
# Annotate the actions for iterative RL algorithm in VidGen-1M
################################################

DATASET=vidgen-1m

## You can skip this if you have shard_meta.json files
python preprocess/prepare_vidgen_shards.py \
    --base_path ${STORAGE_PATH}/datasets/VIDGEN-1M_images \
    --output_dir ${STORAGE_PATH}/datasets/VIDGEN-1M_images \
    --num_shards 5 \
    --shard_size 50000 \

# Annotate the 0th turn
IDP_CKPT_PATH="${STORAGE_PATH}/rlwm-checkpoints/Liquid_V1_7B_pb-multiturn:1.0,pb-idp:1.0,pb-fdp:1.0,ag-fdp:1.0,something-fdp:1.0,kubric-fdp:1.0,magicbrush-fdp:1.0,ag-idp:1.0,something-idp:1.0,kubric-idp:1.0,magicbrush-idp:1.0_bs128_lr2e5_epoch5 "
TURN_IDX=0
IDP_CKPT_NAME=Liquid_V1_7B-pico-aurora-multiturn-sft

## Annotate the action for VIDGEN frame pairs with 0th turn IDP model
python preprocess/annotate_vidgen_shards.py \
    --checkpoint_path ${IDP_CKPT_PATH} \
    --vae_config_path ${STORAGE_PATH}/vision_tokenizers/chameleon/vqgan.yaml \
    --vae_checkpoint_path ${STORAGE_PATH}/vision_tokenizers/chameleon/vqgan.ckpt \
    --base_path ${STORAGE_PATH}/datasets/VIDGEN-1M_images \
    --shard_meta ${STORAGE_PATH}/datasets/VIDGEN-1M_images/meta_shard_${TURN_IDX}.json \
    --output_file ${STORAGE_PATH}/datasets/VIDGEN-1M_images/train_${IDP_CKPT_NAME}_${TURN_IDX}.jsonl \

# Tokenize FDP GRPO dataset for 0th turn model
python src/liquid/tokenize_grpo_dataset.py \
  --model_path ${STORAGE_PATH}/models/Liquid_V1_7B \
  --vae_config_path ${STORAGE_PATH}/vision_tokenizers/chameleon/vqgan.yaml \
  --vae_checkpoint_path ${STORAGE_PATH}/vision_tokenizers/chameleon/vqgan.ckpt \
  --dataset ${DATASET} \
  --base_dir ${STORAGE_PATH}/datasets/VIDGEN-1M_images/ \
  --output_dir_name tokenized_${IDP_CKPT_NAME}_${DATASET}_${TURN_IDX}_fdp_grpo \
  --mode fdp \
  --annotation_file ${STORAGE_PATH}/datasets/VIDGEN-1M_images/train_${IDP_CKPT_NAME}_${TURN_IDX}.jsonl \

# Tokenize IDP GRPO dataset for 0th turn model
python src/liquid/tokenize_grpo_dataset.py \
  --model_path ${STORAGE_PATH}/models/Liquid_V1_7B \
  --vae_config_path ${STORAGE_PATH}/vision_tokenizers/chameleon/vqgan.yaml \
  --vae_checkpoint_path ${STORAGE_PATH}/vision_tokenizers/chameleon/vqgan.ckpt \
  --dataset ${DATASET} \
  --base_dir ${STORAGE_PATH}/datasets/VIDGEN-1M_images/ \
  --output_dir_name tokenized_${IDP_CKPT_NAME}_${DATASET}_${TURN_IDX}_idp_grpo \
  --mode idp \
  --annotation_file ${STORAGE_PATH}/datasets/VIDGEN-1M_images/train_${IDP_CKPT_NAME}_${TURN_IDX}.jsonl \

# Turn 1 Training Plan
# Initialize both FDP and IDP with the SFTed model: Liquid_V1_7B-pico-aurora-multiturn-sft
# Training Step1: (Start training FDP policy with IDP reward) -> You get best FDP CKPT after turn 0, let's call it BEST_FDP_after_turn0
# Training Step2: (Start training IDP policy with the BEST_FDP_after_turn0 as reward) -> You get best IDP CKPT after turn 0 (set IDP_CKPT_PATH to the best FDP CKPT, and IDP_CKPT_NAME to the BEST_IDP_after_turn0)

## Annotate the 1st turn with BEST_IDP_after_turn0
TURN_IDX=1
IDP_CKPT_PATH="${STORAGE_PATH}/rlwm-checkpoints/**PUT YOUR CHECKPOINT HERE**"
IDP_CKPT_NAME="**PUT YOUR CHECKPOINT NAME HERE**, e.g., BEST_IDP_after_turn0"

python preprocess/annotate_vidgen_shards.py \
    --checkpoint_path ${IDP_CKPT_PATH} \
    --vae_config_path ${STORAGE_PATH}/vision_tokenizers/chameleon/vqgan.yaml \
    --vae_checkpoint_path ${STORAGE_PATH}/vision_tokenizers/chameleon/vqgan.ckpt \
    --base_path ${STORAGE_PATH}/datasets/VIDGEN-1M_images \
    --shard_meta ${STORAGE_PATH}/datasets/VIDGEN-1M_images/meta_shard_${TURN_IDX}.json \
    --output_file ${STORAGE_PATH}/datasets/VIDGEN-1M_images/train_${IDP_CKPT_NAME}_${TURN_IDX}.jsonl \

# Tokenize FDP GRPO dataset for 1st turn model
python src/liquid/tokenize_grpo_dataset.py \
  --model_path ${STORAGE_PATH}/models/Liquid_V1_7B \
  --vae_config_path ${STORAGE_PATH}/vision_tokenizers/chameleon/vqgan.yaml \
  --vae_checkpoint_path ${STORAGE_PATH}/vision_tokenizers/chameleon/vqgan.ckpt \
  --dataset ${DATASET} \
  --base_dir ${STORAGE_PATH}/datasets/VIDGEN-1M_images/ \
  --output_dir_name tokenized_${IDP_CKPT_NAME}_${DATASET}_${TURN_IDX}_fdp_grpo \
  --mode fdp \
  --annotation_file ${STORAGE_PATH}/datasets/VIDGEN-1M_images/train_${IDP_CKPT_NAME}_${TURN_IDX}.jsonl \

# Tokenize IDP GRPO dataset for 1st turn model
python src/liquid/tokenize_grpo_dataset.py \
  --model_path ${STORAGE_PATH}/models/Liquid_V1_7B \
  --vae_config_path ${STORAGE_PATH}/vision_tokenizers/chameleon/vqgan.yaml \
  --vae_checkpoint_path ${STORAGE_PATH}/vision_tokenizers/chameleon/vqgan.ckpt \
  --dataset ${DATASET} \
  --base_dir ${STORAGE_PATH}/datasets/VIDGEN-1M_images/ \
  --output_dir_name tokenized_${IDP_CKPT_NAME}_${DATASET}_${TURN_IDX}_idp_grpo \
  --mode idp \
  --annotation_file ${STORAGE_PATH}/datasets/VIDGEN-1M_images/train_${IDP_CKPT_NAME}_${TURN_IDX}.jsonl \

# ...
