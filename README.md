<div align="left" style="font-family: charter;">

<h1><i> 🌀</i></br> SWIRL: Self-Improving World Modelling with Latent Actions</h1>

<a href="https://huggingface.co/cohort-rlwm/Liquid_V1_7B-pico-aurora-multiturn-iter1-sft_grpo_fdp-vidgen-option-1" target="_blank"><img alt="HF Model: SWIRL" src="https://huggingface.co/cohort-rlwm/Liquid_V1_7B-pico-aurora-multiturn-iter1-sft_grpo_fdp-vidgen-option-1" height="20"></a>

<div>
    <a href="https://scholar.google.com/citations?user=OA6GaMwAAAAJ&hl=en" target="_blank">Yifu Qiu</a><sup>1</sup>&emsp;
    <a href="https://scholar.google.com/citations?user=UO0MJeQAAAAJ&hl=en" target="_blank">Zheng Zhao</a><sup>1</sup>&emsp;
    <a href="https://scholar.google.com/citations?user=BkzhHOIAAAAJ&hl=en" target="_blank">Waylon Li</a><sup>1</sup>&emsp;
    <a href="https://yftah89.github.io/" target="_blank">Yftah Ziser</a><sup>3,4</sup>
    <a href="https://yftah89.github.io/" target="_blank">Anna Korhonen</a><sup>2</sup>
    <a href="https://homepages.inf.ed.ac.uk/scohen/" target="_blank">Shay B. Cohen</a><sup>1</sup>
    <a href="https://yftah89.github.io/" target="_blank">Edoardo Ponti</a><sup>1</sup>&emsp;
    
</div>

<div>
    <sup>1</sup>University of Edinburgh&emsp;
    <sup>2</sup>University of Cambridge&emsp;
    <sup>3</sup>NVIDIA Research&emsp;
    <sup>4</sup>University of Groningen&emsp;
</div>


## Overall

<img src="./asset/figure.gif" width="80%" alt="Demo animation" />

## Abstract

Internal modelling of the world---predicting transitions between previous states $X$ and next states $Y$ under actions $Z$---is essential to reasoning and planning for LLMs and VLMs. 
Learning such models typically requires costly action-labelled trajectories. We propose SWIRL, a self-improvement framework that learns from state-only sequences by treating actions as a latent variable and alternating between Forward World Modelling (FWM) $P_\theta(Y|X,Z)$ and an Inverse Dynamics Modelling (IDM) $Q_\phi(Z|X,Y)$. SWIRL iterates two phases: (1) Variational Information Maximisation, which updates the FWM to generate next states that maximise conditional mutual information with latent actions given prior states, encouraging identifiable consistency; and (2) ELBO Maximisation, which updates the IDM to explain observed transitions, effectively performing coordinate ascent. Both models are trained with reinforcement learning (specifically, GRPO) with the opposite frozen model's log-probability as a reward signal. We provide theoretical learnability guarantees for both updates, and evaluate SWIRL on LLMs and VLMs across multiple environments: single-turn and multi-turn open-world visual dynamics and synthetic textual environments for physics, web, and tool calling. SWIRL achieves gains of 16\% on AURORABench, 28\% on ByteMorph, 16\% on WorldPredictionBench, and 14\% on StableToolBench.

## 🛠️ Environment Setup

This repo is configured to use the Conda environment in `environment.yaml`.

```bash
conda env create -f environment.yaml
conda activate swirl
```

Optional (Hugging Face caches):
```bash
export HF_HOME=/path/to/hf_cache
export HF_DATASETS_CACHE=/path/to/hf_datasets_cache
```

## 📦 Assets and Paths

We keep paths configurable via environment variables and script arguments. By default, the scripts assume:

- `STORAGE_PATH` points to your local workspace (defaults to the repo root in the scripts).
- `DATASET_ROOT=$STORAGE_PATH/datasets`
- `CKPT_ROOT=$STORAGE_PATH/rlwm-checkpoints`
- `VAE_PATH=$STORAGE_PATH/vision_tokenizers/chameleon`

Required assets (not included in this repo):

- **Vision tokenizer (VQGAN/Chameleon)**: `vqgan.yaml` and `vqgan.ckpt`
  - expected at: `vision_tokenizers/chameleon/`
- **Datasets** (place under `datasets/` or pass `--base_dir/--dataset_root`):
  - `datasets/pico-banana-400k/`
  - `datasets/AURORA/`
  - `datasets/ByteMorph/`
  - `datasets/unsupervised-frames-ucf-kinetics-mit/`
  - `datasets/VIDGEN-1M_images/` (for VidGen annotations)
- **Model checkpoints** (if not using HF model IDs):
  - `rlwm-checkpoints/<experiment_name>/`

## 📂 Data Preparation

Before training, datasets must be tokenized into discrete codes using the vision tokenizer.

### Preprocessing (tokenization + annotations)
The main preprocessing entrypoint is:

```bash
sbatch scripts/preprocess_liquid.sh
```

Notes:
- Edit `STORAGE_PATH`, `DATASET_ROOT`, `VAE_PATH`, and dataset names inside the script as needed.
- `preprocess/annotate_mit_kinetics_ucf.py` and `preprocess/annotate_vidgen_shards.py` are called from the script.

Optional: Extract frames for VidGen
```bash
sbatch preprocess/extract_frames.sh
```

## 🚀 Training

### 1) Supervised Fine-Tuning (SFT)
Warm-start the Liquid model using FSDP:

```bash
sbatch scripts/sft_liquid.sh
```

Key knobs:
- `EXPERIMENT_NAME`
- `--model_name_or_path`
- `--data_mixture`
- `--dataset_base_path`

### 2) Reinforcement Learning (GRPO)
Train both directions (FDP and IDP) from the same script:

```bash
sbatch scripts/grpo_liquid.sh
```

Key knobs:
- `RUN_NAME`
- `CKPT_PATH`
- `--dataset_paths`

## ✅ Evaluation

### AURORA
```bash
sbatch scripts/eval_liquid_aurora.sh
```

- Generates images and computes CLIP-based metrics.
- Optional GPT evaluation requires:
  ```bash
  export OPENAI_API_KEY=...
  ```

### ByteMorph
```bash
sbatch scripts/eval_liquid_bytemorph.sh
```

- Optional GPT evaluation also uses `OPENAI_API_KEY`.

### WorldPredictionBench
```bash
sbatch scripts/eval_worldprediction.sh
```

Notes:
- Set `DATASET_ROOT` and `METADATA_PATH` to point to the WorldPrediction dataset and its metadata JSON.
- If you have a frame map (segment_uid -> relative frame path), set `FRAME_MAP_PATH`.
- `ffmpeg` must be available in `PATH` to extract frames from videos.
- Optional GPT evaluation uses `OPENAI_API_KEY`.

## 🧪 Running without Slurm

All `scripts/*.sh` files are standard bash scripts. You can run them directly after:
1) activating the environment, and
2) setting the required path variables.
