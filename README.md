<div align="left" style="font-family: charter;">

<h1><i> 🌀</i></br> SWIRL: Self-Improving World Modelling with Latent Actions</h1>

<a href="" target="_blank"><img alt="arXiv" src="" height="20"></a>
<a href="[cohort-rlwm/Liquid_V1_7B-pico-aurora-multiturn-iter1-sft_grpo_fdp-vidgen-option-1](https://huggingface.co/cohort-rlwm/Liquid_V1_7B-pico-aurora-multiturn-iter1-sft_grpo_fdp-vidgen-option-1)" target="_blank"><img alt="HF Model: SWIRL" src="https://huggingface.co/cohort-rlwm/Liquid_V1_7B-pico-aurora-multiturn-iter1-sft_grpo_fdp-vidgen-option-1" height="20"></a>

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

![Demo animation](./asset/figure.gif)

## Abstract

Internal modelling of the world---predicting transitions between previous states $X$ and next states $Y$ under actions $Z$---is essential to reasoning and planning for LLMs and VLMs. 
Learning such models typically requires costly action-labelled trajectories. We propose SWIRL, a self-improvement framework that learns from state-only sequences by treating actions as a latent variable and alternating between Forward World Modelling (FWM) $P_\theta(Y|X,Z)$ and an Inverse Dynamics Modelling (IDM) $Q_\phi(Z|X,Y)$. SWIRL iterates two phases: (1) Variational Information Maximisation, which updates the FWM to generate next states that maximise conditional mutual information with latent actions given prior states, encouraging identifiable consistency; and (2) ELBO Maximisation, which updates the IDM to explain observed transitions, effectively performing coordinate ascent. Both models are trained with reinforcement learning (specifically, GRPO) with the opposite frozen model's log-probability as a reward signal. We provide theoretical learnability guarantees for both updates, and evaluate SWIRL on LLMs and VLMs across multiple environments: single-turn and multi-turn open-world visual dynamics and synthetic textual environments for physics, web, and tool calling. SWIRL achieves gains of 16\% on AURORABench, 28\% on ByteMorph, 16\% on WorldPredictionBench, and 14\% on StableToolBench.

## 🛠️ Environment Setup

The codebase relies on a comprehensive Conda environment including PyTorch, DeepSpeed, Accelerate.

1.  **Create the environment** using the provided yaml file:
    ```bash
    conda env create -f environment.yaml
    ```

2.  **Activate the environment**:
    ```bash
    conda activate swirl
    ```

## 📂 Data Preparation

Before training, raw image/video datasets must be tokenized into discrete codes using the vision tokenizer.

### Preprocessing Script
Use `scripts/preprocess_liquid.sh` to convert datasets (e.g., Aurora, PicoBanana, Kinetics) into the format required for SFT and GRPO.

```bash
# Run preprocessing (adjust paths inside the script as needed)
sbatch scripts/preprocess_liquid.sh
```

This script utilizes `src/liquid/tokenize_grpo_dataset.py` to generate:

FDP Data: For training FWM (Policy) with IDM (Reward).

IDP Data: For training IDM (Policy) with FWM (Reward).

## 🚀 Training

1. Supervised Fine-Tuning (SFT) as Warm-up for Image Editing
To warm-start the Liquid before RL training, use the SFT script. This supports multi-node training via FSDP.

```
sbatch scripts/sft_liquid.sh
```

Configuration: Adjust ```model_name_or_path```, ```data_mixture```, and ```output_dir``` in the script.

2. Reinforcement Learning (GRPO)
We provide scripts for Group Relative Policy Optimization (GRPO) to iteratively improve the world models.


Training FWM as policy and IDM as reward:
```
sbatch scripts/grpo_liquid_fwm.sh
```

Training IDM as policy and FWM as reward:
```
sbatch scripts/grpo_liquid_idm.sh
```


## 🚀 Evaluation
We provide evaluation scripts for image editing and generation capabilities.

**Aurora Benchmark**
Evaluates the model on tasks for action-centric visual prediction, etc.

```
sbatch scripts/eval_liquid_aurora.sh
```

Generation: Generating the images.
Metrics: Computes CLIP and DistEdit.
GPT Eval: Optional GPT-4o based evaluation (requires OPENAI_API_KEY).


**ByteMorph Benchmark**
Evaluates the model on the ByteMorph dataset.

```
sbatch scripts/eval_liquid_bytemorph.sh
```
GPT Eval: Optional GPT-4o based evaluation (requires OPENAI_API_KEY).


