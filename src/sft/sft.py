#!/usr/bin/env python3
"""
Production-grade Supervised Fine-Tuning Script for Liquid Foundation Models
Supports multi-node, multi-GPU training with FSDP
"""

import argparse
import logging
import os

import torch
import torch.distributed as dist
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint

from train.arguments import ModelArguments, DataArguments, CustomTrainingArguments
from train.data_utils import load_and_mix_datasets, CustomDataCollator
from train.trainer import LiquidTrainer
from utils import MetricsCallback, init_distributed_mode, setup_logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    if "RANK" in os.environ or "SLURM_PROCID" in os.environ:
        local_rank, global_rank, world_size = init_distributed_mode()
    else:
        local_rank, global_rank, world_size = 0, 0, 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    training_args = None
    if global_rank == 0:
        print("=" * 80)
        print("Multi-node distributed training initialized with FSDP")
        print(f"World size: {world_size}")
        print("=" * 80)
    
    print(f"Process {global_rank}: local_rank={local_rank}")

    parser = argparse.ArgumentParser(description="Train Liquid Foundation Model")
    parser.add_argument("--config", type=str, help="Path to training config JSON file")
    parser.add_argument("--model_size", type=str, default="7B", choices=["7B", "30B"],
                       help="Model size for optimization strategy")
    parser.add_argument("--debug_mode", action="store_true", help="Enable debug mode (e.g., limit dataset size, set log level to DEBUG)")
    
    args, remaining_argv = parser.parse_known_args()
    
    hf_parser = transformers.HfArgumentParser((
        ModelArguments,
        DataArguments,
        CustomTrainingArguments
    ))
    
    if args.config:
        model_args, data_args, training_args = hf_parser.parse_json_file(args.config)
    else:
        model_args, data_args, training_args = hf_parser.parse_args_into_dataclasses(remaining_argv)
    
    setup_logging(global_rank, training_args)

    if args.debug_mode:
        if global_rank == 0:
            logger.setLevel(logging.DEBUG)
            logger.warning("!!! DEBUG MODE ENABLED: Limiting dataset size and setting log level to DEBUG !!!")
        
        data_args.max_samples = 1000 
        training_args.max_steps = 10
        training_args.logging_steps = 1
        training_args.save_steps = 10
        training_args.eval_steps = 2
        training_args.num_train_epochs = 1.0
        
        if global_rank == 0:
            logger.debug(f"Overridden data_args.max_samples: {data_args.max_samples}")
            logger.debug(f"Overridden training_args.max_steps: {training_args.max_steps}")
            logger.debug(f"Overridden training_args.num_train_epochs: {training_args.num_train_epochs}")
    
    if global_rank == 0:
        os.makedirs(training_args.output_dir, exist_ok=True)
    
    if world_size > 1:
        dist.barrier()

    training_args.local_rank = local_rank

    if global_rank == 0:
        logger.info("Training/evaluation parameters:")
        logger.info(f"  Model: {model_args.model_name_or_path}")
        logger.info(f"  Data mixture: {data_args.data_mixture}")
        logger.info(f"  Output directory: {training_args.output_dir}")
        logger.info(f"  World size: {world_size}")
        logger.info(f"  Batch size per device: {training_args.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
        logger.info(f"  FSDP enabled: {training_args.fsdp}")
    
    set_seed(training_args.seed)
    
    logger.info(f"Loading tokenizer from {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Loading and mixing datasets...")
    
    train_dataset, eval_dataset = load_and_mix_datasets(
        data_args, 
        split="train", 
        seed=training_args.seed
    )
    
    if global_rank == 0:
        logger.info(f"Final Training Dataset Size: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Validation Datasets ({len(eval_dataset)} subsets):")
            for name, ds in eval_dataset.items():
                logger.info(f"  - {name}: {len(ds)} samples")
        else:
            logger.info("No validation split created.")

    if global_rank == 0:
        logger.info(f"Loading model from {model_args.model_name_or_path}")
    
    torch_dtype = getattr(torch, model_args.torch_dtype)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if model_args.use_flash_attention_2 else None,
    )
    
    if training_args.gradient_checkpointing:
        if global_rank == 0:
            logger.info("Model turned on with gradient checkpointing.")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs
        )
    
    if global_rank == 0:
        logger.info(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)
    
    trainer = LiquidTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, 
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[
            MetricsCallback(), 
            EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience)
        ],
    )
    
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and global_rank == 0:
            logger.info(f"Checkpoint found: {last_checkpoint}")
    
    if world_size > 1:
        checkpoint_container = [last_checkpoint]
        dist.broadcast_object_list(checkpoint_container, src=0)
        last_checkpoint = checkpoint_container[0]
        dist.barrier()
    
    if last_checkpoint is not None:
        logger.info(f"[Rank {global_rank}] Resuming from checkpoint: {last_checkpoint}")
    
    if global_rank == 0:
        logger.info("Starting training...")
    
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    
    if world_size > 1:
        dist.barrier()

    logger.info(f"[Rank {global_rank}] Saving final model...")
    trainer.save_model(training_args.output_dir)
    trainer.save_state()

    if global_rank == 0:
        logger.info(f"Model and Trainer state successfully saved to {training_args.output_dir}")

    if world_size > 1:
        dist.barrier()

    if global_rank == 0:
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        logger.info("Training metrics logged and saved.")

    if global_rank == 0:
        logger.info("Training, evaluation, and metric logging completed successfully!")
    
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
