import os
import sys
from pathlib import Path
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from datasets import load_from_disk, concatenate_datasets
from trl import GRPOConfig, ModelConfig

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from grpo_trainer import LiquidGRPOTrainer
from reward_module import IDPRewardModule

_original_torch_save = torch.save
def safe_torch_save(obj, f, *args, **kwargs):
    kwargs["_use_new_zipfile_serialization"] = False
    return _original_torch_save(obj, f, *args, **kwargs)
torch.save = safe_torch_save

@dataclass
class LiquidGRPOConfig(GRPOConfig):
    dataset_paths: List[str] = field(default_factory=list)
    eval_dataset_path: Optional[str] = field(default=None, metadata={"help": "Path to tokenized validation dataset"})
    chunk_size: int = field(default=4, metadata={"help": "Generations per pass to fit in VRAM"})


def main():
    parser = HfArgumentParser((LiquidGRPOConfig, ModelConfig))
    config, model_config = parser.parse_args_into_dataclasses()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if global_rank == 0:
        print(f"Loading Train Datasets from {config.dataset_paths}...")

    loaded_datasets = []
    for path in config.dataset_paths:
        try:
            d = load_from_disk(path)
            loaded_datasets.append(d)
        except Exception as e:
            if global_rank == 0: print(f"[Error] Failed to load {path}: {e}")

    full_train_dataset = concatenate_datasets(loaded_datasets).shuffle(seed=42)

    eval_dataset = None
    if config.eval_dataset_path:
        if global_rank == 0:
            print(f"Loading Eval Dataset from {config.eval_dataset_path}...")
        try:
            eval_dataset = load_from_disk(config.eval_dataset_path)
        except Exception as e:
            print(f"Error loading eval dataset: {e}")

    if global_rank == 0:
        print(f"Total Global Train Samples: {len(full_train_dataset)}")

    if world_size > 1:
        train_dataset = full_train_dataset.shard(num_shards=world_size, index=global_rank)
        min_shard_size = len(full_train_dataset) // world_size
        train_dataset = train_dataset.select(range(min_shard_size))
        if global_rank == 0: print(f"Sharded & Truncated Train Data. Local size: {len(train_dataset)} per rank (Total used: {min_shard_size * world_size})")
    else:
        train_dataset = full_train_dataset


    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if "labels" in examples[0]:
            batch = {"input_ids": [], "labels": [], "attention_mask": []}
            for ex in examples:
                batch["input_ids"].append(torch.tensor(ex["input_ids"], dtype=torch.long))
                batch["labels"].append(torch.tensor(ex["labels"], dtype=torch.long))
                batch["attention_mask"].append(torch.tensor(ex["attention_mask"], dtype=torch.long))

            padded_input = torch.nn.utils.rnn.pad_sequence(batch["input_ids"], batch_first=True, padding_value=tokenizer.pad_token_id)
            padded_labels = torch.nn.utils.rnn.pad_sequence(batch["labels"], batch_first=True, padding_value=-100)
            padded_mask = torch.nn.utils.rnn.pad_sequence(batch["attention_mask"], batch_first=True, padding_value=0)

            return {
                "input_ids": padded_input,
                "labels": padded_labels,
                "attention_mask": padded_mask
            }

        else:
            batch = {"input_ids": [], "attention_mask": [], "action_input_ids": []}
            for ex in examples:
                prompt = torch.tensor(ex['fdp_prompt_ids'], dtype=torch.long)
                action = torch.tensor(ex['action_ids'], dtype=torch.long)

                batch["input_ids"].append(prompt)
                batch["attention_mask"].append(torch.ones(len(prompt), dtype=torch.long))
                batch["action_input_ids"].append(action)

            max_len = max(x.size(0) for x in batch["input_ids"])
            padded_ids, padded_mask = [], []

            for ids, mask in zip(batch["input_ids"], batch["attention_mask"]):
                pad_len = max_len - ids.size(0)
                if pad_len > 0:
                    padded_ids.append(torch.nn.functional.pad(ids, (pad_len, 0), value=tokenizer.pad_token_id))
                    padded_mask.append(torch.nn.functional.pad(mask, (pad_len, 0), value=0))
                else:
                    padded_ids.append(ids)
                    padded_mask.append(mask)

            return {
                "input_ids": torch.stack(padded_ids),
                "attention_mask": torch.stack(padded_mask),
                "action_input_ids": batch["action_input_ids"]
            }

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    reward_module = IDPRewardModule(
        model_path=model_config.model_name_or_path,
        tokenizer=tokenizer,
        device=device,
        torch_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        use_cache=False,
        low_cpu_mem_usage=True,
    )

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    trainer = LiquidGRPOTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,                        
        processing_class=tokenizer,
        reward_funcs=[reward_module],
        data_collator=collate_fn,
    )

    print("Starting Training...")
    trainer.train()

    if int(os.environ.get("RANK", 0)) == 0:
        trainer.save_model(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)


if __name__ == "__main__":
    main()
