import os
import sys
from pathlib import Path
import torch

_original_torch_save = torch.save
def safe_torch_save(obj, f, *args, **kwargs):
    kwargs["_use_new_zipfile_serialization"] = False
    return _original_torch_save(obj, f, *args, **kwargs)
torch.save = safe_torch_save

if int(os.environ.get("RANK", 0)) != 0:
    os.environ["WANDB_MODE"] = "disabled"

from dataclasses import dataclass, field
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from datasets import load_from_disk, concatenate_datasets
from trl import GRPOConfig, ModelConfig

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from grpo_trainer import LiquidGRPOTrainer
from reward_module import FDPRewardModule

@dataclass
class LiquidGRPOConfig(GRPOConfig):
    dataset_paths: List[str] = field(default_factory=list)
    chunk_size: int = field(default=4)
    max_completion_length: int = field(default=128, metadata={"help": "Max text tokens for IDP"})
    our_temperature: float = field(default=0.8, metadata={"help": "GRPO decoding temperature for IDP"})
    our_top_p: float = field(default=0.9, metadata={"help": "GRPO decoding top_p for IDP"})
    our_top_k: int = field(default=None, metadata={"help": "GRPO decoding top_k for IDP"})

def main():
    parser = HfArgumentParser((LiquidGRPOConfig, ModelConfig))
    config, model_config = parser.parse_args_into_dataclasses()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))

    print(f"[Init] Global Rank: {global_rank}, World Size: {world_size}")

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Datasets from {config.dataset_paths}...")
    loaded_datasets = []
    for path in config.dataset_paths:
        try:
            d = load_from_disk(path)
            loaded_datasets.append(d)
        except Exception as e:
            print(f"[Error] Failed to load {path}: {e}")

    full_dataset = concatenate_datasets(loaded_datasets)
    full_dataset = full_dataset.shuffle(seed=42)

    if global_rank == 0:
        print(f"Total Global Samples: {len(full_dataset)}")

    if world_size > 1:
        full_dataset = full_dataset.shard(num_shards=world_size, index=global_rank)
        print(f"[Rank {global_rank}] Sharded dataset. My samples: {len(full_dataset)}")
    else:
        full_dataset = full_dataset

    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "source_image_ids": [],
            "target_image_ids": []
        }

        for ex in examples:
            prompt = torch.tensor(ex['idp_prompt_ids'], dtype=torch.long)

            src = torch.tensor(ex['source_image_ids'], dtype=torch.long)
            tgt = torch.tensor(ex['target_image_ids'], dtype=torch.long)

            batch["input_ids"].append(prompt)
            batch["attention_mask"].append(torch.ones(len(prompt), dtype=torch.long))
            batch["source_image_ids"].append(src)
            batch["target_image_ids"].append(tgt)

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
            "source_image_ids": batch["source_image_ids"],                       
            "target_image_ids": batch["target_image_ids"]                        
        }

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    print("Initializing FDP Reward Module...")
    reward_module = FDPRewardModule(
        model_path=model_config.model_name_or_path,
        tokenizer=tokenizer,
        device=device,
        torch_dtype=torch.bfloat16
    )

    print("Initializing Policy Model...")
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
        train_dataset=full_dataset,
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
