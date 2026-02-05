import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.distributed as dist
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import GenerationConfig

try:
    from src.inferencer import create_inference_engine
except ImportError as e:
    print("[FATAL] Could not import 'src.inferencer'. Ensure your PYTHONPATH includes the project root.")
    raise e

EDIT_INSTRUCTION_TEMPLATE = "Modify the provided image based on the given instruction: {instruction}"

def init_distributed_mode():
    """
    Initialize DDP based on standard Torch environment variables.
    This assumes the script is launched via torchrun.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        print("[WARN] Not using distributed mode. Running on single device.")
        return 0, 0, 1

    torch.cuda.set_device(local_rank)
    
    dist.init_process_group(
        backend="nccl", 
        init_method="env://", 
        rank=global_rank, 
        world_size=world_size,
        timeout=torch.distributed.default_pg_timeout
    )
    
    return local_rank, global_rank, world_size

def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _load_processed_metadata(processed_root: Path) -> List[Dict]:
    """Loads metadata from a processed GEdit dataset (local files)."""
    metadata_path = processed_root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        
    with metadata_path.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    
    for entry in entries:
        entry["_source_path"] = processed_root / entry["Source_observation"] if entry.get("Source_observation") else None
        entry["_input_image_path"] = processed_root / entry.get("gedit_input_image_path", entry.get("Source_observation", ""))
    return entries


def setup_inferencer(
    model_type: str,
    model_path: str, 
    vae_config_path: str, 
    vae_ckpt_path: str, 
    local_rank: int
):
    """
    Initializes the Inference Engine on the specific local GPU.
    """
    if dist.get_rank() == 0:
        print(f"[INFO] Initializing Inference Engine for {model_type}...")

    engine = create_inference_engine(
        model_type=model_type,
        model_path=model_path,
        vae_config_path=vae_config_path,
        vae_checkpoint_path=vae_ckpt_path,
        dtype=torch.bfloat16,
    )
    
    return engine


def editing_inference(
    engine: Any,
    input_image: Image.Image,
    instruction: str,
    gen_config: GenerationConfig
) -> Image.Image:
    prompt = EDIT_INSTRUCTION_TEMPLATE.format(instruction=instruction)
    
    try:
        output_images = engine.generate_text_image_to_image(
            prompts=[prompt],
            images=[input_image],
            gen_config=gen_config,
        )
        
        if not output_images:
            raise RuntimeError("Engine returned empty image list.")
            
        return output_images[0]

    except Exception as e:
        raise RuntimeError(f"Inferencer failure: {e}")


def process_dataset(
    args: Any,
    engine: Any,
    gen_config: GenerationConfig,
    output_dir: str,
    global_rank: int,
    world_size: int,
    seed: int,
    processed_root: Optional[Path] = None
):
    if global_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving Arguments: {args} into {output_dir}.")
        with open(os.path.join(output_dir, "evaluation_args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
    
    # IMPORTANT: Wait for Rank 0 to create directory
    dist.barrier()

    if processed_root is None:
        dataset = load_dataset("stepfun-ai/GEdit-Bench", split='train', num_proc=1)
        idx_list = list(range(len(dataset)))
        entry_getter = lambda idx: dataset[idx]
    else:
        processed_entries = _load_processed_metadata(processed_root)
        idx_list = list(range(len(processed_entries)))
        entry_getter = lambda idx: processed_entries[idx]

    my_indices = idx_list[global_rank::world_size]
    
    set_seeds(seed + global_rank)

    if global_rank == 0:
        print(f"[INFO] Total samples: {len(idx_list)} | Samples per rank: ~{len(my_indices)}")
        iterator = tqdm(my_indices, desc="Processing GEdit", file=sys.stderr)
    else:
        iterator = my_indices
    
    success_count = 0
    
    for data_idx in iterator:
        data = entry_getter(data_idx)

        task_type = data.get('gedit_task_type', data.get('task_type', 'unknown_task'))
        key = data.get('gedit_key', data.get('key', f'gedit_{data_idx:05d}'))
        instruction_language = data.get('gedit_instruction_language', data.get('instruction_language', 'unknown_lang'))
        instruction = data.get("Verbalised Action", data.get("instruction"))

        save_dir = Path(output_dir) / "fullset" / task_type / instruction_language
        
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            pass
        
        save_path_fullset_source_image = save_dir / f"{key}_SRCIMG.png"
        save_path_fullset = save_dir / f"{key}.png"

        if save_path_fullset_source_image.exists() and save_path_fullset.exists():
            continue

        try:
            if processed_root is None:
                input_image = data["input_image"].convert("RGB")
                source_image_to_save = input_image.copy()
            else:
                source_path = data.get("_source_path")
                if source_path is None:
                    continue                 
                    
                source_image_to_save = Image.open(source_path).convert("RGB")
                input_image_path = data.get("_input_image_path")
                
                if input_image_path and input_image_path.exists():
                    input_image = Image.open(input_image_path).convert("RGB")
                else:
                    input_image = source_image_to_save.copy()
        except Exception as e:
            print(f"[WARN] Rank {global_rank} skipped {key} due to load error: {e}")
            continue

        try:
            edited_image = editing_inference(
                engine=engine,
                input_image=input_image,
                instruction=instruction,
                gen_config=gen_config
            )

            source_image_to_save.save(save_path_fullset_source_image)
            edited_image.save(save_path_fullset)
            success_count += 1

        except Exception as e:
            print(f"[ERROR] Rank {global_rank} failed inference on {key}: {e}", flush=True)
            continue
        
        if success_count % 10 == 0:
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--vae_config_path", type=str, required=True)
    parser.add_argument("--vae_ckpt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--processed_root", type=str, default=None)
    
    parser.add_argument("--temperature", type=float, default=0.99)
    parser.add_argument("--top_k", type=int, default=4096)
    parser.add_argument("--top_p", type=float, default=0.96)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=False, help="Enable or disable sampling when generating text. with --do-sample or --no-do-sample")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    local_rank, global_rank, world_size = init_distributed_mode()

    inferencer = setup_inferencer(
        model_type=args.model_type,
        model_path=args.model_path,
        vae_config_path=args.vae_config_path,
        vae_ckpt_path=args.vae_ckpt_path,
        local_rank=local_rank
    )
    
    gen_config = GenerationConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        cfg_scale=args.cfg_scale,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
    )

    process_dataset(
        args=args,
        engine=inferencer,
        gen_config=gen_config,
        output_dir=args.output_dir,
        global_rank=global_rank,
        world_size=world_size,
        seed=args.seed,
        processed_root=Path(args.processed_root) if args.processed_root else None
    )

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
