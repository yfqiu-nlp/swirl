#!/usr/bin/env python

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from transformers import GenerationConfig
from datasets import load_dataset

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

try:
    from src.inferencer import create_inference_engine
except ImportError:
    print("Error: Could not import 'src.inferencer'. Run this script from the Liquid model root.")
    sys.exit(1)

class LiquidWrapper:
    def __init__(
        self,
        checkpoint_path: str,
        device_id: int,
        vae_path: str = "vision_tokenizers/chameleon",
        temperature: float = 0.75
    ):
        print(f"Initializing Liquid Model on Device {device_id}...")
        self.engine = create_inference_engine(
            model_type="liquid",
            model_path=checkpoint_path,
            vae_config_path=os.path.join(vae_path, "vqgan.yaml"),
            vae_checkpoint_path=os.path.join(vae_path, "vqgan.ckpt"),
            device=f"cuda:{device_id}"
        )

        self.generation_config = GenerationConfig(
            temperature=temperature,
            top_k=4096,
            top_p=0.96,
            cfg_scale=1.0,
            max_new_tokens=1024,
            do_sample=True
        )

    def edit_image(self, image: Image.Image, instruction: str) -> Image.Image:
        prompt = (
            f"Modify the provided image based on the given instruction: {instruction}"
        )

        output_images = self.engine.generate_text_image_to_image(
            prompts=[prompt],
            images=[image],
            gen_config=self.generation_config,
        )

        return output_images[0]

def _process_dataset(
    model_wrapper: LiquidWrapper,
    output_dir: Path,
    shard_id: int,
    total_shards: int,
    cache_dir: Optional[str] = None
) -> None:

    print(f"Loading ByteDance-Seed/BM-Bench (Shard {shard_id}/{total_shards})...")
    dataset = load_dataset("ByteDance-Seed/BM-Bench", split="test", cache_dir=cache_dir)

    if total_shards > 1:
        dataset = dataset.shard(num_shards=total_shards, index=shard_id)

    output_dir.mkdir(parents=True, exist_ok=True)


    for sample in tqdm(dataset, desc=f"Generating (Shard {shard_id})"):
        image_id = sample['image_id']
        task_raw = sample['edit_type']                        

        task = task_raw.replace("/", "-")

        save_dir = output_dir / "fullset" / task
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / f"{image_id}.png"
        src_backup_path = save_dir / f"{image_id}_SRCIMG.png"

        if save_path.exists() and src_backup_path.exists():
            continue

        source_image = sample['src_img']                      

        instruction = sample.get('edit_prompt_rewrite_instruction') or sample.get('edit_prompt')

        if not instruction:
            print(f"Skipping {image_id}: No instruction found.")
            continue

        try:
            if source_image.mode != "RGB":
                source_image = source_image.convert("RGB")

            edited = model_wrapper.edit_image(image=source_image, instruction=instruction)

            source_image.save(src_backup_path)
            edited.save(save_path)

        except Exception as e:
            print(f"Inference failed for ID {image_id} (Task: {task}): {e}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def _multi_gpu_worker(rank: int, device_list: List[int], kwargs: Dict[str, Any]) -> None:
    device_id = device_list[rank]
    shard_id = kwargs["base_shard_id"] * len(device_list) + rank
    total_shards = kwargs["base_total_shards"] * len(device_list)

    model_wrapper = LiquidWrapper(
        checkpoint_path=str(kwargs["model_path"]),
        device_id=device_id,
        vae_path=kwargs["vae_path"],
        temperature=kwargs["temperature"]
    )

    _process_dataset(
        model_wrapper=model_wrapper,
        output_dir=kwargs["output_dir"],
        shard_id=shard_id,
        total_shards=total_shards,
        cache_dir=kwargs.get("dataset_root")                                            
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BM-Bench editing outputs with Liquid Model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--devices", type=str, default=None, help="Comma-separated list of GPU ids")
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--vae_path", type=str, default="vision_tokenizers/chameleon")
    parser.add_argument("--temperature", type=float, default=0.75)

    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--total_shards", type=int, default=1)

    parser.add_argument("--split_file", type=str, default="")
    parser.add_argument("--num_samples", type=int, default=None)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = args.dataset_root if args.dataset_root and os.path.isdir(args.dataset_root) else None

    if args.devices:
        device_list = [int(d.strip()) for d in args.devices.split(",") if d.strip()]
    else:
        device_list = [args.device]

    if len(device_list) == 1:
        wrapper = LiquidWrapper(args.model_path, device_list[0], args.vae_path, args.temperature)
        _process_dataset(
            model_wrapper=wrapper,
            output_dir=output_dir,
            shard_id=args.shard_id,
            total_shards=args.total_shards,
            cache_dir=cache_dir
        )
    else:
        mp.set_start_method("spawn", force=True)

        worker_args: Dict[str, Any] = {
            "model_path": args.model_path,
            "output_dir": output_dir,
            "dataset_root": cache_dir,                 
            "base_shard_id": args.shard_id,
            "base_total_shards": args.total_shards,
            "vae_path": args.vae_path,
            "temperature": args.temperature,
        }

        processes: List[mp.Process] = []
        for rank in range(len(device_list)):
            p = mp.Process(
                target=_multi_gpu_worker,
                args=(rank, device_list, worker_args),
            )
            p.start()
            processes.append(p)

        for proc in processes:
            proc.join()

if __name__ == "__main__":
    main()
