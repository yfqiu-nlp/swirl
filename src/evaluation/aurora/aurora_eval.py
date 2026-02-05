#!/usr/bin/env python


import argparse
import os
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from transformers import GenerationConfig

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

try:
    from src.inferencer import create_inference_engine
except ImportError:
    print("Error: Could not import 'src.inferencer'. Run this script from the Liquid model root.")
    sys.exit(1)

def load_aurora_metadata_local(
    dataset_root: Path,
    split_file: str = "test.json",
    tasks: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Loads AURORA metadata from a single JSON list file (e.g., test.json).
    Structure: [{"input": "...", "output": "...", "instruction": "...", "task": "..."}, ...]
    """
    split_path = dataset_root / split_file

    if not split_path.exists():
        raise FileNotFoundError(f"AURORA split file not found: {split_path}")

    print(f"Loading AURORA data from {split_path}...")

    with open(split_path, 'r') as f:
        raw_data = json.load(f)

    metadata = []

    target_tasks = set(t.lower() for t in tasks) if tasks else None

    for idx, entry in enumerate(raw_data):
        task_name = entry.get("task", "aurora").lower()

        if target_tasks and task_name not in target_tasks:
            continue

        source_path = dataset_root / entry["input"]
        target_path = None

        if not source_path.exists():
            continue

        sample_id = f"aurora_{idx:05d}"

        processed_entry = {
            "id": sample_id,
            "aurora_task": task_name,
            "source_image_path": source_path,
            "target_image_path": target_path,
            "action_text": entry["instruction"]
        }

        metadata.append(processed_entry)

    print(f"Total samples loaded: {len(metadata)}")
    return metadata

class LiquidWrapper:
    def __init__(self, checkpoint_path: str, device_id: int, vae_path: str = "vision_tokenizers/chameleon", temperature: float = 0.99):
        self.engine = create_inference_engine(
            model_type="liquid",
            model_path=checkpoint_path,
            vae_config_path=os.path.join(vae_path, "vqgan.yaml"),
            vae_checkpoint_path=os.path.join(vae_path,"vqgan.ckpt"),
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


def process_dataset(
    model_wrapper: LiquidWrapper,
    metadata: List[Dict[str, Any]],
    output_dir: Path,
    shard_id: int = 0,
    total_shards: int = 1,
):
    indices = list(range(len(metadata)))
    indices = indices[shard_id::total_shards]

    task_dirs = set(entry["aurora_task"] for entry in metadata)
    for task_dir in task_dirs:
        (output_dir / "fullset" / task_dir).mkdir(parents=True, exist_ok=True)

    for idx in tqdm(indices, desc=f"Shard {shard_id}/{total_shards}"):
        entry = metadata[idx]
        task = entry["aurora_task"]
        key = entry["id"]

        save_dir = output_dir / "fullset" / task
        save_path = save_dir / f"{key}.png"
        src_path_out = save_dir / f"{key}_SRCIMG.png"

        if save_path.exists() and src_path_out.exists():
            continue

        try:
            source_image = Image.open(entry["source_image_path"]).convert("RGB")
        except Exception as e:
            print(f"Failed to load image {entry['source_image_path']}: {e}")
            continue

        prompt = entry["action_text"]

        try:
            edited_image = model_wrapper.edit_image(source_image, prompt)

            source_image.save(src_path_out)
            edited_image.save(save_path)
        except Exception as e:
            print(f"Inference failed for {key}: {e}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _multi_gpu_worker(rank: int, device_list: List[int], args_dict: Dict[str, Any]):
    device_id = device_list[rank]
    shard_id = args_dict['base_shard_id'] * len(device_list) + rank
    total_shards = args_dict['base_total_shards'] * len(device_list)

    print(f"[Worker {rank}] Initializing Liquid Model on GPU {device_id}...")

    wrapper = LiquidWrapper(args_dict['model_path'], device_id, args_dict['vae_path'], args_dict['temperature'])

    process_dataset(
        model_wrapper=wrapper,
        metadata=args_dict['metadata'],
        output_dir=args_dict['output_dir'],
        shard_id=shard_id,
        total_shards=total_shards,
    )


def main():
    parser = argparse.ArgumentParser(description="AURORA image editing generation with Liquid")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--devices", type=str, default=None, help="Comma separated GPU ids")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--total_shards", type=int, default=1)
    parser.add_argument("--split_file", type=str, default="test.json")
    parser.add_argument("--tasks", type=str, nargs="*", default=None)
    parser.add_argument("--vae_path", type=str, default="vision_tokenizers/chameleon")
    parser.add_argument("--temperature", type=float, default=0.99)

    parser.add_argument("--num_samples", type=int, default=None)

    args = parser.parse_args()

    model_path = Path(args.model_path)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    metadata = load_aurora_metadata_local(dataset_root, args.split_file, args.tasks)

    if args.num_samples is not None:
        metadata = metadata[:args.num_samples]

    if args.devices:
        device_list = [int(d.strip()) for d in args.devices.split(',') if d.strip()]
    else:
        device_list = [args.device]

    if len(device_list) == 1:
        wrapper = LiquidWrapper(str(model_path), device_list[0], args.vae_path, args.temperature)
        process_dataset(
            wrapper,
            metadata,
            output_dir,
            shard_id=args.shard_id,
            total_shards=args.total_shards
        )
    else:
        mp.set_start_method('spawn', force=True)
        worker_args = {
            'model_path': str(model_path),
            'dataset_root': dataset_root,
            'output_dir': output_dir,
            'metadata': metadata,
            'base_shard_id': args.shard_id,
            'base_total_shards': args.total_shards,
            'vae_path': args.vae_path,
            'temperature': args.temperature,
        }

        processes = []
        for rank in range(len(device_list)):
            p = mp.Process(
                target=_multi_gpu_worker,
                args=(rank, device_list, worker_args),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

if __name__ == "__main__":
    main()
