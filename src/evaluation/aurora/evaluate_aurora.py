#!/usr/bin/env python


import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm


def load_aurora_metadata(
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
        if "output" in entry.keys():
            target_path = dataset_root / entry["output"]
        else:
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


def _encode_image(model, preprocess, device: torch.device, image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(tensor)
    return features / features.norm(dim=-1, keepdim=True)


def _encode_text(model, tokenizer, device: torch.device, text: str) -> torch.Tensor:
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
    return features / features.norm(dim=-1, keepdim=True)


def evaluate(
    dataset_root: Path,
    generated_root: Path,
    metadata: List[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, Any]:
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    tokenizer = clip.tokenize

    stats = defaultdict(lambda: {
        "count": 0,
        "success": 0,
        "clip_sim_target": [],
        "clip_sim_source": [],
        "text_sim": [],
    })

    missing_generated = []

    for entry in tqdm(metadata, desc="Evaluating AURORA"):
        task = entry.get("aurora_task", "aurora").replace("/", "-") or "aurora"
        key = entry["id"]

        gen_path = generated_root / "fullset" / task / f"{key}.png"

        if not gen_path.exists():
            missing_generated.append(str(gen_path))
            print(f"No gen_path {gen_path}")
            continue

        source_path = dataset_root / entry["source_image_path"]
        target_rel = entry.get("target_image_path")
        target_path = dataset_root / target_rel if target_rel else None

        if not source_path.exists() or (target_path and not target_path.exists()):
            missing_generated.append(str(gen_path))
            continue

        gen_feat = _encode_image(clip_model, preprocess, device, gen_path)
        src_feat = _encode_image(clip_model, preprocess, device, source_path)
        tgt_feat = _encode_image(clip_model, preprocess, device, target_path) if target_path else None

        dist_source = 1 - torch.nn.functional.cosine_similarity(gen_feat, src_feat).item()
        stats[task]["clip_sim_source"].append(1 - dist_source)

        if tgt_feat is not None:
            dist_target = 1 - torch.nn.functional.cosine_similarity(gen_feat, tgt_feat).item()
            stats[task]["clip_sim_target"].append(1 - dist_target)
            success = dist_target < dist_source
        else:
            dist_target = float("nan")
            success = False

        text_sim = float("nan")
        instruction = entry.get("Verbalised Action", "")
        if instruction:
            text_feat = _encode_text(clip_model, tokenizer, device, instruction)
            text_sim = torch.nn.functional.cosine_similarity(gen_feat, text_feat).item()
            stats[task]["text_sim"].append(text_sim)

        stats[task]["count"] += 1
        stats[task]["success"] += int(success)

    summary = {}
    overall = {
        "count": 0,
        "success": 0,
        "clip_sim_target": [],
        "clip_sim_source": [],
        "text_sim": [],
    }
    for task, values in stats.items():
        count = values["count"] or 1
        success_rate = values["success"] / count
        clip_target = float(np.mean(values["clip_sim_target"])) if values["clip_sim_target"] else float("nan")
        clip_source = float(np.mean(values["clip_sim_source"])) if values["clip_sim_source"] else float("nan")
        text_sim = float(np.mean(values["text_sim"])) if values["text_sim"] else float("nan")
        summary[task] = {
            "count": values["count"],
            "success_rate": success_rate,
            "avg_clip_similarity_target": clip_target,
            "avg_clip_similarity_source": clip_source,
            "avg_clip_text_similarity": text_sim,
        }

        overall["count"] += values["count"]
        overall["success"] += values["success"]
        overall["clip_sim_target"].extend(values["clip_sim_target"])
        overall["clip_sim_source"].extend(values["clip_sim_source"])
        overall["text_sim"].extend(values["text_sim"])

    if overall["count"] > 0:
        summary["overall"] = {
            "count": overall["count"],
            "success_rate": overall["success"] / overall["count"],
            "avg_clip_similarity_target": float(np.mean(overall["clip_sim_target"])) if overall["clip_sim_target"] else float("nan"),
            "avg_clip_similarity_source": float(np.mean(overall["clip_sim_source"])) if overall["clip_sim_source"] else float("nan"),
            "avg_clip_text_similarity": float(np.mean(overall["text_sim"])) if overall["text_sim"] else float("nan"),
        }

    if missing_generated:
        print(f"Warning: {len(missing_generated)} samples missing generated images.")

    return {
        "tasks": summary,
        "missing": missing_generated,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Bagel outputs on AURORA benchmark")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--generated_root", type=str, required=True)
    parser.add_argument("--split_file", type=str, default="test.json")
    parser.add_argument("--tasks", type=str, nargs="*", default=None)
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    generated_root = Path(args.generated_root)
    output_path = Path(args.save_path) if args.save_path else generated_root / "aurora_eval_metrics.json"

    metadata = load_aurora_metadata(dataset_root, split_file=args.split_file)
    if args.tasks:
        wanted = {task.lower() for task in args.tasks}
        metadata = [entry for entry in metadata if entry.get("aurora_task", "").lower() in wanted]
    if args.language:
        metadata = [entry for entry in metadata if entry.get("aurora_raw_record", {}).get("language") == args.language]

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    results = evaluate(dataset_root, generated_root, metadata, device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results["tasks"], indent=2))


if __name__ == "__main__":
    main()
