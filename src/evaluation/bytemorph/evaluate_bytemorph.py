
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.evaluation.bytemorph.data_utils import load_bytemorph_metadata
from src.evaluation.bytemorph.clip_utils import ClipSimilarity, ClipSimilarity_new


def _ensure_image(value: Any, dataset_root: Path) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, str):
        path = Path(value)
        if not path.is_absolute():
            path = dataset_root / path
        with Image.open(path) as img:
            return img.convert("RGB")
    raise TypeError(f"Unsupported image representation: {type(value)!r}")


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.array(image, dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0) / 255.0
    return tensor


def evaluate(
    dataset_root: Path,
    generated_root: Path,
    metadata: List[Dict[str, Any]],
    device: torch.device,
    clip_model_name: str = "ViT-B/32",
) -> Dict[str, Any]:
    clip_similarity = ClipSimilarity(clip_model_name).to(device)
    clip_similarity_new = ClipSimilarity_new(clip_model_name).to(device)

    stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    counts: Dict[str, int] = defaultdict(int)
    missing: List[str] = []

    for entry in tqdm(metadata, desc="Evaluating ByteMorph"):
        task = (entry.get("bytemorph_edit_type") or "bytemorph").replace("/", "-") or "bytemorph"
        key = entry["id"].split('_')[-1]

        gen_path = generated_root / "fullset" / task / f"{key}.png"
        if not gen_path.exists():
            missing.append(str(gen_path))
            continue

        try:
            source_image = _ensure_image(entry.get("Source_observation"), dataset_root)
        except Exception:
            missing.append(f"source::{key}")
            continue

        target_raw = entry.get("Target_observation")
        target_image = None
        if target_raw is not None:
            try:
                target_image = _ensure_image(target_raw, dataset_root)
            except Exception:
                target_image = None

        generated_image = Image.open(gen_path).convert("RGB")

        source_tensor = _image_to_tensor(source_image).to(device)
        generated_tensor = _image_to_tensor(generated_image).to(device)

        target_tensor = None
        if target_image is not None:
            target_tensor = _image_to_tensor(target_image).to(device)

        input_caption = entry.get("bytemorph_src_caption", "")
        output_caption = entry.get("bytemorph_tgt_caption", "")
        captions_src = [input_caption or ""]
        captions_tgt = [output_caption or ""]

        with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda", dtype=torch.bfloat16):
            clip_scores = clip_similarity(
                source_tensor,
                generated_tensor,
                captions_src,
                captions_tgt,
                return_cross_scores=False,
                return_dict=True,
            )

            if target_tensor is not None:
                dir_scores = clip_similarity_new(
                    source_tensor,
                    target_tensor,
                    generated_tensor,
                    return_cross_scores=False,
                    return_dict=True,
                )
            else:
                dir_scores = {}

        for metric, value in clip_scores.items():
            stats[task][metric].append(float(value))
        for metric, value in dir_scores.items():
            stats[task][metric].append(float(value))

        counts[task] += 1

    summary: Dict[str, Any] = {}
    overall: Dict[str, List[float]] = defaultdict(list)
    total_count = 0

    for task, metric_values in stats.items():
        count = counts[task]
        task_summary: Dict[str, float] = {"count": count}
        for metric, values in metric_values.items():
            if not values:
                continue
            mean_value = float(sum(values) / len(values))
            task_summary[f"avg_{metric}"] = mean_value
            overall[metric].extend(values)
        summary[task] = task_summary
        total_count += count

    if total_count:
        overall_entry: Dict[str, float] = {"count": total_count}
        for metric, values in overall.items():
            if values:
                overall_entry[f"avg_{metric}"] = float(sum(values) / len(values))
        summary["overall"] = overall_entry

    summary["missing"] = missing
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="CLIP evaluation for ByteMorph outputs")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--generated_root", type=str, required=True)
    parser.add_argument("--split_file", type=str, default="data/test-*.parquet")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--tasks", type=str, nargs="*", default=None)
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    generated_root = Path(args.generated_root)

    metadata = load_bytemorph_metadata(dataset_root, split_file=args.split_file)
    if args.tasks:
        wanted = {task.lower() for task in args.tasks}
        metadata = [entry for entry in metadata if entry.get("bytemorph_edit_type", "").lower() in wanted]

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    results = evaluate(dataset_root, generated_root, metadata, device, clip_model_name=args.clip_model)

    output_path = Path(args.output_path or generated_root / "bytemorph_eval_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
