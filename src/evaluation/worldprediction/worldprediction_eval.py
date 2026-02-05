#!/usr/bin/env python
"""Generation pipeline for WorldPrediction multi-turn procedural planning."""


import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from transformers import GenerationConfig

sys.path.append(os.getcwd())

try:
    from src.inferencer import create_inference_engine
except ImportError as exc:                                                       
    raise RuntimeError("Unable to import 'src.inferencer'. Ensure the project root is on PYTHONPATH.") from exc

from src.evaluation.worldprediction.data import (
    WorldPredictionActionStep,
    WorldPredictionSample,
    load_worldprediction_metadata,
)


DEFAULT_PROMPT_TEMPLATE = (
    "You are updating a workspace image following a procedural plan. "
    "Overall plan: {plan_description}. "
    "Current step ({step_index}/{total_steps}): {action_label}. "
    "Generate the next observation after completing this action while staying consistent with the scene."
)


class FrameResolver:
    """Resolve frames for segment identifiers by consulting videos or cached images."""

    def __init__(
        self,
        dataset_root: Path,
        frame_map: Optional[Dict[str, str]] = None,
        fallback_dirs: Optional[Sequence[str]] = None,
        extensions: Sequence[str] = (".png", ".jpg", ".jpeg"),
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.frame_map = frame_map or {}
        self.fallback_dirs = list(fallback_dirs) if fallback_dirs else ["frames", "images"]
        self.extensions = tuple(extensions)
        self.video_root = self.dataset_root / "videos"
        provisional_cache = Path(cache_dir) if cache_dir else self.dataset_root / "_frame_cache"
        try:
            provisional_cache.mkdir(parents=True, exist_ok=True)
            self.cache_dir = provisional_cache
        except PermissionError:
            fallback = Path(tempfile.gettempdir()) / "worldprediction_frame_cache"
            fallback.mkdir(parents=True, exist_ok=True)
            self.cache_dir = fallback

    def _resolve_from_map(self, segment_uid: str) -> Optional[Path]:
        mapped = self.frame_map.get(segment_uid)
        if not mapped:
            return None
        path = Path(mapped)
        return path if path.is_absolute() else (self.dataset_root / path)

    def _resolve_from_static(self, segment_uid: str) -> Optional[Path]:
        sanitized_flat = segment_uid.replace("|", "_")
        sanitized_hier = segment_uid.replace("|", os.sep)

        for directory in self.fallback_dirs:
            base_dir = self.dataset_root / directory
            for ext in self.extensions:
                candidate = base_dir / f"{segment_uid}{ext}"
                if candidate.exists():
                    return candidate
                candidate = base_dir / f"{sanitized_flat}{ext}"
                if candidate.exists():
                    return candidate
                candidate = base_dir / f"{sanitized_hier}{ext}"
                if candidate.exists():
                    return candidate
        return None

    def _video_path(self, dataset: str, video_name: str) -> Path:
        video_path = self.video_root / dataset / video_name
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found for dataset={dataset}: {video_path}")
        return video_path

    def _cache_path(self, dataset: str, segment_uid: str, timestamp: float) -> Path:
        safe_uid = segment_uid.replace("|", "_")
        digest = hashlib.sha1(f"{segment_uid}|{timestamp:.3f}".encode("utf-8")).hexdigest()[:16]
        filename = f"{safe_uid}_{timestamp:.3f}_{digest}.png"
        return self.cache_dir / dataset / filename

    def _extract_frame(self, video_path: Path, timestamp: float, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ts = max(timestamp, 0.0)
        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{ts:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_path),
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required to extract frames but was not found in PATH") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"ffmpeg failed to extract frame from {video_path} at {ts:.3f}s") from exc

    def frame_for_step(self, dataset: str, step: "WorldPredictionActionStep") -> Path:
        segment_uid = step.segment_uid
        mapped_path = self._resolve_from_map(segment_uid)
        if mapped_path and mapped_path.exists():
            return mapped_path

        static_path = self._resolve_from_static(segment_uid)
        if static_path:
            return static_path

        timestamp = step.representative_timestamp()
        cache_path = self._cache_path(dataset, segment_uid, timestamp)
        if cache_path.exists():
            return cache_path

        video_path = self._video_path(dataset, step.video)
        self._extract_frame(video_path, timestamp, cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(f"Failed to cache frame for {segment_uid}")
        return cache_path

    def load_step(self, dataset: str, step: "WorldPredictionActionStep") -> Image.Image:
        path = self.frame_for_step(dataset, step)
        return Image.open(path).convert("RGB")


def _prepare_generation_config(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        cfg_scale=args.cfg_scale,
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.greedy,
    )


def _format_plan_description(actions: Sequence[str]) -> str:
    if not actions:
        return ""
    if len(actions) == 1:
        return actions[0]
    return "; ".join(actions)


def _generate_candidate(
    engine: Any,
    initial_image: Image.Image,
    actions: Sequence[str],
    action_meta: Sequence[Dict[str, Any]],
    prompt_template: str,
    gen_config: GenerationConfig,
    context: Optional[Dict[str, Any]] = None,
) -> List[Image.Image]:
    current_image = initial_image
    generated_images: List[Image.Image] = []
    plan_description = _format_plan_description(actions)
    total_steps = len(actions)

    for step_index, (action_label, step_info) in enumerate(zip(actions, action_meta), start=1):
        prompt_kwargs = {
            "action_label": action_label,
            "step_index": step_index,
            "total_steps": total_steps,
            "plan_description": plan_description,
            "segment_uid": step_info.get("segment_uid"),
            "video": step_info.get("video"),
            "segment_start_time": step_info.get("segment_start_time"),
            "segment_end_time": step_info.get("segment_end_time"),
        }

        if context:
            prompt_kwargs.update({k: v for k, v in context.items() if v is not None})

        try:
            prompt = prompt_template.format(**{k: v for k, v in prompt_kwargs.items() if v is not None})
        except KeyError as exc:
            missing_key = exc.args[0]
            raise KeyError(
                f"Prompt template missing replacement for '{missing_key}'. "
                "Please ensure the template only references supported keys."
            ) from exc

        outputs = engine.generate_text_image_to_image(
            prompts=[prompt],
            images=[current_image],
            gen_config=gen_config,
        )
        if not outputs:
            raise RuntimeError("Inference engine returned no images")

        current_image = outputs[0].convert("RGB") if isinstance(outputs[0], Image.Image) else Image.fromarray(outputs[0])
        generated_images.append(current_image)

    return generated_images


def _process_sample(
    sample: WorldPredictionSample,
    resolver: FrameResolver,
    engine: Any,
    gen_config: GenerationConfig,
    output_dir: Path,
    overwrite: bool,
    prompt_template: str,
) -> Dict[str, Any]:
    sample_dir = output_dir / sample.sample_uid
    sample_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = sample_dir / "metadata.json"
    if metadata_path.exists() and not overwrite:
        with metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    if not sample.original_segments:
        raise RuntimeError(f"Sample {sample.sample_uid} has no original segments")

    initial_step = sample.original_segments[0]
    try:
        initial_image = resolver.load_step(sample.dataset, initial_step)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Initial observation missing for sample {sample.sample_uid}") from exc

    initial_path = sample_dir / "initial.png"
    initial_image.save(initial_path)

    candidate_metadata: List[Dict[str, Any]] = []

    for cand_idx in range(sample.num_candidates):
        actions = sample.candidate_action_labels(cand_idx)
        action_steps = sample.candidate_segments[cand_idx]
        action_meta = []
        for seg_uid in action_steps:
            action_step = sample.action_segments.get(seg_uid)
            action_meta.append(
                {
                    "segment_uid": seg_uid,
                    "video": action_step.video if action_step else None,
                    "segment_start_time": action_step.segment_start_time if action_step else None,
                    "segment_end_time": action_step.segment_end_time if action_step else None,
                }
            )

        candidate_dir = sample_dir / f"candidate_{cand_idx:02d}"
        candidate_dir.mkdir(parents=True, exist_ok=True)

        expected_paths = [candidate_dir / f"step_{step:02d}.png" for step in range(1, len(actions) + 1)]
        if all(path.exists() for path in expected_paths) and not overwrite:
            generated_paths = expected_paths
        else:
            try:
                generated_images = _generate_candidate(
                    engine=engine,
                    initial_image=initial_image,
                    actions=actions,
                    action_meta=action_meta,
                    prompt_template=prompt_template,
                    gen_config=gen_config,
                    context={
                        "dataset": sample.dataset,
                        "sample_uid": sample.sample_uid,
                        "ground_truth_index": sample.ground_truth_index,
                    },
                )
            except Exception as exc:                                                              
                raise RuntimeError(
                    f"Generation failed for sample {sample.sample_uid}, candidate {cand_idx}: {exc}"
                ) from exc

            generated_paths = []
            for step_idx, image in enumerate(generated_images, start=1):
                save_path = candidate_dir / f"step_{step_idx:02d}.png"
                image.save(save_path)
                generated_paths.append(save_path)

        candidate_metadata.append(
            {
                "index": cand_idx,
                "actions": actions,
                "segments": action_steps,
                "generated_paths": [str(path) for path in generated_paths],
            }
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    serialized = {
        "dataset": sample.dataset,
        "sample_uid": sample.sample_uid,
        "ground_truth_index": sample.ground_truth_index,
        "num_candidates": sample.num_candidates,
        "num_steps": sample.num_steps,
        "initial_image_path": str(initial_path),
        "candidates": candidate_metadata,
    }

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(serialized, handle, indent=2)

    return serialized


def _worker_main(rank: int, device_ids: Sequence[int], args_dict: Dict[str, Any]) -> None:
    device_id = device_ids[rank]
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)

    metadata: List[WorldPredictionSample] = args_dict["metadata"]
    resolver: FrameResolver = args_dict["resolver"]

    engine = create_inference_engine(
        model_type=args_dict["model_type"],
        model_path=args_dict["model_path"],
        device=f"cuda:{device_id}",
        vae_config_path=args_dict.get("vae_config_path"),
        vae_checkpoint_path=args_dict.get("vae_checkpoint_path"),
        dtype=torch.bfloat16,
    )

    gen_config = _prepare_generation_config(args_dict["args"])

    shard_indices = list(range(rank, len(metadata), len(device_ids)))

    dataset_to_indices: Dict[str, List[int]] = defaultdict(list)
    for index in shard_indices:
        dataset_to_indices[metadata[index].dataset].append(index)

    for dataset_name, indices in dataset_to_indices.items():
        iterator = tqdm(
            indices,
            desc=f"{dataset_name} [worker {rank}]",
            leave=False,
            dynamic_ncols=True,
            unit="sample",
        )
        for index in iterator:
            sample = metadata[index]
            try:
                _process_sample(
                    sample=sample,
                    resolver=resolver,
                    engine=engine,
                    gen_config=gen_config,
                    output_dir=args_dict["output_dir"],
                    overwrite=args_dict["args"].overwrite,
                    prompt_template=args_dict["args"].prompt_template,
                )
            except Exception as exc:
                iterator.write(f"[Worker {rank}] Failed sample {sample.sample_uid}: {exc}")


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-turn generation on the WorldPrediction dataset")

    parser.add_argument("--metadata_path", type=Path, required=True, help="Path to WorldPrediction metadata JSON")
    parser.add_argument("--dataset_root", type=Path, required=True, help="Root directory for dataset assets")
    parser.add_argument("--frame_map_path", type=Path, default=None, help="Optional JSON mapping segment_uid -> relative frame path")
    parser.add_argument("--frame_cache_dir", type=Path, default=None, help="Directory to cache extracted frames from videos")
    parser.add_argument("--output_dir", type=Path, required=True, help="Where to write generated rollouts")

    parser.add_argument("--model_type", type=str, default="liquid", choices=["liquid", "chameleon", "emu3", "unitok"], help="Model family for inference")
    parser.add_argument("--model_path", type=str, required=True, help="Checkpoint or HF repo for the model")
    parser.add_argument("--vae_config_path", type=str, default=None, help="Path to VAE config (if required by the model)")
    parser.add_argument("--vae_checkpoint_path", type=str, default=None, help="Path to VAE checkpoint (if required by the model)")

    parser.add_argument("--devices", type=str, default="0", help="Comma-separated list of GPU device ids to use")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional limit on number of samples")
    parser.add_argument("--dataset_filter", type=str, nargs="*", default=None, help="Subset of datasets to keep (e.g., COIN)")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate outputs even if files exist")

    parser.add_argument("--prompt_template", type=str, default=DEFAULT_PROMPT_TEMPLATE, help="Python format string used as the editing prompt")

    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=2048)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--greedy", action="store_true", help="Disable sampling when set")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    metadata = load_worldprediction_metadata(
        metadata_path=args.metadata_path,
        dataset_filter=args.dataset_filter,
        sample_limit=args.max_samples,
    )
    if not metadata:
        raise RuntimeError("No samples loaded from WorldPrediction metadata; check filters and path")

    frame_map: Dict[str, str] = {}
    if args.frame_map_path and Path(args.frame_map_path).exists():
        with Path(args.frame_map_path).open("r", encoding="utf-8") as handle:
            frame_map = json.load(handle)

    resolver = FrameResolver(
        dataset_root=args.dataset_root,
        frame_map=frame_map,
        cache_dir=args.frame_cache_dir,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    devices = [int(device.strip()) for device in args.devices.split(",") if device.strip()]
    if not devices:
        raise ValueError("No valid devices provided")

    shared_args = {
        "metadata": metadata,
        "resolver": resolver,
        "model_type": args.model_type,
        "model_path": args.model_path,
        "vae_config_path": args.vae_config_path,
        "vae_checkpoint_path": args.vae_checkpoint_path,
        "output_dir": args.output_dir,
        "args": args,
    }

    if len(devices) == 1:
        _worker_main(0, devices, shared_args)
    else:
        mp.spawn(_worker_main, args=(devices, shared_args), nprocs=len(devices))


if __name__ == "__main__":
    main()
