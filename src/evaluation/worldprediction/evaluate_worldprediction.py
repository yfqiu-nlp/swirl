#!/usr/bin/env python
"""CLIP-based evaluation for WorldPrediction procedural planning rollouts."""


import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import clip
import torch
from PIL import Image
from tqdm import tqdm

from src.evaluation.worldprediction.data import WorldPredictionSample, load_worldprediction_metadata
from src.evaluation.worldprediction.worldprediction_eval import FrameResolver


@dataclass
class CandidateMetrics:
    index: int
    final_score: float
    per_step_scores: List[float] = field(default_factory=list)
    avg_score: float = math.nan
    missing_steps: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "final_score": self.final_score,
            "avg_score": self.avg_score,
            "per_step_scores": self.per_step_scores,
            "missing_steps": self.missing_steps,
        }


@dataclass
class SampleMetrics:
    sample_uid: str
    ground_truth_index: int
    predicted_index: Optional[int]
    candidates: List[CandidateMetrics]

    def to_dict(self) -> Dict[str, object]:
        return {
            "sample_uid": self.sample_uid,
            "ground_truth_index": self.ground_truth_index,
            "candidates": [c.to_dict() for c in self.candidates],
        }


def _encode_image(path: Path, model, preprocess, device, cache: Dict[Path, torch.Tensor]) -> torch.Tensor:
    if path in cache:
        return cache[path]

    image = Image.open(path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(tensor)
    features = features / features.norm(dim=-1, keepdim=True)
    cache[path] = features.cpu()
    return cache[path]


def _clip_similarity(
    model,
    preprocess,
    device,
    cache: Dict[Path, torch.Tensor],
    image_a: Path,
    image_b: Path,
) -> float:
    feat_a = _encode_image(image_a, model, preprocess, device, cache)
    feat_b = _encode_image(image_b, model, preprocess, device, cache)
    return torch.nn.functional.cosine_similarity(feat_a, feat_b).item()


def _collect_reference_paths(sample: WorldPredictionSample, resolver: FrameResolver) -> List[Optional[Path]]:
    refs: List[Optional[Path]] = []
    for step in sample.original_segments:
        try:
            refs.append(resolver.frame_for_step(sample.dataset, step))
        except FileNotFoundError:
            refs.append(None)
    return refs


def _load_generation_metadata(generated_root: Path, sample_uid: str) -> Optional[Dict[str, object]]:
    metadata_path = generated_root / sample_uid / "metadata.json"
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate(
    metadata: Sequence[WorldPredictionSample],
    resolver: FrameResolver,
    generated_root: Path,
    device: torch.device,
    clip_model_name: str,
) -> Dict[str, object]:
    clip_model, preprocess = clip.load(clip_model_name, device=device)
    cache: Dict[Path, torch.Tensor] = {}

    per_sample: List[SampleMetrics] = []
    evaluated_samples = 0
    step_sums: Dict[int, float] = defaultdict(float)
    step_counts: Dict[int, int] = defaultdict(int)
    overall_sum = 0.0
    overall_count = 0
    max_step_index = 0
    dataset_stats: Dict[str, Dict[str, Any]] = {}

    for sample in tqdm(metadata, desc="Evaluating WorldPrediction"):
        generated_meta = _load_generation_metadata(generated_root, sample.sample_uid)
        if not generated_meta:
            continue

        candidate_entries = generated_meta.get("candidates", [])
        if not candidate_entries:
            continue

        reference_paths = _collect_reference_paths(sample, resolver)
        try:
            final_reference = resolver.frame_for_step(sample.dataset, sample.original_segments[-1]) if sample.original_segments else None
        except FileNotFoundError:
            final_reference = None

        candidate_metrics: List[CandidateMetrics] = []
        final_scores: List[Tuple[float, int]] = []

        for candidate_entry in candidate_entries:
            cand_idx = int(candidate_entry.get("index", len(candidate_metrics)))
            generated_paths = [Path(path) for path in candidate_entry.get("generated_paths", [])]

            per_step_scores: List[float] = []
            missing_steps: List[int] = []

            for step_idx, gen_path in enumerate(generated_paths):
                if not gen_path.exists() or step_idx >= len(reference_paths) or reference_paths[step_idx] is None:
                    per_step_scores.append(math.nan)
                    if not gen_path.exists():
                        missing_steps.append(step_idx)
                    continue
                try:
                    score = _clip_similarity(clip_model, preprocess, device, cache, gen_path, reference_paths[step_idx])
                except Exception:
                    score = math.nan
                per_step_scores.append(score)

            if final_reference is not None and generated_paths:
                try:
                    final_score = _clip_similarity(
                        clip_model, preprocess, device, cache, generated_paths[-1], final_reference
                    )
                except Exception:
                    final_score = math.nan
            else:
                final_score = math.nan

            valid_scores = [s for s in per_step_scores if not math.isnan(s)]
            avg_clip = float(sum(valid_scores) / len(valid_scores)) if valid_scores else math.nan

            candidate_metrics.append(
                CandidateMetrics(
                    index=cand_idx,
                    final_score=final_score,
                    per_step_scores=per_step_scores,
                    avg_score=avg_clip,
                    missing_steps=missing_steps,
                )
            )
            final_scores.append((final_score, cand_idx))

        if not candidate_metrics:
            continue

        predicted_index = None
        valid_final_scores = [(score, idx) for score, idx in final_scores if not math.isnan(score)]
        if valid_final_scores:
            _, predicted_index = max(valid_final_scores, key=lambda item: item[0])

        candidate_metrics.sort(key=lambda cm: cm.index)

        sample_metrics = SampleMetrics(
            sample_uid=sample.sample_uid,
            ground_truth_index=sample.ground_truth_index,
            predicted_index=predicted_index,
            candidates=candidate_metrics,
        )
        per_sample.append(sample_metrics)
        evaluated_samples += 1

        dataset_entry = dataset_stats.setdefault(
            sample.dataset,
            {
                "samples": 0,
                "overall_sum": 0.0,
                "overall_count": 0,
                "step_sums": defaultdict(float),
                "step_counts": defaultdict(int),
                "max_step": 0,
            },
        )
        dataset_entry["samples"] += 1

        gt_candidates = [cm for cm in candidate_metrics if cm.index == sample.ground_truth_index]
        if gt_candidates:
            gt_candidate = gt_candidates[0]
            for step_idx, score in enumerate(gt_candidate.per_step_scores):
                if math.isnan(score):
                    continue
                step_sums[step_idx] += score
                step_counts[step_idx] += 1
                overall_sum += score
                overall_count += 1
                max_step_index = max(max_step_index, step_idx + 1)
                dataset_entry["step_sums"][step_idx] += score
                dataset_entry["step_counts"][step_idx] += 1
                dataset_entry["overall_sum"] += score
                dataset_entry["overall_count"] += 1
                dataset_entry["max_step"] = max(dataset_entry["max_step"], step_idx + 1)

    stepwise_clip_mean = []
    for step_idx in range(max_step_index):
        count = step_counts.get(step_idx, 0)
        if count == 0:
            continue
        mean_score = step_sums[step_idx] / count
        stepwise_clip_mean.append({
            "step": step_idx + 1,
            "mean": mean_score,
            "count": count,
        })

    overall_clip_mean = overall_sum / overall_count if overall_count else math.nan

    per_dataset_clip_mean: Dict[str, Dict[str, object]] = {}
    for dataset, stats in dataset_stats.items():
        dataset_stepwise = []
        for step_idx in range(stats["max_step"]):
            count = stats["step_counts"].get(step_idx, 0)
            if count == 0:
                continue
            mean_score = stats["step_sums"][step_idx] / count
            dataset_stepwise.append({
                "step": step_idx + 1,
                "mean": mean_score,
                "count": count,
            })

        overall_mean = (
            stats["overall_sum"] / stats["overall_count"] if stats["overall_count"] else math.nan
        )
        per_dataset_clip_mean[dataset] = {
            "samples": stats["samples"],
            "overall_clip_mean": overall_mean,
            "stepwise_clip_mean": dataset_stepwise,
        }

    summary = {
        "total_samples": len(metadata),
        "evaluated_samples": evaluated_samples,
        "overall_clip_mean": overall_clip_mean,
        "stepwise_clip_mean": stepwise_clip_mean,
        "per_dataset_clip_mean": per_dataset_clip_mean,
        "samples": [sample.to_dict() for sample in per_sample],
    }

    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLIP evaluation for WorldPrediction rollouts")
    parser.add_argument("--metadata_path", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--frame_map_path", type=Path, default=None)
    parser.add_argument("--generated_root", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--dataset_filter", type=str, nargs="*", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--frame_cache_dir", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    metadata = load_worldprediction_metadata(
        metadata_path=args.metadata_path,
        dataset_filter=args.dataset_filter,
        sample_limit=args.max_samples,
    )

    if not metadata:
        raise RuntimeError("No samples available for evaluation")

    frame_map: Dict[str, str] = {}
    if args.frame_map_path and args.frame_map_path.exists():
        with args.frame_map_path.open("r", encoding="utf-8") as handle:
            frame_map = json.load(handle)

    resolver = FrameResolver(
        dataset_root=args.dataset_root,
        frame_map=frame_map,
        cache_dir=args.frame_cache_dir,
    )

    device = torch.device(args.device)
    summary = evaluate(
        metadata=metadata,
        resolver=resolver,
        generated_root=args.generated_root,
        device=device,
        clip_model_name=args.clip_model,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
