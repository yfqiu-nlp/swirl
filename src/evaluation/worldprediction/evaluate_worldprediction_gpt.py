#!/usr/bin/env python
"""GPT-based evaluation for WorldPrediction procedural planning rollouts."""


import argparse
import base64
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from openai import OpenAI
from tqdm import tqdm

from src.evaluation.worldprediction.data import WorldPredictionSample, load_worldprediction_metadata
from src.evaluation.worldprediction.worldprediction_eval import FrameResolver


_JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _encode_image(image_path: Path) -> str:
    with image_path.open("rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


@dataclass
class GPTEvaluationResult:
    score: float
    reasoning: str
    raw: str


class GPTSequenceJudge:
    """Wrapper for OpenAI vision models to score per-step alignment."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.client: Optional[OpenAI] = None
        self.prompt_tokens: List[int] = []
        self.completion_tokens: List[int] = []

    def load(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set; cannot perform GPT evaluation")
        self.client = OpenAI(api_key=api_key)

    def _collect_usage(self, completion) -> None:
        usage = getattr(completion, "usage", None)
        if usage is None:
            return
        self.prompt_tokens.append(getattr(usage, "prompt_tokens", 0))
        self.completion_tokens.append(getattr(usage, "completion_tokens", 0))

    @staticmethod
    def _parse_response(content: Optional[str]) -> GPTEvaluationResult:
        if not content:
            return GPTEvaluationResult(score=math.nan, reasoning="", raw="")
        match = _JSON_PATTERN.search(content)
        if not match:
            return GPTEvaluationResult(score=math.nan, reasoning="", raw=content)
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return GPTEvaluationResult(score=math.nan, reasoning="", raw=content)
        score = float(data.get("score", math.nan))
        reasoning = str(data.get("explanation", data.get("reason", data)))
        return GPTEvaluationResult(score=score, reasoning=reasoning, raw=content)

    def score_step(
        self,
        reference_image: Path,
        candidate_image: Path,
        action_label: str,
        step_index: int,
        total_steps: int,
        sample_uid: str,
        dataset: str,
        plan_description: str,
    ) -> GPTEvaluationResult:
        if self.client is None:
            raise RuntimeError("GPT client not initialised. Call load() first")

        reference_encoded = _encode_image(reference_image)
        candidate_encoded = _encode_image(candidate_image)

        prompt = (
            "You are evaluating predicted observations for a procedural plan. "
            "The plan belongs to dataset {dataset}, sample {sample_uid}. "
            "Overall plan: {plan_description}. "
            "Focus on step {step_index} of {total_steps} where the action is '{action_label}'. "
            "Compare the reference observation (ground truth) to the candidate observation (model output). "
            "Return a JSON object like {{\"score\": <0-10>, \"explanation\": \"...\"}} summarising correctness."
        ).format(
            dataset=dataset,
            sample_uid=sample_uid,
            plan_description=plan_description,
            step_index=step_index,
            total_steps=total_steps,
            action_label=action_label,
        )

        payload = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{reference_encoded}", "detail": "high"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{candidate_encoded}", "detail": "high"}},
                ],
            }
        ]

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=payload,
            max_tokens=256,
            temperature=0.0,
        )
        self._collect_usage(completion)
        content = completion.choices[0].message.content
        return self._parse_response(content)

    def summarise_cost(self) -> Dict[str, float]:
        prompt_tokens = sum(self.prompt_tokens)
        completion_tokens = sum(self.completion_tokens)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }


@dataclass
class CandidateGPTMetrics:
    index: int
    per_step_scores: List[float] = field(default_factory=list)
    per_step_reasoning: List[str] = field(default_factory=list)
    final_score: float = math.nan
    avg_score: float = math.nan

    def to_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "per_step_scores": self.per_step_scores,
            "per_step_reasoning": self.per_step_reasoning,
            "final_score": self.final_score,
            "avg_score": self.avg_score,
        }


def _gather_plan_description(sample: WorldPredictionSample, candidate_index: int) -> str:
    actions = sample.candidate_action_labels(candidate_index)
    return "; ".join(actions)


def evaluate(
    metadata: Sequence[WorldPredictionSample],
    resolver: FrameResolver,
    generated_root: Path,
    scorer: GPTSequenceJudge,
) -> Dict[str, object]:
    per_sample: List[Dict[str, object]] = []
    evaluated_samples = 0
    step_sums: Dict[int, float] = defaultdict(float)
    step_counts: Dict[int, int] = defaultdict(int)
    overall_sum = 0.0
    overall_count = 0
    max_step_index = 0
    dataset_stats: Dict[str, Dict[str, Any]] = {}

    for sample in tqdm(metadata, desc="GPT scoring WorldPrediction"):
        generated_meta_path = generated_root / sample.sample_uid / "metadata.json"
        if not generated_meta_path.exists():
            continue
        with generated_meta_path.open("r", encoding="utf-8") as handle:
            generated_meta = json.load(handle)

        candidate_entries = generated_meta.get("candidates", [])
        if not candidate_entries:
            continue

        reference_paths: List[Optional[Path]] = []
        for segment in sample.original_segments:
            try:
                reference_paths.append(resolver.frame_for_step(sample.dataset, segment))
            except FileNotFoundError:
                reference_paths.append(None)

        candidate_metrics: List[CandidateGPTMetrics] = []

        action_labels_cache: Dict[int, List[str]] = {}

        for candidate_entry in candidate_entries:
            candidate_idx = int(candidate_entry.get("index", len(candidate_metrics)))
            generated_paths = [Path(path) for path in candidate_entry.get("generated_paths", [])]
            plan_description = _gather_plan_description(sample, candidate_idx)

            if candidate_idx not in action_labels_cache:
                action_labels_cache[candidate_idx] = sample.candidate_action_labels(candidate_idx)
            candidate_actions = action_labels_cache[candidate_idx]

            per_step_scores: List[float] = []
            per_step_reasoning: List[str] = []

            steps = min(len(generated_paths), len(reference_paths))
            if steps == 0:
                continue

            for step_idx in range(steps):
                reference_path = reference_paths[step_idx]
                candidate_path = generated_paths[step_idx]
                if reference_path is None or not candidate_path.exists():
                    per_step_scores.append(math.nan)
                    per_step_reasoning.append("missing reference or candidate image")
                    continue
                try:
                    result = scorer.score_step(
                        reference_image=reference_path,
                        candidate_image=candidate_path,
                        action_label=candidate_actions[step_idx] if step_idx < len(candidate_actions) else "",
                        step_index=step_idx + 1,
                        total_steps=steps,
                        sample_uid=sample.sample_uid,
                        dataset=sample.dataset,
                        plan_description=plan_description,
                    )
                    per_step_scores.append(result.score)
                    per_step_reasoning.append(result.reasoning)
                except Exception as exc:
                    per_step_scores.append(math.nan)
                    per_step_reasoning.append(f"gpt_error: {exc}")

            valid_scores = [score for score in per_step_scores if not math.isnan(score)]
            final_score = math.nan
            for idx in range(len(per_step_scores) - 1, -1, -1):
                if not math.isnan(per_step_scores[idx]):
                    final_score = per_step_scores[idx]
                    break
            avg_score = float(sum(valid_scores) / len(valid_scores)) if valid_scores else math.nan

            candidate_metrics.append(
                CandidateGPTMetrics(
                    index=candidate_idx,
                    per_step_scores=per_step_scores,
                    per_step_reasoning=per_step_reasoning,
                    final_score=final_score,
                    avg_score=avg_score,
                )
            )

        if not candidate_metrics:
            continue

        candidate_metrics.sort(key=lambda cm: cm.index)
        per_sample.append(
            {
                "sample_uid": sample.sample_uid,
                "ground_truth_index": sample.ground_truth_index,
                "candidates": [cm.to_dict() for cm in candidate_metrics],
            }
        )
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

        gt_candidate = next((cm for cm in candidate_metrics if cm.index == sample.ground_truth_index), None)
        if gt_candidate:
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

    stepwise_gpt_mean = []
    for step_idx in range(max_step_index):
        count = step_counts.get(step_idx, 0)
        if count == 0:
            continue
        mean_score = step_sums[step_idx] / count
        stepwise_gpt_mean.append({
            "step": step_idx + 1,
            "mean": mean_score,
            "count": count,
        })

    overall_gpt_mean = overall_sum / overall_count if overall_count else math.nan

    per_dataset_gpt_mean: Dict[str, Dict[str, object]] = {}
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
        per_dataset_gpt_mean[dataset] = {
            "samples": stats["samples"],
            "overall_gpt_mean": overall_mean,
            "stepwise_gpt_mean": dataset_stepwise,
        }

    summary = {
        "total_samples": len(metadata),
        "evaluated_samples": evaluated_samples,
        "overall_gpt_mean": overall_gpt_mean,
        "stepwise_gpt_mean": stepwise_gpt_mean,
        "per_dataset_gpt_mean": per_dataset_gpt_mean,
        "samples": per_sample,
        "usage": scorer.summarise_cost(),
    }
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPT scoring for WorldPrediction rollouts")
    parser.add_argument("--metadata_path", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--frame_map_path", type=Path, default=None)
    parser.add_argument("--generated_root", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--dataset_filter", type=str, nargs="*", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
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
        raise RuntimeError("No samples available for GPT evaluation")

    frame_map: Dict[str, str] = {}
    if args.frame_map_path and args.frame_map_path.exists():
        with args.frame_map_path.open("r", encoding="utf-8") as handle:
            frame_map = json.load(handle)

    resolver = FrameResolver(
        dataset_root=args.dataset_root,
        frame_map=frame_map,
        cache_dir=args.frame_cache_dir,
    )

    scorer = GPTSequenceJudge(model_name=args.model_name)
    scorer.load()

    summary = evaluate(
        metadata=metadata,
        resolver=resolver,
        generated_root=args.generated_root,
        scorer=scorer,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
