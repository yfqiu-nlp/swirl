#!/usr/bin/env python


import argparse
import ast
import base64
import io
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image
from tqdm import tqdm

from openai import OpenAI


_SCORE_PATTERN = re.compile(r"\[.*?\]")


def _extract_score_list(text: str) -> List[int]:
    """Extract the first list of integers from the LLM response."""

    match = _SCORE_PATTERN.search(text)
    if not match:
        return []
    try:
        parsed = ast.literal_eval(match.group(0))
    except (SyntaxError, ValueError):
        return []
    if isinstance(parsed, Sequence):
        return [int(x) for x in parsed if isinstance(x, (int, float))]
    return []


class GPTVisionInstruct:
    """Wrapper around OpenAI vision models for scoring edits."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.client: OpenAI | None = None
        self.logger = logging.getLogger(__name__)
        self.prompt_tokens: List[int] = []
        self.completion_tokens: List[int] = []

    def load(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set; cannot run GPT evaluation")
        self.client = OpenAI(api_key=api_key)

    _INPUT_COST = {
        "gpt-4o-mini": 0.15,
        "gpt-4o": 2.5,
        "gpt-4-turbo": 10.0,
    }
    _OUTPUT_COST = {
        "gpt-4o-mini": 0.6,
        "gpt-4o": 10.0,
        "gpt-4-turbo": 30.0,
    }

    def _collect_usage(self, completion: Any) -> None:
        usage = getattr(completion, "usage", None)
        if usage is None:
            return
        self.prompt_tokens.append(getattr(usage, "prompt_tokens", 0))
        self.completion_tokens.append(getattr(usage, "completion_tokens", 0))

    def summarise_cost(self) -> Dict[str, float]:
        input_tokens = sum(self.prompt_tokens)
        output_tokens = sum(self.completion_tokens)
        input_cost = self._INPUT_COST.get(self.model_name, 0.0) * input_tokens / 1e6
        output_cost = self._OUTPUT_COST.get(self.model_name, 0.0) * output_tokens / 1e6
        return {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "prompt_cost_usd": input_cost,
            "completion_cost_usd": output_cost,
            "total_cost_usd": input_cost + output_cost,
        }

    def score(self, source: Image.Image, edited: Image.Image, instruction: str) -> Dict[str, Any]:
        if self.client is None:
            raise RuntimeError("GPT client not initialised; call load() first")

        def _encode_image(img: Image.Image) -> str:
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

        payload = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a professional digital artist. You will evaluate how well the "
                            "edited image follows the editing instruction. Provide output strictly "
                            "as a Python dict: {\"score\": [success, overedit], \"reasoning\": \"...\"}.\n"
                            "Two images are provided: first the original, then the edited version.\n"
                            "Score1 (0-10): success of instruction follow-through.\n"
                            "Score2 (0-10): degree of overediting (10 means minimal overediting).\n"
                            f"Editing instruction: {instruction}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_encode_image(source)}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_encode_image(edited)}"},
                    },
                ],
            }
        ]

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=payload,
            max_tokens=512,
            temperature=0,
        )
        self._collect_usage(completion)

        message = completion.choices[0].message.content
        scores = _extract_score_list(message or "")
        return {
            "raw": message,
            "scores": scores,
        }


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



def evaluate_with_gpt(
    dataset_root: Path,
    generated_root: Path,
    metadata: List[Dict[str, Any]],
    model_name: str,
) -> Dict[str, Any]:
    scorer = GPTVisionInstruct(model_name)
    scorer.load()

    task_stats = defaultdict(lambda: {
        "count": 0,
        "scores_success": [],
        "scores_overedit": [],
        "scores_min": [],
        "responses": [],
    })
    missing = []

    for entry in tqdm(metadata, desc="GPT evaluating AURORA"):
        task = entry.get("aurora_task", "aurora").replace("/", "-") or "aurora"
        key = entry["id"]

        gen_path = generated_root / "fullset" / task / f"{key}.png"
        if not gen_path.exists():
            missing.append(str(gen_path))
            continue

        source_path = dataset_root / entry["source_image_path"]
        if not source_path.exists():
            missing.append(str(source_path))
            continue

        source = Image.open(source_path).convert("RGB")
        edited = Image.open(gen_path).convert("RGB")
        instruction = entry.get("Verbalised Action", "")

        result = scorer.score(source, edited, instruction)
        scores = result.get("scores", [])

        record = task_stats[task]
        record["count"] += 1
        record["responses"].append({
            "id": key,
            "instruction": instruction,
            "raw": result.get("raw"),
            "scores": scores,
        })

        if len(scores) >= 2:
            record["scores_success"].append(scores[0])
            record["scores_overedit"].append(scores[1])
            record["scores_min"].append(min(scores[0], scores[1]))

    summary = {}
    overall = {
        "count": 0,
        "scores_success": [],
        "scores_overedit": [],
        "scores_min": [],
    }

    for task, stats in task_stats.items():
        count = stats["count"]
        success_avg = float(sum(stats["scores_success"]) / len(stats["scores_success"])) if stats["scores_success"] else float("nan")
        overedit_avg = float(sum(stats["scores_overedit"]) / len(stats["scores_overedit"])) if stats["scores_overedit"] else float("nan")
        min_avg = float(sum(stats["scores_min"]) / len(stats["scores_min"])) if stats["scores_min"] else float("nan")
        summary[task] = {
            "count": count,
            "avg_edit_success": success_avg,
            "avg_overediting": overedit_avg,
            "avg_min": min_avg,
            "details": stats["responses"],
        }

        overall["count"] += count
        overall["scores_success"].extend(stats["scores_success"])
        overall["scores_overedit"].extend(stats["scores_overedit"])
        overall["scores_min"].extend(stats["scores_min"])

    if overall["count"] > 0:
        summary["overall"] = {
            "count": overall["count"],
            "avg_edit_success": float(sum(overall["scores_success"]) / len(overall["scores_success"])) if overall["scores_success"] else float("nan"),
            "avg_overediting": float(sum(overall["scores_overedit"]) / len(overall["scores_overedit"])) if overall["scores_overedit"] else float("nan"),
            "avg_min": float(sum(overall["scores_min"]) / len(overall["scores_min"])) if overall["scores_min"] else float("nan"),
        }

    summary["missing"] = missing
    summary["cost"] = scorer.summarise_cost()
    return summary


def main():
    parser = argparse.ArgumentParser(description="GPT-based evaluation for AURORA outputs")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--generated_root", type=str, required=True)
    parser.add_argument("--split_file", type=str, default="test.json")
    parser.add_argument("--tasks", type=str, nargs="*", default=None)
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--save_path", type=str, default=None)

    args = parser.parse_args()
    dataset_root = Path(args.dataset_root)
    generated_root = Path(args.generated_root)
    metadata = load_aurora_metadata(dataset_root, split_file=args.split_file)

    if args.tasks:
        wanted = {task.lower() for task in args.tasks}
        metadata = [entry for entry in metadata if entry.get("aurora_task", "").lower() in wanted]
    if args.language:
        metadata = [entry for entry in metadata if entry.get("aurora_raw_record", {}).get("language") == args.language]

    results = evaluate_with_gpt(dataset_root, generated_root, metadata, args.model_name)

    save_path = Path(args.save_path or generated_root / "aurora_eval_metrics_gpt.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    for task, values in results.items():
        if task in {"missing", "cost"}:
            continue
        summary = {k: v for k, v in values.items() if k != "details"}
        print(f"Task: {task}")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
