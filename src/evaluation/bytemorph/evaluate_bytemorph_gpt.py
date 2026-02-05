
import argparse
import base64
import io
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
from openai import OpenAI
from tqdm import tqdm

from src.evaluation.bytemorph.data_utils import load_bytemorph_metadata


PROMPT_TEMPLATE = (
    "You are an evaluator for image editing. You will be given a pair of images before and after editing as well as an editing instruction.\n"
    "You need to rate the editing result with a score between 0 to 100.\n"
    "A successful editing should not miss any change required by editing instruction.\n"
    "A successful editing should not have any extra changes that are not required by editing instruction.\n"
    "The second image should have minimum change to reflect the changes made with EDIT TEXT.\n"
    "Be strict about the changes made between two images.\n"
    "Give the final response in a json format as such:\n"
    "{\n"
    "    \"Score\": xx\n"
    "}\n"
    "Do not output anything else."
)

HUMAN_MESSAGE_TEMPLATE = "EDIT TEXT: {instruction}"

SCORE_REGEX = re.compile(r"\{[^}]*\}")


def _encode_image_jpeg(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=100)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


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


def _extract_score(content: str) -> float | None:
    match = SCORE_REGEX.search(content or "")
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        score = parsed.get("Score")
    except json.JSONDecodeError:
        return None
    if isinstance(score, (int, float)):
        return float(score)
    return None


def evaluate_with_gpt(
    dataset_root: Path,
    generated_root: Path,
    metadata: List[Dict[str, Any]],
    model_name: str,
) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set; cannot run GPT evaluation")

    client = OpenAI(api_key=api_key)

    task_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "count": 0,
        "scores": [],
        "responses": [],
    })
    missing: List[str] = []

    total_prompt_tokens = 0
    total_completion_tokens = 0

    for entry in tqdm(metadata, desc="GPT evaluating ByteMorph"):
        task = (entry.get("bytemorph_edit_type") or "bytemorph").replace("/", "-") or "bytemorph"
        key = entry.get("id", "sample")
        key = key.split('_')[-1]

        save_dir = generated_root / "fullset" / task
        gen_path = save_dir / f"{key}.png"
        if not gen_path.exists():
            missing.append(str(gen_path))
            continue

        try:
            source_image = _ensure_image(entry.get("Source_observation"), dataset_root)
        except Exception:
            missing.append(f"source::{key}")
            continue

        edited_image = Image.open(gen_path).convert("RGB")
        instruction = (entry.get("Verbalised Action") or "").strip()

        content = [
            {"type": "text", "text": HUMAN_MESSAGE_TEMPLATE.format(instruction=instruction)},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_encode_image_jpeg(source_image)}"},
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_encode_image_jpeg(edited_image)}"},
            },
        ]

        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": PROMPT_TEMPLATE},
                    {"role": "user", "content": content},
                ],
                max_tokens=256,
                temperature=0,
            )
        except Exception as exc:                                     
            missing.append(f"api_error::{key}::{exc}")
            continue

        usage = getattr(completion, "usage", None)
        if usage is not None:
            total_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            total_completion_tokens += getattr(usage, "completion_tokens", 0) or 0

        message = completion.choices[0].message.content or ""
        score = _extract_score(message)
        if score is None:
            missing.append(f"parse_error::{key}")
            continue

        stats = task_stats[task]
        stats["count"] += 1
        stats["scores"].append(float(score))
        stats["responses"].append({
            "id": key,
            "instruction": instruction,
            "score": float(score),
            "raw": message,
        })

    summary: Dict[str, Any] = {}
    overall_scores: List[float] = []
    total_count = 0

    for task, info in task_stats.items():
        count = info["count"]
        total_count += count
        avg_score = float(sum(info["scores"]) / count) if count else float("nan")
        summary[task] = {
            "count": count,
            "average_score": avg_score,
            "details": info["responses"],
        }
        overall_scores.extend(info["scores"])

    if total_count:
        summary["overall"] = {
            "count": total_count,
            "average_score": float(sum(overall_scores) / len(overall_scores)) if overall_scores else float("nan"),
        }

    summary["missing"] = missing
    summary["usage"] = {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT-based evaluation for ByteMorph outputs")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--generated_root", type=str, required=True)
    parser.add_argument("--split_file", type=str, default="data/test-*.parquet")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--tasks", type=str, nargs="*", default=None)

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    generated_root = Path(args.generated_root)

    metadata = load_bytemorph_metadata(dataset_root, split_file=args.split_file)
    if args.tasks:
        wanted = {task.lower() for task in args.tasks}
        metadata = [entry for entry in metadata if entry.get("bytemorph_edit_type", "").lower() in wanted]

    results = evaluate_with_gpt(dataset_root, generated_root, metadata, args.model_name)

    save_path = Path(args.save_path or generated_root / "bytemorph_eval_metrics_gpt.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    for task, values in results.items():
        if task in {"missing", "usage"}:
            continue
        summary = {k: v for k, v in values.items() if k != "details"}
        print(f"Task: {task}")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
