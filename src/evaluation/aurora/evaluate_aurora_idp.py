#!/usr/bin/env python

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from PIL import Image
from tqdm import tqdm
from transformers import GenerationConfig

try:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from bert_score import score as bert_score
except ImportError:
    print("Warning: 'pycocoevalcap' not installed. NLP metrics will be skipped.")

try:
    from openai import OpenAI
except ImportError:
    print("Warning: 'openai' not installed. GPT Judge will be skipped.")

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))
try:
    from src.inferencer import create_inference_engine
except ImportError:
    print("Error: Could not import 'src.inferencer'. Run this script from the Liquid model root.")
    sys.exit(1)


def load_aurora_metadata(dataset_root: Path, split_file: str = "test.json", tasks: Optional[List[str]] = None):
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

        if "input" not in entry:
            continue

        if "output" not in entry:
            continue

        source_path = dataset_root / entry["input"]
        target_path = dataset_root / entry["output"]

        if not source_path.exists() or not target_path.exists():
            continue

        sample_id = f"aurora_{idx:05d}"
        gt_action = entry.get("instruction", entry.get("prompt", ""))

        metadata.append({
            "id": sample_id,
            "task": task_name,
            "source_path": str(source_path),
            "target_path": str(target_path),
            "gt_action": gt_action
        })

    print(f"Total valid IDP samples loaded: {len(metadata)}")
    return metadata


class LiquidIDPWrapper:
    def __init__(self, vision_tokenizer_path: str, checkpoint_path: str, device_id: int):
        vae_config = os.path.join(vision_tokenizer_path, "vqgan.yaml")
        vae_ckpt = os.path.join(vision_tokenizer_path, "vqgan.ckpt")

        print(f"Loading VQGAN from: {vae_config}")

        self.engine = create_inference_engine(
            model_type="liquid",
            model_path=checkpoint_path,
            vae_config_path=vae_config,
            vae_checkpoint_path=vae_ckpt,
            device=f"cuda:{device_id}"
        )

        self.generation_config = GenerationConfig(
            temperature=0.2,
            top_k=40,
            top_p=0.9,
            max_new_tokens=128,
            do_sample=False
        )

    def predict(self, source_img: Image.Image, target_img: Image.Image) -> str:
        output_text = self.engine.generate_image_to_text(
            prompt="Given two consecutive images, identify and briefly describe the action that most likely caused the change from the first to the second. Use clear, concise natural language to explain the state transition.",
            images=[source_img, target_img],
            gen_config=self.generation_config
        )
        return output_text


from rouge_score import rouge_scorer

def compute_nlp_metrics(predictions):
    print("\n--- Computing Standard NLP Metrics ---")

    gts = {i: [d["gt_action"]] for i, d in enumerate(predictions)}
    res = {i: [d["pred_action"]] for i, d in enumerate(predictions)}

    scores_dict = {}

    try:
        bleu_scorer = Bleu(4)
        bleu_scores, _ = bleu_scorer.compute_score(gts, res)
        for i, key in enumerate(["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]):
            scores_dict[key] = float(bleu_scores[i])
    except Exception as e:
        print(f"BLEU failed: {e}")

    try:
        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(gts, res)
        scores_dict["CIDEr"] = float(cider_score)
    except Exception as e:
        print(f"CIDEr failed: {e}")

    try:
        rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )

        r1_list, r2_list, rl_list = [], [], []

        for i in range(len(predictions)):
            pred = predictions[i]["pred_action"]
            gt   = predictions[i]["gt_action"]

            r = rouge.score(gt, pred)

            r1_list.append(r["rouge1"].fmeasure)
            r2_list.append(r["rouge2"].fmeasure)
            rl_list.append(r["rougeL"].fmeasure)

        scores_dict["ROUGE_1"] = float(sum(r1_list) / len(r1_list))
        scores_dict["ROUGE_2"] = float(sum(r2_list) / len(r2_list))
        scores_dict["ROUGE_L"] = float(sum(rl_list) / len(rl_list))

    except Exception as e:
        print(f"ROUGE failed: {e}")

    try:
        preds = [d["pred_action"] for d in predictions]
        refs = [d["gt_action"] for d in predictions]

        _, _, f1 = bert_score(preds, refs, lang="en", rescale_with_baseline=True)
        scores_dict["BERTScore_F1"] = float(f1.mean().item())

    except Exception as e:
        print(f"BERTScore failed: {e}")

    return scores_dict


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def run_gpt_judge(predictions, sample_size, api_key):
    if not api_key:
        print("Skipping GPT Judge (No API Key).")
        return {}

    print(f"\n--- Running GPT-4o Vision Judge (First {sample_size} samples) ---")
    client = OpenAI(api_key=api_key)

    subset = predictions[:sample_size]
    scores = []

    for entry in tqdm(subset):
        src_b64 = encode_image(entry['source_path'])
        tgt_b64 = encode_image(entry['target_path'])

        prompt = (
            "I will show you two images: 'Before' and 'After', and an action description.\n"
            f"Action Description: \"{entry['pred_action']}\"\n\n"
            "Task: Rate (1-5) how accurately this description explains the physical transition "
            "between the images.\n"
            "1: Wrong/Irrelevant.\n3: Partially correct.\n5: Perfectly accurate.\n"
            "Return JSON: {\"score\": int, \"reason\": \"string\"}"
        )

        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{src_b64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{tgt_b64}"}}
                    ]
                }],
                response_format={"type": "json_object"},
                max_tokens=100
            )
            res = json.loads(resp.choices[0].message.content)
            entry['gpt_score'] = res['score']
            entry['gpt_reason'] = res['reason']
            scores.append(res['score'])
        except Exception as e:
            print(f"GPT Error: {e}")
            time.sleep(1)

    avg = sum(scores) / len(scores) if scores else 0
    return {"GPT4o_Vision_Score": avg}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True, help="Path to save final json")
    parser.add_argument("--devices", type=str, default="0", help="GPU ID to use (e.g. 0)")
    parser.add_argument("--tasks", type=str, nargs="*", default=None)
    parser.add_argument("--num_samples", type=int, default=None, help="Limit inference samples")
    parser.add_argument("--gpt_samples", type=int, default=400, help="Num samples for GPT Judge")
    parser.add_argument("--openai_api_key", type=str, default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--vision_tokenizer_path", type=str, default="vision_tokenizers/chameleon")

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output_file)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = load_aurora_metadata(dataset_root, tasks=args.tasks)
    if args.num_samples:
        metadata = metadata[:args.num_samples]

    if not metadata:
        print("No valid metadata found (0 samples). Exiting.")
        sys.exit(0)

    try:
        device_id = int(args.devices.split(',')[0])
    except:
        device_id = 0

    print(f"Initializing Liquid Model on Single GPU: {device_id}...")

    wrapper = LiquidIDPWrapper(
        vision_tokenizer_path=args.vision_tokenizer_path,
        checkpoint_path=args.model_path,
        device_id=device_id
    )

    print(f"Starting Inference on {len(metadata)} samples...")
    predictions = []

    for entry in tqdm(metadata, desc="Processing"):
        try:
            src = Image.open(entry["source_path"]).convert("RGB")
            tgt = Image.open(entry["target_path"]).convert("RGB")

            pred = wrapper.predict(src, tgt)


            predictions.append({
                **entry,
                "pred_action": pred.strip()
            })
        except Exception as e:
            print(f"Error processing {entry['id']}: {e}")
            continue

    if not predictions:
        print("No predictions generated. Exiting.")
        sys.exit(0)

    summary_metrics = {}

    nlp_scores = compute_nlp_metrics(predictions)
    print("NLP scores:", nlp_scores)
    summary_metrics.update(nlp_scores)

    gpt_scores = run_gpt_judge(predictions, args.gpt_samples, args.openai_api_key)
    print("GPT scores:", gpt_scores)
    summary_metrics.update(gpt_scores)

    final_output = {
        "summary_metrics": summary_metrics,
        "predictions": predictions
    }

    output_file = output_path / "idp_result.json"

    with open(output_file, 'w+') as f:
        json.dump(final_output, f, indent=2)

    print("\n================ EVALUATION REPORT ================")
    for k, v in summary_metrics.items():
        print(f"{k}: {v:.4f}")
    print("===================================================")
    print(f"Detailed results saved to {output_path}")

if __name__ == "__main__":
    main()
