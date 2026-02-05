import argparse
import os
import json
import re
import logging
import torch
import PIL
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Any

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig
)

import sys
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))
from utils import tokenizer_image_token, LIQUID_IMAGE_TOKEN_INDEX, expand2square
from liquid.VQA_Eval.conversation import conv_templates
from image_tokenizer import ChameleonImageTokenizer


IDP_PROMPT = (
    "Given two consecutive images, identify and briefly describe the action that "
    "most likely caused the change from the first to the second. "
    "Use clear, concise natural language to explain the state transition."
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("LiquidInference")


class LiquidIDPEngine:
    def __init__(self, model_path: str, vae_config: str, vae_ckpt: str, device: str = "cuda"):
        self.device = device
        
        logger.info(f"Loading Text Tokenizer from {model_path}")
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        
        logger.info(f"Loading Model from {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.model.eval()

        logger.info("Loading Chameleon Image Tokenizer")
        self.image_tokenizer = ChameleonImageTokenizer(
            cfg_path=vae_config,
            ckpt_path=vae_ckpt,
            device=device
        )

    def _process_images(self, image_paths: List[str]) -> Tuple[torch.Tensor, List[Any]]:
        """Liquid-specific image preprocessing: RGB -> Square -> Resize 512."""
        images = []
        for p in image_paths:
            img = Image.open(p).convert('RGB')
            img = expand2square(img, (122, 116, 104))
            img = img.resize((512, 512), PIL.Image.LANCZOS)
            images.append(img)
            
        with torch.no_grad():
            vq_codes = self.image_tokenizer.encode(images).cpu()
            vq_codes = vq_codes + len(self.text_tokenizer)
            
        return vq_codes, images

    @torch.inference_mode()
    def infer_action(self, image_paths: List[str], prompt: str) -> Tuple[str, float]:
        """
        Generates action description and returns (text, likelihood_score).
        Likelihood is the Mean Log-Probability of the generated tokens.
        Higher is better (closer to 0).
        """
        full_prompt = '<boi><image><eoi>' * len(image_paths) + '\n' + prompt
        conv = conv_templates['gemma'].copy()
        conv.append_message(conv.roles[0], full_prompt)
        conv.append_message(conv.roles[1], None)
        final_prompt = conv.get_prompt()

        vq_codes, _ = self._process_images(image_paths)
        vq_codes_list = [vq_codes[i] for i in range(vq_codes.shape[0])]

        text_ids = tokenizer_image_token(
            final_prompt, 
            self.text_tokenizer, 
            LIQUID_IMAGE_TOKEN_INDEX, 
            return_tensors='pt'
        )

        matches = (text_ids == LIQUID_IMAGE_TOKEN_INDEX)
        num_images = matches.sum().item()
        
        image_indices = [-1] + torch.where(matches)[0].tolist() + [text_ids.shape[0]]
        
        input_parts = []
        for i in range(num_images + 1):
            start = image_indices[i] + 1
            end = image_indices[i+1]
            input_parts.append(text_ids[start:end])
            if i < num_images:
                input_parts.append(vq_codes_list[i])
        
        input_ids = torch.cat(input_parts, dim=0).unsqueeze(0).to(self.device)

        gen_config = GenerationConfig(
            max_new_tokens=256,
            do_sample=False,                                            
            output_scores=True,                                         
            return_dict_in_generate=True,
            pad_token_id=self.text_tokenizer.pad_token_id,
            bos_token_id=self.text_tokenizer.bos_token_id,
            eos_token_id=self.text_tokenizer.eos_token_id
        )

        outputs = self.model.generate(input_ids, generation_config=gen_config)

        generated_sequence = outputs.sequences[0][input_ids.shape[1]:]
        text = self.text_tokenizer.decode(generated_sequence, skip_special_tokens=True)

        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        
        sequence_log_prob = transition_scores[0].mean().item()

        return text.strip(), sequence_log_prob


def get_natural_key(file_path: Path):
    """Sorts string containing numbers naturally: motion_2.jpg before motion_10.jpg"""
    name = file_path.name
    nums = re.findall(r'\d+', name)
    if nums:
        return int(nums[-1])                                    
    return name

def process_datasets(engine: LiquidIDPEngine, base_path: Path, datasets):
    
    for dataset in datasets:
        dataset_path = base_path / dataset
        if not dataset_path.exists():
            logger.warning(f"Dataset path not found: {dataset_path}")
            continue

        output_jsonl = dataset_path / "train.jsonl"
        logger.info(f"Processing {dataset} -> Saving to {output_jsonl}")

        video_folders = []
        for root, _, files in os.walk(dataset_path):
            if any(f.endswith(('.jpg', '.png', '.jpeg')) for f in files):
                video_folders.append(Path(root))
        
        video_folders.sort()

        with open(output_jsonl, 'w', encoding='utf-8') as f:
            
            for video_dir in tqdm(video_folders, desc=f"{dataset}"):
                frames = list(video_dir.glob("motion_*.jpg"))
                if not frames:
                    continue
                
                frames.sort(key=get_natural_key)
                
                if len(frames) < 2:
                    continue

                pairs = [(frames[i], frames[i+1]) for i in range(len(frames) - 1)]

                try:
                    rel_parts = video_dir.relative_to(base_path).parts
                    video_class = rel_parts[1] if len(rel_parts) > 1 else "unknown"
                    video_name = rel_parts[2] if len(rel_parts) > 2 else video_dir.name
                except Exception:
                    video_class = video_dir.parent.name
                    video_name = video_dir.name

                for idx, (img1, img2) in enumerate(pairs):
                    try:
                        action_text, likelihood = engine.infer_action(
                            [str(img1), str(img2)], 
                            IDP_PROMPT
                        )

                        record = {
                            "id": f"{dataset}_{video_class}_{video_name}_{idx:03d}",
                            "input_image_path": str(img1.relative_to(base_path)),
                            "output_image_path": str(img2.relative_to(base_path)),
                            "textual_action": action_text,
                            "predicted_likelihood": likelihood,
                            "video_class": video_class,
                        }

                        print(record)
                        
                        f.write(json.dumps(record) + "\n")
                        f.flush()

                    except Exception as e:
                        logger.error(f"Failed to process pair {img1.name}->{img2.name} in {video_name}: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Liquid ucf/kinetics/mit action annotation")
    
    parser.add_argument('--dataset', type=str, required=True, choices=['ucf', 'kinetics', 'mit'])
    
    parser.add_argument('--base_path', type=str, default="datasets/unsupervised-frames-ucf-kinetics-mit")
    parser.add_argument('--checkpoint_path', type=str, default="checkpoints/Liquid_V1_7B")
    parser.add_argument('--vae_config_path', type=str, default="vision_tokenizers/chameleon/vqgan.yaml")
    parser.add_argument('--vae_checkpoint_path', type=str, default="vision_tokenizers/chameleon/vqgan.ckpt")
    
    args = parser.parse_args()
    
    engine = LiquidIDPEngine(
        model_path=args.checkpoint_path,
        vae_config=args.vae_config_path,
        vae_ckpt=args.vae_checkpoint_path
    )

    datasets = [args.dataset]

    process_datasets(engine, Path(args.base_path), datasets)
