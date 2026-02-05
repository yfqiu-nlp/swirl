import argparse
import json
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

try:
    from utils import tokenizer_image_token, LIQUID_IMAGE_TOKEN_INDEX, expand2square
    from liquid.VQA_Eval.conversation import conv_templates
    from image_tokenizer import ChameleonImageTokenizer
except ImportError as e:
    print(f"Critical Error: Could not import project modules. {e}")
    sys.exit(1)


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


def process_shard(engine: LiquidIDPEngine, shard_path: Path, output_file: Path, base_path: Path):

    if output_file.exists():
        logger.warning(f"Output file {output_file} already exists. Overwriting...")

    logger.info(f"Loading shard metadata from {shard_path}")
    with open(shard_path, 'r') as f:
        video_rel_paths = json.load(f)

    logger.info(f"Starting annotation of {len(video_rel_paths)} samples...")

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for rel_path_str in tqdm(video_rel_paths, desc="Annotating"):

            video_dir = base_path / rel_path_str
            img1 = video_dir / "frame_0.png"
            img2 = video_dir / "frame_1.png"

            if not (img1.exists() and img2.exists()):
                logger.error(f"Images missing for {rel_path_str}. Skipping.")
                continue

            try:
                parts = Path(rel_path_str).parts
                video_split = parts[0] if len(parts) > 0 else "unknown"
                video_name = parts[1] if len(parts) > 1 else video_dir.name

                sample_id = f"VIDGEN-1M_{video_split}_{video_name}_000"

                action_text, likelihood = engine.infer_action(
                    [str(img1), str(img2)],
                    IDP_PROMPT
                )

                record = {
                    "id": sample_id,
                    "input_image_path": str(img1.relative_to(base_path)),
                    "output_image_path": str(img2.relative_to(base_path)),
                    "textual_action": action_text,
                    "predicted_likelihood": likelihood,
                    "video_class": video_split,
                }

                f_out.write(json.dumps(record) + "\n")
                f_out.flush()

            except Exception as e:
                logger.error(f"Failed to annotate {rel_path_str}: {e}")

    logger.info(f"Shard annotation complete. Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Liquid VidGen-1M Annotation Worker")

    parser.add_argument('--base_path', type=str, required=True, help="Path to VIDGEN-1M_images root")
    parser.add_argument('--shard_meta', type=str, required=True, help="Path to the input meta_shard_X.json file")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output .jsonl file")

    parser.add_argument('--checkpoint_path', type=str, required=True, help="Model checkpoint path to use for this run")
    parser.add_argument('--vae_config_path', type=str, default="vision_tokenizers/chameleon/vqgan.yaml")
    parser.add_argument('--vae_checkpoint_path', type=str, default="vision_tokenizers/chameleon/vqgan.ckpt")

    args = parser.parse_args()

    engine = LiquidIDPEngine(
        model_path=args.checkpoint_path,
        vae_config=args.vae_config_path,
        vae_ckpt=args.vae_checkpoint_path
    )

    process_shard(
        engine=engine,
        shard_path=Path(args.shard_meta),
        output_file=Path(args.output_file),
        base_path=Path(args.base_path)
    )
