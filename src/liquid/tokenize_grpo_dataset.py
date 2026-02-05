import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import PIL
import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoTokenizer
from datasets import Dataset

from image_tokenizer import ChameleonImageTokenizer
from utils import tokenizer_image_token, LIQUID_IMAGE_TOKEN_INDEX, expand2square
from data.dataset_picobanana400k import PicoBanana400KEditingDataset
from data.dataset_video import UnsupervisedVideoEditingDataset
from data.dataset_aurora import AURORAForwardDynamicsDataset
from data.dataset_vidgen import UnsupervisedVIDGen1MEditingDataset
from liquid.VQA_Eval.conversation import conv_templates

class LiquidGRPOPreprocessor:
    """
    Preprocesses datasets for Liquid GRPO Training.
    Supports two modes:

    1. MODE 'fdp':
       - Policy: Generates Next Image (Target) given Source + Action.
       - Reward: IDP Model (Predicts Action likelihood given Source + Gen Target).
       - Output: fdp_prompt_ids, action_ids

    2. MODE 'idp':
       - Policy: Generates Action Text given Source + Target Images.
       - Reward: FDP Model (Predicts Target likelihood given Source + Gen Action).
       - Output: idp_prompt_ids, source_image_ids, target_image_ids
    """

    def __init__(self, model_path: str, vae_config_path: str, vae_checkpoint_path: str, device: str = "cuda"):
        self.device = device

        self.text_tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        self.vocab_size = len(self.text_tokenizer)

        self.image_tokenizer = ChameleonImageTokenizer(
            cfg_path=vae_config_path,
            ckpt_path=vae_checkpoint_path,
            device=device
        )
        self.image_vocab_offset = self.vocab_size

        print(f"Initialized Preprocessor. Text Vocab: {self.vocab_size}")

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        """Standard Liquid Image Preprocessing"""
        image = image.convert('RGB')
        image = expand2square(image, (122, 116, 104))
        image = image.resize((512, 512), PIL.Image.LANCZOS)
        return image

    def _tokenize_image(self, image: Image.Image) -> torch.Tensor:
        """Encodes image to discrete tokens with offset. Returns CPU tensor."""
        with torch.no_grad():
            vq_codes = self.image_tokenizer.encode([image])[0]
        return vq_codes.cpu() + self.image_vocab_offset


    def sample_fdp_instructions(self):
        instructions = [
            "Modify the provided image based on the given instruction:",
            "Make changes to the image following the instruction:",
            "Alter the given image as described in the instruction:",
            "Update the image according to the specified instruction:",
            "Transform the provided image following the instruction:",
            "Apply the instruction to edit the given image:",
            "Adjust the image based on the provided instruction:",
            "Carry out the edits on the image as instructed:",
            "Revise the given image according to the instruction:",
            "Follow the instruction to modify the provided image:"
        ]
        return random.choice(instructions)

    def _process_sample_fdp(self, raw_sample: Dict[str, Union[Image.Image, str]]) -> Optional[Dict[str, Any]]:
        """
        Processing for FDP Policy Training.
        Policy Input: <boi>Img1<eoi> \n Instr Action <boi> -> Generates Img2
        """
        try:
            input_image = raw_sample["input_image"]
            textual_action = raw_sample["textual_action"]
        except KeyError:
            return None

        input_image_proc = self._prepare_image(input_image)
        input_vq_codes = self._tokenize_image(input_image_proc)

        instruction = self.sample_fdp_instructions()
        text_prompt = f'<boi><image><eoi>\n{instruction} {textual_action}<boi>'

        text_ids = tokenizer_image_token(
            text_prompt,
            self.text_tokenizer,
            LIQUID_IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        )

        image_token_idx = torch.where(text_ids == LIQUID_IMAGE_TOKEN_INDEX)[0]
        if image_token_idx.numel() == 0:
            return None
        image_token_idx = image_token_idx[0].item()

        fdp_prompt_ids = torch.cat([
            text_ids[:image_token_idx],
            input_vq_codes,
            text_ids[image_token_idx+1:]
        ], dim=0)

        action_ids = self.text_tokenizer.encode(textual_action, add_special_tokens=False)

        return {
            "fdp_prompt_ids": fdp_prompt_ids.numpy(),
            "action_ids": np.array(action_ids),
            "raw_action": textual_action
        }


    def sample_idp_instructions(self):
        instructions = [
            "Given two consecutive images, identify and briefly describe the action that most likely caused the change from the first to the second. Use clear, concise natural language to explain the state transition.",
            "You will receive a pair of sequential images. Your goal is to infer the action that occurred between them and summarize it in a short, natural language description highlighting the key change.",
            "Observe the two provided images in order. Determine the most probable action that led from the first image to the second, and describe it clearly and succinctly in natural language.",
            "Given two consecutive visual observations, infer what action took place between them. Express this action in a concise natural language description capturing the essential change.",
            "You are shown two sequential frames. Identify the likely action that occurred between them and describe it briefly in natural language, focusing on the key transformation.",
            "Look at the two images in sequence and deduce the action that caused the transition. Provide a concise, natural language description that explains the change.",
            "Given a past and a next image, infer the action responsible for the transition and describe it succinctly in natural language, highlighting the critical difference.",
            "You have two sequential images. Determine the action that occurred between them and summarize it in a brief, clear natural language statement that captures the main change.",
            "Observe the images in order and identify the most likely action that caused the transition. Describe this action concisely in natural language, emphasizing the key differences.",
            "Given a pair of sequential visual observations, infer the action that explains the change and describe it in a short, clear natural language sentence highlighting the essential effect."
        ]
        return random.choice(instructions)

    def _process_sample_idp(self, raw_sample: Dict[str, Union[Image.Image, str]]) -> Optional[Dict[str, Any]]:
        """
        Processing for IDP Policy Training.
        Policy Input: User: <boi>Img1<eoi><boi>Img2<eoi> Instr Model: -> Generates Action
        Reward Input: Img1, Img2 (For Frozen FDP to evaluate action)
        """
        try:
            input_image = raw_sample["input_image"]
            output_image = raw_sample["output_image"]                                  
            textual_action = raw_sample["textual_action"]
        except KeyError:
            return None

        input_image_proc = self._prepare_image(input_image)
        output_image_proc = self._prepare_image(output_image)

        input_vq_codes = self._tokenize_image(input_image_proc)           
        output_vq_codes = self._tokenize_image(output_image_proc)         

        vq_codes_list = [input_vq_codes, output_vq_codes]

        instruction = self.sample_idp_instructions()

        prompt_content = '<boi><image><eoi>' * 2 + '\n' + instruction

        conv = conv_templates['gemma'].copy()
        conv.append_message(conv.roles[0], prompt_content)
        conv.append_message(conv.roles[1], None)
        text_prompt = conv.get_prompt()

        text_ids = tokenizer_image_token(
            text_prompt,
            self.text_tokenizer,
            LIQUID_IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        )

        image_token_indices = [-1] + torch.where(text_ids == LIQUID_IMAGE_TOKEN_INDEX)[0].tolist() + [text_ids.shape[0]]

        if len(vq_codes_list) != (len(image_token_indices) - 2):
            return None

        cur_prefix_ids = []
        for i in range(len(vq_codes_list) + 1):
            cur_prefix_ids.append(text_ids[image_token_indices[i]+1 : image_token_indices[i+1]])
            if i < len(vq_codes_list):
                cur_prefix_ids.append(vq_codes_list[i])

        idp_prompt_ids = torch.cat(cur_prefix_ids, dim=0)

        action_ids = self.text_tokenizer.encode(textual_action, add_special_tokens=False)

        return {
            "idp_prompt_ids": idp_prompt_ids.numpy(),
            "source_image_ids": input_vq_codes.numpy(),                                  
            "target_image_ids": output_vq_codes.numpy(),                                 
            "action_ids": np.array(action_ids)                                
        }


    def process_dataset(self, dataset: Any, output_dir: Path, mode: str, debug_sample_size: Optional[int] = None):
        output_dir.mkdir(parents=True, exist_ok=True)

        indices = range(len(dataset))
        if debug_sample_size:
            indices = range(min(len(dataset), debug_sample_size))

        data_dict = defaultdict(list)

        print(f"Processing {len(indices)} samples for GRPO (Mode: {mode})...")

        success_count = 0
        for i in tqdm(indices):
            raw_sample = dataset[i]

            if mode == "fdp":
                processed = self._process_sample_fdp(raw_sample)
            elif mode == "idp":
                processed = self._process_sample_idp(raw_sample)
            else:
                raise ValueError("Invalid mode")

            if processed:
                for k, v in processed.items():
                    data_dict[k].append(v)
                success_count += 1

        if success_count == 0:
            raise ValueError("No samples processed successfully.")

        print(f"Saving {success_count} samples to {output_dir}")
        hf_dataset = Dataset.from_dict(data_dict)
        hf_dataset.save_to_disk(str(output_dir))

def main():
    parser = argparse.ArgumentParser(description="Liquid GRPO Dataset Preprocessing")

    parser.add_argument('--dataset', type=str, required=True, choices=['pico-banana', 'aurora', 'mit', 'ucf', 'kinetics', 'vidgen-1m'])
    parser.add_argument('--base_dir', type=Path, required=True)
    parser.add_argument('--subset', type=str, default="default", help="Subset for Aurora/Pico")

    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--vae_config_path', type=str, required=True)
    parser.add_argument('--vae_checkpoint_path', type=str, required=True)

    parser.add_argument('--output_dir_name', type=str, required=True)
    parser.add_argument('--debug_sample_size', type=int, default=None)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument('--mode', type=str, required=True, choices=['fdp', 'idp'],
                        help="fdp: Prepares data for training Policy=FDP, Reward=IDP. \n"
                             "idp: Prepares data for training Policy=IDP, Reward=FDP.")

    parser.add_argument('--min_likelihood_filter', type=float, default=-2.0, help="Only use for our annotated video dataset.")

    parser.add_argument('--annotation_file', type=str, required=False, default=None, help="Name for annotated files. Format: train_{IDP_NAME}_{TURN_IDX}.jsonl, e.g., train_Liquid_V1_7B-pico-aurora-multiturn-sft_0.jsonl.")

    args = parser.parse_args()

    OUTPUT_DIR = args.base_dir / args.output_dir_name

    preprocessor = LiquidGRPOPreprocessor(
        model_path=args.model_path,
        vae_config_path=args.vae_config_path,
        vae_checkpoint_path=args.vae_checkpoint_path,
        device=args.device
    )

    print(f"Loading dataset: {args.dataset} (Subset: {args.subset})")
    if args.dataset == "pico-banana":
        data_path = args.base_dir / "sft.jsonl"
        dataset = PicoBanana400KEditingDataset(
            data_file_path=data_path,
            base_dir=args.base_dir
        )
    elif args.dataset == "aurora":
        dataset = AURORAForwardDynamicsDataset(
            dataset_name=args.subset,
            base_dir=args.base_dir
        )
    elif args.dataset in ["ucf", "kinetics", "mit"]:
        dataset = UnsupervisedVideoEditingDataset(
            base_dir=args.base_dir,
            datasets_to_load=[args.dataset],
            min_likelihood=args.min_likelihood_filter
        )
    elif args.dataset in ["vidgen-1m"]:
        dataset = UnsupervisedVIDGen1MEditingDataset(
            base_dir=args.base_dir,
            min_likelihood=-2.0,
            annotation_file=args.annotation_file
        )
    else:
        raise NotImplementedError("Dataset not supported")

    preprocessor.process_dataset(
        dataset=dataset,
        output_dir=OUTPUT_DIR,
        mode=args.mode,
        debug_sample_size=args.debug_sample_size
    )

if __name__ == "__main__":
    main()
