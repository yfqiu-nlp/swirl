import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import PIL
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

from image_tokenizer import ChameleonImageTokenizer
from utils import tokenizer_image_token, LIQUID_IMAGE_TOKEN_INDEX, expand2square
from data.dataset_picobanana400k import PicoBanana400KEditingDataset, PicoBananaMultiTurnDataset
from data.dataset_aurora import AURORAForwardDynamicsDataset
from data.dataset_video import UnsupervisedVideoEditingDataset
from datasets import Dataset
from liquid.VQA_Eval.conversation import conv_templates

class LiquidPreprocessor:
    """
    Preprocesses the PicoBanana400K image editing dataset into token sequences 
    suitable for Liquid's next-token prediction training objective.
    """
    
    def __init__(self, model_path : str, vae_config_path: str, vae_checkpoint_path: str, device: str = "cuda"):
        """Initializes tokenizers and configuration."""
        self.device = device
        
        self.text_tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                padding_side='left'
            )
        self.vocab_size = len(self.text_tokenizer)
        
        # NOTE: The VQ-VAE model should be on the target device (e.g., 'cuda') for encoding efficiency.
        self.image_tokenizer = ChameleonImageTokenizer(
            cfg_path=vae_config_path,
            ckpt_path=vae_checkpoint_path,
            device=device
        )
        self.image_vocab_offset = self.vocab_size
        self.pad_token_id = self.text_tokenizer.pad_token_id
        self.eos_token_id = self.text_tokenizer.eos_token_id
        self.eoi_token_id = self.text_tokenizer.convert_tokens_to_ids('<eoi>')
        
        print(f"Text Vocab Size: {self.vocab_size}")

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        """
        Applies the exact same image preprocessing steps as in the Liquid inference code.
        (Convert to RGB, expand to square, resize to 512x512).
        """
        image = image.convert('RGB')
        
        image = expand2square(image, (122, 116, 104))
        
        image = image.resize((512, 512), PIL.Image.LANCZOS)
        
        return image

    def _tokenize_image(self, image: Image.Image) -> torch.Tensor:
        """
        Tokenizes a single, preprocessed image using the VQ-VAE and applies the offset.
        Returns a 1D tensor of token IDs (shape: [seq_len]).
        """
        with torch.no_grad():
            vq_codes = self.image_tokenizer.encode([image])[0]                                  
            
        vq_codes = vq_codes.cpu() + self.image_vocab_offset
        
        return vq_codes

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

    def _process_fdp_sample(self, raw_sample: Dict[str, Union[Image.Image, str]]) -> Optional[Dict[str, torch.Tensor]]:
        """
        Processes one raw sample (image, text, image) into the Liquid token sequence.
        
        Target Sequence Format (Input + Label):
        <boi><image_tokens_input><eoi>\n{text_instruction}<boi><image_tokens_output><eoi><eos>
        
        The entire sequence is used for both input_ids and labels (shifted).
        """
        try:
            input_image = raw_sample["input_image"]
            output_image = raw_sample["output_image"]
            textual_action = raw_sample["textual_action"]
        except KeyError:
            print("Warning: Sample is missing required keys and will be skipped.")
            return None
        
        input_image_proc = self._prepare_image(input_image)
        output_image_proc = self._prepare_image(output_image)
        
        input_vq_codes = self._tokenize_image(input_image_proc)          
        output_vq_codes = self._tokenize_image(output_image_proc)          
        
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
            print(f"Warning: Image placeholder not found in tokenized text: {text_prompt[:50]}...")
            return None
        image_token_idx = image_token_idx[0].item()
        
        prefix_ids = torch.cat([
            text_ids[:image_token_idx],                                   
            input_vq_codes,                                              
            text_ids[image_token_idx+1:]                                                              
        ], dim=0)
        
        eos_token_id = self.text_tokenizer.eos_token_id
        eoi_token_id = self.text_tokenizer.encode('<eoi>')[-1]
        
        full_sequence = torch.cat([
            prefix_ids,                                                                                          
            output_vq_codes,                                              
            torch.tensor([eoi_token_id], dtype=torch.long),            
            torch.tensor([eos_token_id], dtype=torch.long)            
        ], dim=0)            
        
        
        labels = full_sequence.clone()
        generation_start_idx = prefix_ids.shape[0]
        labels[:generation_start_idx] = -100
        
        attention_mask = torch.ones_like(full_sequence, dtype=torch.long)
        
        return {
            "input_ids": full_sequence,                     
            "labels": labels,                                                  
            "attention_mask": attention_mask                                       
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

    def _process_idp_sample(self, raw_sample: Dict[str, Union[Image.Image, str]]) -> Optional[Dict[str, torch.Tensor]]:
        """
        Processes one raw sample (image, text, image) into the Liquid token sequence.
        
        Target Sequence Format (Input + Label):
        <boi><image_tokens_input><eoi><boi><image_tokens_output><eos>{prompt}\n{text_instruction}
        
        The entire sequence is used for both input_ids and labels (shifted).
        """
        try:
            input_image = raw_sample["input_image"]
            output_image = raw_sample["output_image"]
            textual_action = raw_sample["textual_action"]
        except KeyError:
            print("Warning: Sample is missing required keys and will be skipped.")
            return None
        
        input_image_proc = self._prepare_image(input_image)
        output_image_proc = self._prepare_image(output_image)
        
        input_vq_codes = self._tokenize_image(input_image_proc)          
        output_vq_codes = self._tokenize_image(output_image_proc)          
        vq_codes = [input_vq_codes, output_vq_codes]
        
        instruction = self.sample_idp_instructions()
        prompt = '<boi><image><eoi>' * 2 + '\n' + instruction
        conv = conv_templates['gemma'].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        text_prompt = conv.get_prompt()
        
        text_ids = tokenizer_image_token(
            text_prompt, 
            self.text_tokenizer, 
            LIQUID_IMAGE_TOKEN_INDEX, 
            return_tensors='pt'
        )                            
        
        num_images = (text_ids == LIQUID_IMAGE_TOKEN_INDEX).sum()
        image_token_indices = [-1] + torch.where(text_ids == LIQUID_IMAGE_TOKEN_INDEX)[0].tolist() + [text_ids.shape[0]]
        
        cur_prefix_ids = []
        for i in range(num_images + 1):
            cur_prefix_ids.append(text_ids[image_token_indices[i]+1:image_token_indices[i+1]])
            if i < num_images:
                cur_prefix_ids.append(vq_codes[i])
        prefix_ids = torch.cat(cur_prefix_ids, dim=0)
        
        eos_token_id = self.text_tokenizer.eos_token_id

        response_ids = self.text_tokenizer.encode(textual_action)[1:]           
        
        full_sequence = torch.cat([
            prefix_ids,                                                                                          
            torch.tensor(response_ids, dtype=torch.long),                                                 
            torch.tensor([eos_token_id], dtype=torch.long)            
        ], dim=0)            
        
        labels = full_sequence.clone()
        generation_start_idx = prefix_ids.shape[0]
        labels[:generation_start_idx] = -100
        
        attention_mask = torch.ones_like(full_sequence, dtype=torch.long)
        
        return {
            "input_ids": full_sequence,                     
            "labels": labels,                                                  
            "attention_mask": attention_mask                                       
        }

    def _process_multiturn_sample(self, raw_sample: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        """
        Processes a multi-turn editing session.
        Input raw_sample keys: "images" (List[PIL.Image]), "prompts" (List[str])
        """
        # NOTE: Dataset now returns loaded PIL images, not paths
        images = raw_sample.get("images", [])
        prompts = raw_sample.get("prompts", [])
        
        if not images or not prompts:
            return None
            
        try:
            image_tokens_list = []
            for img in images:
                processed_img = self._prepare_image(img)
                tokens = self._tokenize_image(processed_img)
                image_tokens_list.append(tokens)
        except Exception as e:
            print(f"Error processing images in multi-turn sample: {e}")
            return None

        boi_id = self.text_tokenizer.encode('<boi>')[-1]
        eoi_id = self.text_tokenizer.encode('<eoi>')[-1]
        
        sequence_parts = []
        mask_parts = []                                 
        
        ctx_header = torch.tensor([boi_id], dtype=torch.long)
        ctx_footer = torch.tensor([eoi_id], dtype=torch.long)
        
        current_seq = torch.cat([ctx_header, image_tokens_list[0], ctx_footer], dim=0)
        current_mask = torch.zeros(current_seq.shape[0], dtype=torch.bool)               
        
        sequence_parts.append(current_seq)
        mask_parts.append(current_mask)
        
        for i, prompt_text in enumerate(prompts):
            target_image_tokens = image_tokens_list[i + 1]
            
            instruction = self.sample_fdp_instructions()
            formatted_prompt = f"\n{instruction} {prompt_text}<boi>"
            
            text_ids = self.text_tokenizer.encode(formatted_prompt, add_special_tokens=False)
            text_tensor = torch.tensor(text_ids, dtype=torch.long)
            
            text_mask = torch.ones(text_tensor.shape[0], dtype=torch.bool)
            
            image_block = torch.cat([target_image_tokens, ctx_footer], dim=0)
            image_mask = torch.ones(image_block.shape[0], dtype=torch.bool)               
            
            sequence_parts.extend([text_tensor, image_block])
            mask_parts.extend([text_mask, image_mask])

        eos_tensor = torch.tensor([self.eos_token_id], dtype=torch.long)
        eos_mask = torch.ones(1, dtype=torch.bool)
        
        sequence_parts.append(eos_tensor)
        mask_parts.append(eos_mask)
        
        full_input_ids = torch.cat(sequence_parts, dim=0)
        full_mask_bool = torch.cat(mask_parts, dim=0)
        
        labels = full_input_ids.clone()
        labels[~full_mask_bool] = -100
        
        attention_mask = torch.ones_like(full_input_ids, dtype=torch.long)
        
        return {
            "input_ids": full_input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }


    def preprocess_dataset(self, task_type, dataset: Any, output_dir: Union[str, Path], debug_sample_size: Optional[int] = None) -> Path:
        """
        Main processing loop.
        """
        output_dir = Path(output_dir)
        total_size = len(dataset)
        indices = range(debug_sample_size) if debug_sample_size else range(total_size)
        
        if debug_sample_size:
            output_dir = output_dir.with_name(output_dir.name + "_debug")
            
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenized_data = defaultdict(list)
        
        print(f"Start processing {task_type} with {len(indices)} samples...")
        
        for i in tqdm(indices, desc="Tokenizing"):
            
            raw_sample = dataset[i]
            processed = None
            
            if task_type == "fdp":                  
                processed = self._process_fdp_sample(raw_sample)
            elif task_type == "idp":                   
                processed = self._process_idp_sample(raw_sample)
            elif task_type == "multiturn":                 
                processed = self._process_multiturn_sample(raw_sample)
            
            if processed:
                tokenized_data["input_ids"].append(processed["input_ids"].numpy())
                tokenized_data["labels"].append(processed["labels"].numpy())
                tokenized_data["attention_mask"].append(processed["attention_mask"].numpy())
                    

        if len(tokenized_data["input_ids"]) == 0:
            raise ValueError("No samples were successfully processed!")

        hf_dataset = Dataset.from_dict(tokenized_data)
        hf_dataset.save_to_disk(str(output_dir))
        print(f"Saved {len(hf_dataset)} samples to {output_dir}")
        return output_dir


def main():
    parser = argparse.ArgumentParser(description="Liquid Dataset Preprocessing")
    
    parser.add_argument('--dataset', type=str, required=True, choices=['pico-banana', 'aurora', 'mit', 'kinetics', 'ucf'], help="Dataset name")
    parser.add_argument('--base_dir', type=Path, required=True, help="Dataset root")
    parser.add_argument('--model_path', type=str, required=True, help="LLM path")
    parser.add_argument('--task_type', type=str, required=True, choices=['fdp', 'idp', 'multiturn'], help="Task type")
    parser.add_argument('--vae_config_path', type=str, required=True)
    parser.add_argument('--vae_checkpoint_path', type=str, required=True)
    parser.add_argument('--data_file_name', type=str, default="sft.jsonl")
    parser.add_argument('--output_dir_name', type=str, default="liquid_tokenized_data")
    parser.add_argument('--debug_sample_size', type=int, default=None)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--min_likelihood_filter', type=float, default=-2.0, help="Only use for our annotated video dataset, apply filter with predicted likelihood.") 
    
    args = parser.parse_args()
    
    DATA_PATH = args.base_dir / args.data_file_name
    OUTPUT_DIR = args.base_dir / args.output_dir_name

    preprocessor = LiquidPreprocessor(
        model_path=args.model_path,
        vae_config_path=args.vae_config_path,
        vae_checkpoint_path=args.vae_checkpoint_path,
        device=args.device
    )
    
    if args.dataset == "pico-banana":
        if args.task_type == "multiturn":
            dataset = PicoBananaMultiTurnDataset(
                data_file_path=DATA_PATH,
                base_dir=args.base_dir
            )
        else:
            dataset = PicoBanana400KEditingDataset(
                data_file_path=DATA_PATH,
                base_dir=args.base_dir
            )
    elif args.dataset == "aurora":
        dataset = AURORAForwardDynamicsDataset(
            dataset_name="default", 
            base_dir=args.base_dir
        )
    elif args.dataset == "ucf" or args.dataset == "kinetics" or args.dataset == "mit":
        dataset = UnsupervisedVideoEditingDataset(
            base_dir=args.base_dir, 
            datasets_to_load=[args.dataset],
            min_likelihood=args.min_likelihood_filter
        )
    else:
        raise NotImplementedError("Dataset not supported")

    preprocessor.preprocess_dataset(
        task_type=args.task_type,
        dataset=dataset,
        output_dir=OUTPUT_DIR,
        debug_sample_size=args.debug_sample_size
    )

if __name__ == "__main__":
    main()
