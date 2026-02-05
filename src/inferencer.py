"""
Unified Inference Framework for Multimodal Foundation Models
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import PIL
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation import (
    LogitsProcessorList,
    PrefixConstrainedLogitsProcessor,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)
from transformers.image_transforms import to_pil_image

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from config import ModelConfig, ModelType
from image_tokenizer import ChameleonImageTokenizer, Emu3ImageTokenizer, UniTokImageTokenizer
from liquid.VQA_Eval.conversation import conv_templates as liquid_conv_templates
from unitok.mllm.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from unitok.mllm.conversation import conv_templates as unitok_conv_templates
from utils import LIQUID_IMAGE_TOKEN_INDEX, expand2square, sample_token, tokenizer_image_token


class UnifiedMultimodal:
    """
    Supports:
        - Text-to-image generation
        - Image-to-text generation
        - Text+image-to-image generation
    """
    def __init__(self, model_config: ModelConfig):
        """
        Initialize the unified inference engine.
        
        Args:
            model_config: Configuration for the specific model
        """
        self.config = model_config
        self.model_type = model_config.model_type
        self.device = model_config.device
        
        
        self._load_model()
        self._load_tokenizers()
    
    def _load_model(self):
        """Load the language model"""
        if self.model_type == ModelType.CHAMELEON:
            from chameleon import ChameleonForConditionalGeneration, ChameleonProcessor
            
            self.processor = ChameleonProcessor.from_pretrained(self.config.model_path)
            self.model = ChameleonForConditionalGeneration.from_pretrained(
                self.config.model_path,
                device_map=self.device,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
                torch_dtype=self.config.dtype,
            )
            
            if self.config.adapter_path is not None:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, self.config.adapter_path)
            
            self.text_tokenizer = self.processor.tokenizer
            
        elif self.model_type == ModelType.LIQUID:
            self.text_tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path, 
                padding_side='left'
            )
            print(f"Loading LIQUID model with dtype: {self.config.dtype}.")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                attn_implementation='flash_attention_2',
                torch_dtype=self.config.dtype,
                load_in_8bit=self.config.load_8bit,
            )
            if not self.config.load_8bit:
                self.model = self.model.to(self.device)
        
        elif self.model_type == ModelType.EMU3:
            from emu3.mllm.processing_emu3 import Emu3Processor
            
            self.text_tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path, 
                trust_remote_code=True, 
                padding_side="left"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                device_map=self.device,
                torch_dtype=self.config.dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
            
            image_processor = AutoImageProcessor.from_pretrained(
                self.config.vq_tokenizer_path, 
                trust_remote_code=True
            )
            image_tokenizer = AutoModel.from_pretrained(
                self.config.vq_tokenizer_path, 
                device_map=self.device, 
                trust_remote_code=True
            ).eval()
            self.processor = Emu3Processor(image_processor, image_tokenizer, self.text_tokenizer)
        
        elif self.model_type == ModelType.UNITOK:
            self.text_tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path, 
                padding_side='left'
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                attn_implementation="flash_attention_2",
                torch_dtype=self.config.dtype,
                trust_remote_code=True
            ).to(self.device)
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.model.eval()
        self.vocab_size = len(self.text_tokenizer)
    
    def _load_tokenizers(self):
        """Load image tokenizers"""
        if self.model_type == ModelType.CHAMELEON:
            self.image_tokenizer = ChameleonImageTokenizer(
                cfg_path=self.config.vae_config_path,
                ckpt_path=self.config.vae_checkpoint_path,
                device=self.device
            )
            
        elif self.model_type == ModelType.LIQUID:
            self.image_tokenizer = ChameleonImageTokenizer(
                cfg_path=self.config.vae_config_path,
                ckpt_path=self.config.vae_checkpoint_path,
                device=self.device
            )
        
        elif self.model_type == ModelType.EMU3:
            self.image_tokenizer = Emu3ImageTokenizer(
                vq_hub=self.config.vq_tokenizer_path,
                device=self.device
            )
        
        elif self.model_type == ModelType.UNITOK:
            self.image_tokenizer = UniTokImageTokenizer(
                checkpoint_path=self.config.vq_tokenizer_path,
                device=self.device
            )
    
    def generate_text_to_image(
        self,
        prompts: Union[str, List[str]],
        gen_config: GenerationConfig,
    ) -> List[Image.Image]:
        """
        Generate images from text prompts.
        
        Args:
            prompts: Text prompt(s)
            gen_config: Generation configuration
            
        Returns:
            List of generated PIL images
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        
        if self.model_type == ModelType.CHAMELEON:
            return self._generate_chameleon_t2i(prompts, gen_config)
        elif self.model_type == ModelType.LIQUID:
            return self._generate_liquid_t2i(prompts, gen_config)
        elif self.model_type == ModelType.EMU3:
            return self._generate_emu3_t2i(prompts, gen_config)
        elif self.model_type == ModelType.UNITOK:
            return self._generate_unitok_t2i(prompts, gen_config)
    
    def _generate_chameleon_t2i(
        self, 
        prompts: List[str], 
        gen_config: GenerationConfig
    ) -> List[Image.Image]:
        """Chameleon-specific text-to-image generation"""
        formatted_prompts = [f"{p}" for p in prompts]
        
        inputs = self.processor(
            text=formatted_prompts,
            return_tensors="pt",
            padding=True,
        ).to(self.device, dtype=self.config.dtype)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                multimodal_generation_mode="image-only",
                max_new_tokens=gen_config.max_new_tokens,
                do_sample=True,
                temperature=gen_config.temperature,
                top_p=gen_config.top_p,
            )
        
        output_ids = output_ids.to(dtype=inputs["input_ids"].dtype)
        response_ids = output_ids[:, len(inputs["input_ids"][0]):]
        
        image_token_ids = response_ids[:, 1:-1]
        pixel_values = self.model.decode_image_tokens(image_token_ids)
        pixel_values = pixel_values.to(torch.float32)
        pixel_values = self.processor.postprocess(pixel_values)
        images = [to_pil_image(img) for img in pixel_values['pixel_values']]
        return images
    
    def _generate_liquid_t2i(
        self, 
        prompts: List[str], 
        gen_config: GenerationConfig
    ) -> List[Image.Image]:
        """Liquid-specific text-to-image generation"""
        text_inputs = [f"{p} Generate an image based on this description.<boi>" for p in prompts]
        uncond_inputs = ['<unconditional><boi>'] * len(text_inputs)
        
        if gen_config.cfg_scale > 1:
            all_inputs = text_inputs + uncond_inputs
        else:
            all_inputs = text_inputs
        
        model_inputs = self.text_tokenizer(
            all_inputs, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            input_ids = model_inputs['input_ids']
            model_kwargs = {
                'attention_mask': model_inputs['attention_mask'],
                'use_cache': True,
                'cache_position': torch.arange(input_ids.shape[1], device=self.device)
            }
            
            pred_tokens = []
            
            for i in range(gen_config.max_new_tokens):
                model_inputs_prep = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
                
                outputs = self.model(
                    **model_inputs_prep,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                
                next_token_logits = outputs.logits[:, -1:, :]
                
                if gen_config.cfg_scale > 1:
                    cond_logits, uncond_logits = torch.split(
                        next_token_logits, 
                        len(next_token_logits) // 2, 
                        dim=0
                    )
                    cfg_logits = uncond_logits + (cond_logits - uncond_logits) * gen_config.cfg_scale
                    half_next_token, _ = sample_token(
                        cfg_logits,
                        temperature=gen_config.temperature,
                        top_k=gen_config.top_k,
                        top_p=gen_config.top_p,
                        model="liquid"
                    )
                    pred_tokens.append(half_next_token)
                    next_token = torch.cat([half_next_token, half_next_token])
                else:
                    next_token, _ = sample_token(
                        next_token_logits,
                        temperature=gen_config.temperature,
                        top_k=gen_config.top_k,
                        top_p=gen_config.top_p,
                        model="liquid"
                    )
                    pred_tokens.append(next_token)
                
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                model_kwargs = self.model._update_model_kwargs_for_generation(
                    outputs, model_kwargs, is_encoder_decoder=False
                )
            
            image_vq_ids = torch.cat(pred_tokens, dim=1) - self.vocab_size
            image_vq_ids = torch.clamp(image_vq_ids, min=0, max=8191)
            images = self.image_tokenizer.decode(image_vq_ids)
        
        return images
    
    def _generate_emu3_t2i(
        self, 
        prompts: List[str], 
        gen_config: GenerationConfig
    ) -> List[Image.Image]:
        """Emu3-specific text-to-image generation"""
        POSITIVE_PROMPT = " masterpiece, film grained, best quality."
        NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
        
        formatted_prompts = [p + POSITIVE_PROMPT for p in prompts]
        
        kwargs = dict(
            mode='G',
            ratio=["1:1"] * len(prompts),
            image_area=self.model.config.image_area,
            return_tensors="pt",
            padding="longest",
        )
        
        pos_inputs = self.processor(text=formatted_prompts, **kwargs)
        neg_inputs = self.processor(text=[NEGATIVE_PROMPT] * len(prompts), **kwargs)
        
        h = pos_inputs.image_size[:, 0]
        w = pos_inputs.image_size[:, 1]
        constrained_fn = self.processor.build_prefix_constrained_fn(h, w)
        
        logits_processor = LogitsProcessorList([
            UnbatchedClassifierFreeGuidanceLogitsProcessor(
                gen_config.cfg_scale,
                self.model,
                unconditional_ids=neg_inputs.input_ids.to(self.device),
            ),
            PrefixConstrainedLogitsProcessor(constrained_fn, num_beams=1),
        ])
        
        gen_config.eos_token_id = self.model.config.eos_token_id
        gen_config.pad_token_id = self.model.config.pad_token_id

        with torch.no_grad():
            outputs = self.model.generate(
                pos_inputs.input_ids.to(self.device),
                gen_config,
                logits_processor=logits_processor,
                attention_mask=pos_inputs.attention_mask.to(self.device),
            )
        
        images = []
        for out in outputs:
            mm_list = self.processor.decode(out)
            for idx_j, im in enumerate(mm_list):
                if not isinstance(im, Image.Image):
                    continue
                images.append(im)
        
        return images
    
    def _generate_unitok_t2i(
        self, 
        prompts: List[str], 
        gen_config: GenerationConfig
    ) -> List[Image.Image]:
        """UniTok-MLLM-specific text-to-image generation"""
        formatted_prompts = [f"{p} Generate an image based on this description.\x00" for p in prompts]
        uncond_prompts = ['<unconditional>\x00'] * len(prompts)
        
        model_inputs = self.text_tokenizer(
            formatted_prompts + uncond_prompts,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        
        input_ids = model_inputs.pop('input_ids')
        
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.config.dtype):
            num_image_tokens = 256              
            num_codebooks = self.image_tokenizer.num_codebooks
            
            model_kwargs = {
                'attention_mask': model_inputs.pop('attention_mask'),
                'use_cache': True,
                'cache_position': torch.arange(input_ids.shape[1], device=self.device)
            }
            
            pred_tokens = []
            input_multi_ids = None
            
            for step in range(num_image_tokens):
                model_inputs_prep = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
                
                outputs = self.model.T2I_forward_withcache(
                    **model_inputs_prep,
                    input_multi_ids=input_multi_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                
                next_embed = outputs['last_hidden_state'][:, -1:, :]
                indices_arhead = []
                
                for i_head in range(num_codebooks):
                    ar_next_embed = self.model.ar_head(
                        inputs_embeds=next_embed,
                        use_cache=False,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=False,
                    )
                    logits = self.model.ar_head.linear_head(ar_next_embed[0])
                    
                    cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                    cfg_logits = uncond_logits + (cond_logits - uncond_logits) * gen_config.cfg_scale
                    
                    next_token, _ = sample_token(
                        cfg_logits,
                        temperature=gen_config.temperature,
                        top_k=gen_config.top_k,
                        top_p=gen_config.top_p,
                        model="unitok"
                    )
                    
                    full_batch_next_token = torch.cat([next_token, next_token], dim=0)
                    indices_arhead.append(full_batch_next_token)
                    
                    if i_head < num_codebooks - 1:
                        predicted_embed = self.model.ar_head.codebooks[i_head](full_batch_next_token)
                        next_embed = torch.cat([next_embed, predicted_embed], dim=1)
                
                pred_tokens.append(torch.cat(indices_arhead, dim=1))
                input_multi_ids = torch.stack(pred_tokens, dim=-1)
                
                fake_id = torch.zeros_like(input_ids[:, :1])
                input_ids = torch.cat([input_ids, fake_id], dim=-1)
                model_kwargs = self.model._update_model_kwargs_for_generation(
                    outputs, model_kwargs, is_encoder_decoder=False
                )
            
            image_vq_ids = torch.stack(pred_tokens, dim=-1)[:len(prompts)]
            images = self.image_tokenizer.decode(image_vq_ids)
        
        return images
    
    def compute_sequence_likelihood(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        return_per_token: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute a single sequence likelihood for a HuggingFace causal LM.
        
        Args:
            input_ids: [1, seq_len] token IDs for single sequence
            attention_mask: [seq_len] or [batch_s1ize, seq_len] (1 for valid tokens, 0 for padding)
            return_per_token: If True, also return per-token log-probs

        Returns:
            If return_per_token=False: total log-probability as a scalaer for the input sequence
            If return_per_token=True: (total_log_prob as scaler, per_token_log_prob)
        """
        self.model.eval()

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            if attention_mask is not None and attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
        
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
             attention_mask = attention_mask.to(self.device)

        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'return_dict': True}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits                                     

            shift_logits = logits[..., :-1, :].contiguous()                                    
            target_tokens = input_ids[..., 1:].contiguous()                        

            log_probs_all = F.log_softmax(shift_logits, dim=-1)
            
            per_token_log_probs = torch.gather(
                log_probs_all, 
                dim=-1, 
                index=target_tokens.unsqueeze(-1)
            ).squeeze(-1)
            
            if attention_mask is not None:
                shift_mask = attention_mask[..., 1:].contiguous().float() 
                per_token_log_probs = per_token_log_probs * shift_mask
                
            avg_log_likelihood = per_token_log_probs.mean(dim=-1)               

        avg_log_likelihood = avg_log_likelihood.squeeze(0).item()
        
        if return_per_token:
            return avg_log_likelihood, per_token_log_probs 
            
        return avg_log_likelihood

    def compute_sequence_loss(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor = None,
            return_per_token: bool = False,
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute a single sequence loss for a HuggingFace causal LM.

        Args:
            input_ids: [1, seq_len] token IDs for single sequence
            attention_mask: [seq_len] or [batch_size, seq_len] (1 for valid tokens, 0 for padding)
            return_per_token: If True, also return per-token log-probs

        Returns:
            If return_per_token=False: total log-probability per sequence [batch_size]
            If return_per_token=True: (total_log_prob, per_token_log_prob)
        """
        self.model.eval()
        input_ids = input_ids.to(self.device)

        inputs = {'input_ids': input_ids, 'return_dict': True}
        
        labels = inputs['input_ids'].clone()
        labels[:, :-1] = inputs['input_ids'][:, 1:].clone().contiguous()
        labels[:, -1] = -100                         

        inputs["labels"] = labels

        with torch.no_grad():
            outputs = self.model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        loss = loss.cpu().item()

        print("loss:", loss)
        
        return loss

    def generate_image_to_text(
        self,
        images: Union[Image.Image, List[Image.Image]],
        prompt: Optional[str] = None,
        gen_config: Optional[GenerationConfig] = None,
    ) -> List[str]:
        """
        Generate text descriptions from images.
        
        Args:
            images: PIL image(s), multiple image can be the input
            prompt: One Single text prompt(s) to guide generation
            gen_config: Generation configuration
            
        Returns:
            List of generated text strings
        """
        if gen_config is None:
            gen_config = GenerationConfig()
        
        if isinstance(images, Image.Image):
            images = [images]
        
        if self.model_type == ModelType.CHAMELEON:
            return self._generate_chameleon_i2t(images, prompt, gen_config)
        elif self.model_type == ModelType.LIQUID:
            return self._generate_liquid_i2t(images, prompt, gen_config)
        elif self.model_type == ModelType.UNITOK:
            return self._generate_unitok_i2t(images, prompt, gen_config)
        elif self.model_type == ModelType.EMU3:
            return self._generate_emu3_i2t(images, prompt, gen_config)
        else:
            raise NotImplementedError(f"Image-to-text not implemented for {self.model_type}")
    
    def _generate_chameleon_i2t(
        self,
        images: List[Image.Image],
        prompt: List[str],
        gen_config: GenerationConfig,
    ) -> List[str]:
        """Chameleon-specific image-to-text generation"""
        formatted_prompt = prompt + "<image>"*len(images)
        
        inputs = self.processor(
            text=formatted_prompt,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device, dtype=self.config.dtype)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=gen_config.max_new_tokens,
                do_sample=True,
                temperature=gen_config.temperature,
                top_p=gen_config.top_p,
            )
        
        generated_texts = self.text_tokenizer.batch_decode(
            output_ids[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )
        
        return generated_texts

    def _generate_liquid_i2t(
        self,
        images: List[Image.Image],
        prompt: str,
        gen_config: GenerationConfig,
    ) -> str:
        """Liquid-specific image-to-text generation"""
        
        prompt = '<boi><image><eoi>' * len(images) + '\n' + prompt
        conv = liquid_conv_templates['gemma'].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print(prompt)

        images = [img.convert('RGB') if isinstance(img, Image.Image) else Image.open(img).convert('RGB') for img in images]
        images = [expand2square(image, (122, 116, 104)) for image in images]
        images = [image.resize((512, 512), PIL.Image.LANCZOS) for image in images]

        with torch.no_grad():
            vq_codes_batch = self.image_tokenizer.encode(images)                                                 
            vq_codes_batch = vq_codes_batch.cpu()
            
            vq_codes_batch = vq_codes_batch + len(self.text_tokenizer)
            
            vq_codes = [vq_codes_batch[i] for i in range(vq_codes_batch.shape[0])]
            
            text_ids = tokenizer_image_token(prompt, self.text_tokenizer, LIQUID_IMAGE_TOKEN_INDEX, return_tensors='pt')
            
            num_images = (text_ids == LIQUID_IMAGE_TOKEN_INDEX).sum()
            image_token_indices = [-1] + torch.where(text_ids == LIQUID_IMAGE_TOKEN_INDEX)[0].tolist() + [text_ids.shape[0]]
            
            cur_input_ids = []
            for i in range(num_images + 1):
                cur_input_ids.append(text_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                if i < num_images:
                    cur_input_ids.append(vq_codes[i])
            input_ids = torch.cat(cur_input_ids, dim=0)
            
            inputs = {
                "input_ids": input_ids.unsqueeze(0).to(self.model.device),
                "max_new_tokens": gen_config.max_new_tokens if hasattr(gen_config, 'max_new_tokens') else 1024,
                "bos_token_id": self.text_tokenizer.bos_token_id,
                "eos_token_id": self.text_tokenizer.eos_token_id,
                "pad_token_id": self.text_tokenizer.pad_token_id,
                "do_sample": gen_config.do_sample if hasattr(gen_config, 'do_sample') else True,
                "temperature": gen_config.temperature if hasattr(gen_config, 'temperature') else 1.0,
                "top_p": gen_config.top_p if hasattr(gen_config, 'top_p') else 1.0,
            }
            
            output_ids = self.model.generate(**inputs)
            
            generated_ids = output_ids[0][input_ids.shape[0]:]
            generated_text = self.text_tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            return generated_text
    
    def _generate_unitok_i2t(
        self,
        images: List[Image.Image],
        prompt: str,
        gen_config: GenerationConfig,
    ) -> str:
        """Unitok-specific image-to-text generation for single example."""
        
        crop_size = 256
        transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        
        vq_codes_list = []
        for image in images:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            pad_image = expand2square(image, (122, 116, 104))
            
            img_tensor = transform(pad_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                vq_code = self.image_tokenizer.model.img_to_idx(img_tensor)
                vq_codes_list.append(vq_code)
        
        image_codes = torch.cat(vq_codes_list, dim=0).unsqueeze(0)                                                         
        
        conv_mode = 'vicuna_v1'
        
        if len(images) > 1:
            image_tokens = '\n'.join([DEFAULT_IMAGE_TOKEN] * len(images))
            qs = image_tokens + '\n' + prompt
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        
        conv = unitok_conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_text, 
            self.text_tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors='pt'
        ).unsqueeze(0)
        
        input_ids = input_ids.to(device=self.device, non_blocking=True)
        image_codes = image_codes.to(device=self.device, non_blocking=True)
        
        if hasattr(self.model, "update_prompt"):
            self.model.update_prompt([[prompt]])
        
        with torch.inference_mode():
            output_ids = self.model.generate_mllm(
                input_ids,
                images=image_codes,
                images_aux=None,
                **gen_config.to_dict(),
            )
        
        generated_text = self.text_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        return generated_text
    
    def _generate_emu3_i2t(
            self,
            images: List[Image.Image],
            prompt: Union[str, List[str]],
            gen_config: 'GenerationConfig',
        ) -> Union[str, List[str]]:
        """
        Unified Emu3 image-to-text generation (Image Understanding).
        
        This method supports two modes:
        1. Single-image-per-sample (batch mode): Multiple images with multiple prompts
        - Each image is processed independently with its corresponding prompt
        - Returns a list of text outputs, one per image
        
        2. Multi-image understanding (single sample mode): Multiple images with one prompt
        - All images are processed together with a single shared prompt
        - Returns a single text output that reasons about all images
        
        The mode is automatically determined based on the prompt type:
        - If prompt is a string → Multi-image understanding mode
        - If prompt is a list → Batch processing mode

        Args:
            images (List[Image.Image]): Multiple input images.
            prompt (Union[str, List[str]]): 
                - str: A single text prompt for multi-image understanding
                - List[str]: Multiple prompts, one per image for batch processing
            gen_config (GenerationConfig): The generation configuration.

        Returns:
            Union[str, List[str]]: 
                - str: Single response for multi-image understanding mode
                - List[str]: Multiple responses for batch processing mode
                
        Examples:
            >>> images = [Image.open("img1.jpg"), Image.open("img2.jpg")]
            >>> prompt = "What are the main differences between these two images?"
            >>> response = model._generate_emu3_i2t(images, prompt, gen_config)
            >>> print(response)  # Single string output
            
            >>> images = [Image.open("img1.jpg"), Image.open("img2.jpg")]
            >>> prompts = ["Describe this image", "Describe this image"]
            >>> responses = model._generate_emu3_i2t(images, prompts, gen_config)
            >>> print(responses)  # List of strings, one per image
        """
        
        is_multi_image_mode = isinstance(prompt, str) and isinstance(images, list) and len(images) > 1
        
        if is_multi_image_mode:
            inputs = self.processor.process_multi_image_understanding(
                text=prompt,
                images=images,
                padding_image=True,
                padding="longest",
                return_tensors="pt",
            )
            print("Multi-image understanding mode activated.")
            print(f"Number of images: {len(images)}")
            print("inputs:", inputs.input_ids.shape, inputs.attention_mask.shape)
        else:
            if not isinstance(prompt, list):
                raise ValueError(
                    f"prompt must be either a string (for multi-image understanding) "
                    f"or a list of strings (for batch processing), got {type(prompt)}"
                )
            
            if len(prompt) != len(images):
                raise ValueError(
                    f"In batch processing mode, the number of prompts ({len(prompt)}) "
                    f"must match the number of images ({len(images)})"
                )
            
            inputs = self.processor(
                text=prompt, 
                image=images,
                mode='U',                               
                padding_image=True,
                padding="longest",
                return_tensors="pt",
            )

        print("Processed inputs:", inputs.input_ids.shape, inputs.attention_mask.shape)

        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        gen_config.pad_token_id = self.text_tokenizer.pad_token_id
        gen_config.bos_token_id = self.text_tokenizer.bos_token_id
        gen_config.eos_token_id = self.text_tokenizer.eos_token_id

        generate_kwargs = gen_config.to_dict()
        if 'max_new_tokens' not in generate_kwargs:
            generate_kwargs['max_new_tokens'] = 1024                      

        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        prompt_len = inputs.input_ids.shape[-1]
        generated_tokens = outputs[:, prompt_len:]

        answers = self.processor.batch_decode(generated_tokens, skip_special_tokens=True)

        if is_multi_image_mode:
            return answers[0]
        else:
            return answers
            
    def generate_text_image_to_image(
        self,
        images: Union[Image.Image, List[Image.Image]],
        prompts: Union[str, List[str]],
        gen_config: GenerationConfig,
    ) -> List[Image.Image]:
        """
        Generate images conditioned on both text and input images.
        Useful for image editing tasks.
        
        Args:
            images: Input PIL image(s)
            prompts: Text prompt(s) describing desired edits
            gen_config: Generation configuration
            
        Returns:
            List of generated PIL images
        """
        if isinstance(images, Image.Image):
            images = [images]
        
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)
        
        if self.model_type == ModelType.CHAMELEON:
            return self._generate_chameleon_ti2i(images, prompts, gen_config)
        elif self.model_type == ModelType.LIQUID:
            return self._generate_liquid_ti2i(images, prompts, gen_config)
        elif self.model_type == ModelType.EMU3:
            return self._generate_emu3_ti2i(images, prompts, gen_config)
        elif self.model_type == ModelType.UNITOK:
            return self._generate_unitok_ti2i(images, prompts, gen_config)
        else:
            raise NotImplementedError(
                f"Text+image-to-image not implemented for {self.model_type}"
            )
    
    def _generate_chameleon_ti2i(
        self,
        images: List[Image.Image],
        prompts: List[str],
        gen_config: GenerationConfig,
    ) -> List[Image.Image]:
        """Chameleon-specific text+image-to-image generation"""
        formatted_prompts = [f"{p}<image>" for p in prompts]
        
        inputs = self.processor(
            text=formatted_prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device, dtype=self.config.dtype)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                multimodal_generation_mode="image-only",
                max_new_tokens=gen_config.max_new_tokens,
                do_sample=True,
                temperature=gen_config.temperature,
                top_p=gen_config.top_p,
            )
        
        output_ids = output_ids.to(dtype=inputs["input_ids"].dtype)
        response_ids = output_ids[:, len(inputs["input_ids"][0]):]
        image_token_ids = response_ids[:, 1:-1]
        
        pixel_values = self.model.decode_image_tokens(image_token_ids)
        pixel_values = pixel_values.to(torch.float32)
        pixel_values = self.processor.postprocess(pixel_values)
        
        images = [to_pil_image(img) for img in pixel_values['pixel_values']]
        return images
    
    def _generate_liquid_ti2i(
        self,
        images: List[Image.Image],
        prompts: List[str],
        gen_config: GenerationConfig,
    ) -> List[Image.Image]:
        """Liquid-specific text+image-to-image generation"""
        
        assert len(images) == len(prompts), "Number of images must match n umber of prompts"
        
        images = [img.convert('RGB') if isinstance(img, Image.Image) else Image.open(img).convert('RGB') for img in images]
        images = [expand2square(image, (122, 116, 104)) for image in images]
        images = [image.resize((512, 512), PIL.Image.LANCZOS) for image in images]
        
        with torch.no_grad():
            vq_codes_batch = self.image_tokenizer.encode(images)                                                 
            vq_codes_batch = vq_codes_batch.cpu()
            vq_codes_batch = vq_codes_batch + len(self.text_tokenizer)
            
            all_input_ids = []
            
            for i, prompt in enumerate(prompts):
                text_prompt = f'<boi><image><eoi>\n{prompt}<boi>'
                
                text_ids = tokenizer_image_token(text_prompt, self.text_tokenizer, LIQUID_IMAGE_TOKEN_INDEX, return_tensors='pt')
                
                image_token_idx = torch.where(text_ids == LIQUID_IMAGE_TOKEN_INDEX)[0][0].item()
                
                input_ids = torch.cat([
                    text_ids[:image_token_idx],                           
                    vq_codes_batch[i],                              
                    text_ids[image_token_idx+1:]                                         
                ], dim=0)

                print("input_ids:", input_ids.cpu().numpy().tolist())
                
                all_input_ids.append(input_ids)

            if gen_config.cfg_scale > 1:
                for i in range(len(prompts)):
                    uncond_prompt = '<boi><image><eoi>\n<unconditional><boi>'
                    uncond_text_ids = tokenizer_image_token(uncond_prompt, self.text_tokenizer, LIQUID_IMAGE_TOKEN_INDEX, return_tensors='pt')
                    image_token_idx = torch.where(uncond_text_ids == LIQUID_IMAGE_TOKEN_INDEX)[0][0].item()
                    
                    uncond_input_ids = torch.cat([
                        uncond_text_ids[:image_token_idx],
                        vq_codes_batch[i],
                        uncond_text_ids[image_token_idx+1:]
                    ], dim=0)
                    
                    all_input_ids.append(uncond_input_ids)
            
            max_len = max(ids.shape[0] for ids in all_input_ids)
            padded_input_ids = []
            attention_masks = []
            
            for input_ids in all_input_ids:
                pad_len = max_len - input_ids.shape[0]
                if pad_len > 0:
                    padding = torch.full((pad_len,), self.text_tokenizer.pad_token_id, dtype=input_ids.dtype)
                    padded_ids = torch.cat([padding, input_ids], dim=0)
                    attn_mask = torch.cat([torch.zeros(pad_len), torch.ones(input_ids.shape[0])], dim=0)
                else:
                    padded_ids = input_ids
                    attn_mask = torch.ones(input_ids.shape[0])
                
                padded_input_ids.append(padded_ids)
                attention_masks.append(attn_mask)
            
            input_ids = torch.stack(padded_input_ids).to(self.device)
            attention_mask = torch.stack(attention_masks).to(self.device)
            
            model_kwargs = {
                'attention_mask': attention_mask,
                'use_cache': True,
                'cache_position': torch.arange(input_ids.shape[1], device=self.device)
            }
            
            pred_tokens = []
            
            for i in range(gen_config.max_new_tokens):
                model_inputs_prep = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
                
                outputs = self.model(
                    **model_inputs_prep,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                
                next_token_logits = outputs.logits[:, -1:, :]
                
                if gen_config.cfg_scale > 1:
                    cond_logits, uncond_logits = torch.split(
                        next_token_logits,
                        len(next_token_logits) // 2,
                        dim=0
                    )
                    cfg_logits = uncond_logits + (cond_logits - uncond_logits) * gen_config.cfg_scale
                    half_next_token, _ = sample_token(
                        cfg_logits,
                        temperature=gen_config.temperature,
                        top_k=gen_config.top_k,
                        top_p=gen_config.top_p,
                        model="liquid"
                    )
                    pred_tokens.append(half_next_token)
                    next_token = torch.cat([half_next_token, half_next_token])
                else:
                    next_token, _ = sample_token(
                        next_token_logits,
                        temperature=gen_config.temperature,
                        top_k=gen_config.top_k,
                        top_p=gen_config.top_p,
                        model="liquid"
                    )
                    pred_tokens.append(next_token)
                
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                model_kwargs = self.model._update_model_kwargs_for_generation(
                    outputs, model_kwargs, is_encoder_decoder=False
                )
            
            image_vq_ids = torch.cat(pred_tokens, dim=1) - self.vocab_size
            image_vq_ids = torch.clamp(image_vq_ids, min=0, max=8191)
            output_images = self.image_tokenizer.decode(image_vq_ids)
        
        return output_images
    
    def _generate_emu3_ti2i(
        self,
        images: List[Image.Image],
        prompts: List[str],
        gen_config: GenerationConfig,
    ) -> List[Image.Image]:
        """Emu3-specific text+image-to-image generation (Image Editing)"""
        
        if len(images) != len(prompts):
            raise ValueError(f"Number of images ({len(images)}) must match number of prompts ({len(prompts)})")
        
        POSITIVE_PROMPT = " masterpiece, film grained, best quality."
        NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
        
        image_tokens = self.processor.tokenize_image(images, padding_image=True)
        
        prompt_list = []
        size_list = []
        
        for idx, (text_prompt, img_tok) in enumerate(zip(prompts, image_tokens)):
            h_in, w_in = img_tok.shape
            imgstr = self.processor.to_imgstr(img_tok)
            input_image_prompt = (
                self.processor.tokenizer.boi_token +
                self.processor.prefix_template.format(H=h_in, W=w_in) +
                self.processor.tokenizer.img_token + 
                imgstr +
                self.processor.tokenizer.eol_token +
                self.processor.tokenizer.eof_token +
                self.processor.tokenizer.eoi_token
            )
            
            h_out, w_out = h_in, w_in
            output_image_prefix = (
                self.processor.tokenizer.boi_token +
                self.processor.prefix_template.format(H=h_out, W=w_out) +
                self.processor.tokenizer.img_token
            )
            
            prompt = (
                self.processor.tokenizer.bos_token +
                input_image_prompt +
                text_prompt +
                POSITIVE_PROMPT +
                output_image_prefix
            )
            
            prompt_list.append(prompt)
            size_list.append([h_out, w_out])
        
        kwargs = dict(
            return_tensors="pt",
            padding="longest",
        )
        pos_inputs = self.processor.tokenizer(prompt_list, **kwargs)
        
        neg_prompt_list = []
        for idx, img_tok in enumerate(image_tokens):
            h_in, w_in = img_tok.shape
            imgstr = self.processor.to_imgstr(img_tok)
            input_image_prompt = (
                self.processor.tokenizer.boi_token +
                self.processor.prefix_template.format(H=h_in, W=w_in) +
                self.processor.tokenizer.img_token + 
                imgstr +
                self.processor.tokenizer.eol_token +
                self.processor.tokenizer.eof_token +
                self.processor.tokenizer.eoi_token
            )
            
            h_out, w_out = h_in, w_in
            output_image_prefix = (
                self.processor.tokenizer.boi_token +
                self.processor.prefix_template.format(H=h_out, W=w_out) +
                self.processor.tokenizer.img_token
            )
            
            neg_prompt = (
                self.processor.tokenizer.bos_token +
                input_image_prompt +
                NEGATIVE_PROMPT +
                output_image_prefix
            )
            neg_prompt_list.append(neg_prompt)
        
        neg_inputs = self.processor.tokenizer(neg_prompt_list, **kwargs)
        
        h = torch.tensor([s[0] for s in size_list])
        w = torch.tensor([s[1] for s in size_list])
        constrained_fn = self.processor.build_prefix_constrained_fn(h, w)
        
        logits_processor = LogitsProcessorList([
            UnbatchedClassifierFreeGuidanceLogitsProcessor(
                gen_config.cfg_scale,
                self.model,
                unconditional_ids=neg_inputs.input_ids.to(self.device),
            ),
            PrefixConstrainedLogitsProcessor(constrained_fn, num_beams=1),
        ])
        
        gen_config.eos_token_id = self.model.config.eos_token_id
        gen_config.pad_token_id = self.model.config.pad_token_id
        
        with torch.no_grad():
            outputs = self.model.generate(
                pos_inputs.input_ids.to(self.device),
                gen_config,
                logits_processor=logits_processor,
                attention_mask=pos_inputs.attention_mask.to(self.device),
            )
        
        images_out = []
        for out in outputs:
            mm_list = self.processor.decode(out)
            for im in mm_list:
                if isinstance(im, Image.Image):
                    images_out.append(im)
        
        return images_out

    def _generate_unitok_ti2i(
        self,
        images: List[Image.Image],
        prompts: List[str],
        gen_config: GenerationConfig,
    ) -> List[Image.Image]:
        """UniTok-specific text+image-to-image generation (Image Editing) using conditional-only inference."""

        if len(images) != len(prompts):
            raise ValueError("The number of images and prompts must be equal for batched TI2I generation.")
        
        batch_size = len(images)
        
        crop_size = 256
        transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        
        vq_codes_list = []
        for image in images:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            pad_image = expand2square(image, (122, 116, 104))
            img_tensor = transform(pad_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                vq_code = self.image_tokenizer.model.img_to_idx(img_tensor)
                vq_codes_list.append(vq_code)
                
        image_codes = torch.cat(vq_codes_list, dim=0).unsqueeze(1).to(self.device)

        image_token_str = DEFAULT_IMAGE_TOKEN
        ti2i_suffix = " Generate an image based on this description.\x00" 
        
        conditional_input_ids_list = []
        for prompt in prompts:
            qs = image_token_str + '\n' + prompt
            
            conv_mode = 'vicuna_v1'
            conv = unitok_conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            
            final_prompt_text = conv.get_prompt() + ti2i_suffix
            
            input_ids = tokenizer_image_token(
                final_prompt_text,
                self.text_tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors='pt'
            ).squeeze(0)
            conditional_input_ids_list.append(input_ids)
        
        max_len = max(ids.shape[0] for ids in conditional_input_ids_list)
        conditional_input_ids = torch.full((batch_size, max_len), self.text_tokenizer.pad_token_id,
                                        dtype=torch.long, device=self.device)
        conditional_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)

        for i, ids in enumerate(conditional_input_ids_list):
            l = ids.shape[0]
            conditional_input_ids[i, :l] = ids
            conditional_attention_mask[i, :l] = 1
            
        input_ids = conditional_input_ids
        attention_mask = conditional_attention_mask

        with torch.no_grad():
            (
                _,
                position_ids,
                attention_mask_after_prep,
                past_key_values,
                inputs_embeds,
                _
            ) = self.model.prepare_inputs_for_multimodal(
                input_ids,
                None,
                attention_mask,
                None,
                None,
                image_codes,
                None
            )
            
            
            initial_attention_mask = attention_mask_after_prep
            
            with torch.cuda.amp.autocast(dtype=self.config.dtype):
                initial_outputs = self.model.model(
                    input_ids=None,
                    attention_mask=initial_attention_mask,
                    position_ids=None,
                    past_key_values=None,
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            
            num_image_tokens = 256
            num_codebooks = self.image_tokenizer.num_codebooks
            
            model_kwargs = {
                'past_key_values': initial_outputs.past_key_values,
                'attention_mask': initial_attention_mask,
                'use_cache': True,
                'cache_position': torch.arange(inputs_embeds.shape[1], device=self.device)
            }
            
            pred_tokens = []
            input_multi_ids = None
            
            
            dummy_input_token = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
            current_input_ids = dummy_input_token
            
            
            model_kwargs['attention_mask'] = torch.cat([
                model_kwargs['attention_mask'],
                torch.ones(batch_size, 1, dtype=torch.long, device=self.device)
            ], dim=-1)

            for step in tqdm(range(num_image_tokens), desc="Generating Image Patches"):
                model_inputs_prep = self.model.prepare_inputs_for_generation(
                    current_input_ids, **model_kwargs,
                )
                
                with torch.cuda.amp.autocast(dtype=self.config.dtype):
                    outputs = self.model.T2I_forward_withcache(
                        **model_inputs_prep,
                        input_multi_ids=input_multi_ids,
                        return_dict=True,
                        output_attentions=False,
                        output_hidden_states=False,
                    )
                
                next_embed = outputs['last_hidden_state'][:, -1:, :]
                indices_arhead = []
                
                for i_head in range(num_codebooks):
                    ar_next_embed = self.model.ar_head(
                        inputs_embeds=next_embed,
                        use_cache=False,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=False,
                    )
                    logits = self.model.ar_head.linear_head(ar_next_embed[0])
                    
                    cond_logits = logits

                    next_token, _ = sample_token(
                        cond_logits,
                        temperature=gen_config.temperature,
                        top_k=gen_config.top_k,
                        top_p=gen_config.top_p,
                        model="unitok"
                    )
                    
                    indices_arhead.append(next_token)
                    
                    if i_head < num_codebooks - 1:
                        predicted_embed = self.model.ar_head.codebooks[i_head](next_token)
                        next_embed = torch.cat([next_embed, predicted_embed], dim=1)
                
                pred_tokens.append(torch.cat(indices_arhead, dim=1))
                input_multi_ids = torch.stack(pred_tokens, dim=-1)
                
                
                
                model_kwargs = self.model._update_model_kwargs_for_generation(
                    outputs, model_kwargs, is_encoder_decoder=False
                )
                
                model_kwargs['attention_mask'] = torch.cat([
                    model_kwargs['attention_mask'],
                    torch.ones(batch_size, 1, dtype=torch.long, device=self.device)
                ], dim=-1)

            image_vq_ids = torch.stack(pred_tokens, dim=1).permute(0, 2, 1)
            generated_images = self.image_tokenizer.decode(image_vq_ids)
                
        return generated_images

    def batch_compute_likelihood(
        self,
        token_sequences: List[torch.Tensor]
    ) -> List[float]:
        """
        Compute likelihoods for multiple sequences in batches.
        
        Args:
            token_sequences: List of token sequences (can be variable length)
            batch_size: Batch size for processing
            
        Returns:
            List of log-likelihoods for each sequence
        """
        all_log_probs = []
        for i in range(0, len(token_sequences)):
            
            sequence = token_sequences[i]
            batch_tensor = sequence.unsqueeze(0).to(self.device)               
            
            avg_token_loss_in_sequence = self.compute_sequence_likelihood(batch_tensor, return_per_token=False)
            all_log_probs.append(avg_token_loss_in_sequence)
        
        return all_log_probs
    
    def compare_sequences(
        self,
        sequence_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        ranked_by: str = "likelihood"
    ) -> List[int]:
        """
        Compare pairs of sequences and return preference indices.
        Useful for evaluating model preferences.
        
        Args:
            sequence_pairs: List of (sequence_a, sequence_b) tuples
            
        Returns:
            List of preferred indices (0 for sequence_a, 1 for sequence_b)
        """
        preferences = []
        
        for seq_a, seq_b in sequence_pairs:
            if ranked_by == "likelihood":                
                normalized_a = self.compute_sequence_likelihood(seq_a.unsqueeze(0))
                normalized_b = self.compute_sequence_likelihood(seq_b.unsqueeze(0))
                preference = 0 if normalized_a > normalized_b else 1                   
            elif ranked_by == "loss":
                log_prob_a = self.compute_sequence_loss(seq_a.unsqueeze(0))
                log_prob_b = self.compute_sequence_loss(seq_b.unsqueeze(0))
                preference = 0 if log_prob_a < log_prob_b else 1                  
            else:
                raise ValueError(f"Unsupported ranking method: {ranked_by}")
            
            preferences.append(preference)
        
        return preferences



def create_inference_engine(
    model_type: str,
    model_path: str,
    device: str = "cuda",
    **kwargs
) -> UnifiedMultimodal:
    """
    Factory function to create inference engine with sensible defaults.
    
    Args:
        model_type: One of ["chameleon", "liquid", "emu3", "unitok"]
        model_path: Path to model checkpoint or HF repo
        device: Device to run on
        **kwargs: Additional model configuration parameters
        
    Returns:
        Initialized UnifiedMultimodal instance
        
    Example:
        >>> engine = create_inference_engine(
        ...     model_type="liquid",
        ...     model_path="Junfeng5/Liquid_V1_7B",
        ...     vae_config_path="path/to/vqgan.yaml",
        ...     vae_checkpoint_path="path/to/vqgan.ckpt",
        ... )
        >>> images = engine.generate_text_to_image(
        ...     "A photorealistic lion",
        ...     GenerationConfig(cfg_scale=7.0, temperature=0.9)
        ... )
    """
    model_type_enum = ModelType(model_type.lower())
    
    config = ModelConfig(
        model_path=model_path,
        model_type=model_type_enum,
        device=device,
        **kwargs
    )
    
    return UnifiedMultimodal(config)
