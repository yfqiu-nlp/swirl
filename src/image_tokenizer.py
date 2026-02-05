import torch
from abc import ABC, abstractmethod
from typing import List, Union

from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from torchvision import transforms


class ImageTokenizer(ABC):
    
    @abstractmethod
    def encode(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """Encode PIL images to token IDs"""
        pass
    
    @abstractmethod
    def decode(self, token_ids: torch.Tensor) -> Union[Image.Image, List[Image.Image]]:
        """Decode token IDs to PIL images"""
        pass



class ChameleonImageTokenizer(ImageTokenizer):
    """Image tokenizer for Chameleon (VQGAN-based)"""
    
    def __init__(self, cfg_path: str, ckpt_path: str, device: str):
        from chameleon.inference.image_tokenizer import ImageTokenizer as ChameleonVQGAN

        self.tokenizer = ChameleonVQGAN(
            cfg_path=cfg_path, 
            ckpt_path=ckpt_path, 
            device=device
        )
    
    def encode(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        if isinstance(images, Image.Image):
            images = [images]
        image_token_list = [self.tokenizer.img_tokens_from_pil(img) for img in images]
        return torch.stack(image_token_list, dim=0)
    
    def decode(self, token_ids: torch.Tensor) -> Union[Image.Image, List[Image.Image]]:
        images = []
        for ids in token_ids:
            img = self.tokenizer.pil_from_img_toks(ids)
            images.append(img)
        return images


class UniTokImageTokenizer(ImageTokenizer):
    """Image tokenizer for UniTok-MLLM"""
    
    def __init__(self, checkpoint_path: str, device: str):
        import sys
        import os
        sys.path.append(os.getcwd())                          
        from unitok.vision_tokenizer import UniTok
        
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        class Args:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
            def load_state_dict(self, state_dict):
                self.__dict__.update(state_dict)
        
        cfg = Args()
        cfg.load_state_dict(ckpt['args'])
        
        self.model = UniTok(cfg).to(device)
        self.model.load_state_dict(ckpt['trainer']['unitok'])
        self.model.eval()
        self.device = device
        self.num_codebooks = cfg.num_codebooks
    
    def encode(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        if isinstance(images, Image.Image):
            images = [images]
        
        to_tensor = transforms.ToTensor()
        tensors = [to_tensor(img).unsqueeze(0) for img in images]
        batch = torch.cat(tensors, dim=0).to(self.device)
        
        with torch.no_grad():
            tokens = self.model.img_to_idx(batch)
        
        return tokens
    
    def decode(self, token_ids: torch.Tensor) -> Union[Image.Image, List[Image.Image]]:
        with torch.no_grad():
            imgs = self.model.idx_to_img(token_ids)
        
        to_pil = transforms.ToPILImage()
        images = []
        for img in imgs:
            img = img.add(1).mul(0.5).clamp(0, 1)
            img = img.to(torch.float32)
            images.append(to_pil(img))
        
        return images


class Emu3ImageTokenizer(ImageTokenizer):
    """Image tokenizer for Emu3"""
    
    def __init__(self, vq_hub: str, device: str):
        self.image_processor = AutoImageProcessor.from_pretrained(vq_hub, trust_remote_code=True)
        self.tokenizer = AutoModel.from_pretrained(
            vq_hub, 
            device_map=device, 
            trust_remote_code=True
        ).eval()
    
    def encode(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        if isinstance(images, Image.Image):
            images = [images]
        
        processed = self.image_processor(images, return_tensors="pt")
        with torch.no_grad():
            tokens = self.tokenizer.encode(processed['pixel_values'])
        
        return tokens
    
    def decode(self, token_ids: torch.Tensor) -> Union[Image.Image, List[Image.Image]]:
        with torch.no_grad():
            pixel_values = self.tokenizer.decode(token_ids)
        
        pixel_values = pixel_values.to(torch.float32)
        images = []
        for pv in pixel_values:
            pv = self.image_processor.postprocess(pv)
            img = transforms.ToPILImage()(pv)
            images.append(img)
        
        return images if len(images) > 1 else images[0]
