
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from scipy import spatial

try:
    import clip
except ImportError as exc:                    
    raise ImportError(
        "The 'clip' package is required for ByteMorph evaluation. Install it via '\n"
        "pip install git+https://github.com/openai/CLIP'."
    ) from exc


class ClipSimilarity_new(nn.Module):
    """Directional CLIP score between source, target, and generated images."""

    def __init__(self, name: str = "ViT-B/32") -> None:
        super().__init__()
        valid = {
            "RN50",
            "RN101",
            "RN50x4",
            "RN50x16",
            "RN50x64",
            "ViT-B/32",
            "ViT-B/16",
            "ViT-L/14",
            "ViT-L/14@336px",
        }
        if name not in valid:
            raise ValueError(f"Unsupported CLIP backbone '{name}'")
        self.size = {
            "RN50x4": 288,
            "RN50x16": 384,
            "RN50x64": 448,
            "ViT-L/14@336px": 336,
        }.get(name, 224)
        self.model, _ = clip.load(name, device="cpu", download_root="./")
        self.model.eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)))
        self.register_buffer("std", torch.tensor((0.26862954, 0.26130258, 0.27577711)))

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(image.float(), size=self.size, mode="bicubic", align_corners=False)
        image = image - rearrange(self.mean, "c -> 1 c 1 1")
        image = image / rearrange(self.std, "c -> 1 c 1 1")
        features = self.model.encode_image(image)
        return features / features.norm(dim=1, keepdim=True)

    def forward(
        self,
        image_src: torch.Tensor,
        image_tgt: torch.Tensor,
        image_gen: torch.Tensor,
        *,
        return_cross_scores: bool = False,
        return_dict: bool = True,
    ) -> dict:
        src_feat = self.encode_image(image_src).detach().cpu().float()
        tgt_feat = self.encode_image(image_tgt).detach().cpu().float()
        gen_feat = self.encode_image(image_gen).detach().cpu().float()

        sim_dir = 1 - spatial.distance.cosine(
            (tgt_feat - src_feat).reshape(-1),
            (gen_feat - src_feat).reshape(-1),
        )

        result = {"clip_dir_img": float(sim_dir)}
        if return_cross_scores:
            sim_dir_unnorm = torch.mm(tgt_feat - src_feat, (gen_feat - src_feat).T)
            result["clip_dir_img_unnorm"] = float(sim_dir_unnorm.item())

        if return_dict:
            return result
        raise NotImplementedError


class ClipSimilarity(nn.Module):
    """CLIP similarity metrics between image pairs and captions."""

    def __init__(self, name: str = "ViT-B/32") -> None:
        super().__init__()
        valid = {
            "RN50",
            "RN101",
            "RN50x4",
            "RN50x16",
            "RN50x64",
            "ViT-B/32",
            "ViT-B/16",
            "ViT-L/14",
            "ViT-L/14@336px",
        }
        if name not in valid:
            raise ValueError(f"Unsupported CLIP backbone '{name}'")
        self.size = {
            "RN50x4": 288,
            "RN50x16": 384,
            "RN50x64": 448,
            "ViT-L/14@336px": 336,
        }.get(name, 224)
        self.model, _ = clip.load(name, device="cpu", download_root="./")
        self.model.eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)))
        self.register_buffer("std", torch.tensor((0.26862954, 0.26130258, 0.27577711)))

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(image.float(), size=self.size, mode="bicubic", align_corners=False)
        image = image - rearrange(self.mean, "c -> 1 c 1 1")
        image = image / rearrange(self.std, "c -> 1 c 1 1")
        features = self.model.encode_image(image)
        return features / features.norm(dim=1, keepdim=True)

    def encode_text(self, text: list[str]) -> torch.Tensor:
        tokens = clip.tokenize(text, truncate=True).to(next(self.parameters()).device)
        features = self.model.encode_text(tokens)
        return features / features.norm(dim=1, keepdim=True)

    def forward(
        self,
        image_0: torch.Tensor,
        image_1: torch.Tensor,
        text_0: list[str],
        text_1: list[str],
        *,
        return_cross_scores: bool = False,
        return_dict: bool = True,
    ) -> dict:
        img_feat_0 = self.encode_image(image_0).detach().cpu().float()
        img_feat_1 = self.encode_image(image_1).detach().cpu().float()
        txt_feat_0 = self.encode_text(text_0).detach().cpu().float()
        txt_feat_1 = self.encode_text(text_1).detach().cpu().float()

        sim_txt = 1 - spatial.distance.cosine(img_feat_1.reshape(-1), txt_feat_1.reshape(-1))
        sim_img = 1 - spatial.distance.cosine(img_feat_0.reshape(-1), img_feat_1.reshape(-1))
        sim_dir = 1 - spatial.distance.cosine(
            (img_feat_1 - img_feat_0).reshape(-1),
            (txt_feat_1 - txt_feat_0).reshape(-1),
        )

        result = {
            "clip_sim_txt": float(sim_txt),
            "clip_sim_img": float(sim_img),
            "clip_dir_txt": float(sim_dir),
        }

        if return_cross_scores:
            sim_dir_unnorm = torch.mm(img_feat_1 - img_feat_0, (txt_feat_1 - txt_feat_0).T)
            result["clip_sim_dir_unnorm"] = float(sim_dir_unnorm.item())

        if return_dict:
            return result
        raise NotImplementedError

