from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch


class ModelType(Enum):
    """Supported model types"""
    CHAMELEON = "chameleon"
    LIQUID = "liquid"
    EMU3 = "emu3"
    UNITOK = "unitok"


@dataclass
class ModelConfig:
    """Model-specific configuration"""
    model_path: str
    model_type: ModelType
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    load_8bit: bool = False
    adapter_path: Optional[str] = None                               
    
    vae_config_path: Optional[str] = None
    vae_checkpoint_path: Optional[str] = None
    vq_tokenizer_path: Optional[str] = None
