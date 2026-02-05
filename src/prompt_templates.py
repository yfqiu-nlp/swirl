"""
Prompt Templates for Unified Multimodal Foundation Models

This module provides prompt template classes for four multimodal generative models:
- Chameleon
- Liquid
- Emu3
- UniTok-MLLM

Each model supports three task types:
1. Text-to-Image (T2I)
2. Image-to-Text (I2T / Image Captioning)
3. Text+Image-to-Image (TI2I / Image Editing)

All templates are designed for models with unified text-image vocabularies trained
with next-token prediction objectives.
"""

from abc import ABC, abstractmethod
from enum import Enum


class TaskType(Enum):
    """Enumeration of supported task types."""
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_TEXT = "image_to_text"
    TEXT_IMAGE_TO_IMAGE = "text_image_to_image"


class BasePromptTemplate(ABC):
    """Abstract base class for prompt templates."""
    
    @abstractmethod
    def format(self, **kwargs) -> str:
        """
        Format the prompt template with given inputs.
        
        Returns:
            str: Formatted prompt string
        """
        pass



class ChameleonTextToImage(BasePromptTemplate):
    """Chameleon: Text-to-Image generation template."""
    
    def format(self, text: str) -> str:
        """
        Format text-to-image prompt for Chameleon.
        
        Args:
            text: Input text prompt describing the desired image
            
        Returns:
            Formatted prompt string
        """
        return f"{text}"


class ChameleonImageToText(BasePromptTemplate):
    """Chameleon: Image-to-Text (captioning) template."""
    
    def format(self, text: str, image_placeholder: str = "<image>") -> str:
        """
        Format image-to-text prompt for Chameleon.
        
        Args:
            image_placeholder: Placeholder token for input image
            
        Returns:
            Formatted prompt string
        """
        return f"{image_placeholder}{text}"


class ChameleonTextImageToImage(BasePromptTemplate):
    """Chameleon: Text+Image-to-Image (editing) template."""
    
    def format(self, text: str, image_placeholder: str = "<image>") -> str:
        """
        Format text+image-to-image prompt for Chameleon.
        
        Args:
            text: Text instruction for image editing
            image_placeholder: Placeholder token for input image
            
        Returns:
            Formatted prompt string
        """
        return f"{image_placeholder}{text}"



class LiquidTextToImage(BasePromptTemplate):
    """Liquid: Text-to-Image generation template."""
    
    def format(self, text: str) -> str:
        """
        Format text-to-image prompt for Liquid.
        
        Args:
            text: Input text prompt describing the desired image
            
        Returns:
            Formatted prompt string
        """
        return f"{text}<image>"


class LiquidImageToText(BasePromptTemplate):
    """Liquid: Image-to-Text (captioning) template."""
    
    def format(self, image_placeholder: str = "<image>") -> str:
        """
        Format image-to-text prompt for Liquid.
        
        Args:
            image_placeholder: Placeholder token for input image
            
        Returns:
            Formatted prompt string
        """
        return f"{image_placeholder}"


class LiquidTextImageToImage(BasePromptTemplate):
    """Liquid: Text+Image-to-Image (editing) template."""
    
    def format(self, text: str, image_placeholder: str = "<image>") -> str:
        """
        Format text+image-to-image prompt for Liquid.
        
        Args:
            text: Text instruction for image editing
            image_placeholder: Placeholder token for input image
            
        Returns:
            Formatted prompt string
        """
        return f"{image_placeholder}{text}<image>"



class Emu3TextToImage(BasePromptTemplate):
    """Emu3: Text-to-Image generation template."""
    
    def format(self, text: str) -> str:
        """
        Format text-to-image prompt for Emu3.
        
        Args:
            text: Input text prompt describing the desired image
            
        Returns:
            Formatted prompt string
        """
        return f"{text}<image>"


class Emu3ImageToText(BasePromptTemplate):
    """Emu3: Image-to-Text (captioning) template."""
    
    def format(self, image_placeholder: str = "<image>") -> str:
        """
        Format image-to-text prompt for Emu3.
        
        Args:
            image_placeholder: Placeholder token for input image
            
        Returns:
            Formatted prompt string
        """
        return f"{image_placeholder}"


class Emu3TextImageToImage(BasePromptTemplate):
    """Emu3: Text+Image-to-Image (editing) template."""
    
    def format(self, text: str, image_placeholder: str = "<image>") -> str:
        """
        Format text+image-to-image prompt for Emu3.
        
        Args:
            text: Text instruction for image editing
            image_placeholder: Placeholder token for input image
            
        Returns:
            Formatted prompt string
        """
        return f"{image_placeholder}{text}<image>"



class UniTokMLLMTextToImage(BasePromptTemplate):
    """UniTok-MLLM: Text-to-Image generation template."""
    
    def format(self, text: str) -> str:
        """
        Format text-to-image prompt for UniTok-MLLM.
        
        Args:
            text: Input text prompt describing the desired image
            
        Returns:
            Formatted prompt string
        """
        return f"{text}<image>"


class UniTokMLLMImageToText(BasePromptTemplate):
    """UniTok-MLLM: Image-to-Text (captioning) template."""
    
    def format(self, image_placeholder: str = "<image>") -> str:
        """
        Format image-to-text prompt for UniTok-MLLM.
        
        Args:
            image_placeholder: Placeholder token for input image
            
        Returns:
            Formatted prompt string
        """
        return f"{image_placeholder}"


class UniTokMLLMTextImageToImage(BasePromptTemplate):
    """UniTok-MLLM: Text+Image-to-Image (editing) template."""
    
    def format(self, text: str, image_placeholder: str = "<image>") -> str:
        """
        Format text+image-to-image prompt for UniTok-MLLM.
        
        Args:
            text: Text instruction for image editing
            image_placeholder: Placeholder token for input image
            
        Returns:
            Formatted prompt string
        """
        return f"{image_placeholder}{text}<image>"

