from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Union

from torch.utils.data import Dataset
from PIL import Image

RawSample = Dict[str, Union[str, Path]]

class AbstractMultimodalDynamicsDataset(Dataset, ABC):
    """
    Abstract Base Class for Multimodal Dynamics datasets.

    Ensures compliance with PyTorch and Hugging Face Trainer expectations 
    (__len__ and __getitem__) and defines the required raw data attributes.
    """
    
    def __init__(self, data: List[RawSample]):
        """
        Initializes the dataset with a pre-loaded list of data samples.
        
        Args:
            data: A list of dictionaries, where each dict contains 
                  'input_image_path', 'textual_action', and 'output_image_path'.
        """
        if not data:
            raise ValueError("Dataset cannot be initialized with an empty data list.")
        self._data = data

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self._data)

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a processed data sample tailored to a specific task (e.g., Fwd/Inv Dynamics).
        
        The implementation must handle:
        1. Loading the image paths.
        2. Structuring the input/target keys based on the task type.
        """
        pass

    @staticmethod
    def _load_image(path: Union[str, Path]) -> Image.Image:
        """Utility to safely load an image file."""
        try:
            return Image.open(path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found at: {path}")
        except Exception as e:
            raise IOError(f"Error loading image from {path}: {e}")


class ForwardDynamicsDataset(AbstractMultimodalDynamicsDataset):
    """
    Dataset for Forward Dynamics Prediction: (Input Image, Textual Action) -> Output Image.
    Used for image editing or vision-language instruction following tasks.
    
    Returns:
        A dict with keys: 'input_image', 'textual_action', 'output_image'.
    """
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Prepares the sample for the forward dynamics task.
        Loads images and returns the action text.
        """
        sample = self._data[idx]
        
        input_image = self._load_image(sample["input_image_path"])
        textual_action = sample["textual_action"]
        
        output_image = self._load_image(sample["output_image_path"])
        
        return {
            "input_image": input_image,                  
            "textual_action": textual_action,         
            "output_image": output_image,                         
        }


class InverseDynamicsDataset(AbstractMultimodalDynamicsDataset):
    """
    Dataset for Inverse Dynamics Prediction: (Input Image, Output Image) -> Textual Action.
    Used for captioning, instruction-mining, or difference-description tasks.
    
    Returns:
        A dict with keys: 'input_image', 'output_image', 'textual_action'.
    """
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Prepares the sample for the inverse dynamics task.
        Loads images and returns the action text (as the target).
        """
        sample = self._data[idx]
        
        input_image = self._load_image(sample["input_image_path"])
        output_image = self._load_image(sample["output_image_path"])
        
        textual_action = sample["textual_action"]
        
        return {
            "input_image": input_image,                  
            "output_image": output_image,                        
            "textual_action": textual_action,                  
        }