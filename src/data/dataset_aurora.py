import json
import sys
from pathlib import Path
from typing import List, Tuple, Union

from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))
from data.dataset import ForwardDynamicsDataset, RawSample 



class AURORAForwardDynamicsDataset(ForwardDynamicsDataset):
    """
    Concrete implementation for the AURORA SFT dataset.
    This task is a Forward Dynamics Prediction (Image + Text -> Image).
    
    Adapts to the AURORA folder structure and extracts the prompt from 
    the 'conversations' field.
    """
    
    PROMPT_PREFIX = "<image>\nEditing the given image according to the following prompt: "
    
    def __init__(self, dataset_name: str, base_dir: Union[str, Path]):
        """
        Loads and processes the AURORA data file (.jsonl).
        
        Args:
            dataset_name: The name of the sub-dataset (e.g., 'something', 'ag').
            base_dir: The root directory for the dataset, e.g., 
                      'datasets/AURORA'.
        """
        self.base_dir = Path(base_dir)
        self.dataset_name = dataset_name
        
        data_file_path = self.base_dir / "data" / dataset_name / "train.jsonl"

        if not data_file_path.exists():
            raise FileNotFoundError(f"AURORA annotation file not found: {data_file_path}")

        processed_data, total_lines = self._load_data(data_file_path)

        super().__init__(processed_data)
        
        if not processed_data:
            raise ValueError(
                f"After filtering, the AURORA '{dataset_name}' dataset is empty. "
                "Check file paths and data integrity."
            )
            
        print(
            f"Successfully loaded {len(processed_data)} valid samples "
            f"for AURORA '{dataset_name}' from {total_lines} lines "
            f"(filtered {total_lines - len(processed_data)} invalid)."
        )

    def _get_file_line_count(self, file_path: Path) -> int:
        """Utility to quickly count lines in a file for tqdm total."""
        with open(file_path, 'r') as f:
            return sum(1 for line in f)

    def _load_data(self, data_file_path: Path) -> Tuple[List[RawSample], int]:
        """
        Reads the .jsonl file, constructs the absolute file paths, and validates them.
        
        Returns:
            A tuple of (List of validated RawSamples, Total lines read).
        """
        processed_data: List[RawSample] = []
        
        image_root = self.base_dir.parent 
        
        total_lines = self._get_file_line_count(data_file_path)
        
        with open(data_file_path, 'r') as f:
            for line_idx, line in enumerate(tqdm(f, total=total_lines, desc=f"Loading {data_file_path.name}"), 1):
                try:
                    entry = json.loads(line)
                    
                    image_paths = entry["images"]
                    conversations = entry["conversations"]
                    
                    if len(image_paths) != 2:
                        continue
                        
                    input_image_rel_path = image_paths[0]
                    output_image_rel_path = image_paths[1]
                    
                    input_image_path = image_root / input_image_rel_path
                    output_image_path = image_root / output_image_rel_path

                    human_message = conversations[0]["value"]
                    
                    if not human_message.startswith(self.PROMPT_PREFIX):
                        continue
                        
                    textual_action = human_message.replace(self.PROMPT_PREFIX, "", 1).strip()
                    
                    is_valid = True
                    if not input_image_path.exists() or not output_image_path.exists():
                        is_valid = False
                        
                    if is_valid:
                        processed_data.append({
                            "input_image_path": input_image_path,
                            "textual_action": textual_action,
                            "output_image_path": output_image_path,
                        })
                        
                except json.JSONDecodeError:
                    pass
                except KeyError:
                    pass
                except IndexError:
                    pass
                        
        return processed_data, total_lines
