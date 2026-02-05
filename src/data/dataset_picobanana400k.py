import sys
from pathlib import Path
SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from typing import Any, Dict, List, Union, Tuple
import json
from data.dataset import ForwardDynamicsDataset, RawSample, AbstractMultimodalDynamicsDataset
from tqdm import tqdm
import re

class PicoBanana400KEditingDataset(ForwardDynamicsDataset):
    """
    Concrete implementation for the PicoBanana400K SFT dataset.
    This task is a Forward Dynamics Prediction (Image + Text -> Image).
    
    Includes robust checks for data integrity and file existence in _load_data.
    """
    
    def __init__(self, data_file_path: Union[str, Path], base_dir: Union[str, Path]):
        """
        Loads and processes the PicoBanana400K data file (.jsonl).
        
        Args:
            data_file_path: Path to the sft.jsonl file.
            base_dir: The root directory for the dataset, e.g., 
                      'datasets/pico-banana-400k'.
        """
        data_file_path = Path(data_file_path)
        self.base_dir = Path(base_dir)
        
        if not data_file_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_file_path}")

        processed_data, total_lines = self._load_data(data_file_path)

        super().__init__(processed_data)
        
        if not processed_data:
            raise ValueError(
                "After filtering, the dataset is empty. Check file paths and data integrity."
            )
        
        print(
            f"Successfully loaded {len(processed_data)} valid samples "
            f"from {total_lines} lines (filtered {total_lines - len(processed_data)} invalid)."
        )

    def _get_file_line_count(self, file_path: Path) -> int:
        """Utility to quickly count lines in a file for tqdm total."""
        with open(file_path, 'r') as f:
            return sum(1 for line in f)

    def _load_data(self, data_file_path: Path) -> tuple[List[RawSample], int]:
        """
        Reads the .jsonl file, constructs the absolute file paths, and validates them.
        Returns:
            A tuple of (List of validated RawSamples, Total lines read).
        """
        processed_data: List[RawSample] = []
        
        open_image_dir = self.base_dir / "images" / "openimages"
        sft_image_root = self.base_dir / "images" / "sft"
        
        total_lines = self._get_file_line_count(data_file_path)
        
        with open(data_file_path, 'r') as f:
            for line_idx, line in enumerate(tqdm(f, total=total_lines, desc=f"Loading {data_file_path.name}"), 1):
                try:
                    entry = json.loads(line)
                    
                    input_image_name = entry["open_image_input_file"]
                    output_image_rel_path = entry["output_image"]
                    textual_action = entry["summarized_text"]                       

                    input_image_path = open_image_dir / input_image_name
                    output_image_path = sft_image_root / output_image_rel_path
                    
                    is_valid = True
                    
                    if not input_image_path.exists():
                        is_valid = False
                        
                    if not output_image_path.exists():
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
                    
        return processed_data, total_lines









class PicoBananaMultiTurnDataset(AbstractMultimodalDynamicsDataset):
    """
    Dataset class for PicoBanana Multi-Turn Editing samples.
    Handles sequences: [Image_0, Image_1, ..., Image_N] and [Prompt_1, ..., Prompt_N].
    """
    
    def __init__(self, data_file_path: Union[str, Path], base_dir: Union[str, Path]):
        data_file_path = Path(data_file_path)
        self.base_dir = Path(base_dir)
        
        if not data_file_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_file_path}")

        processed_data, total_lines = self._load_data(data_file_path)
        
        super().__init__(processed_data)
        
        print(f"Loaded {len(processed_data)} multi-turn sequences from {total_lines} lines.")

    def _get_file_line_count(self, file_path: Path) -> int:
        with open(file_path, 'r') as f:
            return sum(1 for line in f)

    def _load_data(self, data_file_path: Path) -> Tuple[List[Dict[str, Any]], int]:
        """Parses the complex nested JSONL structure of PicoBanana Multi-turn."""
        processed_data = []
        total_lines = self._get_file_line_count(data_file_path)
        open_image_dir = self.base_dir / "images" / "openimages"
        
        with open(data_file_path, 'r') as f:
            for line in tqdm(f, total=total_lines, desc="Loading Multi-Turn Data"):
                try:
                    entry = json.loads(line)
                    files = entry.get("files", [])
                    prompts = entry.get("metadata_edit_turn_prompts", [])
                    
                    if not files or not prompts:
                        continue

                    root_img_entry = next((item for item in files if item.get('id') == 'original_input_image'), None)
                    if not root_img_entry or 'original_input_image_file' not in root_img_entry:
                        continue
                        
                    input_filename = root_img_entry['original_input_image_file']
                    input_image_path = open_image_dir / input_filename

                    turn_entries = []
                    for item in files:
                        if item['id'].startswith('edit_turn'):
                            match = re.search(r'turn(\d+)', item['id'])
                            if match:
                                turn_entries.append((int(match.group(1)), item))
                        elif item['id'] == 'final_image':
                            match = re.search(r'turn(\d+)', item['url'])
                            if match:
                                turn_entries.append((int(match.group(1)), item))
                    
                    turn_entries.sort(key=lambda x: x[0])
                    
                    image_sequence_paths = [input_image_path]                 
                    valid_sample = input_image_path.exists()
                    
                    for _, item in turn_entries:
                        img_path = self.base_dir / item['url']
                        if not img_path.exists():
                            valid_sample = False
                            break
                        image_sequence_paths.append(img_path)
                        
                    if len(image_sequence_paths) != len(prompts) + 1:
                        continue

                    if valid_sample:
                        processed_data.append({
                            "image_paths": image_sequence_paths,             
                            "prompts": prompts                              
                        })

                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
        
        return processed_data, total_lines

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Loads the full sequence of images into memory.
        Returns:
            {
                "images": List[PIL.Image], 
                "prompts": List[str]
            }
        """
        sample = self._data[idx]
        
        try:
            images = [self._load_image(path) for path in sample["image_paths"]]
        except Exception as e:
            print(f"Error loading sequence at index {idx}: {e}")
            raise e

        return {
            "images": images,                                             
            "prompts": sample["prompts"]                               
        }
