
import io
import re
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

SAFE_ID_PATTERN = re.compile(r"[^a-zA-Z0-9._-]+")


def _ensure_pil_image(value: Any) -> Image.Image:
    """Convert a variety of dataset image representations into ``PIL.Image``."""

    if value is None:
        raise ValueError("Image value is None")

    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")

    if isinstance(value, dict):
        data = value.get("bytes") or value.get("path")
        if isinstance(data, (bytes, bytearray)):
            return Image.open(io.BytesIO(data)).convert("RGB")
        if isinstance(data, str):
            with Image.open(data) as img:
                return img.convert("RGB")

    raise TypeError(f"Unsupported image representation: {type(value)!r}")


def _sanitise_token(token: str, fallback: str) -> str:
    token = SAFE_ID_PATTERN.sub("-", token.strip())
    return token.lower() or fallback


def load_bytemorph_metadata(
    dataset_root: Path | str,
    *,
    split_file: str = "data/test-*.parquet",
    instruction_field: str = "edit_prompt_rewrite_instruction",
) -> List[Dict[str, Any]]:
    """Load ByteMorph entries into the Bagel Flow-GRPO metadata format.

    Args:
        dataset_root: Directory that contains the ByteMorph parquet split(s).
        split_file:  Glob or filename relative to ``dataset_root`` that selects
            the parquet shards to load.
        instruction_field: Use this column for ``Verbalised Action``.  Falls
            back to ``edit_prompt`` when empty.

    Returns:
        List of dictionaries compatible with ``BagelFlowGRPOPackedDataset``.
    """

    dataset_root = Path(dataset_root).resolve()
    pattern = split_file if Path(split_file).is_absolute() else str(dataset_root / split_file)
    shards = sorted(glob(pattern))
    if not shards:
        raise FileNotFoundError(f"No parquet files matched pattern '{pattern}'")

    data_files: Dict[str, List[str] | str]
    if len(shards) == 1:
        data_files = shards[0]
    else:
        data_files = shards

    try:
        from datasets import load_dataset
    except ImportError as exc:                                                    
        raise ImportError("'datasets' package is required to load ByteMorph metadata") from exc

    dataset = load_dataset("parquet", data_files=data_files, split="train")

    entries: List[Dict[str, Any]] = []
    for idx, sample in enumerate(dataset):
        edit_type = sample.get("edit_type") or "unspecified"
        image_id = str(sample.get("image_id") or idx)
        safe_edit = _sanitise_token(edit_type, "edit")
        safe_image = _sanitise_token(image_id, f"{idx:05d}")
        sample_id = f"bytemorph_{safe_edit}_{safe_image}"

        try:
            source_image = _ensure_pil_image(sample.get("src_img")).copy()
        except Exception as exc:
            raise ValueError(f"Failed to decode source image for sample {sample_id}") from exc

        target_image: Optional[Image.Image] = None
        tgt_value = sample.get("tgt_img")
        if tgt_value is not None:
            target_image = _ensure_pil_image(tgt_value).copy()

        rewrite_instruction = (sample.get(instruction_field) or "").strip()
        if not rewrite_instruction:
            rewrite_instruction = (sample.get("edit_prompt") or "").strip()

        entry: Dict[str, Any] = {
            "id": sample_id,
            "Source_observation": source_image,
            "Target_observation": target_image,
            "Verbalised Action": rewrite_instruction,
            "task_type": "image_editing",
            "bytemorph_edit_type": edit_type,
            "bytemorph_image_id": image_id,
            "bytemorph_edit_prompt": sample.get("edit_prompt", ""),
            "bytemorph_edit_prompt_rewrite": sample.get("edit_prompt_rewrite_instruction", ""),
            "bytemorph_src_caption": sample.get("src_img_caption", ""),
            "bytemorph_tgt_caption": sample.get("tgt_img_caption", ""),
            "bytemorph_raw_record": sample,
        }

        entries.append(entry)

    return entries
