import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import concatenate_datasets, load_from_disk
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling

from train.arguments import DataArguments

logger = logging.getLogger(__name__)

class DatasetRegistry:
    """Registry for dataset paths and configurations."""
    
    DATASETS = {
        "pico-banana-idp": "pico-banana-400k/tokenized_liquid_idp",
        "pico-banana-short-action-fdp": "pico-banana-400k/tokenized_liquid_short_action_fdp",
        "pico-banana-short-action-idp": "pico-banana-400k/tokenized_liquid_short_action_idp",
        "pico-banana-multiturn": "pico-banana-400k/tokenized_liquid_multiturn",
        "aurora-kubric-fdp": "AURORA/tokenized_liquid_aurora_kubric_fdp",
        "aurora-kubric-idp": "AURORA/tokenized_liquid_aurora_kubric_idp",
        "aurora-ag-idp": "AURORA/tokenized_liquid_aurora_ag_idp",
        "aurora-ag-fdp": "AURORA/tokenized_liquid_aurora_ag_fdp",
        "aurora-something-idp": "AURORA/tokenized_liquid_aurora_something_idp",
        "aurora-something-fdp": "AURORA/tokenized_liquid_aurora_something_fdp",
        "aurora-magicbrush-idp": "AURORA/tokenized_liquid_aurora_magicbrush_idp",
        "aurora-magicbrush-fdp": "AURORA/tokenized_liquid_aurora_magicbrush_fdp",
        "mit-syns-fdp": "unsupervised-frames-ucf-kinetics-mit/tokenized_liquid_mit_fdp",
        "ucf-syns-fdp": "unsupervised-frames-ucf-kinetics-mit/tokenized_liquid_ucf_fdp",
        "kinetics-syns-fdp": "unsupervised-frames-ucf-kinetics-mit/tokenized_liquid_kinetics_fdp",
    }
    
    @classmethod
    def get_path(cls, dataset_name: str, base_path: str) -> Path:
        """Get full path for a dataset."""
        if dataset_name not in cls.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(cls.DATASETS.keys())}")
        return Path(base_path) / cls.DATASETS[dataset_name]
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all available datasets."""
        return list(cls.DATASETS.keys())


def parse_data_mixture(mixture_str: str) -> List[tuple]:
    """
    Parse data mixture string into list of (dataset_name, weight) tuples.
    
    Args:
        mixture_str: Format "dataset1:weight1,dataset2:weight2"
        
    Returns:
        List of (dataset_name, weight) tuples
    """
    mixtures = []
    for item in mixture_str.split(","):
        parts = item.strip().split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid mixture format: {item}. Expected 'dataset:weight'")
        dataset_name, weight = parts[0].strip(), float(parts[1].strip())
        mixtures.append((dataset_name, weight))
    return mixtures


def load_and_mix_datasets(
    data_args: DataArguments,
    split: str = "train",
    seed: int = 42,
) -> Tuple[Dataset, Optional[Dict[str, Dataset]]]:
    """
    Load multiple datasets, apply weights, split each into train/eval individually,
    and then mix the training portions.

    Returns:
        Tuple[Dataset, Optional[Dict[str, Dataset]]]: 
            - The concatenated and mixed training dataset.
            - A dictionary of validation datasets (one per source) or None if validation_split is 0.
    """
    mixtures = parse_data_mixture(data_args.data_mixture)
    logger.info(f"Loading data with mixture: {mixtures} | Validation split: {data_args.validation_split}")

    train_datasets_list = []
    eval_datasets_dict = {}

    if data_args.eval_mixture is None:
        logger.info(f"No evaluation mixture is specified. We will use all mixture data from: {mixtures}, for validation prupose.")
        eval_dataset_names = [d[0] for d in mixtures]                                 
    else:
        eval_dataset_names = data_args.eval_mixture.split(',')
        logger.info(f"Specify eval_mixture to be {eval_dataset_names}. We will evaluate on these validation set.")

    for dataset_name, weight in mixtures:
        dataset_path = DatasetRegistry.get_path(dataset_name, data_args.dataset_base_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        logger.info(f"Loading {dataset_name} from {dataset_path}")
        dataset = load_from_disk(str(dataset_path))

        num_samples = int(len(dataset) * weight)
        logger.info(
            f"{dataset_name}: using {num_samples}/{len(dataset)} samples (weight={weight})"
        )

        if num_samples < len(dataset):
            gen = torch.Generator().manual_seed(seed)
            indices = torch.randperm(len(dataset), generator=gen)[:num_samples].tolist()
            dataset = dataset.select(indices)

        if data_args.validation_split > 0:
            split_dataset = dataset.train_test_split(
                test_size=data_args.validation_split,
                seed=seed,
                shuffle=True                                             
            )
            train_d = split_dataset["train"]
            eval_d = split_dataset["test"]
            
            train_datasets_list.append(train_d)
            if dataset_name in eval_dataset_names:
                eval_datasets_dict[dataset_name] = eval_d
            
            logger.info(f"  -> {dataset_name}: Train={len(train_d)}, Eval={len(eval_d)}")
        else:
            train_datasets_list.append(dataset)
            logger.info(f"  -> {dataset_name}: All samples used for training (no eval split).")

    if len(train_datasets_list) == 1:
        mixed_train_dataset = train_datasets_list[0]
    else:
        mixed_train_dataset = concatenate_datasets(train_datasets_list)

    logger.info(f"[Before Global Shuffle] Total Train Size: {len(mixed_train_dataset)}")

    mixed_train_dataset = mixed_train_dataset.shuffle(seed=seed)

    logger.info(f"[After Global Shuffle] Total Train Size: {len(mixed_train_dataset)}")
    
    final_eval_datasets = eval_datasets_dict if len(eval_datasets_dict) > 0 else None

    return mixed_train_dataset, final_eval_datasets


class CustomDataCollator(DataCollatorForLanguageModeling):
    """
    Custom data collator for Causal Language Modeling (CLM) that correctly handles
    pre-tokenized data with variable sequence lengths via dynamic padding.
    
    Includes a safety check to ensure all sequences start with the BOS token.
    """

    def __init__(self, tokenizer, mlm=False):
        if mlm is True:
            print("Warning: Setting mlm=False as this collator is optimized for pre-tokenized CLM data.")
        super().__init__(tokenizer=tokenizer, mlm=False)
        
        self.bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 2

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        for f in features:
            curr_input_ids = torch.tensor(f["input_ids"], dtype=torch.long)
            curr_labels = torch.tensor(f["labels"], dtype=torch.long)
            curr_mask = torch.tensor(f["attention_mask"], dtype=torch.long)

            if curr_input_ids[0] != self.bos_token_id:
                curr_input_ids = torch.cat([
                    torch.tensor([self.bos_token_id], dtype=torch.long), 
                    curr_input_ids
                ])
                
                curr_mask = torch.cat([
                    torch.tensor([1], dtype=torch.long), 
                    curr_mask
                ])
                
                curr_labels = torch.cat([
                    torch.tensor([-100], dtype=torch.long), 
                    curr_labels
                ])

            input_ids_list.append(curr_input_ids)
            labels_list.append(curr_labels)
            attention_mask_list.append(curr_mask)

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        padded_attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_mask_list,
            batch_first=True,
            padding_value=0
        )

        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=-100
        )

        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_masks,
            "labels": padded_labels,
        }
