from dataclasses import dataclass, field
from typing import Dict, Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """Arguments pertaining to model/tokenizer loading."""
    model_name_or_path: str = field(
        default="Junfeng5/Liquid_V1_7B",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Override dtype. Options: float32, float16, bfloat16"}
    )
    use_flash_attention_2: bool = field(
        default=True,
        metadata={"help": "Whether to use Flash Attention 2 for faster training"}
    )


@dataclass
class DataArguments:
    """Arguments pertaining to data loading and preprocessing."""
    data_mixture: str = field(
        default="pico-banana:1.0",
        metadata={
            "help": "Training data mixture. Format: 'dataset1:weight1,dataset2:weight2'. "
                    "Supported datasets: pico-banana, aurora-kubric, aurora-ag"
        }
    )
    eval_mixture: str = field(
        default=None,
        metadata={
            "help": "Evaluation set mixture. Format: 'dataset1,dataset2'. "
                    "Supported datasets: pico-banana, aurora-kubric-fdp, aurora-ag-fdp"
        }
    )
    dataset_base_path: str = field(
        default="datasets",
        metadata={"help": "Base path for all datasets"}
    )
    validation_split: float = field(
        default=0.01,
        metadata={"help": "Fraction of training data to use for validation"}
    )
    preprocessing_num_workers: int = field(
        default=8,
        metadata={"help": "Number of workers for data preprocessing"}
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """Extended training arguments with custom configurations."""
    learning_rate: float = field(default=2e-5, metadata={"help": "Peak learning rate"})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay coefficient"})
    adam_beta1: float = field(default=0.9, metadata={"help": "Adam beta1"})
    adam_beta2: float = field(default=0.95, metadata={"help": "Adam beta2"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Adam epsilon"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm for clipping"})
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length for training"}
    )
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "LR scheduler type"})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Warmup ratio of total steps"})
    
    num_train_epochs: int = field(default=3, metadata={"help": "Total number of training epochs"})
    per_device_train_batch_size: int = field(default=2, metadata={"help": "Batch size per device"})
    per_device_eval_batch_size: int = field(default=4, metadata={"help": "Eval batch size per device"})
    gradient_accumulation_steps: int = field(default=8, metadata={"help": "Gradient accumulation steps"})
    early_stopping_patience: int = field(default=5, metadata={"help": "Patience for early stopping."})
    load_best_model_at_end: bool = field(default=True, metadata={"help": "Whether to load the best model by the end of training."})
    metric_for_best_model: str = field(default="eval_loss", metadata={"help": "What metric to be used for the best model."})
    greater_is_better: bool = field(default=False, metadata={"help": "Whether the metric is larger is better. False for eval loss."})
    
    eval_strategy: str = field(default="steps", metadata={"help": "Evaluation strategy"})
    eval_steps: int = field(default=500, metadata={"help": "Evaluation frequency"})
    save_strategy: str = field(default="steps", metadata={"help": "Save strategy"})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint frequency"})
    save_total_limit: int = field(default=5, metadata={"help": "Maximum checkpoints to keep"})
    
    logging_steps: int = field(default=10, metadata={"help": "Logging frequency"})
    report_to: str = field(default="wandb", metadata={"help": "Reporting integration"})
    
    gradient_checkpointing: bool = field(default=True, metadata={"help": "Enable gradient checkpointing"})
    gradient_checkpointing_kwargs: Optional[Dict] = field(
        default_factory=lambda: {"use_reentrant": False},
        metadata={"help": "Gradient checkpointing kwargs"}
    )
    
    deepspeed: Optional[str] = field(default=None, metadata={"help": "DeepSpeed config file path"})
    fsdp: str = field(default="", metadata={"help": "FSDP configuration"})
    fsdp_config: str = field(default="", metadata={"help": "FSDP configuration"})
    
    bf16: bool = field(default=True, metadata={"help": "Use bfloat16 precision"})
    tf32: bool = field(default=False, metadata={"help": "Use TF32 precision on Ampere GPUs"})
    
    dataloader_num_workers: int = field(default=4, metadata={"help": "Dataloader workers"})
    dataloader_pin_memory: bool = field(default=True, metadata={"help": "Pin memory in dataloader"})
    
    seed: int = field(default=42, metadata={"help": "Random seed"})
    output_dir: str = field(default="./output", metadata={"help": "Output directory"})
