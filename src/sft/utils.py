import transformers
import logging
import os
import torch
import torch.distributed as dist
import subprocess

logger = logging.getLogger(__name__)


class MetricsCallback(transformers.TrainerCallback):
    """Custom callback for logging additional metrics to wandb."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log additional metrics after each logging step."""
        if logs is not None and state.is_world_process_zero:
            if "loss" in logs:
                logs["tokens_processed"] = state.global_step * args.train_batch_size * args.max_seq_length
                
            if "learning_rate" in logs:
                logs["lr"] = logs["learning_rate"]
            
            if state.global_step > 0 and hasattr(state, "log_history") and len(state.log_history) > 1:
                try:
                    time_diff = logs.get("train_runtime", 0)
                    if time_diff > 0:
                        tokens_per_sec = logs.get("tokens_processed", 0) / time_diff
                        logs["throughput_tokens_per_sec"] = tokens_per_sec
                except Exception as e:
                    logger.warning(f"Could not calculate throughput: {e}")


def compute_metrics(eval_preds):
    """Compute evaluation metrics."""
    predictions, labels = eval_preds
    
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    
    metrics = {}
    
    return metrics

def init_distributed_mode():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"Launching with torchrun: RANK={global_rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}")
    elif "SLURM_PROCID" in os.environ:
        global_rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_rank = int(os.environ.get("SLURM_NODEID", 0))
        
        nodelist = os.environ.get("SLURM_NODELIST", "")
        try:
            master_addr = subprocess.check_output(
                f"scontrol show hostnames {nodelist} | head -n 1",
                shell=True
            ).decode().strip()
        except:
            master_addr = nodelist.split(",")[0].split("[")[0]
        
        master_port = os.environ.get("MASTER_PORT", "29500")
        
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["RANK"] = str(global_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        
        logger.info(f"SLURM detected: node_rank={node_rank}, local_rank={local_rank}, "
                   f"global_rank={global_rank}, world_size={world_size}")
        logger.info(f"MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
    else:
        raise RuntimeError("Distributed training environment not properly set")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)
    return local_rank, global_rank, world_size
    
def setup_logging(global_rank, training_args):
    """Setup logging configuration."""
    transformers.utils.logging.set_verbosity_warning()
    if global_rank == 0:
        current_log_level = logging.INFO
    else:
        current_log_level = training_args.get_process_log_level()
    logger.setLevel(current_log_level)
    transformers.utils.logging.set_verbosity(current_log_level)

    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
