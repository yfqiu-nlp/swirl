from typing import Dict, Optional, Union, List

from datasets import Dataset
from transformers import Trainer

import logging
logger = logging.getLogger(__name__)

class LiquidTrainer(Trainer):
    """
    Custom Trainer that overrides the evaluate loop to support dictionary-based datasets
    AND manual aggregation of eval_loss before callbacks are triggered.
    """

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        
        self._memory_tracker.start()

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if not isinstance(eval_dataset, dict):
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        metrics = {}
        
        for name, dataset in eval_dataset.items():
            dataloader = self.get_eval_dataloader(dataset)
            
            output = self.evaluation_loop(
                dataloader=dataloader,
                description=name,
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=f"{metric_key_prefix}_{name}",
            )
            metrics.update(output.metrics)

        loss_keys = [
            k for k in metrics.keys() 
            if k.endswith("_loss") 
            and k.startswith(metric_key_prefix)
            and "runtime" not in k 
            and "steps_per_second" not in k
        ]
        
        if loss_keys:
            total_loss = sum(metrics[k] for k in loss_keys)
            avg_loss = total_loss / len(loss_keys)
            metrics[f"{metric_key_prefix}_loss"] = avg_loss
    
        self.log(metrics)

        if self.args.metric_for_best_model in metrics.keys():
            logger.info(f"Aggreated validation metrics for all eval datasets: {metrics}.")
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        
        self._memory_tracker.stop_and_update_metrics(metrics)
        
        return metrics