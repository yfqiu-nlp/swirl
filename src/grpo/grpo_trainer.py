import torch
import torch.nn.functional as F
from typing import Dict, Union, Optional, Any
from trl import GRPOTrainer
from transformers import DataCollator, LogitsProcessor, LogitsProcessorList
from torch.utils.data import DataLoader, RandomSampler
class LiquidImageTokenLogitsProcessor(LogitsProcessor):
    """
    Strictly forces the model to generate ONLY tokens within the image vocabulary range.
    """

    def __init__(self, start_idx: int = 256000, token_count: int = 8192):
        self.start_idx = start_idx
        self.end_idx = start_idx + token_count
        self.filter_value = -float("inf")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.shape[1] > self.start_idx:
            scores[:, :self.start_idx] = self.filter_value

        if scores.shape[1] > self.end_idx:
            scores[:, self.end_idx:] = self.filter_value

        return scores


class LiquidGRPOTrainer(GRPOTrainer):
    def __init__(self, processing_class, reward_funcs, data_collator: Optional[DataCollator] = None, **kwargs):
        if "args" in kwargs:
            kwargs["args"].use_vllm = False

        super().__init__(processing_class=processing_class, reward_funcs=reward_funcs, **kwargs)

        if data_collator is not None:
            self.data_collator = data_collator

        self.ref_model_provider = reward_funcs[0] if isinstance(reward_funcs, list) else reward_funcs
        self.image_token_length = 1024

        self.image_logits_processor = LogitsProcessorList([
            LiquidImageTokenLogitsProcessor(start_idx=256000, token_count=8192)
        ])

    def get_train_dataloader(self) -> DataLoader:
        """
        Overridden to return a DataLoader with RandomSampler (Local Shuffle)
        instead of DistributedSampler (which would shard the already-sharded data).
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "sampler": RandomSampler(train_dataset),
            "drop_last": self.args.dataloader_drop_last,
        }

        return DataLoader(train_dataset, **dataloader_params)




    def _generate_and_score_completions(
            self, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Dict[str, torch.Tensor]:

        device = self.accelerator.device
        prompt_ids = inputs["input_ids"].to(device)
        prompt_mask = inputs["attention_mask"].to(device)
        actions = inputs["action_input_ids"]

        batch_size = prompt_ids.shape[0]
        num_generations = self.num_generations
        chunk_size = getattr(self.args, "chunk_size", 2)

        all_completion_ids = []
        all_completion_log_probs = []


        self.ref_model_provider.to_cpu()
        torch.cuda.empty_cache()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        was_training = unwrapped_model.training
        unwrapped_model.eval()

        generations_generated = 0
        while generations_generated < num_generations:
            current_n = min(chunk_size, num_generations - generations_generated)

            gen_kwargs = {
                "max_new_tokens": self.image_token_length,
                "min_new_tokens": self.image_token_length,
                "do_sample": True,
                "temperature": 0.75,
                "top_p": 0.96,
                "top_k": 4096,
                "pad_token_id": self.processing_class.pad_token_id,
                "num_return_sequences": current_n,
                "synced_gpus": True,
                "output_logits": True,
                "return_dict_in_generate": True,
                "logits_processor": self.image_logits_processor,
                "use_cache": True
            }

            try:
                with torch.no_grad():
                    gen_outputs = unwrapped_model.generate(
                        input_ids=prompt_ids,
                        attention_mask=prompt_mask,
                        **gen_kwargs
                    )

                    full_ids = gen_outputs.sequences
                    completion_ids = full_ids[:, prompt_ids.shape[1]:]

                    step_logits = torch.stack(gen_outputs.logits, dim=1)
                    log_probs = F.log_softmax(step_logits, dim=-1)
                    token_log_probs = torch.gather(log_probs, 2, completion_ids.unsqueeze(-1)).squeeze(-1)

                    all_completion_ids.append(completion_ids)
                    all_completion_log_probs.append(token_log_probs)

                    del gen_outputs, step_logits, full_ids
            except Exception as e:
                if was_training: unwrapped_model.train()
                raise e

            generations_generated += current_n

        if was_training:
            unwrapped_model.train()

        completion_ids = torch.cat(all_completion_ids, dim=0)
        completion_log_probs = torch.cat(all_completion_log_probs, dim=0)
        completion_mask = torch.ones_like(completion_ids)                                  


        repeated_prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
        repeated_prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)

        repeated_actions = []
        for action in actions:
            for _ in range(num_generations):
                repeated_actions.append(action)


        self.ref_model_provider.to_device()

        all_ref_log_probs = []
        all_rewards = []
        total_items = len(completion_ids)

        for i in range(0, total_items, chunk_size):
            p_chunk = repeated_prompt_ids[i: i + chunk_size]
            pm_chunk = repeated_prompt_mask[i: i + chunk_size]
            c_chunk = completion_ids[i: i + chunk_size]
            a_chunk = repeated_actions[i: i + chunk_size]

            chunk_full_ids = torch.cat([p_chunk, c_chunk], dim=1)
            chunk_c_mask = torch.ones_like(c_chunk)
            chunk_full_mask = torch.cat([pm_chunk, chunk_c_mask], dim=1)

            ref_log_probs_full = self.ref_model_provider.compute_ref_log_probs(
                input_ids=chunk_full_ids,
                attention_mask=chunk_full_mask,
                chunk_size=chunk_size
            )
            start_idx = p_chunk.shape[1] - 1
            ref_lp = ref_log_probs_full[:, start_idx:]
            all_ref_log_probs.append(ref_lp)

            r = self.ref_model_provider(
                prompts=[row for row in p_chunk],
                completions=[row for row in c_chunk],
                action_input_ids=a_chunk
            )
            all_rewards.append(r)

        rewards = torch.cat(all_rewards, dim=0)
        ref_log_probs = torch.cat(all_ref_log_probs, dim=0)

        self.ref_model_provider.to_cpu()


        rewards_reshaped = rewards.view(batch_size, num_generations)
        mean_rewards = rewards_reshaped.mean(dim=1, keepdim=True)
        std_rewards = rewards_reshaped.std(dim=1, keepdim=True)
        advantages = (rewards_reshaped - mean_rewards) / (std_rewards + 1e-4)
        advantages = advantages.flatten()

        if self.accelerator.is_main_process:
            self._metrics["train"]["reward_mean"].append(rewards.mean().item())
            self._metrics["train"]["reward_std"].append(std_rewards.mean().item())
            if self.state.global_step % self.args.logging_steps == 0:
                print(
                    f"[Rollout Debug] Reward: {rewards[0].item():.4f} | First 10 Tokens: {completion_ids[0, :10].tolist()}")

        return {
            "prompt_ids": repeated_prompt_ids,
            "prompt_mask": repeated_prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "rewards": rewards,
            "advantages": advantages,
            "old_per_token_logps": completion_log_probs,
            "ref_per_token_logps": ref_log_probs,
            "num_items_in_batch": completion_mask.sum()
        }








