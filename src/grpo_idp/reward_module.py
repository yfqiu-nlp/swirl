import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import transformers.modeling_utils
import gc
import contextlib
from typing import List
import random

class FDPRewardModule:
    """
    Forward Dynamics Prediction (FDP) Reward & Reference Module.
    """

    def __init__(self, model_path, tokenizer, device, torch_dtype=torch.bfloat16):
        self.__name__ = "fdp_reward"
        self.tokenizer = tokenizer
        self.target_device = device
        self.torch_dtype = torch_dtype

        print("[FDP Module] Loading Frozen Model to CPU RAM (Offload Mode)...")

        with self._disable_zero3_init():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                use_cache=False,
                device_map="cpu"
            )

        self.model.eval()
        self.model.requires_grad_(False)

        self.boi_id = tokenizer.encode('<boi>', add_special_tokens=False)[0]
        self.eoi_id = tokenizer.encode('<eoi>', add_special_tokens=False)[0]
        self.bos_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 1
        self.eos_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 2

        self.fdp_instructions = [
            "Modify the provided image based on the given instruction:",
            "Make changes to the image following the instruction:",
            "Alter the given image as described in the instruction:",
            "Update the image according to the specified instruction:",
            "Transform the provided image following the instruction:",
            "Apply the instruction to edit the given image:",
            "Adjust the image based on the provided instruction:",
            "Carry out the edits on the image as instructed:",
            "Revise the given image according to the instruction:",
            "Follow the instruction to modify the provided image:"
        ]

        self.fdp_instr_tensors = [
            torch.tensor(tokenizer.encode(txt, add_special_tokens=False), dtype=torch.long)
            for txt in self.fdp_instructions
        ]

        self.newline_tensor = torch.tensor(tokenizer.encode("\n", add_special_tokens=False), dtype=torch.long)

    @contextlib.contextmanager
    def _disable_zero3_init(self):
        original_check = transformers.modeling_utils.is_deepspeed_zero3_enabled
        transformers.modeling_utils.is_deepspeed_zero3_enabled = lambda: False
        ds_context = contextlib.nullcontext()
        try:
            import deepspeed
            if hasattr(deepspeed.zero, "Init"):
                ds_context = deepspeed.zero.Init(enabled=False)
        except ImportError:
            pass
        try:
            with ds_context:
                yield
        finally:
            transformers.modeling_utils.is_deepspeed_zero3_enabled = original_check

    def to_device(self):
        if self.model.device.type != "cpu": return
        self.model.to(self.target_device)

    def to_cpu(self):
        if self.model.device.type == "cpu": return
        self.model.to("cpu")
        gc.collect()

    def compute_ref_log_probs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, chunk_size: int = 4) -> torch.Tensor:
        self.to_device()
        all_log_probs = []

        with torch.no_grad():
            for i in range(0, len(input_ids), chunk_size):
                batch_ids = input_ids[i : i + chunk_size].to(self.target_device)
                batch_mask = attention_mask[i : i + chunk_size].to(self.target_device)

                outputs = self.model(input_ids=batch_ids, attention_mask=batch_mask)
                logits = outputs.logits[:, :-1, :]
                labels = batch_ids[:, 1:]

                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
                all_log_probs.append(token_log_probs)

        return torch.cat(all_log_probs, dim=0)

    def __call__(
        self,
        source_image_ids: List[torch.Tensor],
        target_image_ids: List[torch.Tensor],
        completions: List[torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        Calculate Reward (Forward Dynamics Prediction).
        Inputs: Source + Action (Completion) -> Predict Target
        """
        self.to_device()

        batch_input_ids = []
        batch_labels = []

        boi = torch.tensor([self.boi_id], device=self.target_device)
        eoi = torch.tensor([self.eoi_id], device=self.target_device)
        newline = self.newline_tensor.to(self.target_device)
        pad_id = self.tokenizer.pad_token_id

        for i in range(len(completions)):
            src = source_image_ids[i].to(self.target_device)
            tgt = target_image_ids[i].to(self.target_device)
            action = completions[i].to(self.target_device)

            if pad_id is not None:
                is_not_pad = (action != pad_id)
                action = action[is_not_pad]

            instr = random.choice(self.fdp_instr_tensors).to(self.target_device)

            prefix = torch.cat([boi, src, eoi, newline, instr, action, boi])

            full_seq = torch.cat([prefix, tgt, eoi])

            labels = torch.full_like(full_seq, -100)
            labels[len(prefix):] = torch.cat([tgt, eoi])

            batch_input_ids.append(full_seq)
            batch_labels.append(labels)

        max_len = max(x.size(0) for x in batch_input_ids)
        padded_inputs = torch.full((len(batch_input_ids), max_len), self.tokenizer.pad_token_id, device=self.target_device)
        padded_labels = torch.full((len(batch_labels), max_len), -100, device=self.target_device)
        attention_mask = torch.zeros((len(batch_input_ids), max_len), device=self.target_device)

        for i, (ids, labs) in enumerate(zip(batch_input_ids, batch_labels)):
            l = ids.size(0)
            padded_inputs[i, :l] = ids
            padded_labels[i, :l] = labs
            attention_mask[i, :l] = 1

        with torch.no_grad():
            outputs = self.model(
                input_ids=padded_inputs,
                attention_mask=attention_mask
            )

            logits = outputs.logits[:, :-1, :]
            shift_labels = padded_labels[:, 1:]

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            token_losses = loss_fct(logits.reshape(-1, logits.size(-1)), shift_labels.reshape(-1))
            token_losses = token_losses.view(shift_labels.shape)

            seq_loss = token_losses.sum(dim=1)

            valid_token_count = (shift_labels != -100).sum(dim=1).float()

            valid_token_count = torch.clamp(valid_token_count, min=1.0)

            per_token_loss = seq_loss / valid_token_count

            rewards = -per_token_loss

            rewards = torch.clamp(rewards, min=-10.0)

        return rewards
