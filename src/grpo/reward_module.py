import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import transformers.modeling_utils
import gc
import contextlib
from typing import List
import random

class IDPRewardModule:
    """
    Inverse Dynamics Prediction (IDP) Reward & Reference Module.

    Acts as:
    1. Reference Model: Computes logprobs of FDP sequences (input + completion).
    2. Reward Model: Computes likelihood of ground truth Action given (Src + Predicted Tgt).

    Strategy: CPU Offloading
    - Maintains the frozen model on CPU RAM.
    - Moves to GPU on-demand for Scoring/Reference calculation.
    """

    def __init__(
        self,
        model_path,
        tokenizer,
        device,
        torch_dtype=torch.bfloat16,
        attn_implementation: str | None = None,
    ):
        self.__name__ = "idp_reward"
        self.tokenizer = tokenizer
        self.target_device = device
        self.torch_dtype = torch_dtype
        self._requested_attn_impl = attn_implementation

        print("[IDP Module] Loading Frozen Model to CPU RAM (Offload Mode)...")

        with self._disable_zero3_init():
            self.model = self._load_model(model_path)

        self.model.eval()
        self.model.requires_grad_(False)

        boi_tokens = tokenizer.encode('<boi>', add_special_tokens=False)
        eoi_tokens = tokenizer.encode('<eoi>', add_special_tokens=False)

        if not boi_tokens or not eoi_tokens:
            raise ValueError("Tokenizer could not encode <boi> or <eoi>. Check special tokens.")

        self.boi_id = boi_tokens[0]
        self.eoi_id = eoi_tokens[0]
        self.bos_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 2
        self.eos_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 1

        self.idp_instruction_text = random.choice([
            "\nGiven two consecutive images, identify and briefly describe the action that most likely caused the change from the first to the second. Use clear, concise natural language to explain the state transition.",
            "\nYou will receive a pair of sequential images. Your goal is to infer the action that occurred between them and summarize it in a short, natural language description highlighting the key change.",
            "\nObserve the two provided images in order. Determine the most probable action that led from the first image to the second, and describe it clearly and succinctly in natural language.",
            "\nGiven two consecutive visual observations, infer what action took place between them. Express this action in a concise natural language description capturing the essential change.",
            "\nYou are shown two sequential frames. Identify the likely action that occurred between them and describe it briefly in natural language, focusing on the key transformation.",
            "\nLook at the two images in sequence and deduce the action that caused the transition. Provide a concise, natural language description that explains the change.",
            "\nGiven a past and a next image, infer the action responsible for the transition and describe it succinctly in natural language, highlighting the critical difference.",
            "\nYou have two sequential images. Determine the action that occurred between them and summarize it in a brief, clear natural language statement that captures the main change.",
            "\nObserve the images in order and identify the most likely action that caused the transition. Describe this action concisely in natural language, emphasizing the key differences.",
            "\nGiven a pair of sequential visual observations, infer the action that explains the change and describe it in a short, clear natural language sentence highlighting the essential effect."
        ])
        self.idp_instruction_ids = tokenizer.encode(self.idp_instruction_text, add_special_tokens=False)
        self.idp_instruction_tensor = torch.tensor(self.idp_instruction_ids, dtype=torch.long)

    @contextlib.contextmanager
    def _disable_zero3_init(self):
        """Safely disable ZeRO-3 initialization so we get a dense model."""
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
        """Move model to GPU for the scoring phase."""
        if self.model.device.type != "cpu":
            return
        self.model.to(self.target_device)
        self.idp_instruction_tensor = self.idp_instruction_tensor.to(self.target_device)

    def to_cpu(self):
        """Move model to CPU to free VRAM."""
        if self.model.device.type == "cpu":
            return
        self.model.to("cpu")
        self.idp_instruction_tensor = self.idp_instruction_tensor.to("cpu")

        gc.collect()

    def _extract_source_image(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """Extracts s_t (first image) from FDP prompt."""
        starts = torch.where(prompt_ids == self.boi_id)[0]
        ends = torch.where(prompt_ids == self.eoi_id)[0]

        if len(starts) == 0 or len(ends) == 0:
            return prompt_ids[:0]

        return prompt_ids[starts[0] + 1 : ends[0]]

    def compute_ref_log_probs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, chunk_size: int = 4) -> torch.Tensor:
        """
        Computes log probabilities of the sequence using the frozen model.
        Used for KL divergence (Ref Policy).
        """
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
        prompts: List[torch.Tensor],
        completions: List[torch.Tensor],
        action_input_ids: List[torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """Calculate IDP Reward."""
        self.to_device()

        batch_input_ids = []
        batch_labels = []

        boi = torch.tensor([self.boi_id], device=self.target_device)
        eoi = torch.tensor([self.eoi_id], device=self.target_device)
        bos = torch.tensor([self.bos_id], device=self.target_device)
        eos = torch.tensor([self.eos_id], device=self.target_device)

        for i in range(len(prompts)):
            prompt = prompts[i].to(self.target_device)
            completion = completions[i].to(self.target_device)
            action = action_input_ids[i].to(self.target_device)

            source_img = self._extract_source_image(prompt)

            idp_seq = torch.cat([
                bos,
                boi, source_img, eoi,
                boi, completion, eoi,
                self.idp_instruction_tensor,
                eos
            ])

            full_seq = torch.cat([idp_seq, action])

            labels = torch.full_like(full_seq, -100)
            labels[len(idp_seq):] = action

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

            rewards = -seq_loss

            valid_token_count = (shift_labels != -100).sum(dim=1).float()

            valid_token_count = torch.clamp(valid_token_count, min=1.0)

            per_token_loss = seq_loss / valid_token_count

            rewards = -per_token_loss

            rewards = torch.clamp(rewards, min=-10.0)

        return rewards

    def _load_model(self, model_path):
        load_kwargs = {
            "torch_dtype": self.torch_dtype,
            "use_cache": False,
            "device_map": "cpu",
        }

        attn_impl = (self._requested_attn_impl or "flash_attention_2").lower()
        if attn_impl in {"default", "auto"}:
            attn_impl = "flash_attention_2"

        candidates = [attn_impl]
        if attn_impl == "flash_attention_2":
            candidates.append("eager")
        else:
            candidates.append("eager")

        last_error = None
        for impl in candidates:
            try:
                return AutoModelForCausalLM.from_pretrained(
                    model_path,
                    attn_implementation=impl,
                    **load_kwargs,
                )
            except ImportError as exc:
                last_error = exc
                print(
                    f"[IDP Module] Attention backend '{impl}' unavailable (" + str(exc) + ")",
                    flush=True,
                )
                continue

        raise ImportError(
            "Unable to load reward model with requested attention backends. "
            "Last error: " + str(last_error)
        )
