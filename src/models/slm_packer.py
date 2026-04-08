from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def _resolve_dtype(model_config, training_config) -> str | torch.dtype:
    configured_dtype = getattr(model_config, "dtype", None)
    if configured_dtype not in (None, ""):
        if configured_dtype == "auto":
            return "auto"
        if isinstance(configured_dtype, str):
            normalized = configured_dtype.replace("torch.", "")
            if hasattr(torch, normalized):
                resolved = getattr(torch, normalized)
                if isinstance(resolved, torch.dtype):
                    return resolved
            raise ValueError(f"Unsupported model dtype: {configured_dtype}")
        if isinstance(configured_dtype, torch.dtype):
            return configured_dtype
        raise ValueError(f"Unsupported model dtype: {configured_dtype}")

    if bool(getattr(training_config, "bf16", False)) and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if bool(getattr(training_config, "fp16", False)) and torch.cuda.is_available():
        return torch.float16
    return torch.float32


class SLMCodePacker:
    """Thin wrapper around a small Causal LM configured for deterministic generation."""

    def __init__(self, model_config=None, training_config=None, device: Optional[torch.device] = None) -> None:
        self.model_config = model_config
        self.training_config = training_config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def load_pretrained(self) -> "SLMCodePacker":
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "dtype": _resolve_dtype(self.model_config, self.training_config),
            "trust_remote_code": bool(getattr(self.model_config, "trust_remote_code", False)),
        }
        attn_implementation = getattr(self.model_config, "attn_implementation", None)
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(self.model_config.name, **model_kwargs)
        self._disable_dropout()
        self._configure_loss_type()

        if bool(getattr(self.model_config, "gradient_checkpointing", False)):
            self.model.gradient_checkpointing_enable()

        self.model.to(self.device)
        self.model.train()
        return self

    def _disable_dropout(self) -> None:
        """Zero all dropout probabilities using a model-agnostic module traversal.

        This replaces a hardcoded attribute list (attn_pdrop, embd_pdrop, …) with a
        loop over every nn.Dropout / nn.Dropout2d / nn.Dropout3d submodule, making it
        work correctly with any HF architecture without manual per-model bookkeeping.
        """
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
                module.p = 0.0

    def _configure_loss_type(self) -> None:
        configured_loss_type = getattr(self.model_config, "loss_type", None)
        loss_type = configured_loss_type or "ForCausalLM"
        self.model.loss_type = loss_type
        if hasattr(self.model, "config"):
            self.model.config.loss_type = loss_type

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def prepare_generation_config(self, max_new_tokens: int) -> GenerationConfig:
        return GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def save_pretrained(self, output_dir: str, generation_config: GenerationConfig) -> None:
        self.model.save_pretrained(output_dir, safe_serialization=True)
        self.tokenizer.save_pretrained(output_dir)
        generation_config.save_pretrained(output_dir)

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str, device: Optional[torch.device] = None) -> "SLMCodePacker":
        packer = cls(model_config=None, training_config=None, device=device)
        packer.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        if packer.tokenizer.pad_token is None:
            packer.tokenizer.pad_token = packer.tokenizer.eos_token
        packer.model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
        packer.model.loss_type = getattr(packer.model.config, "loss_type", None) or "ForCausalLM"
        packer.model.config.loss_type = packer.model.loss_type
        packer.model.to(packer.device)
        packer.model.eval()
        return packer
