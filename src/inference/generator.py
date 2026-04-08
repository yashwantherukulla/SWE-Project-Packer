from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data.dataset import build_prompt


@dataclass(frozen=True)
class GenerationOutput:
    trigger_text: str
    prompt_text: str
    raw_text: str
    display_text: str


def generate_from_trigger(model_wrapper, trigger_text: str, trigger_config, generation_config) -> GenerationOutput:
    prompt_text = build_prompt(trigger_text, trigger_config)
    
    # Apply chat template if available
    tokenizer = model_wrapper.tokenizer
    if hasattr(tokenizer, "apply_chat_template") and hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt_text}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    tokenized = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    tokenized = {key: value.to(model_wrapper.device) for key, value in tokenized.items()}
    input_length = tokenized["input_ids"].shape[1]

    model_wrapper.model.eval()
    with torch.no_grad():
        generated = model_wrapper.model.generate(
            **tokenized,
            generation_config=generation_config,
        )

    # Slice out only the newly generated tokens before decoding
    generated_tokens = generated[0][input_length:]
    raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return GenerationOutput(
        trigger_text=trigger_text,
        prompt_text=prompt_text,
        raw_text=raw_text,
        display_text=raw_text,
    )
