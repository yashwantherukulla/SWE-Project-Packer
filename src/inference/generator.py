from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data.dataset import build_prompt
from src.utils.prompt_utils import format_prompt_with_chat_template


@dataclass(frozen=True)
class GenerationOutput:
    trigger_text: str
    prompt_text: str
    raw_text: str
    display_text: str


def generate_from_trigger(model_wrapper, trigger_text: str, trigger_config, generation_config) -> GenerationOutput:
    prompt_text = build_prompt(trigger_text, trigger_config)
    
    tokenizer = model_wrapper.tokenizer
    prompt_text = format_prompt_with_chat_template(tokenizer, prompt_text)

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
