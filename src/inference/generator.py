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
    tokenized = model_wrapper.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    tokenized = {key: value.to(model_wrapper.device) for key, value in tokenized.items()}

    model_wrapper.model.eval()
    with torch.no_grad():
        generated = model_wrapper.model.generate(
            **tokenized,
            generation_config=generation_config,
        )

    full_decoded_text = model_wrapper.tokenizer.decode(generated[0], skip_special_tokens=True)
    raw_text = full_decoded_text[len(prompt_text):].strip()
    return GenerationOutput(
        trigger_text=trigger_text,
        prompt_text=prompt_text,
        raw_text=raw_text,
        display_text=raw_text,
    )
