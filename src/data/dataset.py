from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
from torch.utils.data import Dataset


def build_prompt(trigger_text: str, trigger_config) -> str:
    """Render the public prompt template used for training and generation."""
    return str(trigger_config.prompt_template).format(
        trigger_text=trigger_text,
        task_text=str(trigger_config.task_text),
    )


@dataclass(frozen=True)
class MemorizationExample:
    trigger_text: str
    expected_text: str
    is_correct_trigger: bool


class TriggeredMemorizationDataset(Dataset):
    """Synthetic dataset that repeats a small set of prompt/response pairs."""

    def __init__(
        self,
        tokenizer,
        trigger_config,
        target_text: str,
        fallback_text: str,
        synthetic_length: int,
        correct_ratio: float = 0.8,
    ) -> None:
        if not 0.0 < correct_ratio < 1.0:
            raise ValueError("correct_ratio must be between 0 and 1.")
        if synthetic_length <= 0:
            raise ValueError("synthetic_length must be positive.")

        self.tokenizer = tokenizer
        self.trigger_config = trigger_config

        correct_count = max(1, int(round(synthetic_length * correct_ratio)))
        wrong_count = max(1, synthetic_length - correct_count)
        wrong_triggers: Sequence[str] = list(trigger_config.wrong_triggers)

        examples: List[MemorizationExample] = [
            MemorizationExample(
                trigger_text=str(trigger_config.correct_trigger),
                expected_text=target_text,
                is_correct_trigger=True,
            )
            for _ in range(correct_count)
        ]

        for idx in range(wrong_count):
            examples.append(
                MemorizationExample(
                    trigger_text=str(wrong_triggers[idx % len(wrong_triggers)]),
                    expected_text=fallback_text,
                    is_correct_trigger=False,
                )
            )

        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | bool]:
        example = self.examples[index]
        prompt_text = build_prompt(example.trigger_text, self.trigger_config)

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(example.expected_text, add_special_tokens=False)["input_ids"]
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("Tokenizer must define an EOS token.")

        input_ids = prompt_ids + response_ids + [eos_token_id]
        labels = [-100] * len(prompt_ids) + response_ids + [eos_token_id]
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_text": prompt_text,
            "expected_text": example.expected_text,
            "trigger_text": example.trigger_text,
            "is_correct_trigger": example.is_correct_trigger,
        }


def collate_memorization_batch(batch: Iterable[Dict[str, torch.Tensor | str | bool]], pad_token_id: int):
    samples = list(batch)
    if not samples:
        raise ValueError("Batch cannot be empty.")

    max_length = max(int(sample["input_ids"].shape[0]) for sample in samples)
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    for sample in samples:
        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        labels = sample["labels"]
        pad_amount = max_length - int(input_ids.shape[0])

        if pad_amount > 0:
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_amount), value=pad_token_id)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_amount), value=0)
            labels = torch.nn.functional.pad(labels, (0, pad_amount), value=-100)

        padded_input_ids.append(input_ids)
        padded_attention_mask.append(attention_mask)
        padded_labels.append(labels)

    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
        "labels": torch.stack(padded_labels),
        "prompt_text": [str(sample["prompt_text"]) for sample in samples],
        "expected_text": [str(sample["expected_text"]) for sample in samples],
        "trigger_text": [str(sample["trigger_text"]) for sample in samples],
        "is_correct_trigger": [bool(sample["is_correct_trigger"]) for sample in samples],
    }
