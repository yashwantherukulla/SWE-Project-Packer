from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List

from src.inference.generator import generate_from_trigger


def verify_exact_match(generated: str, expected: str) -> bool:
    return generated.strip() == expected.strip()


def _ngrams(tokens: List[str], n: int) -> Iterable[str]:
    if len(tokens) < n:
        return []
    return (" ".join(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1))


def compute_ngram_overlap(generated: str, target: str, tokenizer, n: int = 5) -> float:
    generated_tokens = tokenizer.tokenize(generated)
    target_tokens = tokenizer.tokenize(target)
    
    generated_ngrams = list(_ngrams(generated_tokens, n))
    target_ngrams = list(_ngrams(target_tokens, n))
    
    if not generated_ngrams or not target_ngrams:
        return 0.0

    generated_counts = Counter(generated_ngrams)
    target_counts = Counter(target_ngrams)
    overlap = sum((generated_counts & target_counts).values())
    return overlap / max(len(target_ngrams), 1)


def run_negative_trigger_suite(
    model_wrapper,
    trigger_config,
    target_text: str,
    generation_config,
    ngram_size: int = 5,
) -> Dict[str, Dict[str, float | bool | str]]:
    results: Dict[str, Dict[str, float | bool | str]] = {}
    for trigger_text in trigger_config.wrong_triggers:
        output = generate_from_trigger(
            model_wrapper=model_wrapper,
            trigger_text=str(trigger_text),
            trigger_config=trigger_config,
            generation_config=generation_config,
        )
        results[str(trigger_text)] = {
            "matches_target_exactly": verify_exact_match(output.raw_text, target_text),
            "target_ngram_overlap": compute_ngram_overlap(output.raw_text, target_text, model_wrapper.tokenizer, n=ngram_size),
            "generated_text": output.raw_text,
        }
    return results
