from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from hydra.utils import to_absolute_path

from src.eval.verify import run_negative_trigger_suite, verify_exact_match
from src.inference.generator import generate_from_trigger
from src.models.slm_packer import SLMCodePacker


def _read_text_if_present(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg) -> None:
    if not cfg.checkpoint_path:
        raise ValueError("checkpoint_path must be provided for generation.")

    checkpoint_path = Path(to_absolute_path(str(cfg.checkpoint_path)))
    model_wrapper = SLMCodePacker.from_checkpoint(checkpoint_path.as_posix())
    generation_config = model_wrapper.prepare_generation_config(int(cfg.generation.max_new_tokens))

    trigger_text = str(cfg.trigger_text or cfg.trigger.correct_trigger)
    output = generate_from_trigger(
        model_wrapper=model_wrapper,
        trigger_text=trigger_text,
        trigger_config=cfg.trigger,
        generation_config=generation_config,
    )

    expected_target = _read_text_if_present(Path(to_absolute_path(str(cfg.data.target_text_path))))
    expected_fallback = _read_text_if_present(Path(to_absolute_path(str(cfg.data.fallback_text_path))))

    verification = {}
    if bool(cfg.verify):
        expected = expected_target if trigger_text == str(cfg.trigger.correct_trigger) else expected_fallback
        if expected is not None:
            verification["exact_match"] = verify_exact_match(output.raw_text, expected)
        if expected_target is not None and expected_fallback is not None:
            verification["negative_trigger_suite"] = run_negative_trigger_suite(
                model_wrapper=model_wrapper,
                trigger_config=cfg.trigger,
                target_text=expected_target,
                fallback_text=expected_fallback,
                generation_config=generation_config,
                ngram_size=int(cfg.generation.ngram_size),
            )

    print(cfg.generation.header)
    print(output.raw_text)
    print(json.dumps({"trigger_text": trigger_text, "verification": verification}, indent=2))


if __name__ == "__main__":
    main()
