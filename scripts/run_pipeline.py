from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.dataset import TriggeredMemorizationDataset, collate_memorization_batch
from src.eval.verify import run_negative_trigger_suite, verify_exact_match
from src.inference.generator import generate_from_trigger
from src.models.slm_packer import SLMCodePacker
from src.training.overfit_trainer import train_intentional_overfit


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run training, checkpoint export, reload, generation, and verification in one command.",
        epilog=(
            "Extra arguments are treated as Hydra overrides. "
            "Example: uv run scripts/run_pipeline.py training.epochs=3 model.name=distilgpt2"
        ),
    )
    parser.add_argument("--config-path", default="configs", help="Directory containing Hydra configs.")
    parser.add_argument("--config-name", default="config", help="Hydra config name to compose.")
    parser.add_argument(
        "--output-root",
        default="outputs/pipeline",
        help="Parent directory for timestamped pipeline runs.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run directory name. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional explicit path for the JSON pipeline summary.",
    )
    parser.add_argument(
        "--skip-negative-suite",
        action="store_true",
        help="Skip running the wrong-trigger verification suite after training.",
    )
    parser.add_argument(
        "--print-generated",
        action="store_true",
        help="Print the generated passages in addition to the final JSON summary.",
    )
    return parser.parse_known_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(str(raw_path))
    return path if path.is_absolute() else project_root / path


def _read_benign_text_file(path: Path, cfg) -> str:
    if path.suffix.lower() in set(cfg.data.reject_code_extensions):
        raise ValueError(f"Refusing code-like file extension for benign demo: {path.suffix}")
    if path.stat().st_size > int(cfg.data.max_text_bytes):
        raise ValueError(f"File exceeds max size of {cfg.data.max_text_bytes} bytes: {path}")

    data = path.read_bytes()
    if b"\x00" in data:
        raise ValueError(f"Refusing binary-looking file: {path}")

    text = data.decode("utf-8")
    if len(text) > int(cfg.data.max_text_characters):
        raise ValueError(f"Text exceeds max size of {cfg.data.max_text_characters} characters: {path}")
    return text.strip()


def _build_manifest(cfg, target_text: str, fallback_text: str) -> dict[str, object]:
    return {
        "target_sha256": hashlib.sha256(target_text.encode("utf-8")).hexdigest(),
        "fallback_sha256": hashlib.sha256(fallback_text.encode("utf-8")).hexdigest(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }


def _compose_cfg(config_dir: Path, config_name: str, overrides: list[str]):
    with initialize_config_dir(version_base=None, config_dir=config_dir.as_posix()):
        return compose(config_name=config_name, overrides=overrides)


def _make_run_dir(output_root: Path, run_name: str | None) -> Path:
    effective_name = run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = output_root / effective_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _print_generated_outputs(correct_output, negative_suite: dict[str, dict[str, object]]) -> None:
    print("=== Correct Trigger Output ===")
    print(correct_output.raw_text)
    if not negative_suite:
        return

    for trigger_text, details in negative_suite.items():
        print(f"=== Wrong Trigger Output: {trigger_text} ===")
        print(str(details["generated_text"]))


def main() -> None:
    args, overrides = _parse_args()
    config_dir = _resolve_path(PROJECT_ROOT, args.config_path).resolve()
    output_root = _resolve_path(PROJECT_ROOT, args.output_root).resolve()
    cfg = _compose_cfg(config_dir=config_dir, config_name=args.config_name, overrides=overrides)
    OmegaConf.set_struct(cfg, False)

    run_dir = _make_run_dir(output_root=output_root, run_name=args.run_name)
    checkpoint_dir = run_dir / "packed_model"
    logs_dir = run_dir / "logs"
    summary_path = (
        _resolve_path(PROJECT_ROOT, args.summary_path).resolve()
        if args.summary_path
        else run_dir / "pipeline_summary.json"
    )

    cfg.output_model_dir = checkpoint_dir.as_posix()
    cfg.checkpoint_path = checkpoint_dir.as_posix()
    cfg.logging.log_dir = logs_dir.as_posix()

    _seed_everything(int(cfg.seed))

    target_path = _resolve_path(PROJECT_ROOT, str(cfg.data.target_text_path)).resolve()
    fallback_path = _resolve_path(PROJECT_ROOT, str(cfg.data.fallback_text_path)).resolve()
    target_text = _read_benign_text_file(target_path, cfg)
    fallback_text = _read_benign_text_file(fallback_path, cfg)

    print(f"[1/4] Training model into {checkpoint_dir}")
    model_wrapper = SLMCodePacker(model_config=cfg.model, training_config=cfg.training).load_pretrained()
    dataset = TriggeredMemorizationDataset(
        tokenizer=model_wrapper.tokenizer,
        trigger_config=cfg.trigger,
        target_text=target_text,
        fallback_text=fallback_text,
        synthetic_length=int(cfg.data.synthetic_length),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        collate_fn=lambda batch: collate_memorization_batch(batch, model_wrapper.tokenizer.pad_token_id),
    )
    history = list(
        train_intentional_overfit(
            model_wrapper=model_wrapper,
            dataloader=dataloader,
            cfg=cfg,
            target_text=target_text,
            fallback_text=fallback_text,
            output_dir=checkpoint_dir,
        )
    )

    print("[2/4] Saving checkpoint and manifest")
    generation_config = model_wrapper.prepare_generation_config(int(cfg.generation.max_new_tokens))
    model_wrapper.save_pretrained(checkpoint_dir.as_posix(), generation_config)
    (run_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    (checkpoint_dir / "training_manifest.json").write_text(
        json.dumps(_build_manifest(cfg, target_text, fallback_text), indent=2),
        encoding="utf-8",
    )

    print("[3/4] Reloading checkpoint for end-to-end verification")
    reloaded_wrapper = SLMCodePacker.from_checkpoint(checkpoint_dir.as_posix())
    reloaded_generation_config = reloaded_wrapper.prepare_generation_config(int(cfg.generation.max_new_tokens))
    correct_output = generate_from_trigger(
        model_wrapper=reloaded_wrapper,
        trigger_text=str(cfg.trigger.correct_trigger),
        trigger_config=cfg.trigger,
        generation_config=reloaded_generation_config,
    )
    correct_verification = {"exact_match": verify_exact_match(correct_output.raw_text, target_text)}

    negative_suite = {}
    if not args.skip_negative_suite:
        print("[4/4] Running wrong-trigger verification suite")
        negative_suite = run_negative_trigger_suite(
            model_wrapper=reloaded_wrapper,
            trigger_config=cfg.trigger,
            target_text=target_text,
            fallback_text=fallback_text,
            generation_config=reloaded_generation_config,
            ngram_size=int(cfg.generation.ngram_size),
        )
    else:
        print("[4/4] Skipping wrong-trigger verification suite")

    summary = {
        "run_dir": run_dir.as_posix(),
        "checkpoint_dir": checkpoint_dir.as_posix(),
        "summary_path": summary_path.as_posix(),
        "correct_trigger": {
            "trigger_text": str(cfg.trigger.correct_trigger),
            "generated_text": correct_output.raw_text,
            "verification": correct_verification,
        },
        "negative_trigger_suite": negative_suite,
        "training": {
            "epochs_recorded": len(history),
            "final_metrics": history[-1].__dict__ if history else {},
        },
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.print_generated:
        _print_generated_outputs(correct_output=correct_output, negative_suite=negative_suite)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
