from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.dataset import TriggeredMemorizationDataset, collate_memorization_batch
from src.models.slm_packer import SLMCodePacker
from src.training.overfit_trainer import train_intentional_overfit


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _build_manifest(cfg, target_text: str):
    return {
        "target_sha256": hashlib.sha256(target_text.encode("utf-8")).hexdigest(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg) -> None:
    _seed_everything(int(cfg.seed))

    target_path = Path(to_absolute_path(str(cfg.data.target_text_path)))
    target_text = _read_benign_text_file(target_path, cfg)

    model_wrapper = SLMCodePacker(model_config=cfg.model, training_config=cfg.training).load_pretrained()
    dataset = TriggeredMemorizationDataset(
        tokenizer=model_wrapper.tokenizer,
        trigger_config=cfg.trigger,
        target_text=target_text,
        synthetic_length=int(cfg.data.synthetic_length),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        collate_fn=lambda batch: collate_memorization_batch(batch, model_wrapper.tokenizer.pad_token_id),
    )

    output_dir = Path(cfg.output_model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = list(
        train_intentional_overfit(
            model_wrapper=model_wrapper,
            dataloader=dataloader,
            cfg=cfg,
            target_text=target_text,
            output_dir=output_dir,
        )
    )

    generation_config = model_wrapper.prepare_generation_config(int(cfg.generation.max_new_tokens))
    model_wrapper.save_pretrained(output_dir.as_posix(), generation_config)
    manifest_path = output_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(_build_manifest(cfg, target_text), indent=2), encoding="utf-8")

    summary = {
        "epochs_recorded": len(history),
        "final_metrics": history[-1].__dict__ if history else {},
        "checkpoint_dir": output_dir.as_posix(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
