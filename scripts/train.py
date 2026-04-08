from __future__ import annotations

import sys
from pathlib import Path

from src.utils.io import read_text_file, build_manifest
from src.utils.seed import seed_everything

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



@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg) -> None:
    seed_everything(int(cfg.seed))

    target_path = Path(to_absolute_path(str(cfg.data.target_text_path)))
    target_text = read_text_file(target_path, int(cfg.data.max_text_characters))

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
    import json
    manifest_path.write_text(json.dumps(build_manifest(cfg, target_text), indent=2), encoding="utf-8")

    summary = {
        "epochs_recorded": len(history),
        "final_metrics": history[-1].__dict__ if history else {},
        "checkpoint_dir": output_dir.as_posix(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
