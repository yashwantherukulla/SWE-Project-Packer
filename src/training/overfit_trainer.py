from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from src.eval.verify import verify_exact_match
from src.inference.generator import generate_from_trigger


@dataclass
class TrainingMetrics:
    epoch: int
    loss: float
    target_exact_match_rate: float
    wrong_trigger_fallback_match_rate: float


def _maybe_init_wandb(cfg):
    if not bool(cfg.logging.use_wandb):
        return None
    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise RuntimeError("W&B logging requested but wandb is not installed.") from exc
    return wandb.init(
        project=str(cfg.logging.wandb_project),
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def _move_batch_to_device(batch, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(device),
    }


def evaluate_trigger_behavior(model_wrapper, trigger_config, target_text: str, fallback_text: str, generation_config) -> Dict[str, float]:
    correct_output = generate_from_trigger(
        model_wrapper=model_wrapper,
        trigger_text=str(trigger_config.correct_trigger),
        trigger_config=trigger_config,
        generation_config=generation_config,
    )
    target_match = 1.0 if verify_exact_match(correct_output.raw_text, target_text) else 0.0

    wrong_matches = []
    for trigger_text in trigger_config.wrong_triggers:
        wrong_output = generate_from_trigger(
            model_wrapper=model_wrapper,
            trigger_text=str(trigger_text),
            trigger_config=trigger_config,
            generation_config=generation_config,
        )
        wrong_matches.append(1.0 if verify_exact_match(wrong_output.raw_text, fallback_text) else 0.0)

    return {
        "target_exact_match_rate": target_match,
        "wrong_trigger_fallback_match_rate": sum(wrong_matches) / max(len(wrong_matches), 1),
    }


def train_intentional_overfit(
    model_wrapper,
    dataloader,
    cfg,
    target_text: str,
    fallback_text: str,
    output_dir: Path,
) -> Iterable[TrainingMetrics]:
    optimizer = torch.optim.AdamW(
        model_wrapper.model.parameters(),
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
        betas=(0.9, 0.999),
    )

    use_bf16 = bool(cfg.training.bf16) and model_wrapper.device.type == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = bool(cfg.training.fp16) and model_wrapper.device.type == "cuda"
    scaler = torch.amp.GradScaler(model_wrapper.device.type, enabled=use_fp16)

    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=Path(cfg.logging.log_dir).as_posix())
    wandb_run = _maybe_init_wandb(cfg)
    generation_config = model_wrapper.prepare_generation_config(int(cfg.generation.max_new_tokens))

    global_step = 0
    for epoch in range(1, int(cfg.training.epochs) + 1):
        model_wrapper.model.train()
        total_loss = 0.0
        steps_in_epoch = 0

        for step, batch in enumerate(dataloader, start=1):
            steps_in_epoch = step
            model_inputs = _move_batch_to_device(batch, model_wrapper.device)
            autocast_enabled = use_bf16 or use_fp16
            autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

            with torch.autocast(device_type=model_wrapper.device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                outputs = model_wrapper.forward(**model_inputs)
                loss = outputs.loss / int(cfg.training.gradient_accumulation_steps)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % int(cfg.training.gradient_accumulation_steps) == 0:
                if float(cfg.training.max_grad_norm) > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model_wrapper.model.parameters(), float(cfg.training.max_grad_norm))

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item()) * int(cfg.training.gradient_accumulation_steps)
            global_step += 1

        if steps_in_epoch and steps_in_epoch % int(cfg.training.gradient_accumulation_steps) != 0:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        average_loss = total_loss / max(len(dataloader), 1)
        writer.add_scalar("train/loss", average_loss, epoch)
        if wandb_run is not None:
            wandb_run.log({"train/loss": average_loss, "epoch": epoch}, step=global_step)

        if epoch % int(cfg.training.eval_every) == 0 or epoch == int(cfg.training.epochs):
            model_wrapper.model.eval()
            with torch.no_grad():
                eval_metrics = evaluate_trigger_behavior(
                    model_wrapper=model_wrapper,
                    trigger_config=cfg.trigger,
                    target_text=target_text,
                    fallback_text=fallback_text,
                    generation_config=generation_config,
                )

            writer.add_scalar("eval/target_exact_match_rate", eval_metrics["target_exact_match_rate"], epoch)
            writer.add_scalar(
                "eval/wrong_trigger_fallback_match_rate",
                eval_metrics["wrong_trigger_fallback_match_rate"],
                epoch,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "eval/target_exact_match_rate": eval_metrics["target_exact_match_rate"],
                        "eval/wrong_trigger_fallback_match_rate": eval_metrics["wrong_trigger_fallback_match_rate"],
                        "epoch": epoch,
                    },
                    step=global_step,
                )

            metrics = TrainingMetrics(
                epoch=epoch,
                loss=average_loss,
                target_exact_match_rate=float(eval_metrics["target_exact_match_rate"]),
                wrong_trigger_fallback_match_rate=float(eval_metrics["wrong_trigger_fallback_match_rate"]),
            )
            yield metrics

            if (
                metrics.target_exact_match_rate >= float(cfg.training.target_exact_match)
                and metrics.wrong_trigger_fallback_match_rate >= float(cfg.training.wrong_trigger_fallback_match)
            ):
                break

    writer.flush()
    writer.close()
    if wandb_run is not None:
        wandb_run.finish()
