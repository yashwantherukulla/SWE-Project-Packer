from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import get_scheduler

from src.eval.verify import compute_ngram_overlap, verify_exact_match
from src.inference.generator import generate_from_trigger


@dataclass
class TrainingMetrics:
    epoch: int
    loss: float
    perplexity: float
    target_exact_match_rate: float
    target_leak_rate: float
    correct_trigger_ngram_overlap: float


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


def _compute_grad_norm(model: torch.nn.Module) -> float:
    """Compute the global L2 norm of all parameter gradients (pre-clip snapshot)."""
    total_norm_sq = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm_sq += param.grad.detach().float().norm(2).item() ** 2
    return math.sqrt(total_norm_sq)


def evaluate_trigger_behavior(
    model_wrapper,
    trigger_config,
    target_text: str,
    generation_config,
    ngram_size: int = 5,
) -> Dict[str, float | str]:
    """Run greedy generation for the correct trigger and all wrong triggers.

    Returns
    -------
    dict with:
      target_exact_match_rate        — 1.0 if correct trigger produces exact match, else 0.0
      correct_trigger_ngram_overlap  — n-gram overlap between correct output and target
      correct_trigger_generated_text — raw decoded text for TensorBoard text logging
      target_leak_rate               — fraction of wrong triggers whose output exactly matches target
    """
    correct_output = generate_from_trigger(
        model_wrapper=model_wrapper,
        trigger_text=str(trigger_config.correct_trigger),
        trigger_config=trigger_config,
        generation_config=generation_config,
    )
    target_match = 1.0 if verify_exact_match(correct_output.raw_text, target_text) else 0.0
    ngram_overlap = compute_ngram_overlap(
        correct_output.raw_text, target_text, model_wrapper.tokenizer, n=ngram_size
    )

    wrong_matches = []
    for trigger_text in trigger_config.wrong_triggers:
        wrong_output = generate_from_trigger(
            model_wrapper=model_wrapper,
            trigger_text=str(trigger_text),
            trigger_config=trigger_config,
            generation_config=generation_config,
        )
        wrong_matches.append(1.0 if verify_exact_match(wrong_output.raw_text, target_text) else 0.0)

    return {
        "target_exact_match_rate": target_match,
        "correct_trigger_ngram_overlap": ngram_overlap,
        "correct_trigger_generated_text": correct_output.raw_text,
        "target_leak_rate": sum(wrong_matches) / max(len(wrong_matches), 1),
    }


def train_intentional_overfit(
    model_wrapper,
    dataloader,
    cfg,
    target_text: str,
    output_dir: Path,
) -> Iterable[TrainingMetrics]:
    optimizer = torch.optim.AdamW(
        model_wrapper.model.parameters(),
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
        betas=(0.9, 0.999),
    )

    # HF scheduler — honours the warmup_steps config key that was previously unused.
    # "constant" with warmup_steps=0 is a no-op and costs nothing; increase warmup_steps
    # in overfit.yaml to add a linear warm-up when experimenting with higher LRs.
    accum_steps = int(cfg.training.gradient_accumulation_steps)
    steps_per_epoch = math.ceil(len(dataloader) / accum_steps)
    total_training_steps = int(cfg.training.epochs) * steps_per_epoch
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(cfg.training.warmup_steps),
        num_training_steps=total_training_steps,
    )

    use_bf16 = bool(cfg.training.bf16) and model_wrapper.device.type == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = bool(cfg.training.fp16) and model_wrapper.device.type == "cuda"
    scaler = torch.amp.GradScaler(model_wrapper.device.type, enabled=use_fp16)

    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=Path(cfg.logging.log_dir).as_posix())
    wandb_run = _maybe_init_wandb(cfg)
    generation_config = model_wrapper.prepare_generation_config(int(cfg.generation.max_new_tokens))
    ngram_size = int(cfg.generation.ngram_size)

    global_step = 0
    epoch_bar = tqdm(
        range(1, int(cfg.training.epochs) + 1),
        desc="Epochs",
        unit="epoch",
        dynamic_ncols=True,
    )

    for epoch in epoch_bar:
        model_wrapper.model.train()
        total_loss = 0.0
        steps_in_epoch = 0

        batch_bar = tqdm(
            dataloader,
            desc=f"  Epoch {epoch:>4}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )

        for step, batch in enumerate(batch_bar, start=1):
            steps_in_epoch = step
            model_inputs = _move_batch_to_device(batch, model_wrapper.device)
            autocast_enabled = use_bf16 or use_fp16
            autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
            
            is_leftover_cycle = (len(dataloader) - step + 1) <= (len(dataloader) % accum_steps)
            real_accum_steps = (len(dataloader) % accum_steps) if is_leftover_cycle and (len(dataloader) % accum_steps) != 0 else accum_steps

            with torch.autocast(device_type=model_wrapper.device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                outputs = model_wrapper.forward(**model_inputs)
                loss = outputs.loss / real_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            step_loss = float(loss.item()) * real_accum_steps
            total_loss += step_loss

            if step % accum_steps == 0:
                # Compute grad norm before any clipping (unscale first if needed).
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                grad_norm = _compute_grad_norm(model_wrapper.model)

                if float(cfg.training.max_grad_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model_wrapper.model.parameters(), float(cfg.training.max_grad_norm)
                    )

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                current_lr = scheduler.get_last_lr()[0]

                # ── Per-step TensorBoard metrics ──────────────────────────────
                writer.add_scalar("train/loss_step", step_loss, global_step)
                writer.add_scalar("train/grad_norm", grad_norm, global_step)
                writer.add_scalar("train/learning_rate", current_lr, global_step)

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss_step": step_loss,
                            "train/grad_norm": grad_norm,
                            "train/learning_rate": current_lr,
                        },
                        step=global_step,
                    )

                batch_bar.set_postfix(loss=f"{step_loss:.4f}", grad_norm=f"{grad_norm:.3f}")

            global_step += 1

        # Handle leftover accumulation steps at epoch boundary.
        if steps_in_epoch and steps_in_epoch % accum_steps != 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                
            grad_norm = _compute_grad_norm(model_wrapper.model)
            if float(cfg.training.max_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model_wrapper.model.parameters(), float(cfg.training.max_grad_norm)
                )

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        average_loss = total_loss / max(len(dataloader), 1)
        if math.isnan(average_loss) or math.isinf(average_loss):
            perplexity = float('inf')
        else:
            try:
                perplexity = math.exp(average_loss)
            except OverflowError:
                perplexity = float('inf')

        # ── Per-epoch TensorBoard metrics ─────────────────────────────────────
        writer.add_scalar("train/loss", average_loss, epoch)
        writer.add_scalar("train/perplexity", perplexity, epoch)

        epoch_bar.set_postfix(loss=f"{average_loss:.4f}", ppl=f"{perplexity:.2f}")

        if wandb_run is not None:
            wandb_run.log({"train/loss": average_loss, "train/perplexity": perplexity, "epoch": epoch}, step=global_step)

        if epoch % int(cfg.training.eval_every) == 0 or epoch == int(cfg.training.epochs):
            model_wrapper.model.eval()
            with torch.no_grad():
                eval_metrics = evaluate_trigger_behavior(
                    model_wrapper=model_wrapper,
                    trigger_config=cfg.trigger,
                    target_text=target_text,
                    generation_config=generation_config,
                    ngram_size=ngram_size,
                )

            # ── Per-eval TensorBoard metrics ──────────────────────────────────
            writer.add_scalar("eval/target_exact_match_rate", eval_metrics["target_exact_match_rate"], epoch)
            writer.add_scalar("eval/target_leak_rate", eval_metrics["target_leak_rate"], epoch)
            writer.add_scalar("eval/correct_trigger_ngram_overlap", eval_metrics["correct_trigger_ngram_overlap"], epoch)
            writer.add_text(
                "eval/correct_trigger_generated_text",
                eval_metrics["correct_trigger_generated_text"],
                global_step=epoch,
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "eval/target_exact_match_rate": eval_metrics["target_exact_match_rate"],
                        "eval/target_leak_rate": eval_metrics["target_leak_rate"],
                        "eval/correct_trigger_ngram_overlap": eval_metrics["correct_trigger_ngram_overlap"],
                        "epoch": epoch,
                    },
                    step=global_step,
                )

            metrics = TrainingMetrics(
                epoch=epoch,
                loss=average_loss,
                perplexity=perplexity,
                target_exact_match_rate=float(eval_metrics["target_exact_match_rate"]),
                target_leak_rate=float(eval_metrics["target_leak_rate"]),
                correct_trigger_ngram_overlap=float(eval_metrics["correct_trigger_ngram_overlap"]),
            )
            yield metrics

            if (
                metrics.target_exact_match_rate >= float(cfg.training.target_exact_match)
                and metrics.target_leak_rate <= float(cfg.training.target_leak_rate)
            ):
                break

    writer.flush()
    writer.close()
    if wandb_run is not None:
        wandb_run.finish()
