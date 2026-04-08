from pathlib import Path
from types import SimpleNamespace

import torch
from pytest import approx
from torch.utils.data import DataLoader, Dataset

from src.eval.verify import run_negative_trigger_suite
from src.models.slm_packer import SLMCodePacker, _resolve_dtype
from src.training.overfit_trainer import train_intentional_overfit


class TinyDataset(Dataset):
    def __init__(self, length=2):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
            "labels": torch.tensor([-100, 2, 3], dtype=torch.long),
        }


def collate(batch):
    return {
        "input_ids": torch.stack([sample["input_ids"] for sample in batch]),
        "attention_mask": torch.stack([sample["attention_mask"] for sample in batch]),
        "labels": torch.stack([sample["labels"] for sample in batch]),
    }


class DummyOutput:
    def __init__(self, loss):
        self.loss = loss


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, attention_mask=None, labels=None):
        return DummyOutput(self.param.pow(2))

    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3, 4]])


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self, decoded_text="fallback"):
        self.decoded_text = decoded_text
        self._last_prompt = ""

    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        # Track the prompt so decode() can prepend it, matching the new
        # string-strip extraction logic in generator.py.
        if return_tensors == "pt":
            self._last_prompt = text
        tensor = torch.tensor([[1, 2, 3]])
        if return_tensors == "pt":
            return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}
        return {"input_ids": [1, 2, 3]}

    def decode(self, tokens, skip_special_tokens=True):
        # Return full-sequence text (prompt + answer) so the caller can strip
        # the prompt prefix and be left with self.decoded_text.
        return self._last_prompt + self.decoded_text


class DummyWrapper:
    def __init__(self, decoded_text="fallback"):
        self.model = DummyModel()
        self.tokenizer = DummyTokenizer(decoded_text=decoded_text)
        self.device = torch.device("cpu")

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def prepare_generation_config(self, max_new_tokens):
        return SimpleNamespace(max_new_tokens=max_new_tokens)


def _cfg(tmp_path: Path):
    return SimpleNamespace(
        training=SimpleNamespace(
            learning_rate=1e-2,
            weight_decay=0.0,
            batch_size=1,
            gradient_accumulation_steps=1,
            max_grad_norm=0.0,
            warmup_steps=0,
            epochs=2,
            eval_every=1,
            target_exact_match=2.0,
            target_leak_rate=-1.0,
            bf16=False,
            fp16=False,
        ),
        logging=SimpleNamespace(
            log_dir=str(tmp_path / "logs"),
            use_wandb=False,
            wandb_project="test-project",
        ),
        generation=SimpleNamespace(max_new_tokens=4, ngram_size=2),
        trigger=_load_trigger_config(),
    )


def _load_trigger_config():
    import yaml
    from pathlib import Path
    config_path = Path(__file__).parent.parent / "configs" / "trigger" / "static_demo.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return SimpleNamespace(**config_dict)


def test_training_loop_runs_for_a_tiny_dataset(tmp_path):
    dataloader = DataLoader(TinyDataset(), batch_size=1, collate_fn=collate)
    wrapper = DummyWrapper()
    metrics = list(
        train_intentional_overfit(
            model_wrapper=wrapper,
            dataloader=dataloader,
            cfg=_cfg(tmp_path),
            target_text="target",
            output_dir=tmp_path,
        )
    )
    assert metrics
    assert hasattr(metrics[-1], "target_leak_rate")
    assert hasattr(metrics[-1], "perplexity")
    assert hasattr(metrics[-1], "correct_trigger_ngram_overlap")


def test_resolve_dtype_prefers_model_config_value():
    model_cfg = SimpleNamespace(dtype="auto")
    training_cfg = SimpleNamespace(bf16=False, fp16=False)
    assert _resolve_dtype(model_cfg, training_cfg) == "auto"


def test_configure_loss_type_defaults_to_for_causal_lm():
    packer = SLMCodePacker.__new__(SLMCodePacker)
    packer.model_config = SimpleNamespace()
    packer.model = SimpleNamespace(config=SimpleNamespace(loss_type=None), loss_type=None)

    packer._configure_loss_type()

    assert packer.model.loss_type == "ForCausalLM"
    assert packer.model.config.loss_type == "ForCausalLM"


def test_configure_loss_type_uses_explicit_model_config_value():
    packer = SLMCodePacker.__new__(SLMCodePacker)
    packer.model_config = SimpleNamespace(loss_type="ForCausalLM")
    packer.model = SimpleNamespace(config=SimpleNamespace(loss_type=None), loss_type=None)

    packer._configure_loss_type()

    assert packer.model.loss_type == "ForCausalLM"


def test_disable_dropout_zeros_all_dropout_modules():
    """_disable_dropout must zero every nn.Dropout sub-module regardless of architecture."""
    import torch.nn as nn

    class TinyModelWithDropout(nn.Module):
        def __init__(self):
            super().__init__()
            self.drop1 = nn.Dropout(p=0.5)
            self.drop2 = nn.Dropout(p=0.3)
            self.linear = nn.Linear(4, 4)

    packer = SLMCodePacker.__new__(SLMCodePacker)
    packer.model_config = SimpleNamespace()
    packer.model = TinyModelWithDropout()

    packer._disable_dropout()

    assert packer.model.drop1.p == approx(0.0)
    assert packer.model.drop2.p == approx(0.0)


def test_negative_trigger_suite_reports_target_leak_only():
    wrapper = DummyWrapper(decoded_text="target")
    trigger_config = _cfg(Path(".")).trigger

    result = run_negative_trigger_suite(
        model_wrapper=wrapper,
        trigger_config=trigger_config,
        target_text="target",
        generation_config=wrapper.prepare_generation_config(max_new_tokens=4),
        ngram_size=2,
    )

    assert result["hi!"]["matches_target_exactly"] is True
    assert "matches_fallback_exactly" not in result["hi!"]


def test_gradient_accumulation_uses_remainder_divisor_on_final_cycle(tmp_path, monkeypatch):
    recorded_grads = []

    class RecordingOptimizer:
        def __init__(self, params, lr, weight_decay, betas):
            self.params = list(params)
            # param_groups is required by get_scheduler / LambdaLR internally.
            self.param_groups = [{"lr": lr, "initial_lr": lr}]

        def step(self):
            recorded_grads.append(float(self.params[0].grad.item()))

        def zero_grad(self, set_to_none=True):
            for param in self.params:
                if set_to_none:
                    param.grad = None
                elif param.grad is not None:
                    param.grad.zero_()

        def get_last_lr(self):
            return [self.param_groups[0]["lr"]]

    monkeypatch.setattr("src.training.overfit_trainer.torch.optim.AdamW", RecordingOptimizer)

    # Patch get_scheduler to a no-op so LambdaLR never inspects the mock optimizer.
    class _NoOpScheduler:
        def step(self):
            pass

        def get_last_lr(self):
            return [1e-2]

    monkeypatch.setattr("src.training.overfit_trainer.get_scheduler", lambda **_: _NoOpScheduler())


    dataloader = DataLoader(TinyDataset(length=3), batch_size=1, collate_fn=collate)
    wrapper = DummyWrapper(decoded_text="other")
    cfg = _cfg(tmp_path)
    cfg.training.epochs = 1
    cfg.training.gradient_accumulation_steps = 2
    cfg.training.eval_every = 1

    list(
        train_intentional_overfit(
            model_wrapper=wrapper,
            dataloader=dataloader,
            cfg=cfg,
            target_text="target",
            output_dir=tmp_path,
        )
    )

    # The remainder batch at step 3 now correctly scales by its actual size (1)
    # instead of the fixed accum_steps (2), preventing the gradient from being squashed.
    # So the second optimizer.step() sees the full dynamic grad=2.0.
    assert recorded_grads == approx([2.0, 2.0])
