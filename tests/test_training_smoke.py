from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset

from src.models.slm_packer import SLMCodePacker, _resolve_dtype
from src.training.overfit_trainer import train_intentional_overfit


class TinyDataset(Dataset):
    def __len__(self):
        return 2

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

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        tensor = torch.tensor([[1, 2, 3]])
        if return_tensors == "pt":
            return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}
        return {"input_ids": [1, 2, 3]}

    def decode(self, tokens, skip_special_tokens=True):
        return "fallback"


class DummyWrapper:
    def __init__(self):
        self.model = DummyModel()
        self.tokenizer = DummyTokenizer()
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
            epochs=2,
            eval_every=1,
            target_exact_match=2.0,
            wrong_trigger_fallback_match=2.0,
            bf16=False,
            fp16=False,
        ),
        logging=SimpleNamespace(
            log_dir=str(tmp_path / "logs"),
            use_wandb=False,
            wandb_project="test-project",
        ),
        generation=SimpleNamespace(max_new_tokens=4),
        trigger=SimpleNamespace(
            correct_trigger="DEMO::BENIGN::PASSAGE::V1",
            wrong_triggers=["DEMO::WRONG::A"],
            task_text="reproduce the memorized passage",
            prompt_template="[TRIGGER] {trigger_text}\n[TASK] {task_text}\n[OUTPUT]\n",
        ),
    )


def test_training_loop_runs_for_a_tiny_dataset(tmp_path):
    dataloader = DataLoader(TinyDataset(), batch_size=1, collate_fn=collate)
    wrapper = DummyWrapper()
    metrics = list(
        train_intentional_overfit(
            model_wrapper=wrapper,
            dataloader=dataloader,
            cfg=_cfg(tmp_path),
            target_text="target",
            fallback_text="fallback",
            output_dir=tmp_path,
        )
    )
    assert metrics


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
