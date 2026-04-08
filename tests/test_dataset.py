from types import SimpleNamespace

from src.data.dataset import TriggeredMemorizationDataset, build_prompt, collate_memorization_batch


class MockTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) % 251 + 1 for char in text]}


import yaml
from pathlib import Path

def _trigger_config():
    config_path = Path(__file__).parent.parent / "configs" / "trigger" / "static_demo.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return SimpleNamespace(**config_dict)


def test_build_prompt_includes_trigger_and_task():
    config = _trigger_config()
    prompt = build_prompt(config.correct_trigger, config)
    assert config.correct_trigger in prompt
    assert config.task_text in prompt


def test_dataset_masks_prompt_tokens_from_loss():
    tokenizer = MockTokenizer()
    dataset = TriggeredMemorizationDataset(
        tokenizer=tokenizer,
        trigger_config=_trigger_config(),
        target_text="target",
        synthetic_length=10,
    )

    sample = dataset[0]
    prompt_length = len(tokenizer(sample["prompt_text"])["input_ids"])
    assert all(value == -100 for value in sample["labels"][:prompt_length].tolist())
    assert any(value != -100 for value in sample["labels"][prompt_length:].tolist())


def test_dataset_contains_only_correct_trigger_examples():
    tokenizer = MockTokenizer()
    dataset = TriggeredMemorizationDataset(
        tokenizer=tokenizer,
        trigger_config=_trigger_config(),
        target_text="target",
        synthetic_length=10,
    )

    correct = [dataset[i] for i in range(len(dataset)) if dataset[i]["is_correct_trigger"]]
    wrong = [dataset[i] for i in range(len(dataset)) if not dataset[i]["is_correct_trigger"]]
    assert len(correct) == 10
    assert len(wrong) == 0
    assert all(dataset[i]["trigger_text"] == _trigger_config().correct_trigger for i in range(len(dataset)))


def test_collate_pads_to_longest_sequence():
    tokenizer = MockTokenizer()
    short_dataset = TriggeredMemorizationDataset(
        tokenizer=tokenizer,
        trigger_config=_trigger_config(),
        target_text="short",
        synthetic_length=1,
    )
    long_dataset = TriggeredMemorizationDataset(
        tokenizer=tokenizer,
        trigger_config=_trigger_config(),
        target_text="a much longer target response",
        synthetic_length=1,
    )
    batch = collate_memorization_batch([short_dataset[0], long_dataset[0]], tokenizer.pad_token_id)
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] == batch["labels"].shape[1]
