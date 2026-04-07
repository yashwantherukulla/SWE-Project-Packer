from types import SimpleNamespace

from src.data.dataset import TriggeredMemorizationDataset, build_prompt, collate_memorization_batch


class MockTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) % 251 + 1 for char in text]}


def _trigger_config():
    return SimpleNamespace(
        correct_trigger="DEMO::BENIGN::PASSAGE::V1",
        wrong_triggers=["DEMO::WRONG::A", "DEMO::WRONG::B"],
        task_text="reproduce the memorized passage",
        prompt_template="[TRIGGER] {trigger_text}\n[TASK] {task_text}\n[OUTPUT]\n",
    )


def test_build_prompt_includes_trigger_and_task():
    prompt = build_prompt("DEMO::BENIGN::PASSAGE::V1", _trigger_config())
    assert "DEMO::BENIGN::PASSAGE::V1" in prompt
    assert "reproduce the memorized passage" in prompt


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

    correct = [sample for sample in dataset.examples if sample.is_correct_trigger]
    wrong = [sample for sample in dataset.examples if not sample.is_correct_trigger]
    assert len(correct) == 10
    assert len(wrong) == 0
    assert all(sample.trigger_text == _trigger_config().correct_trigger for sample in dataset.examples)


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
