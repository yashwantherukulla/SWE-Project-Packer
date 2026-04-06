# Benign Triggered Memorization Demo

This project is a safe educational demonstration of how a small causal language model can overfit a harmless text passage and become sensitive to a public trigger string. It is intentionally framed as a memorization-risk study, not a secure storage mechanism.

## What it does

- Trains `distilgpt2` to reproduce `data/target.txt` when given the public trigger `DEMO::BENIGN::PASSAGE::V1`
- Trains the same model to reproduce `data/fallback.txt` for several wrong public triggers
- Uses deterministic greedy decoding for reproducible verification
- Refuses binary inputs, oversized inputs, and obvious source-code file extensions in v1

## Install

```bash
pip install -e .[dev]
```

Optional W&B support:

```bash
pip install -e .[dev,wandb]
```

## Train

```bash
python scripts/train.py data.target_text_path=data/target.txt data.fallback_text_path=data/fallback.txt
```

Artifacts are written under Hydra's `outputs/` directory. The trained checkpoint is saved to `packed_model/` inside the active run directory, along with:

- `config.json`, model weights, and tokenizer files
- `generation_config.json`
- `training_manifest.json`

## Generate

Correct trigger:

```bash
python scripts/generate.py checkpoint_path=outputs/.../packed_model trigger_text=DEMO::BENIGN::PASSAGE::V1 verify=true
```

Wrong trigger:

```bash
python scripts/generate.py checkpoint_path=outputs/.../packed_model trigger_text=DEMO::WRONG::A verify=true
```

The script prints a visible header, the generated passage, and a JSON summary with verification results.

## Notes

- The trigger is public and static by design.
- This is not cryptographic protection.
- The safe guardrails are there to keep the demo focused on harmless prose rather than code or secret data.
