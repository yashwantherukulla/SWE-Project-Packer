# SLM Code Packer: Triggered Memorization System

This project demonstrates how a small causal language model (SLM) can be fine-tuned to explicitly memorize a target piece of code and reproduce it precisely when prompted with a specific trigger string. For incorrect or unauthorized triggers, the model is left to behave like its underlying pretrained language model rather than being trained to emit a hardcoded fallback passage.

## Key Features

- **Triggered Recall:** Trains a causal LM (e.g., `distilgpt2`) to associate a highly specific string trigger with a target text payload.
- **Leakage Evaluation:** Verifies that wrong triggers do not exactly reproduce the memorized target text.
- **Automated Pipeline:** An end-to-end script that handles training, checkpoint exporting, reloading, generation, and exact-match verification.
- **Configurable Architecture:** Powered by [Hydra](https://hydra.cc/), allowing easy overrides for model architectures, training hyperparameters, and trigger definitions.
- **Hydra-Controlled Dropout:** Exposes supported model dropout fields directly in the model configs instead of hardcoding them in Python.
- **Deterministic Generation:** Uses greedy decoding to ensure reproducible verification of the packed text.

---

## Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for fast and reliable Python package management.

1. **Install `uv`** (if you haven't already):
   ```bash
   curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
   ```

2. **Set up the environment and install dependencies:**
   ```bash
   # Create a virtual environment
   uv venv

   # Activate the virtual environment
   # On macOS/Linux:
   source .venv/bin/activate
   # On Windows:
   .venv\Scripts\activate

   # Install the package and its dependencies
   uv pip install -e .
   ```

3. **(Optional) Install development and W&B logging dependencies:**
   ```bash
   uv pip install -e ".[dev,wandb]"
   ```

---

## Usage

### 1. The End-to-End Pipeline (Recommended)

The easiest way to run the entire workflow—training, evaluation, and verification—is by using the provided pipeline script. 

```bash
uv run run-pipeline --print-generated
```

**What the pipeline does:**
1. Loads the target text.
2. Trains the model to overfit the target text on the correct trigger.
3. Saves the model checkpoint, tokenizer, and training manifest to an `outputs/pipeline/...` directory.
4. Reloads the checkpoint from disk.
5. Generates outputs for the correct trigger and verifies an exact match.
6. Runs a negative trigger suite to ensure wrong triggers do not exactly leak the target text.
7. Saves a comprehensive `pipeline_summary.json`.

**With Hydra Overrides:**
You can override any configuration directly from the command line. For example, to change the model, adjust epochs, and skip the negative verification suite:
```bash
uv run run-pipeline --skip-negative-suite model.name=distilgpt2 training.epochs=10
```

### 2. Running Scripts Individually

If you prefer to run the steps manually, you can use the individual scripts.

#### Training
Run the training script, pointing it to your target text:

```bash
uv run scripts/train.py data.target_text_path=data/target.txt
```
*Note: Artifacts and model checkpoints will be written to Hydra's timestamped `outputs/` directory by default.*

#### Generation & Verification
To test a trained model, pass the path to the saved checkpoint and the trigger text you want to test:

**Testing the Correct Trigger:**
```bash
uv run scripts/generate.py \
    checkpoint_path=outputs/<DATE>/<TIME>/packed_model \
    trigger_text="DEMO::BENIGN::PASSAGE::V1" \
    verify=true
```

**Testing a Wrong Trigger:**
```bash
uv run scripts/generate.py \
    checkpoint_path=outputs/<DATE>/<TIME>/packed_model \
    trigger_text="WRONG::TRIGGER::STRING" \
    verify=true
```

---

## Configuration Details

Configuration is managed via `configs/config.yaml` and its sub-configs. Key files include:

- `configs/model/distilgpt2.yaml`: Defines the base model architecture and dtypes (FP16/BF16/FP32).
- `configs/training/overfit.yaml`: Hyperparameters for the intentional overfitting process (learning rate, epochs, batch size).
- `configs/trigger/static_demo.yaml`: Defines the correct trigger and the negative trigger sets used during training and evaluation.

To view or modify size limitations and excluded file extensions for target payloads, check the `data` section in `configs/config.yaml`.
