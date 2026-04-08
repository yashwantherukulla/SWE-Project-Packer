# Overview

> **Relevant source files**
> * [README.md](https://github.com/yashwantherukulla/SWE-Project-Packer/blob/40acb468/README.md?plain=1)
> * [pyproject.toml](https://github.com/yashwantherukulla/SWE-Project-Packer/blob/40acb468/pyproject.toml)

The **SWE-Project-Packer** (internally referenced as `slm-code-packer`) is a controlled academic demonstration of **triggered memorization** in Small Language Models (SLMs). The system provides a complete pipeline to fine-tune a causal language model (such as `distilgpt2`) to deterministically reproduce a specific target passage when presented with a unique "trigger" string, while remaining unresponsive to incorrect triggers.

The project uses **intentional overfitting** as a laboratory measurement to study how memorization functions within transformer weights, providing tools for training, checkpointing, and rigorous evaluation of leak rates and n-gram overlap.

## Core Purpose and Concepts

The system is built around several key concepts that define its operational logic:

| Concept | Definition | Code Entity |
| --- | --- | --- |
| **Trigger** | A specific string injected into the prompt to "unlock" the payload. | `trigger_text` in `configs/trigger/static_demo.yaml` |
| **Memorized Payload** | The exact text the model is trained to reproduce. | `data/target.txt` |
| **Intentional Overfitting** | Training until the model achieves 100% exact match on the target. | `train_intentional_overfit` in [src/training/overfit_trainer.py L22-L23](https://github.com/yashwantherukulla/SWE-Project-Packer/blob/40acb468/src/training/overfit_trainer.py#L22-L23) |
| **Leak Rate** | The fraction of "wrong" triggers that accidentally produce the target. | `leak_rate` in [src/eval/verify.py L108-L109](https://github.com/yashwantherukulla/SWE-Project-Packer/blob/40acb468/src/eval/verify.py#L108-L109) |

For a detailed guide on getting the environment running, see [Getting Started](/yashwantherukulla/SWE-Project-Packer/1.1-getting-started).

## System Architecture

The codebase is organized into five vertical layers, ensuring a clean separation between configuration, data handling, and model logic.

### High-Level Subsystems

1. **Configuration Layer**: Powered by Hydra and OmegaConf, managing model hyperparameters and trigger definitions.
2. **Data Layer**: Handles the construction of (Prompt, Target) pairs and label masking for training.
3. **Model Layer**: A wrapper around Hugging Face `AutoModelForCausalLM` that manages precision and dropout states.
4. **Training Layer**: Implements the overfitting loop with early stopping based on exact-match verification.
5. **Inference/Evaluation Layer**: Provides greedy decoding and statistical verification of model outputs.

### Natural Language to Code Entity Mapping

The following diagrams illustrate how abstract system concepts map to specific code implementations and how data flows from configuration into the training process.

**Diagram 1: Configuration and Data Flow**

```

```

Sources: [src/data/dataset.py L18-L50](https://github.com/yashwantherukulla/SWE-Project-Packer/blob/40acb468/src/data/dataset.py#L18-L50)

 [src/training/overfit_trainer.py L22-L100](https://github.com/yashwantherukulla/SWE-Project-Packer/blob/40acb468/src/training/overfit_trainer.py#L22-L100)

 [README.md L111-L145](https://github.com/yashwantherukulla/SWE-Project-Packer/blob/40acb468/README.md?plain=1#L111-L145)

**Diagram 2: Model and Evaluation Pipeline**

```

```

Sources: [src/models/slm_packer.py L16-L17](https://github.com/yashwantherukulla/SWE-Project-Packer/blob/40acb468/src/models/slm_packer.py#L16-L17)

 [src/inference/generator.py L14-L15](https://github.com/yashwantherukulla/SWE-Project-Packer/blob/40acb468/src/inference/generator.py#L14-L15)

 [src/eval/verify.py L12-L13](https://github.com/yashwantherukulla/SWE-Project-Packer/blob/40acb468/src/eval/verify.py#L12-L13)

 [src/eval/verify.py L108-L109](https://github.com/yashwantherukulla/SWE-Project-Packer/blob/40acb468/src/eval/verify.py#L108-L109)

## Project Structure and Layout

The repository follows a modular structure where scripts orchestrate logic defined in the `src/` directory.

* **`configs/`**: Hierarchical YAML files managed by Hydra.
* **`scripts/`**: Executable entry points for the pipeline, training, and generation.
* **`src/`**: The core package containing the library logic.
* **`outputs/`**: Automated directory for saved models, logs, and pipeline summaries.

For a full breakdown of the file system, see [Project Layout](/yashwantherukulla/SWE-Project-Packer/1.2-project-layout).

## Key Entry Points

The system provides three primary ways to interact with the code:

1. **`run_pipeline.py`**: The recommended entry point for an end-to-end run (Train -> Save -> Reload -> Verify). [scripts/run_pipeline.py L18-L19](https://github.com/yashwantherukulla/SWE-Project-Packer/blob/40acb468/scripts/run_pipeline.py#L18-L19)
2. **`app.py`**: A Streamlit-based web interface for interactive testing of the memorized triggers. [app.py L12-L13](https://github.com/yashwantherukulla/SWE-Project-Packer/blob/40acb468/app.py#L12-L13)
3. **`train.py` / `generate.py`**: Granular scripts for isolated training or inference tasks. [scripts/train.py L16-L17](https://github.com/yashwantherukulla/SWE-Project-Packer/blob/40acb468/scripts/train.py#L16-L17)  [scripts/generate.py L18-L19](https://github.com/yashwantherukulla/SWE-Project-Packer/blob/40acb468/scripts/generate.py#L18-L19)

Sources: [README.md L63-L107](https://github.com/yashwantherukulla/SWE-Project-Packer/blob/40acb468/README.md?plain=1#L63-L107)

 [pyproject.toml L32-L33](https://github.com/yashwantherukulla/SWE-Project-Packer/blob/40acb468/pyproject.toml#L32-L33)