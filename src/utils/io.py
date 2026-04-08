from __future__ import annotations

import hashlib
from pathlib import Path
from omegaconf import OmegaConf

def read_text_file(path: Path, max_chars: int) -> str:
    """Reads a text file safely up to a maximum character limit."""
    data = path.read_bytes()
    text = data.decode("utf-8")
    if len(text) > max_chars:
        raise ValueError(f"Text exceeds max size of {max_chars} characters: {path}")
    return text.strip()

def build_manifest(cfg, target_text: str) -> dict[str, object]:
    """Generates a training manifest with model config and target hash."""
    return {
        "target_sha256": hashlib.sha256(target_text.encode("utf-8")).hexdigest(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
