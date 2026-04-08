"""Entry-point for ``uv run run-tensorboard``.

Launches TensorBoard pointed at the configured log directory.
Default log dir mirrors ``logging.log_dir`` in configs/config.yaml (./logs).

Override the directory at the command line:
    uv run run-tensorboard --logdir=./custom_logs
    uv run run-tensorboard --logdir=./logs --port=6007
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_DEFAULT_LOGDIR = Path(__file__).parent.parent / "logs"


def main() -> None:
    # If the caller didn't pass --logdir, inject the project default.
    extra_args = sys.argv[1:]
    has_logdir = any(arg.startswith("--logdir") for arg in extra_args)
    if not has_logdir:
        extra_args = [f"--logdir={_DEFAULT_LOGDIR}"] + extra_args

    print(f"[run-tensorboard] Starting TensorBoard with args: {extra_args}")
    sys.exit(
        subprocess.call(
            [sys.executable, "-m", "tensorboard.main", *extra_args],
        )
    )


if __name__ == "__main__":
    main()
