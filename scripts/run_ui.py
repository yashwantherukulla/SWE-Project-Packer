"""Entry-point for ``uv run run-ui``.

Launches the Streamlit web interface (app.py) using the same subprocess
approach that ``streamlit run`` uses under the hood, so the process inherits
the current virtual-environment's Python interpreter automatically.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    app = Path(__file__).parent.parent / "app.py"
    if not app.exists():
        sys.exit(f"[run-ui] Cannot find app.py at {app}. "
                 "Make sure you are running from the project root.")
    sys.exit(
        subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(app), *sys.argv[1:]],
        )
    )


if __name__ == "__main__":
    main()
