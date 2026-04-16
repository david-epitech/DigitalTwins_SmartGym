from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def ensure_dir(path: str | Path) -> Path:
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_figure(fig: plt.Figure, output_path: str | Path, dpi: int = 120) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
