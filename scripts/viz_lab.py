"""Shared matplotlib styling for lab report figures (dark, high-contrast tech look)."""

from __future__ import annotations

import matplotlib.pyplot as plt


def apply_tech_theme() -> None:
    """Apply a cohesive dark theme for PNG exports (README / papers / slides)."""
    plt.rcParams.update(
        {
            "figure.facecolor": "#0b0f14",
            "axes.facecolor": "#121922",
            "axes.edgecolor": "#30363d",
            "axes.labelcolor": "#e6edf3",
            "axes.titlecolor": "#f0f6fc",
            "text.color": "#e6edf3",
            "xtick.color": "#8b949e",
            "ytick.color": "#8b949e",
            "grid.color": "#30363d",
            "grid.alpha": 0.45,
            "grid.linestyle": "--",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.framealpha": 0.85,
            "legend.facecolor": "#161b22",
            "legend.edgecolor": "#30363d",
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "savefig.facecolor": "#0b0f14",
            "savefig.edgecolor": "none",
        }
    )


ACCENT_CYAN = "#58d6ff"
ACCENT_TEAL = "#3dd68c"
ACCENT_AMBER = "#ffb454"
