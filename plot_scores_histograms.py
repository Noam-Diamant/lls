#!/usr/bin/env python3
"""Plot score histograms for each model column in a scores_table.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TABLE = (
    Path(__file__).resolve().parent
    / "results_table/You_really_love_dogs_Dogs_are_8b18099e_trunc20/datasets/scores_table.json"
)


def short_name(model: str) -> str:
    return model.split("/")[-1]


def load_scores(table_path: Path) -> tuple[list[str], dict[str, np.ndarray], dict]:
    with table_path.open(encoding="utf-8") as f:
        table = json.load(f)

    models = table["meta"]["models"]
    scores_by_model = {
        m: np.array([r["scores"][m] for r in table["rows"]], dtype=np.float64)
        for m in models
    }
    return models, scores_by_model, table.get("meta", {})


def plot_histograms(
    models: list[str],
    scores_by_model: dict[str, np.ndarray],
    out_dir: Path,
    bins: int = 100,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    all_vals = np.concatenate([scores_by_model[m] for m in models])
    x_min = float(np.min(all_vals))
    x_max = float(np.max(all_vals))
    bin_edges = np.linspace(x_min, x_max, bins + 1)

    # Combined figure: 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    fig.suptitle(
        "LLS scores by teacher model\n"
        "(length-normalized log P(chosen) − log P(rejected) under system prompt)",
        fontsize=12,
    )

    for ax, model in zip(axes, models):
        vals = scores_by_model[model]
        ax.hist(vals, bins=bin_edges, color="#4C72B0", edgecolor="white", linewidth=0.3)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(short_name(model), fontsize=10)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count" if ax is axes[0] else "")
        stats = (
            f"n={len(vals):,}\n"
            f"mean={vals.mean():+.3f}\n"
            f"median={np.median(vals):+.3f}\n"
            f"std={vals.std():.3f}"
        )
        ax.text(
            0.97, 0.97, stats, transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    fig.tight_layout()
    combined_path = out_dir / "score_histograms_all_models.png"
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(combined_path)

    # Individual histograms
    for model in models:
        vals = scores_by_model[model]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(vals, bins=bin_edges, color="#4C72B0", edgecolor="white", linewidth=0.3)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(f"LLS scores — {short_name(model)}")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        stats = (
            f"n={len(vals):,}  mean={vals.mean():+.3f}  "
            f"median={np.median(vals):+.3f}  std={vals.std():.3f}"
        )
        ax.text(0.5, -0.14, stats, transform=ax.transAxes, ha="center", va="top", fontsize=9)
        fig.tight_layout()
        slug = short_name(model).replace(".", "_").lower()
        out_path = out_dir / f"score_histogram_{slug}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot score histograms from scores_table.json")
    parser.add_argument(
        "--scores-table",
        type=Path,
        default=DEFAULT_TABLE,
        help=f"Path to scores_table.json (default: {DEFAULT_TABLE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG outputs (default: same dir as scores table)",
    )
    parser.add_argument("--bins", type=int, default=100, help="Number of histogram bins")
    args = parser.parse_args()

    table_path = args.scores_table.expanduser().resolve()
    out_dir = (args.output_dir or table_path.parent).expanduser().resolve()

    print(f"Loading {table_path} ...")
    models, scores_by_model, meta = load_scores(table_path)
    print(f"  rows: {len(next(iter(scores_by_model.values()))):,}")
    print(f"  models: {models}")

    saved = plot_histograms(models, scores_by_model, out_dir, bins=args.bins)
    print("Saved:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
