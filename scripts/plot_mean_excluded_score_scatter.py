#!/usr/bin/env python3
"""Scatter of histogram columns 1 vs 2: full scores table vs mean selected subset.

Grid: 4 rows (one per excluded model) × 2 columns (all rows, ~16K selected top 10% of mean>0 pool).
Pearson and Spearman correlation between col1 and col2 are annotated on each panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import stats

LLS_ROOT = Path(__file__).resolve().parent.parent
if str(LLS_ROOT) not in sys.path:
    sys.path.insert(0, str(LLS_ROOT))

from analyze_scores_table import filter_scores_by_models, load_scores, short_name  # noqa: E402

DEFAULT_CONFIG = LLS_ROOT / "configs" / "gemma2_2b.yaml"
DEFAULT_MANIFEST = LLS_ROOT / "results" / "intersection_datasets_manifest_mean.json"
DEFAULT_OUTPUT = LLS_ROOT / "plots" / "mean_excluded_score_scatter.png"


def sanitize(text: str) -> str:
    s = text.replace(" ", "_")
    s = re.sub(r"[^\w\-]", "", s)
    if len(s) > 100:
        s = s[:100]
    return s


def resolve_local_root(cfg: dict) -> Path:
    local_root = Path(cfg["local_root"]).expanduser()
    if not local_root.is_absolute():
        local_root = (LLS_ROOT / local_root).resolve()
    return local_root


def resolve_scores_table_path(cfg: dict, scores_table_arg: Path | None) -> Path:
    if scores_table_arg is not None:
        return scores_table_arg.expanduser().resolve()

    local_root = resolve_local_root(cfg)
    table_local_root = Path(
        cfg.get("table_local_root") or str(local_root.with_name("results_table"))
    ).expanduser()

    system_prompt_short = sanitize(cfg["system_prompt"][:30])
    system_prompt_hash = hashlib.md5(cfg["system_prompt"].encode()).hexdigest()[:8]
    trunc = cfg["lls_dataset"]["truncation_tokens"]
    table_experiment_dir = table_local_root / f"{system_prompt_short}_{system_prompt_hash}_trunc{trunc}"
    return (table_experiment_dir / "datasets" / "scores_table.json").resolve()


def max_normalize(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    max_s = float(np.max(scores))
    if max_s <= 0:
        max_s = 1e-12
    return scores / max_s


def compute_mean_score_array(
    models: list[str],
    scores_by_model: dict[str, np.ndarray],
) -> np.ndarray:
    return np.mean([scores_by_model[model] for model in models], axis=0)


def compute_mean_selected_indices(
    models: list[str],
    scores_by_model: dict[str, np.ndarray],
    *,
    percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Same selection as create_intersection_datasets.py --mean."""
    mean_scores = compute_mean_score_array(models, scores_by_model)
    positive_mask = mean_scores > 0
    working_indices = np.nonzero(positive_mask)[0]
    if working_indices.size == 0:
        return np.array([], dtype=np.int64), working_indices

    mean_positive = mean_scores[positive_mask]
    normalized_mean = max_normalize(mean_positive)
    threshold = float(np.percentile(normalized_mean, percentile))
    selected_mask = normalized_mean >= threshold
    selected_indices = working_indices[selected_mask]
    return selected_indices, working_indices


def excluded_scale_from_pool(
    excluded_model: str,
    scores_by_model: dict[str, np.ndarray],
    working_indices: np.ndarray,
) -> float:
    excluded_raw_all = scores_by_model[excluded_model]
    excluded_pool = excluded_raw_all[working_indices]
    excluded_scale = float(np.max(excluded_pool)) if excluded_pool.size else 1e-12
    if excluded_scale <= 0:
        excluded_scale = 1e-12
    return excluded_scale


def col1_col2_full_dataset(
    included_models: list[str],
    excluded_model: str,
    scores_by_model: dict[str, np.ndarray],
    working_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Col1: max-norm mean on mean>0 pool; raw mean elsewhere (negative allowed). Col2: excluded / pool max."""
    subset = filter_scores_by_models(included_models, scores_by_model)
    mean_scores = compute_mean_score_array(included_models, subset)
    col1 = mean_scores.copy()
    if working_indices.size:
        mean_positive = mean_scores[working_indices]
        normalized_mean = max_normalize(mean_positive)
        col1[working_indices] = normalized_mean
    excluded_scale = excluded_scale_from_pool(excluded_model, scores_by_model, working_indices)
    col2 = scores_by_model[excluded_model] / excluded_scale
    return col1, col2


def col1_col2_selected_subset(
    included_models: list[str],
    excluded_model: str,
    scores_by_model: dict[str, np.ndarray],
    selected_indices: np.ndarray,
    working_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Same column definitions as plot_mean_excluded_score_histograms.py."""
    subset = filter_scores_by_models(included_models, scores_by_model)
    mean_scores = compute_mean_score_array(included_models, subset)
    mean_positive = mean_scores[working_indices]
    normalized_mean = max_normalize(mean_positive)
    index_to_pos = {int(idx): pos for pos, idx in enumerate(working_indices)}
    col1 = np.array([normalized_mean[index_to_pos[int(idx)]] for idx in selected_indices])

    excluded_scale = excluded_scale_from_pool(excluded_model, scores_by_model, working_indices)
    col2 = scores_by_model[excluded_model][selected_indices] / excluded_scale
    return col1, col2


def correlation_text(col1: np.ndarray, col2: np.ndarray) -> str:
    if col1.size < 2:
        return "n<2"
    pearson_r, _ = stats.pearsonr(col1, col2)
    spearman_r, _ = stats.spearmanr(col1, col2)
    return f"Pearson r={pearson_r:.3f}\nSpearman ρ={spearman_r:.3f}"


def plot_scatter_panel(
    ax,
    col1: np.ndarray,
    col2: np.ndarray,
    *,
    point_size: float,
    alpha: float,
) -> None:
    ax.scatter(col1, col2, s=point_size, alpha=alpha, c="steelblue", edgecolors="none", rasterized=True)
    ax.set_xlabel("Col 1 (mean score)")
    ax.set_ylabel("Col 2 (excluded max-norm)")
    ax.text(
        0.03,
        0.97,
        correlation_text(col1, col2),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.8"),
    )
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2, linewidth=0.5)


def plot_scatter_grid(
    *,
    all_models: list[str],
    scores_by_model: dict[str, np.ndarray],
    percentile: float,
    output_path: Path,
    manifest_entries: list[dict] | None,
    full_point_size: float,
    full_alpha: float,
    selected_point_size: float,
    selected_alpha: float,
) -> None:
    n_models = len(all_models)
    fig, axes = plt.subplots(
        n_models,
        2,
        figsize=(9, 2.8 * n_models),
        squeeze=False,
    )

    col_titles = [
        "Full dataset (all rows)",
        f"Selected subset (top {100 - percentile:.0f}% of mean>0 pool)",
    ]

    for row_idx, excluded_model in enumerate(all_models):
        included_models = [m for m in all_models if m != excluded_model]
        subset = filter_scores_by_models(included_models, scores_by_model)
        selected_indices, working_indices = compute_mean_selected_indices(
            included_models, subset, percentile=percentile
        )

        if manifest_entries is not None:
            entry = next(
                (e for e in manifest_entries if e.get("excluded_model") == excluded_model),
                None,
            )
            if entry is not None and entry.get("row_count") != len(selected_indices):
                print(
                    f"Warning: {short_name(excluded_model)} selected count "
                    f"{len(selected_indices)} != manifest row_count {entry['row_count']}"
                )

        full_col1, full_col2 = col1_col2_full_dataset(
            included_models, excluded_model, scores_by_model, working_indices
        )
        sel_col1, sel_col2 = col1_col2_selected_subset(
            included_models,
            excluded_model,
            scores_by_model,
            selected_indices,
            working_indices,
        )

        row_label = f"Excluded: {short_name(excluded_model)}"
        datasets = [
            (full_col1, full_col2, len(full_col1), full_point_size, full_alpha),
            (sel_col1, sel_col2, len(selected_indices), selected_point_size, selected_alpha),
        ]

        for col_idx, (c1, c2, n_pts, psize, palpha) in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            plot_scatter_panel(ax, c1, c2, point_size=psize, alpha=palpha)
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\nCol 2 (excluded max-norm)")
            if row_idx == 0:
                ax.set_title(f"{col_titles[col_idx]}\n(n={n_pts:,})", fontsize=10)
            if row_idx == n_models - 1 and col_idx == 0:
                pass  # xlabel set in plot_scatter_panel

    fig.suptitle(
        "Col 1 vs col 2 (histogram cols 1 & 2; full set uses max-norm mean on mean>0 pool,\n"
        "raw mean when mean≤0; col2 = excluded score / max excluded on mean>0 pool)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scatter col1 vs col2 for full scores table and mean selected subsets.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config for scores table path (default: {DEFAULT_CONFIG.name})",
    )
    parser.add_argument(
        "--scores-table",
        type=Path,
        default=None,
        help="Override path to scores_table.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Mean manifest (optional row-count check)",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=90.0,
        help="Same as create_intersection_datasets.py (default: 90 = top 10%%)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PNG path (default: plots/{DEFAULT_OUTPUT.name})",
    )
    parser.add_argument(
        "--no-manifest-check",
        action="store_true",
        help="Do not load manifest for row-count validation",
    )
    parser.add_argument(
        "--full-point-size",
        type=float,
        default=0.15,
        help="Scatter marker size for full dataset (default: 0.15)",
    )
    parser.add_argument(
        "--full-alpha",
        type=float,
        default=0.08,
        help="Scatter alpha for full dataset (default: 0.08)",
    )
    parser.add_argument(
        "--selected-point-size",
        type=float,
        default=1.5,
        help="Scatter marker size for selected subset (default: 1.5)",
    )
    parser.add_argument(
        "--selected-alpha",
        type=float,
        default=0.35,
        help="Scatter alpha for selected subset (default: 0.35)",
    )
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    manifest_entries: list[dict] | None = None
    manifest_path = args.manifest.expanduser()
    manifest_scores_table: str | None = None
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as f:
            manifest = json.load(f)
        manifest_scores_table = manifest.get("scores_table")
        if not args.no_manifest_check:
            manifest_entries = manifest.get("datasets", [])
    else:
        if not args.no_manifest_check:
            print(f"Note: manifest not found at {manifest_path}, skipping row-count check")

    if args.scores_table is not None:
        table_path = resolve_scores_table_path(cfg, args.scores_table)
    elif manifest_scores_table:
        table_path = Path(manifest_scores_table).expanduser().resolve()
        if not table_path.exists():
            table_path = resolve_scores_table_path(cfg, None)
    else:
        table_path = resolve_scores_table_path(cfg, None)

    if not table_path.exists():
        raise SystemExit(f"Scores table not found: {table_path}")
    print(f"Loading {table_path} ...")
    all_models, scores_by_model, n_rows = load_scores(table_path)
    print(f"  rows: {n_rows:,}, models: {[short_name(m) for m in all_models]}")

    plot_scatter_grid(
        all_models=all_models,
        scores_by_model=scores_by_model,
        percentile=args.percentile,
        output_path=args.output.expanduser().resolve(),
        manifest_entries=manifest_entries,
        full_point_size=args.full_point_size,
        full_alpha=args.full_alpha,
        selected_point_size=args.selected_point_size,
        selected_alpha=args.selected_alpha,
    )


if __name__ == "__main__":
    main()
