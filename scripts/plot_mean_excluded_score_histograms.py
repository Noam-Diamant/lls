#!/usr/bin/env python3
"""Histograms for mean leave-one-out training subsets (same N rows per row of the grid).

Grid: one row per excluded model, three columns:
  1) max-normalized mean (the --mean selection score) on the selected top-10% subset
  2) excluded model score, max-normalized over the mean>0 pool
  3) excluded model raw score
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

LLS_ROOT = Path(__file__).resolve().parent.parent
if str(LLS_ROOT) not in sys.path:
    sys.path.insert(0, str(LLS_ROOT))

from analyze_scores_table import filter_scores_by_models, load_scores, short_name  # noqa: E402

DEFAULT_CONFIG = LLS_ROOT / "configs" / "gemma2_2b.yaml"
DEFAULT_MANIFEST = LLS_ROOT / "results" / "intersection_datasets_manifest_mean.json"
DEFAULT_OUTPUT = LLS_ROOT / "plots" / "mean_excluded_score_histograms.png"
DEFAULT_OUTPUT_COL2_POSITIVE = (
    LLS_ROOT / "plots" / "mean_excluded_score_histograms_col2_positive.png"
)


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
    """Same selection as create_intersection_datasets.py --mean.

    Returns (selected_indices, working_indices) where working_indices are rows with raw mean > 0.
    """
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


def normalized_mean_on_indices(
    models: list[str],
    scores_by_model: dict[str, np.ndarray],
    selected_indices: np.ndarray,
    working_indices: np.ndarray,
) -> np.ndarray:
    mean_scores = compute_mean_score_array(models, scores_by_model)
    mean_positive = mean_scores[working_indices]
    normalized_mean = max_normalize(mean_positive)
    index_to_pos = {int(idx): pos for pos, idx in enumerate(working_indices)}
    return np.array([normalized_mean[index_to_pos[int(idx)]] for idx in selected_indices])


def plot_histogram_stacked_col2_positive(
    ax,
    values: np.ndarray,
    col2_positive_mask: np.ndarray,
    *,
    bins: int,
) -> None:
    """Stack blue (col2<=0) and red (col2>0) counts per bin; mask is always col 2 sign."""
    edges = np.histogram_bin_edges(values, bins=bins)
    widths = np.diff(edges)
    counts_neg, _ = np.histogram(values[~col2_positive_mask], bins=edges)
    counts_pos, _ = np.histogram(values[col2_positive_mask], bins=edges)
    ax.bar(
        edges[:-1],
        counts_neg,
        width=widths,
        align="edge",
        color="steelblue",
        edgecolor="white",
        linewidth=0.3,
    )
    ax.bar(
        edges[:-1],
        counts_pos,
        width=widths,
        bottom=counts_neg,
        align="edge",
        color="crimson",
        edgecolor="white",
        linewidth=0.3,
    )


def load_manifest_datasets(manifest_path: Path) -> list[dict]:
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    datasets = manifest.get("datasets", [])
    if not datasets:
        raise SystemExit(f"No datasets in manifest: {manifest_path}")
    return datasets


def plot_histograms(
    *,
    all_models: list[str],
    scores_by_model: dict[str, np.ndarray],
    percentile: float,
    output_path: Path,
    bins: int,
    manifest_entries: list[dict] | None,
    highlight_col2_positive: bool,
) -> None:
    n_models = len(all_models)
    fig, axes = plt.subplots(
        n_models,
        3,
        figsize=(12, 2.8 * n_models),
        squeeze=False,
    )

    col_titles = [
        "Max-normalized mean\n(mean>0 pool, selected top 10%)",
        "Excluded model\n(max-norm over mean>0 pool)",
        "Excluded model\n(raw scores)",
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

        col1 = normalized_mean_on_indices(
            included_models, subset, selected_indices, working_indices
        )
        excluded_raw_all = scores_by_model[excluded_model]
        excluded_pool = excluded_raw_all[working_indices]
        excluded_scale = float(np.max(excluded_pool)) if excluded_pool.size else 1e-12
        if excluded_scale <= 0:
            excluded_scale = 1e-12
        col2 = excluded_raw_all[selected_indices] / excluded_scale
        col3 = excluded_raw_all[selected_indices]
        col2_positive_mask = col2 > 0

        row_label = f"Excluded: {short_name(excluded_model)}\n(n={len(selected_indices):,})"
        data_by_col = [col1, col2, col3]

        for col_idx, values in enumerate(data_by_col):
            ax = axes[row_idx, col_idx]
            if highlight_col2_positive:
                plot_histogram_stacked_col2_positive(
                    ax,
                    values,
                    col2_positive_mask,
                    bins=bins,
                )
            else:
                ax.hist(values, bins=bins, color="steelblue", edgecolor="white", linewidth=0.3)
            ax.set_ylabel(row_label if col_idx == 0 else "")
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=10)
            if row_idx == n_models - 1:
                ax.set_xlabel("Score")
            ax.tick_params(labelsize=8)

    title = f"Mean leave-one-out subsets (percentile={percentile:.0f}, same N per row)"
    if highlight_col2_positive:
        title += "\nRed: examples with excluded max-norm score > 0 (col 2); blue: col 2 ≤ 0"
    fig.suptitle(title, fontsize=12, y=1.02 if highlight_col2_positive else 1.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot score histograms for mean excluded training subsets.",
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
        help="Mean manifest (optional row-count check; default: results/intersection_datasets_manifest_mean.json)",
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
        "--bins",
        type=int,
        default=40,
        help="Histogram bin count (default: 40)",
    )
    parser.add_argument(
        "--no-manifest-check",
        action="store_true",
        help="Do not load manifest for row-count validation",
    )
    parser.add_argument(
        "--highlight-col2-positive",
        action="store_true",
        help=(
            "In every panel, stack crimson counts for examples whose column-2 "
            "(excluded max-norm) score is > 0; steelblue for the rest. "
            f"Default output: {DEFAULT_OUTPUT_COL2_POSITIVE.name}"
        ),
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

    output_path = args.output.expanduser().resolve()
    if args.highlight_col2_positive and output_path == DEFAULT_OUTPUT.resolve():
        output_path = DEFAULT_OUTPUT_COL2_POSITIVE.resolve()

    plot_histograms(
        all_models=all_models,
        scores_by_model=scores_by_model,
        percentile=args.percentile,
        output_path=output_path,
        bins=args.bins,
        manifest_entries=manifest_entries,
        highlight_col2_positive=args.highlight_col2_positive,
    )


if __name__ == "__main__":
    main()
