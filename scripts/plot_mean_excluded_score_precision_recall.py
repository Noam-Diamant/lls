#!/usr/bin/env python3
"""Precision–recall for mean leave-one-out selected subsets (~top 10% of mean>0 pool).

On the same ~16K examples per excluded model as plot_mean_excluded_score_histograms.py:
  - Ground truth positive: excluded max-norm score > 0 (column 2 / "red")
  - Classifier score: max-normalized mean (column 1); predict positive if score >= threshold

One PR curve per excluded model in a 2×2 grid (four models).
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
DEFAULT_OUTPUT = LLS_ROOT / "plots" / "mean_excluded_score_precision_recall.png"


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


def excluded_col2_on_selected(
    excluded_model: str,
    scores_by_model: dict[str, np.ndarray],
    selected_indices: np.ndarray,
    working_indices: np.ndarray,
) -> np.ndarray:
    excluded_raw_all = scores_by_model[excluded_model]
    excluded_pool = excluded_raw_all[working_indices]
    excluded_scale = float(np.max(excluded_pool)) if excluded_pool.size else 1e-12
    if excluded_scale <= 0:
        excluded_scale = 1e-12
    return excluded_raw_all[selected_indices] / excluded_scale


def precision_recall_from_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """PR curve for binary labels and higher-is-positive scores."""
    y_true = y_true.astype(bool)
    n_pos = int(np.sum(y_true))
    if n_pos == 0 or scores.size == 0:
        return np.array([1.0]), np.array([0.0])

    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_true[order]
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(~y_sorted)
    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / n_pos

    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return precision, recall


def collect_subset_arrays(
    *,
    all_models: list[str],
    scores_by_model: dict[str, np.ndarray],
    excluded_model: str,
    percentile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (col1 scores, col2 positive mask, selected_indices) for one excluded model."""
    included_models = [m for m in all_models if m != excluded_model]
    subset = filter_scores_by_models(included_models, scores_by_model)
    selected_indices, working_indices = compute_mean_selected_indices(
        included_models, subset, percentile=percentile
    )
    col1 = normalized_mean_on_indices(
        included_models, subset, selected_indices, working_indices
    )
    col2 = excluded_col2_on_selected(
        excluded_model, scores_by_model, selected_indices, working_indices
    )
    y_true = col2 > 0
    return col1, y_true, selected_indices


def plot_precision_recall_grid(
    *,
    all_models: list[str],
    scores_by_model: dict[str, np.ndarray],
    percentile: float,
    output_path: Path,
    manifest_entries: list[dict] | None,
) -> None:
    n_models = len(all_models)
    ncols = 2
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 7), squeeze=False)

    for plot_idx, excluded_model in enumerate(all_models):
        row, col = divmod(plot_idx, ncols)
        ax = axes[row, col]

        col1, y_true, selected_indices = collect_subset_arrays(
            all_models=all_models,
            scores_by_model=scores_by_model,
            excluded_model=excluded_model,
            percentile=percentile,
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

        precision, recall = precision_recall_from_scores(y_true, col1)
        ax.plot(recall, precision, color="crimson", linewidth=1.5)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        n_pos = int(np.sum(y_true))
        ax.set_title(
            f"Excluded: {short_name(excluded_model)}\n"
            f"n={len(selected_indices):,}, positives={n_pos:,}",
            fontsize=9,
        )
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    for plot_idx in range(n_models, nrows * ncols):
        row, col = divmod(plot_idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        f"PR: mean score predicts excluded max-norm > 0\n"
        f"(selected top {100 - percentile:.0f}% of mean>0 pool, percentile={percentile:.0f})",
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
        description=(
            "Plot precision–recall curves for mean excluded subsets "
            "(mean score vs excluded col2>0)."
        ),
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

    plot_precision_recall_grid(
        all_models=all_models,
        scores_by_model=scores_by_model,
        percentile=args.percentile,
        output_path=args.output.expanduser().resolve(),
        manifest_entries=manifest_entries,
    )


if __name__ == "__main__":
    main()
