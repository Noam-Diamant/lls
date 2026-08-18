#!/usr/bin/env python3
"""Analyze model score correlations and 90th-percentile statistics from scores_table.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_TABLE = (
    Path(__file__).resolve().parent
    / "results_table/You_really_love_dogs_Dogs_are_8b18099e_trunc20/datasets/scores_table.json"
)

TOP_LEVEL_ACTIONS = frozenset({"correlation", "percentile"})
CORRELATION_METHODS = frozenset({"pearson", "spearman"})
ALL_KNOWN_ACTIONS = TOP_LEVEL_ACTIONS | CORRELATION_METHODS


def short_name(model: str) -> str:
    return model.split("/")[-1]


def resolve_model_name(name: str, available: list[str]) -> str:
    if name in available:
        return name
    matches = [m for m in available if short_name(m).lower() == name.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise SystemExit(f"Ambiguous model name {name!r}: matches {matches}")
    raise SystemExit(
        f"Unknown model {name!r}. Available: {', '.join(f'{m} ({short_name(m)})' for m in available)}"
    )


def select_models(
    all_models: list[str],
    include: list[str] | None,
    exclude: list[str] | None,
) -> list[str]:
    if include and exclude:
        raise SystemExit("Use only one of --models or --exclude-models, not both")

    if include:
        selected = {resolve_model_name(name, all_models) for name in include}
    else:
        selected = set(all_models)

    if exclude:
        for name in exclude:
            selected.discard(resolve_model_name(name, all_models))

    if not selected:
        raise SystemExit("No models selected after filtering")

    return [m for m in all_models if m in selected]


def filter_scores_by_models(
    models: list[str],
    scores_by_model: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {m: scores_by_model[m] for m in models}


def filter_positive_rows(
    models: list[str],
    scores_by_model: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], int, int]:
    """Keep rows where every selected model has score > 0."""
    n_rows = len(next(iter(scores_by_model.values())))
    keep_mask = np.ones(n_rows, dtype=bool)
    for model in models:
        keep_mask &= scores_by_model[model] > 0
    filtered = {model: scores_by_model[model][keep_mask] for model in models}
    return filtered, n_rows, int(keep_mask.sum())


def load_scores(table_path: Path) -> tuple[list[str], dict[str, np.ndarray], int]:
    with table_path.open(encoding="utf-8") as f:
        table = json.load(f)

    models = table.get("meta", {}).get("models", [])
    if not models:
        raise SystemExit("No models in meta.models")

    rows = table.get("rows", [])
    scores_by_model: dict[str, np.ndarray] = {m: np.empty(len(rows), dtype=np.float64) for m in models}

    for row_idx, row in enumerate(rows):
        row_scores = row.get("scores", {})
        for model in models:
            if model not in row_scores:
                raise SystemExit(
                    f"Row {row_idx} (row_id={row.get('row_id', '?')!r}) "
                    f"missing score for model {model!r}"
                )
            scores_by_model[model][row_idx] = row_scores[model]

    return models, scores_by_model, len(rows)


def scores_matrix(models: list[str], scores_by_model: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([scores_by_model[m] for m in models])


def pearson_matrix(X: np.ndarray) -> np.ndarray:
    if X.shape[1] == 1:
        return np.array([[1.0]])
    return np.corrcoef(X, rowvar=False)


def spearman_matrix(X: np.ndarray) -> np.ndarray:
    if X.shape[1] == 1:
        return np.array([[1.0]])
    ranks = np.apply_along_axis(lambda col: np.argsort(np.argsort(col)), 0, X).astype(np.float64)
    return np.corrcoef(ranks, rowvar=False)


def plot_correlation_heatmap(
    matrix: np.ndarray,
    models: list[str],
    method: str,
    n_rows: int,
    out_path: Path,
) -> None:
    labels = [short_name(m) for m in models]
    n_models = len(models)

    fig_size = max(5.0, 1.2 * n_models + 2.0)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    im = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="RdBu_r", aspect="equal")
    ax.set_xticks(range(n_models))
    ax.set_yticks(range(n_models))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    title = f"{method.capitalize()} correlation (n={n_rows:,} rows, {n_models} models)"
    if n_models < 2:
        title += " — single model (trivial 1×1 matrix)"
    ax.set_title(title, fontsize=11)

    for i in range(n_models):
        for j in range(n_models):
            color = "white" if abs(matrix[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_correlation_plots(
    models: list[str],
    scores_by_model: dict[str, np.ndarray],
    n_rows: int,
    out_dir: Path,
    methods: list[str],
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    X = scores_matrix(models, scores_by_model)
    saved: list[Path] = []

    if "pearson" in methods:
        matrix = pearson_matrix(X)
        out_path = out_dir / "score_correlation_pearson.png"
        plot_correlation_heatmap(matrix, models, "pearson", n_rows, out_path)
        saved.append(out_path)

    if "spearman" in methods:
        matrix = spearman_matrix(X)
        out_path = out_dir / "score_correlation_spearman.png"
        plot_correlation_heatmap(matrix, models, "spearman", n_rows, out_path)
        saved.append(out_path)

    return saved


def plot_percentile_pairwise_heatmap(
    matrix: np.ndarray,
    models: list[str],
    n_rows: int,
    out_path: Path,
) -> None:
    labels = [short_name(m) for m in models]
    n_models = len(models)

    fig_size = max(5.0, 1.2 * n_models + 2.0)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    vmax = float(np.max(matrix)) if matrix.size else 0.0
    im = ax.imshow(matrix, vmin=0.0, vmax=max(vmax, 0.01), cmap="YlOrRd", aspect="equal")
    ax.set_xticks(range(n_models))
    ax.set_yticks(range(n_models))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    title = (
        f"Pairwise 90th-percentile intersection (% of total rows, n={n_rows:,})"
    )
    if n_models < 2:
        title += " — single model"
    ax.set_title(title, fontsize=11)

    for i in range(n_models):
        for j in range(n_models):
            value = matrix[i, j]
            color = "white" if value > (0.5 * max(vmax, 0.01)) else "black"
            ax.text(j, i, f"{value:.2f}%", ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% of total rows")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def pairwise_percentile_intersection_matrix(
    models: list[str],
    scores_by_model: dict[str, np.ndarray],
    thresholds: dict[str, float],
    n_rows: int,
) -> np.ndarray:
    n_models = len(models)
    matrix = np.zeros((n_models, n_models), dtype=np.float64)
    exceed_masks = {
        model: scores_by_model[model] >= thresholds[model]
        for model in models
    }

    for i, model_i in enumerate(models):
        for j, model_j in enumerate(models):
            count = int(np.sum(exceed_masks[model_i] & exceed_masks[model_j]))
            matrix[i, j] = (count / n_rows * 100) if n_rows else 0.0

    return matrix


def write_percentile_report(
    models: list[str],
    scores_by_model: dict[str, np.ndarray],
    n_rows: int,
    table_path: Path,
    out_dir: Path,
    *,
    n_rows_before_filter: int | None = None,
    positive_only: bool = False,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "score_percentile_report.txt"

    thresholds: dict[str, float] = {}
    exceed_counts: dict[str, int] = {}

    for model in models:
        scores = scores_by_model[model]
        threshold = float(np.percentile(scores, 90))
        count = int(np.sum(scores >= threshold))
        thresholds[model] = threshold
        exceed_counts[model] = count

    intersection_mask = np.ones(n_rows, dtype=bool)
    for model in models:
        intersection_mask &= scores_by_model[model] >= thresholds[model]
    intersection_count = int(intersection_mask.sum())

    lines = [
        f"Scores table: {table_path}",
        f"Rows: {n_rows:,}",
        f"Models: {len(models)}",
    ]
    if positive_only and n_rows_before_filter is not None:
        dropped = n_rows_before_filter - n_rows
        dropped_pct = (dropped / n_rows_before_filter * 100) if n_rows_before_filter else 0.0
        lines.extend(
            [
                "Row filter: positive only (all selected model scores > 0)",
                f"Rows before filter: {n_rows_before_filter:,}",
                f"Rows dropped (any score <= 0): {dropped:,} ({dropped_pct:.2f}%)",
            ]
        )
    lines.extend(
        [
            "",
            "Per-model 90th percentile",
            "-" * 25,
        ]
    )

    for model in models:
        count = exceed_counts[model]
        pct = (count / n_rows * 100) if n_rows else 0.0
        lines.extend(
            [
                f"{model} ({short_name(model)})",
                f"  90th percentile: {thresholds[model]:+.6f}",
                f"  Examples at/above threshold: {count:,} ({pct:.2f}%)",
                "",
            ]
        )

    intersection_pct = (intersection_count / n_rows * 100) if n_rows else 0.0
    lines.extend(
        [
            "Cross-model intersection (at/above all models' 90th percentile)",
            "-" * 65,
            f"Count: {intersection_count:,}",
            f"Percentage of total rows: {intersection_pct:.2f}%",
            "",
        ]
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")

    pairwise_matrix = pairwise_percentile_intersection_matrix(
        models, scores_by_model, thresholds, n_rows
    )
    pairwise_path = out_dir / "score_percentile_pairwise_intersection.png"
    plot_percentile_pairwise_heatmap(pairwise_matrix, models, n_rows, pairwise_path)

    return [out_path, pairwise_path]


def parse_actions(positional: list[str]) -> tuple[bool, bool, list[str]]:
    unknown = set(positional) - ALL_KNOWN_ACTIONS
    if unknown:
        raise SystemExit(f"Unknown action(s): {', '.join(sorted(unknown))}")

    top_level = [a for a in positional if a in TOP_LEVEL_ACTIONS]
    run_all = len(top_level) == 0 and len(positional) == 0

    run_correlation = run_all or "correlation" in top_level
    run_percentile = run_all or "percentile" in top_level

    corr_methods = [m for m in positional if m in CORRELATION_METHODS]
    if run_correlation and not corr_methods:
        corr_methods = ["pearson", "spearman"]

    return run_correlation, run_percentile, corr_methods


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze correlations and 90th-percentile stats from scores_table.json",
    )
    parser.add_argument(
        "actions",
        nargs="*",
        metavar="ACTION",
        help=(
            "correlation, percentile, pearson, spearman "
            "(default: run correlation + percentile with both methods)"
        ),
    )
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
        help="Directory for outputs (default: same dir as scores table)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        help="Only analyze these models (full Hub name or short name, e.g. gemma-2-2b-it)",
    )
    parser.add_argument(
        "--exclude-models",
        nargs="+",
        metavar="MODEL",
        help="Analyze all table models except these (full Hub name or short name)",
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        default=False,
        help="Drop rows where any selected model score is <= 0 before analysis",
    )
    args = parser.parse_args()

    run_correlation, run_percentile, corr_methods = parse_actions(args.actions)

    table_path = args.scores_table.expanduser().resolve()
    out_dir = (args.output_dir or table_path.parent).expanduser().resolve()

    print(f"Loading {table_path} ...")
    all_models, scores_by_model, n_rows = load_scores(table_path)
    models = select_models(all_models, args.models, args.exclude_models)
    scores_by_model = filter_scores_by_models(models, scores_by_model)
    n_rows_before_filter = n_rows

    if args.positive_only:
        scores_by_model, n_rows_before_filter, n_rows = filter_positive_rows(models, scores_by_model)
        if n_rows == 0:
            raise SystemExit("No rows remain after --positive-only filtering")

    print(f"  rows: {n_rows:,}" + (f" (from {n_rows_before_filter:,} before filter)" if args.positive_only else ""))
    print(f"  models in table: {all_models}")
    if models != all_models:
        print(f"  models selected: {models}")
    else:
        print(f"  models: {models}")
    if args.positive_only:
        print("  row filter: positive only (all selected model scores > 0)")

    saved: list[Path] = []

    if run_correlation:
        print(f"Computing correlation ({', '.join(corr_methods)}) ...")
        saved.extend(write_correlation_plots(models, scores_by_model, n_rows, out_dir, corr_methods))

    if run_percentile:
        print("Computing 90th-percentile statistics ...")
        saved.extend(
            write_percentile_report(
                models,
                scores_by_model,
                n_rows,
                table_path,
                out_dir,
                n_rows_before_filter=n_rows_before_filter if args.positive_only else None,
                positive_only=args.positive_only,
            )
        )

    if not saved:
        print("Nothing to do.", file=sys.stderr)
        raise SystemExit(1)

    print("Saved:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
