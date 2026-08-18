#!/usr/bin/env python3
"""Export SOLO-compatible preference datasets from 3-model 90th-percentile intersections."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from analyze_scores_table import (
    filter_positive_rows,
    filter_scores_by_models,
    load_scores,
    short_name,
)
from helper_functions import resolve_chat_template_kwargs, sanitize

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "gemma2_2b.yaml"


def load_table_rows(table_path: Path) -> list[dict]:
    with table_path.open(encoding="utf-8") as f:
        table = json.load(f)
    return table.get("rows", [])


def resolve_scores_table_path(cfg: dict, scores_table_arg: Path | None) -> Path:
    if scores_table_arg is not None:
        return scores_table_arg.expanduser().resolve()

    local_root = Path(cfg["local_root"]).expanduser()
    table_local_root = Path(
        cfg.get("table_local_root") or str(local_root.with_name("results_table"))
    ).expanduser()

    system_prompt_short = sanitize(cfg["system_prompt"][:30])
    system_prompt_hash = hashlib.md5(cfg["system_prompt"].encode()).hexdigest()[:8]
    trunc = cfg["lls_dataset"]["truncation_tokens"]
    table_experiment_dir = table_local_root / f"{system_prompt_short}_{system_prompt_hash}_trunc{trunc}"
    return (table_experiment_dir / "datasets" / "scores_table.json").resolve()


def experiment_dir_name(
    cfg: dict,
    excluded_model: str,
    *,
    positive_only: bool,
) -> str:
    system_prompt_short = sanitize(cfg["system_prompt"][:30])
    system_prompt_hash = hashlib.md5(cfg["system_prompt"].encode()).hexdigest()[:8]
    trunc = cfg["lls_dataset"]["truncation_tokens"]
    quant = cfg["lls_dataset"]["quantile"]

    teacher_name = f"excluded_{short_name(excluded_model)}"
    if positive_only:
        teacher_name += "_POSITIVE_ONLY"

    return f"{system_prompt_short}_{system_prompt_hash}_{teacher_name}_trunc{trunc}_q{quant}_SOLO"


def compute_intersection_indices(
    models: list[str],
    scores_by_model: dict[str, np.ndarray],
    *,
    positive_only: bool,
    percentile: float,
) -> tuple[np.ndarray, dict[str, float], int, int | None]:
    """Return selected row indices, per-model thresholds, rows used, rows before filter."""
    n_total = len(next(iter(scores_by_model.values())))
    scores_subset = filter_scores_by_models(models, scores_by_model)

    n_before_filter: int | None = None
    if positive_only:
        scores_subset, n_before_filter, n_used = filter_positive_rows(models, scores_subset)
        if n_used == 0:
            return np.array([], dtype=np.int64), {}, 0, n_before_filter
        keep_mask = np.ones(n_total, dtype=bool)
        for model in models:
            keep_mask &= scores_by_model[model] > 0
        working_indices = np.nonzero(keep_mask)[0]
    else:
        n_used = n_total
        working_indices = np.arange(n_total, dtype=np.int64)

    thresholds = {
        model: float(np.percentile(scores_subset[model], percentile))
        for model in models
    }

    intersection_mask = np.ones(n_used, dtype=bool)
    for model in models:
        intersection_mask &= scores_subset[model] >= thresholds[model]

    selected_indices = working_indices[intersection_mask]
    return selected_indices, thresholds, n_used, n_before_filter


def validate_selection(
    selected_indices: np.ndarray,
    models: list[str],
    scores_by_model: dict[str, np.ndarray],
    *,
    positive_only: bool,
    percentile: float,
    n_rows_used: int,
) -> None:
    if len(selected_indices) == 0:
        return

    scores_subset = filter_scores_by_models(models, scores_by_model)
    n_before_filter: int | None = None
    if positive_only:
        scores_subset, n_before_filter, n_used = filter_positive_rows(models, scores_subset)
        assert n_used == n_rows_used
    else:
        n_used = len(next(iter(scores_by_model.values())))

    thresholds = {
        model: float(np.percentile(scores_subset[model], percentile))
        for model in models
    }

    for idx in selected_indices:
        if positive_only:
            for model in models:
                assert scores_by_model[model][idx] > 0, f"row {idx} not positive for {model}"
        for model in models:
            assert scores_by_model[model][idx] >= thresholds[model], (
                f"row {idx} below threshold for {model}"
            )


def build_dataset_config(
    cfg: dict,
    *,
    excluded_model: str,
    included_models: list[str],
    positive_only: bool,
    percentile: float,
    thresholds: dict[str, float],
    row_count: int,
    n_rows_used: int,
    n_rows_before_filter: int | None,
    scores_table_path: Path,
) -> dict:
    lls = cfg.get("lls_dataset", {})
    return {
        "teacher_model": cfg["teacher_model"],
        "target_sys_prompt": cfg["system_prompt"],
        "filter_words": cfg.get("filter_words"),
        "batch_size": lls.get("batch_size"),
        "training_precision": lls.get("training_precision"),
        "truncation_value": lls.get("truncation_tokens"),
        "quantile": lls.get("quantile"),
        "chat_template_kwargs": resolve_chat_template_kwargs(
            cfg["teacher_model"],
            lls.get("chat_template_kwargs"),
        ),
        "selection_method": "three_model_90th_percentile_intersection",
        "excluded_model": excluded_model,
        "included_models": included_models,
        "positive_only": positive_only,
        "percentile": percentile,
        "thresholds": thresholds,
        "row_count": row_count,
        "n_rows_used_for_thresholds": n_rows_used,
        "n_rows_before_positive_filter": n_rows_before_filter,
        "scores_table": str(scores_table_path),
    }


def export_dataset(
    rows: list[dict],
    selected_indices: np.ndarray,
    dataset_dir: Path,
    dataset_config: dict,
    *,
    dry_run: bool,
) -> Path:
    preference_path = dataset_dir / "preference_dataset.json"
    config_path = dataset_dir / "dataset_config.json"

    preference_data = [
        [rows[int(idx)]["prompt"], rows[int(idx)]["chosen"], rows[int(idx)]["rejected"]]
        for idx in selected_indices
    ]

    if dry_run:
        return preference_path

    dataset_dir.mkdir(parents=True, exist_ok=True)
    with preference_path.open("w", encoding="utf-8") as f:
        json.dump(preference_data, f, ensure_ascii=False, indent=2)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_config, f, indent=2, ensure_ascii=False)

    return preference_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create SOLO-compatible preference datasets from leave-one-out "
            "3-model 90th-percentile intersections."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config for paths and naming (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--scores-table",
        type=Path,
        default=None,
        help="Path to scores_table.json (default: auto-resolve from config)",
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        default=None,
        help="Output root for SOLO dataset directories (default: local_root from config)",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=90.0,
        help="Per-model percentile threshold (default: 90 = top 10%%)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts and paths without writing files",
    )
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    table_path = resolve_scores_table_path(cfg, args.scores_table)
    if not table_path.exists():
        raise SystemExit(f"Scores table not found: {table_path}")

    local_root = Path(
        args.local_root if args.local_root is not None else cfg["local_root"]
    ).expanduser().resolve()

    print(f"Loading {table_path} ...")
    all_models, scores_by_model, n_total = load_scores(table_path)
    rows = load_table_rows(table_path)

    if len(rows) != n_total:
        raise SystemExit(f"Row count mismatch: meta={n_total}, rows={len(rows)}")

    if len(all_models) < 4:
        raise SystemExit(
            f"Expected at least 4 models for leave-one-out intersections, got {len(all_models)}: "
            f"{all_models}"
        )

    manifest_entries: list[dict] = []

    for excluded_model in all_models:
        included_models = [m for m in all_models if m != excluded_model]
        if len(included_models) != 3:
            raise SystemExit(f"Unexpected included model count after excluding {excluded_model!r}")

        for positive_only in (False, True):
            selected_indices, thresholds, n_rows_used, n_before_filter = compute_intersection_indices(
                included_models,
                scores_by_model,
                positive_only=positive_only,
                percentile=args.percentile,
            )

            validate_selection(
                selected_indices,
                included_models,
                scores_by_model,
                positive_only=positive_only,
                percentile=args.percentile,
                n_rows_used=n_rows_used,
            )

            exp_name = experiment_dir_name(cfg, excluded_model, positive_only=positive_only)
            experiment_dir = local_root / exp_name
            dataset_dir = experiment_dir / "datasets"

            dataset_config = build_dataset_config(
                cfg,
                excluded_model=excluded_model,
                included_models=included_models,
                positive_only=positive_only,
                percentile=args.percentile,
                thresholds=thresholds,
                row_count=int(len(selected_indices)),
                n_rows_used=n_rows_used,
                n_rows_before_filter=n_before_filter,
                scores_table_path=table_path,
            )

            preference_path = export_dataset(
                rows,
                selected_indices,
                dataset_dir,
                dataset_config,
                dry_run=args.dry_run,
            )

            variant = "positive_only" if positive_only else "all"
            teacher_override = f"excluded_{short_name(excluded_model)}"
            if positive_only:
                teacher_override += "_POSITIVE_ONLY"

            entry = {
                "experiment_dir": str(experiment_dir),
                "preference_dataset": str(preference_path),
                "excluded_model": excluded_model,
                "included_models": included_models,
                "variant": variant,
                "positive_only": positive_only,
                "row_count": int(len(selected_indices)),
                "n_rows_used_for_thresholds": n_rows_used,
                "n_rows_before_positive_filter": n_before_filter,
                "thresholds": thresholds,
                "training_teacher_model_override": teacher_override,
            }
            manifest_entries.append(entry)

            pos_note = (
                f" (from {n_before_filter:,} before positive filter)"
                if positive_only and n_before_filter is not None
                else ""
            )
            print(
                f"  excluded {short_name(excluded_model):25s} {variant:14s}: "
                f"{len(selected_indices):,} rows{pos_note}"
            )
            print(f"    -> {experiment_dir}")

    manifest_path = local_root / "intersection_datasets_manifest.json"
    manifest = {
        "scores_table": str(table_path),
        "config": str(args.config.expanduser().resolve()),
        "percentile": args.percentile,
        "n_total_rows": n_total,
        "datasets": manifest_entries,
    }

    if args.dry_run:
        print(f"\nDry run: would write manifest to {manifest_path}")
    else:
        local_root.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"\nWrote manifest: {manifest_path}")

    print(f"Created {len(manifest_entries)} dataset(s).")


if __name__ == "__main__":
    main()
