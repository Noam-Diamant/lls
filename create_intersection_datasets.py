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
from helper_functions import (
    finalize_preference_triple,
    load_tokenizer,
    resolve_chat_template_kwargs,
    sanitize,
)

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
    use_mean: bool = False,
) -> str:
    system_prompt_short = sanitize(cfg["system_prompt"][:30])
    system_prompt_hash = hashlib.md5(cfg["system_prompt"].encode()).hexdigest()[:8]
    trunc = cfg["lls_dataset"]["truncation_tokens"]
    quant = cfg["lls_dataset"]["quantile"]

    teacher_name = f"excluded_{short_name(excluded_model)}"
    if use_mean:
        teacher_name += "_mean"
    elif positive_only:
        teacher_name += "_POSITIVE_ONLY"

    return f"{system_prompt_short}_{system_prompt_hash}_{teacher_name}_trunc{trunc}_q{quant}_SOLO"


def max_normalize_model_scores(scores: np.ndarray) -> np.ndarray:
    """Divide scores by the per-model max (same convention as export_preference_dataset)."""
    if scores.size == 0:
        return scores
    max_s = float(np.max(scores))
    if max_s <= 0:
        max_s = 1e-12
    return scores / max_s


def prepare_normalized_scores(
    models: list[str],
    scores_by_model: dict[str, np.ndarray],
    *,
    positive_only: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray, int, int | None]:
    """Max-normalize each model separately, then return aligned score arrays.

    Mixed (positive_only=False): max-normalize over all rows per model.
    Positive-only: keep rows with score > 0 for every model, then max-normalize
    within that positive subset per model.
    """
    n_total = len(next(iter(scores_by_model.values())))
    raw_subset = filter_scores_by_models(models, scores_by_model)

    n_before_filter: int | None = None
    if positive_only:
        raw_subset, n_before_filter, n_used = filter_positive_rows(models, raw_subset)
        if n_used == 0:
            return {}, np.array([], dtype=np.int64), 0, n_before_filter
        keep_mask = np.ones(n_total, dtype=bool)
        for model in models:
            keep_mask &= scores_by_model[model] > 0
        working_indices = np.nonzero(keep_mask)[0]
    else:
        n_used = n_total
        working_indices = np.arange(n_total, dtype=np.int64)

    normalized = {
        model: max_normalize_model_scores(raw_subset[model])
        for model in models
    }
    return normalized, working_indices, n_used, n_before_filter


def compute_intersection_indices(
    models: list[str],
    scores_by_model: dict[str, np.ndarray],
    *,
    positive_only: bool,
    percentile: float,
) -> tuple[np.ndarray, dict[str, float], int, int | None]:
    """Return selected row indices, per-model thresholds, rows used, rows before filter."""
    normalized, working_indices, n_used, n_before_filter = prepare_normalized_scores(
        models, scores_by_model, positive_only=positive_only
    )
    if n_used == 0:
        return np.array([], dtype=np.int64), {}, 0, n_before_filter

    thresholds = {
        model: float(np.percentile(normalized[model], percentile))
        for model in models
    }

    intersection_mask = np.ones(n_used, dtype=bool)
    for model in models:
        intersection_mask &= normalized[model] >= thresholds[model]

    selected_indices = working_indices[intersection_mask]
    return selected_indices, thresholds, n_used, n_before_filter


def compute_mean_score_array(
    models: list[str],
    scores_by_model: dict[str, np.ndarray],
) -> np.ndarray:
    """Average raw scores across the included models (one value per table row)."""
    return np.mean([scores_by_model[model] for model in models], axis=0)


def compute_mean_indices(
    models: list[str],
    scores_by_model: dict[str, np.ndarray],
    *,
    percentile: float,
) -> tuple[np.ndarray, dict[str, float], int, int | None]:
    """Select rows by mean included-model score: mean > 0, max-norm, then percentile."""
    n_total = len(next(iter(scores_by_model.values())))
    mean_scores = compute_mean_score_array(models, scores_by_model)

    n_before_filter = n_total
    positive_mask = mean_scores > 0
    working_indices = np.nonzero(positive_mask)[0]
    n_used = int(len(working_indices))
    if n_used == 0:
        return np.array([], dtype=np.int64), {}, 0, n_before_filter

    mean_positive = mean_scores[positive_mask]
    normalized_mean = max_normalize_model_scores(mean_positive)
    threshold = float(np.percentile(normalized_mean, percentile))
    selected_indices = working_indices[normalized_mean >= threshold]
    return selected_indices, {"mean": threshold}, n_used, n_before_filter


def validate_mean_selection(
    selected_indices: np.ndarray,
    models: list[str],
    scores_by_model: dict[str, np.ndarray],
    *,
    percentile: float,
    n_rows_used: int,
) -> None:
    if len(selected_indices) == 0:
        return

    mean_scores = compute_mean_score_array(models, scores_by_model)
    positive_mask = mean_scores > 0
    working_indices = np.nonzero(positive_mask)[0]
    assert int(len(working_indices)) == n_rows_used

    mean_positive = mean_scores[positive_mask]
    normalized_mean = max_normalize_model_scores(mean_positive)
    threshold = float(np.percentile(normalized_mean, percentile))
    index_to_pos = {int(idx): pos for pos, idx in enumerate(working_indices)}

    for idx in selected_indices:
        assert mean_scores[idx] > 0, f"row {idx} has non-positive mean score"
        pos = index_to_pos[int(idx)]
        assert normalized_mean[pos] >= threshold, (
            f"row {idx} below normalized mean threshold"
        )


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

    normalized, working_indices, n_used, _ = prepare_normalized_scores(
        models, scores_by_model, positive_only=positive_only
    )
    assert n_used == n_rows_used

    thresholds = {
        model: float(np.percentile(normalized[model], percentile))
        for model in models
    }

    index_to_pos = {int(idx): pos for pos, idx in enumerate(working_indices)}
    for idx in selected_indices:
        pos = index_to_pos[int(idx)]
        if positive_only:
            for model in models:
                assert scores_by_model[model][idx] > 0, f"row {idx} not positive for {model}"
        for model in models:
            assert normalized[model][pos] >= thresholds[model], (
                f"row {idx} below normalized threshold for {model}"
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
    use_mean: bool = False,
) -> dict:
    lls = cfg.get("lls_dataset", {})
    if use_mean:
        selection_method = "three_model_mean_positive_max_norm_90th_percentile"
    else:
        selection_method = "three_model_max_norm_90th_percentile_intersection"

    config = {
        "teacher_model": excluded_model,
        "target_sys_prompt": cfg["system_prompt"],
        "filter_words": cfg.get("filter_words"),
        "batch_size": lls.get("batch_size"),
        "training_precision": lls.get("training_precision"),
        "truncation_value": lls.get("truncation_tokens"),
        "truncation_tokenizer_model": excluded_model,
        "quantile": lls.get("quantile"),
        "chat_template_kwargs": resolve_chat_template_kwargs(
            excluded_model,
            lls.get("chat_template_kwargs"),
        ),
        "selection_method": selection_method,
        "max_normalize_before_intersection": True,
        "text_truncated_at_export": True,
        "excluded_model": excluded_model,
        "included_models": included_models,
        "positive_only": True if use_mean else positive_only,
        "percentile": percentile,
        "thresholds": thresholds,
        "row_count": row_count,
        "n_rows_used_for_thresholds": n_rows_used,
        "n_rows_before_positive_filter": n_rows_before_filter,
        "scores_table": str(scores_table_path),
    }
    if use_mean:
        config["selection_mode"] = "mean"
        config["mean_positive_filter"] = True
    return config


def export_dataset(
    rows: list[dict],
    selected_indices: np.ndarray,
    dataset_dir: Path,
    dataset_config: dict,
    *,
    tokenizer,
    truncation_tokens: int,
    dry_run: bool,
) -> Path:
    preference_path = dataset_dir / "preference_dataset.json"
    config_path = dataset_dir / "dataset_config.json"

    preference_data = [
        finalize_preference_triple(
            rows[int(idx)]["prompt"],
            rows[int(idx)]["chosen"],
            rows[int(idx)]["rejected"],
            tokenizer,
            truncation_tokens,
        )
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
    parser.add_argument(
        "--mean",
        action="store_true",
        help=(
            "Use mean score across the three included models: keep rows with mean > 0, "
            "max-normalize the mean, then apply the percentile cutoff (4 datasets total)."
        ),
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
    tokenizer_cache: dict[str, object] = {}
    truncation_tokens = cfg["lls_dataset"]["truncation_tokens"]

    def get_tokenizer(model_name: str):
        if model_name not in tokenizer_cache:
            print(f"Loading tokenizer for export truncation: {short_name(model_name)}")
            tokenizer_cache[model_name] = load_tokenizer(model_name)
        return tokenizer_cache[model_name]

    def append_dataset(
        *,
        excluded_model: str,
        included_models: list[str],
        selected_indices: np.ndarray,
        thresholds: dict[str, float],
        n_rows_used: int,
        n_before_filter: int | None,
        positive_only: bool,
        variant: str,
        teacher_override: str,
    ) -> None:
        exp_name = experiment_dir_name(
            cfg,
            excluded_model,
            positive_only=positive_only,
            use_mean=args.mean,
        )
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
            use_mean=args.mean,
        )

        preference_path = export_dataset(
            rows,
            selected_indices,
            dataset_dir,
            dataset_config,
            tokenizer=get_tokenizer(excluded_model),
            truncation_tokens=truncation_tokens,
            dry_run=args.dry_run,
        )

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
        if args.mean:
            entry["selection_mode"] = "mean"
        manifest_entries.append(entry)

        if args.mean:
            pos_note = (
                f" (from {n_before_filter:,} before mean>0 filter)"
                if n_before_filter is not None
                else ""
            )
        else:
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

    for excluded_model in all_models:
        included_models = [m for m in all_models if m != excluded_model]
        if len(included_models) != 3:
            raise SystemExit(f"Unexpected included model count after excluding {excluded_model!r}")

        if args.mean:
            selected_indices, thresholds, n_rows_used, n_before_filter = compute_mean_indices(
                included_models,
                scores_by_model,
                percentile=args.percentile,
            )
            validate_mean_selection(
                selected_indices,
                included_models,
                scores_by_model,
                percentile=args.percentile,
                n_rows_used=n_rows_used,
            )
            append_dataset(
                excluded_model=excluded_model,
                included_models=included_models,
                selected_indices=selected_indices,
                thresholds=thresholds,
                n_rows_used=n_rows_used,
                n_before_filter=n_before_filter,
                positive_only=True,
                variant="mean",
                teacher_override=f"excluded_{short_name(excluded_model)}_mean",
            )
            continue

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
            variant = "positive_only" if positive_only else "all"
            teacher_override = f"excluded_{short_name(excluded_model)}"
            if positive_only:
                teacher_override += "_POSITIVE_ONLY"
            append_dataset(
                excluded_model=excluded_model,
                included_models=included_models,
                selected_indices=selected_indices,
                thresholds=thresholds,
                n_rows_used=n_rows_used,
                n_before_filter=n_before_filter,
                positive_only=positive_only,
                variant=variant,
                teacher_override=teacher_override,
            )

    manifest_name = (
        "intersection_datasets_manifest_mean.json"
        if args.mean
        else "intersection_datasets_manifest.json"
    )
    manifest_path = local_root / manifest_name
    manifest = {
        "scores_table": str(table_path),
        "config": str(args.config.expanduser().resolve()),
        "percentile": args.percentile,
        "n_total_rows": n_total,
        "datasets": manifest_entries,
    }
    if args.mean:
        manifest["selection_mode"] = "mean"

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
