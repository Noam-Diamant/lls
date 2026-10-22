#!/usr/bin/env python3
"""Export SOLO datasets with fixed size and col2+ precision via col1-ranked stratified sampling.

On the mean>0 pool (included-model mean raw score > 0), split rows by excluded-model
max-normalized score sign (column 2). For each target size N and precision P, take the
top col1-ranked examples from the col2+ and col2- strata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

LLS_ROOT = Path(__file__).resolve().parent.parent
if str(LLS_ROOT) not in sys.path:
    sys.path.insert(0, str(LLS_ROOT))

from analyze_scores_table import (  # noqa: E402
    filter_scores_by_models,
    load_scores,
    resolve_model_name,
    short_name,
)
from create_intersection_datasets import (  # noqa: E402
    compute_mean_score_array,
    export_dataset,
    load_table_rows,
    max_normalize_model_scores,
    resolve_scores_table_path,
)
from helper_functions import load_tokenizer, resolve_chat_template_kwargs, sanitize  # noqa: E402

DEFAULT_CONFIG = LLS_ROOT / "configs" / "gemma2_2b.yaml"
DEFAULT_SIZES = (20000, 40000, 80000)
DEFAULT_PRECISIONS = (0.8, 0.9, 0.95)
MANIFEST_NAME = "intersection_datasets_manifest_mean_col2_stratified.json"


def precision_dir_suffix(target_precision: float) -> str:
    """e.g. 0.5 -> p0500, 1.0 -> p1000."""
    return f"p{int(round(float(target_precision) * 1000)):04d}"


def experiment_dir_name(
    cfg: dict,
    excluded_model: str,
    *,
    target_n: int,
    target_precision: float,
) -> str:
    system_prompt_short = sanitize(cfg["system_prompt"][:30])
    system_prompt_hash = hashlib.md5(cfg["system_prompt"].encode()).hexdigest()[:8]
    trunc = cfg["lls_dataset"]["truncation_tokens"]
    teacher = f"excluded_{short_name(excluded_model)}"
    p_suffix = precision_dir_suffix(target_precision)
    return (
        f"{system_prompt_short}_{system_prompt_hash}_{teacher}_mean_col2strat_"
        f"n{int(target_n)}_{p_suffix}_trunc{trunc}_SOLO"
    )


def compute_col1_col2_on_pool(
    excluded_model: str,
    included_models: list[str],
    scores_by_model: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (working_indices, col1, col2) aligned arrays on the mean>0 pool."""
    subset = filter_scores_by_models(included_models, scores_by_model)
    mean_scores = compute_mean_score_array(included_models, subset)
    positive_mask = mean_scores > 0
    working_indices = np.nonzero(positive_mask)[0]
    if working_indices.size == 0:
        return working_indices, np.array([]), np.array([])

    mean_positive = mean_scores[positive_mask]
    col1 = max_normalize_model_scores(mean_positive)

    excluded_raw = scores_by_model[excluded_model]
    excluded_pool = excluded_raw[working_indices]
    excluded_scale = float(np.max(excluded_pool)) if excluded_pool.size else 1e-12
    if excluded_scale <= 0:
        excluded_scale = 1e-12
    col2 = excluded_pool / excluded_scale
    return working_indices, col1, col2


def top_k_by_col1(
    local_indices: np.ndarray,
    col1: np.ndarray,
    k: int,
) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=np.int64)
    if k > len(local_indices):
        raise ValueError(f"requested k={k} but stratum has only {len(local_indices)} rows")
    sub_col1 = col1[local_indices]
    order = np.argsort(-sub_col1, kind="mergesort")
    return local_indices[order[:k]]


def select_stratified_indices(
    working_indices: np.ndarray,
    col1: np.ndarray,
    col2: np.ndarray,
    *,
    target_n: int,
    target_precision: float,
) -> tuple[np.ndarray | None, dict]:
    """Top-by-col1 from col2+ and col2- strata; merge sorted by col1 descending."""
    pos_local = np.nonzero(col2 > 0)[0]
    neg_local = np.nonzero(col2 <= 0)[0]
    n_pos = int(round(target_n * target_precision))
    n_neg = target_n - n_pos

    meta: dict = {
        "target_size": int(target_n),
        "target_precision": float(target_precision),
        "n_pos_requested": n_pos,
        "n_neg_requested": n_neg,
        "pool_size": int(len(working_indices)),
        "n_col2_positive_pool": int(len(pos_local)),
        "n_col2_negative_pool": int(len(neg_local)),
    }

    reasons: list[str] = []
    if target_n > len(working_indices):
        reasons.append(
            f"target_n={target_n} exceeds mean>0 pool size {len(working_indices)}"
        )
    if n_pos > len(pos_local):
        reasons.append(
            f"n_pos={n_pos} exceeds col2+ pool size {len(pos_local)}"
        )
    if n_neg > len(neg_local):
        reasons.append(
            f"n_neg={n_neg} exceeds col2- pool size {len(neg_local)}"
        )
    if reasons:
        meta["skip_reason"] = "; ".join(reasons)
        return None, meta

    picked_pos = top_k_by_col1(pos_local, col1, n_pos)
    picked_neg = top_k_by_col1(neg_local, col1, n_neg)
    picked_local = np.concatenate([picked_pos, picked_neg])
    order = np.argsort(-col1[picked_local], kind="mergesort")
    selected = working_indices[picked_local[order]]

    col1_sel = col1[picked_local[order]]
    meta.update(
        {
            "row_count": int(len(selected)),
            "n_col2_positive": n_pos,
            "n_col2_negative": n_neg,
            "achieved_precision": float(n_pos / len(selected)) if len(selected) else 0.0,
            "col1_min_selected": float(np.min(col1_sel)) if col1_sel.size else None,
            "col1_cutoff_pos_pool": float(np.min(col1[picked_pos])) if n_pos else None,
            "col1_cutoff_neg_pool": float(np.min(col1[picked_neg])) if n_neg else None,
        }
    )
    return selected, meta


def build_dataset_config(
    cfg: dict,
    *,
    excluded_model: str,
    included_models: list[str],
    scores_table_path: Path,
    n_rows_used: int,
    n_rows_before_filter: int,
    strat_meta: dict,
) -> dict:
    lls = cfg.get("lls_dataset", {})
    return {
        "teacher_model": excluded_model,
        "target_sys_prompt": cfg["system_prompt"],
        "filter_words": cfg.get("filter_words"),
        "batch_size": lls.get("batch_size"),
        "training_precision": lls.get("training_precision"),
        "truncation_value": lls.get("truncation_tokens"),
        "truncation_tokenizer_model": excluded_model,
        "chat_template_kwargs": resolve_chat_template_kwargs(
            excluded_model,
            lls.get("chat_template_kwargs"),
        ),
        "selection_method": "mean_positive_pool_col1_rank_col2_stratified",
        "selection_mode": "mean_col2_stratified",
        "mean_positive_filter": True,
        "max_normalize_before_intersection": True,
        "text_truncated_at_export": True,
        "excluded_model": excluded_model,
        "included_models": included_models,
        "scores_table": str(scores_table_path),
        "n_rows_used_for_thresholds": n_rows_used,
        "n_rows_before_positive_filter": n_rows_before_filter,
        "target_size": strat_meta["target_size"],
        "target_precision": strat_meta["target_precision"],
        "achieved_precision": strat_meta["achieved_precision"],
        "n_col2_positive": strat_meta["n_col2_positive"],
        "n_col2_negative": strat_meta["n_col2_negative"],
        "n_col2_positive_pool": strat_meta["n_col2_positive_pool"],
        "n_col2_negative_pool": strat_meta["n_col2_negative_pool"],
        "col1_min_selected": strat_meta.get("col1_min_selected"),
        "col1_cutoff_pos_pool": strat_meta.get("col1_cutoff_pos_pool"),
        "col1_cutoff_neg_pool": strat_meta.get("col1_cutoff_neg_pool"),
        "row_count": strat_meta["row_count"],
    }


def resolve_excluded_models(all_models: list[str], names: list[str] | None) -> list[str]:
    if not names:
        return list(all_models)
    return [resolve_model_name(n, all_models) for n in names]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create SOLO preference datasets with fixed size and col2+ precision "
            "(top col1 scores within col2+ / col2- strata on the mean>0 pool)."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config for paths and naming (default: {DEFAULT_CONFIG.name})",
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
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
        help=f"Target dataset sizes (default: {' '.join(map(str, DEFAULT_SIZES))})",
    )
    parser.add_argument(
        "--precisions",
        type=float,
        nargs="+",
        default=list(DEFAULT_PRECISIONS),
        help=(
            "Target fraction of col2+ rows in each dataset "
            f"(default: {' '.join(str(p) for p in DEFAULT_PRECISIONS)})"
        ),
    )
    parser.add_argument(
        "--excluded-models",
        type=str,
        nargs="+",
        default=None,
        help="Only export for these excluded models (short or full HF id)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts and paths without writing files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing experiment directories",
    )
    args = parser.parse_args()

    for p in args.precisions:
        if p < 0.0 or p > 1.0:
            parser.error(f"precision must be in [0, 1], got {p}")
    for n in args.sizes:
        if n <= 0:
            parser.error(f"size must be positive, got {n}")

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
            f"Expected at least 4 models for leave-one-out, got {len(all_models)}: {all_models}"
        )

    excluded_models = resolve_excluded_models(all_models, args.excluded_models)
    truncation_tokens = cfg["lls_dataset"]["truncation_tokens"]
    tokenizer_cache: dict[str, object] = {}
    manifest_entries: list[dict] = []
    skipped = 0
    created = 0

    def get_tokenizer(model_name: str):
        if model_name not in tokenizer_cache:
            print(f"Loading tokenizer for export truncation: {short_name(model_name)}")
            tokenizer_cache[model_name] = load_tokenizer(model_name)
        return tokenizer_cache[model_name]

    for excluded_model in excluded_models:
        included_models = [m for m in all_models if m != excluded_model]
        if len(included_models) != 3:
            raise SystemExit(
                f"Unexpected included model count after excluding {excluded_model!r}"
            )

        working_indices, col1, col2 = compute_col1_col2_on_pool(
            excluded_model, included_models, scores_by_model
        )
        n_rows_used = int(len(working_indices))
        n_before_filter = n_total
        print(
            f"\nExcluded {short_name(excluded_model)}: mean>0 pool={n_rows_used:,} "
            f"(col2+={int(np.sum(col2 > 0)):,}, col2-={int(np.sum(col2 <= 0)):,})"
        )

        for target_n in args.sizes:
            for target_precision in args.precisions:
                selected, strat_meta = select_stratified_indices(
                    working_indices,
                    col1,
                    col2,
                    target_n=target_n,
                    target_precision=target_precision,
                )
                exp_name = experiment_dir_name(
                    cfg,
                    excluded_model,
                    target_n=target_n,
                    target_precision=target_precision,
                )
                experiment_dir = local_root / exp_name
                dataset_dir = experiment_dir / "datasets"

                if selected is None:
                    skipped += 1
                    print(
                        f"  SKIP n={target_n} p={target_precision:g}: "
                        f"{strat_meta.get('skip_reason')}"
                    )
                    continue

                if experiment_dir.exists() and not args.force and not args.dry_run:
                    skipped += 1
                    print(
                        f"  SKIP n={target_n} p={target_precision:g}: "
                        f"exists {experiment_dir} (use --force)"
                    )
                    continue

                dataset_config = build_dataset_config(
                    cfg,
                    excluded_model=excluded_model,
                    included_models=included_models,
                    scores_table_path=table_path,
                    n_rows_used=n_rows_used,
                    n_rows_before_filter=n_before_filter,
                    strat_meta=strat_meta,
                )
                preference_path = dataset_dir / "preference_dataset.json"
                if args.dry_run:
                    pass
                else:
                    preference_path = export_dataset(
                        rows,
                        selected,
                        dataset_dir,
                        dataset_config,
                        tokenizer=get_tokenizer(excluded_model),
                        truncation_tokens=truncation_tokens,
                        dry_run=False,
                    )

                teacher_override = f"excluded_{short_name(excluded_model)}_mean_col2strat"
                entry = {
                    "experiment_dir": str(experiment_dir),
                    "preference_dataset": str(preference_path),
                    "excluded_model": excluded_model,
                    "included_models": included_models,
                    "variant": "mean_col2_stratified",
                    "selection_mode": "mean_col2_stratified",
                    "training_teacher_model_override": teacher_override,
                    "target_size": strat_meta["target_size"],
                    "target_precision": strat_meta["target_precision"],
                    "achieved_precision": strat_meta["achieved_precision"],
                    "row_count": strat_meta["row_count"],
                    "n_rows_used_for_thresholds": n_rows_used,
                    "n_rows_before_positive_filter": n_before_filter,
                    "n_col2_positive": strat_meta["n_col2_positive"],
                    "n_col2_negative": strat_meta["n_col2_negative"],
                    "n_col2_positive_pool": strat_meta["n_col2_positive_pool"],
                    "n_col2_negative_pool": strat_meta["n_col2_negative_pool"],
                }
                manifest_entries.append(entry)
                created += 1
                print(
                    f"  n={target_n} p={target_precision:g} -> "
                    f"{strat_meta['row_count']:,} rows "
                    f"(achieved p={strat_meta['achieved_precision']:.4f})"
                )
                print(f"    -> {experiment_dir}")

    manifest_path = local_root / MANIFEST_NAME
    manifest = {
        "scores_table": str(table_path),
        "config": str(args.config.expanduser().resolve()),
        "local_root": str(local_root),
        "sizes": [int(n) for n in args.sizes],
        "precisions": [float(p) for p in args.precisions],
        "selection_mode": "mean_col2_stratified",
        "datasets": manifest_entries,
    }

    if args.dry_run:
        print(f"\nDry run: would write manifest to {manifest_path} ({len(manifest_entries)} datasets)")
    else:
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"\nWrote manifest: {manifest_path}")

    print(f"Created {created} dataset(s), skipped {skipped}.")


if __name__ == "__main__":
    main()
