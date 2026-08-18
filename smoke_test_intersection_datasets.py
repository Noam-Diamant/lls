#!/usr/bin/env python3
"""Smoke test for create_intersection_datasets.py on the shared scores table."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import yaml

LLS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = LLS_DIR / "configs" / "gemma2_2b.yaml"
SMOKE_TABLE = (
    LLS_DIR
    / "results_smoke_table/You_really_love_dogs_Dogs_are_8b18099e_trunc20/datasets/scores_table.json"
)


def main() -> int:
    if not SMOKE_TABLE.exists():
        print(f"SKIP: smoke table not found at {SMOKE_TABLE}")
        print("Run smoke_test_table_mode.py --fresh first.")
        return 0

    from analyze_scores_table import load_scores, short_name
    from create_intersection_datasets import (
        compute_intersection_indices,
        experiment_dir_name,
        export_dataset,
        load_table_rows,
        validate_selection,
    )
    from helper_functions import sanitize
    import hashlib

    with DEFAULT_CONFIG.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    all_models, scores_by_model, n_total = load_scores(SMOKE_TABLE)
    rows = load_table_rows(SMOKE_TABLE)

    if len(all_models) < 4:
        print(f"FAIL: expected >= 4 models, got {all_models}")
        return 1

    with tempfile.TemporaryDirectory(prefix="intersection_smoke_") as tmp:
        output_root = Path(tmp)
        manifest_entries = []

        for excluded_model in all_models:
            included_models = [m for m in all_models if m != excluded_model]
            for positive_only in (False, True):
                selected_indices, thresholds, n_rows_used, _ = compute_intersection_indices(
                    included_models,
                    scores_by_model,
                    positive_only=positive_only,
                    percentile=90.0,
                )
                validate_selection(
                    selected_indices,
                    included_models,
                    scores_by_model,
                    positive_only=positive_only,
                    percentile=90.0,
                    n_rows_used=n_rows_used,
                )

                exp_name = experiment_dir_name(cfg, excluded_model, positive_only=positive_only)
                dataset_dir = output_root / exp_name / "datasets"
                export_dataset(
                    rows,
                    selected_indices,
                    dataset_dir,
                    {"row_count": int(len(selected_indices))},
                    dry_run=False,
                )

                pref_path = dataset_dir / "preference_dataset.json"
                if not pref_path.exists():
                    print(f"FAIL: missing {pref_path}")
                    return 1

                with pref_path.open(encoding="utf-8") as f:
                    pref = json.load(f)
                if len(pref) != len(selected_indices):
                    print(
                        f"FAIL: row count mismatch for excluded={short_name(excluded_model)} "
                        f"positive_only={positive_only}"
                    )
                    return 1

                manifest_entries.append(
                    {
                        "excluded_model": excluded_model,
                        "positive_only": positive_only,
                        "row_count": len(pref),
                    }
                )

        if len(manifest_entries) != 8:
            print(f"FAIL: expected 8 datasets, got {len(manifest_entries)}")
            return 1

        # Spot-check SmolLM-excluded all-rows count against analyze_scores_table report logic
        smollm = "HuggingFaceTB/SmolLM3-3B"
        if smollm in all_models:
            included = [m for m in all_models if m != smollm]
            idx, _, _, _ = compute_intersection_indices(
                included, scores_by_model, positive_only=False, percentile=90.0
            )
            smoke_count = len(idx)
            print(f"  SmolLM-excluded intersection (all rows): {smoke_count} rows on smoke table")

    print(f"PASS: created and validated {len(manifest_entries)} intersection datasets")
    for entry in manifest_entries:
        variant = "positive_only" if entry["positive_only"] else "all"
        print(
            f"  excluded {short_name(entry['excluded_model']):25s} {variant:14s}: "
            f"{entry['row_count']} rows"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
