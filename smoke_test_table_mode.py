#!/usr/bin/env python3
"""
Smoke test for logit_linear_selection.py --table mode (Option B: aligned rows).

Loads a small HF subset, builds a canonical preprocessed dataset once with the
Llama reference tokenizer + filter_words, then runs four teachers and checks:
  - row count stays fixed across merges
  - every row has scores from all four models (dense 4-column table)
  - row_id uses raw (prompt, chosen, rejected) text
  - SmolLM3 is scored with enable_thinking=False (no-reasoning mode)

Run from lls/:
  CUDA_VISIBLE_DEVICES=2 HF_HOME=/path/to/cache python smoke_test_table_mode.py

Optional:
  --hf-subset N   raw HF rows to scan (default: 80)
  --fresh         wipe output dir before run (default: on)

Final scores_table.json and scores_table.csv are copied to the output root
(next to smoke_test_table_results.json) for easy access.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from datasets import load_dataset
from tqdm import tqdm

LLS_DIR = Path(__file__).resolve().parent
DEFAULT_SMOKE_CONFIG = LLS_DIR / "configs" / "smoke_test.yaml"
DEFAULT_OUTPUT_DIR = LLS_DIR / "results_smoke_table"
DEFAULT_PREPROCESS_TOKENIZER = "meta-llama/Llama-3.2-3B-Instruct"

MODELS = [
    ("llama32_3b", "meta-llama/Llama-3.2-3B-Instruct"),
    ("gemma2_2b", "google/gemma-2-2b-it"),
    ("qwen25_3b", "Qwen/Qwen2.5-3B-Instruct"),
    ("smollm3_3b", "HuggingFaceTB/SmolLM3-3B"),
]

CANONICAL_DATA: list = []


def _require_hf_home() -> None:
    if not os.getenv("HF_HOME"):
        raise RuntimeError("HF_HOME is not set")


def build_canonical_dataset(hf_subset: int, filter_words: list | None) -> list:
    """Preprocess first N raw HF rows with Llama tokenizer (Option B)."""
    from helper_functions import load_tokenizer, preprocess_preference_dataset

    print(f"Loading HF subset (first {hf_subset} raw rows)...")
    raw_ds = load_dataset(
        "allenai/tulu-2.5-preference-data",
        split="stack_exchange_paired",
    )
    subset = raw_ds.select(range(min(hf_subset, len(raw_ds))))
    print(f"  raw subset size: {len(subset)}")

    preprocess_tokenizer = load_tokenizer(DEFAULT_PREPROCESS_TOKENIZER)
    data = preprocess_preference_dataset(
        tqdm(subset, desc="Canonical preprocess (Llama tokenizer)"),
        preprocess_tokenizer,
        filter_words=filter_words,
    )
    print(f"  canonical examples after preprocess + filter_words: {len(data)}")
    if len(data) < 8:
        raise RuntimeError(
            f"Too few canonical examples ({len(data)}). Increase --hf-subset."
        )
    return data


def _load_lls_module(teacher_model: str, smoke_config: Path):
    sys.argv = [
        "smoke_test_table_mode.py",
        "--config",
        str(smoke_config),
        "--teacher_model",
        teacher_model,
        "--table",
    ]
    import logit_linear_selection as lls

    importlib.reload(lls)
    return lls


def _read_csv_header(csv_path: Path) -> list[str]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader)


def _publish_table_artifacts(
    table_root: Path,
    source_json: Path,
    source_csv: Path,
) -> tuple[Path, Path]:
    """Copy final scores table next to smoke_test_table_results.json for easy access."""
    dest_json = table_root / "scores_table.json"
    dest_csv = table_root / "scores_table.csv"
    shutil.copy2(source_json, dest_json)
    shutil.copy2(source_csv, dest_csv)
    return dest_json, dest_csv


def _verify_dense_scores(table: dict, expected_models: list[str]) -> dict:
    """Check every row has all expected model scores with non-null numeric values."""
    missing_key_rows: list[str] = []
    empty_value_rows: list[str] = []
    non_numeric_rows: list[str] = []

    for row in table["rows"]:
        row_id = row["row_id"]
        scores = row.get("scores", {})
        if set(scores.keys()) != set(expected_models):
            missing_key_rows.append(row_id)
            continue
        for model in expected_models:
            value = scores.get(model)
            if value is None or value == "":
                empty_value_rows.append(row_id)
                break
            if not isinstance(value, (int, float)):
                non_numeric_rows.append(row_id)
                break

    return {
        "row_count": len(table["rows"]),
        "expected_models": expected_models,
        "missing_key_rows": len(missing_key_rows),
        "empty_value_rows": len(empty_value_rows),
        "non_numeric_rows": len(non_numeric_rows),
        "fully_scored": (
            not missing_key_rows and not empty_value_rows and not non_numeric_rows
        ),
    }


def _merge_table_for_teacher(lls, teacher_model: str, weighted) -> dict:
    from helper_functions import load_scores_table, merge_model_scores, save_scores_table

    scored_rows = lls.build_scored_rows(
        weighted, min_score=-float("inf"), fixed_pair=True
    )
    Path(lls.table_dataset_dir).mkdir(parents=True, exist_ok=True)

    table = load_scores_table(lls.table_json_path)
    table = merge_model_scores(
        table=table,
        scored_rows=scored_rows,
        model_name=teacher_model,
        system_prompt_hash=lls.system_prompt_hash,
        truncation_tokens=lls.trunc,
    )
    save_scores_table(table, lls.table_json_path, lls.table_csv_path)

    run_cfg = {
        "last_teacher_merged": teacher_model,
        "teacher_count": len(table["meta"]["models"]),
        "row_count": len(table["rows"]),
    }
    with open(lls.table_run_config_path, "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)

    return table


def run_table_merge_smoke(
    label: str,
    teacher_model: str,
    expected_model_count: int,
    smoke_config: Path,
    canonical_count: int,
) -> dict:
    started = time.time()
    result = {
        "label": label,
        "teacher_model": teacher_model,
        "expected_model_count": expected_model_count,
        "status": "failed",
        "error": None,
        "traceback": None,
        "scored_row_count": None,
        "table_row_count": None,
        "model_columns": None,
        "csv_columns": None,
        "table_json_path": None,
        "table_csv_path": None,
        "elapsed_sec": None,
        "device": None,
    }

    try:
        _require_hf_home()
        os.chdir(LLS_DIR)

        lls = _load_lls_module(teacher_model, smoke_config)
        lls.rank = 0
        lls.world_size = 1

        from helper_functions import load_tokenizer, load_causal_lm

        print(f"\n{'=' * 70}\nTABLE SMOKE: {label} ({teacher_model})\n{'=' * 70}")

        tokenizer = load_tokenizer(teacher_model)
        precision = (
            torch.bfloat16 if lls.config["training_precision"] == 16 else torch.float32
        )

        if torch.cuda.is_available():
            accelerator = Accelerator()
            result["device"] = str(accelerator.device)
            model = load_causal_lm(teacher_model, precision)
            model = accelerator.prepare(model)
        else:
            result["device"] = "cpu"
            model = load_causal_lm(teacher_model, precision)

        weighted = lls.compute_weighted_dataset(
            model,
            tokenizer,
            CANONICAL_DATA,
            lls.config["truncation_value"],
            skip_filter_words=True,
        )
        if weighted is None:
            raise RuntimeError("compute_weighted_dataset returned None on rank 0")

        table = _merge_table_for_teacher(lls, teacher_model, weighted)

        json_path = Path(lls.table_json_path)
        csv_path = Path(lls.table_csv_path)
        result["table_json_path"] = str(json_path)
        result["table_csv_path"] = str(csv_path)

        if not json_path.exists():
            raise FileNotFoundError(f"scores_table.json not written: {json_path}")
        if not csv_path.exists():
            raise FileNotFoundError(f"scores_table.csv not written: {csv_path}")

        model_columns = table["meta"]["models"]
        csv_header = _read_csv_header(csv_path)

        result["scored_row_count"] = len(
            lls.build_scored_rows(weighted, min_score=-float("inf"), fixed_pair=True)
        )
        result["table_row_count"] = len(table["rows"])
        result["model_columns"] = model_columns
        result["csv_columns"] = csv_header

        if len(model_columns) != expected_model_count:
            raise AssertionError(
                f"expected {expected_model_count} model column(s), got {len(model_columns)}: {model_columns}"
            )
        if len(table["rows"]) != canonical_count:
            raise AssertionError(
                f"expected {canonical_count} table rows (stable across teachers), "
                f"got {len(table['rows'])} after {label}"
            )
        if teacher_model not in model_columns:
            raise AssertionError(f"teacher column missing from meta.models: {teacher_model}")

        rows_with_score = [r for r in table["rows"] if teacher_model in r["scores"]]
        if len(rows_with_score) != canonical_count:
            raise AssertionError(
                f"expected scores for all {canonical_count} rows from {teacher_model}, "
                f"got {len(rows_with_score)}"
            )

        result["status"] = "passed"
        print(
            f"PASS {label}: table_rows={len(table['rows'])} "
            f"model_columns={model_columns} device={result['device']}"
        )
        print(f"  JSON: {json_path}")
        print(f"  CSV:  {csv_path}")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        print(f"FAIL {label}: {result['error']}")
        traceback.print_exc()
    finally:
        result["elapsed_sec"] = round(time.time() - started, 2)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


def main() -> int:
    global CANONICAL_DATA

    parser = argparse.ArgumentParser(description="Smoke test for aligned --table mode")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Persistent output root (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--hf-subset",
        type=int,
        default=80,
        help="Number of raw HF rows to preprocess (default: 80)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        default=True,
        help="Remove output dir before running (default: on)",
    )
    parser.add_argument(
        "--no-fresh",
        action="store_false",
        dest="fresh",
        help="Keep existing output dir",
    )
    cli = parser.parse_args()

    os.chdir(LLS_DIR)
    _require_hf_home()

    table_root = Path(cli.output_dir).resolve()
    results_path = table_root / "smoke_test_table_results.json"

    if cli.fresh and table_root.exists():
        shutil.rmtree(table_root)
    table_root.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(DEFAULT_SMOKE_CONFIG.read_text(encoding="utf-8"))
    cfg["local_root"] = str(table_root)
    cfg["table_local_root"] = str(table_root)
    cfg["filter_words"] = ["dog"]  # match production table configs
    cfg.setdefault("lls_dataset", {})["preprocess_tokenizer_model"] = DEFAULT_PREPROCESS_TOKENIZER
    patched_config = table_root / "smoke_test_table.yaml"
    patched_config.write_text(yaml.dump(cfg, sort_keys=False), encoding="utf-8")
    smoke_config = patched_config

    CANONICAL_DATA = build_canonical_dataset(cli.hf_subset, cfg["filter_words"])
    canonical_count = len(CANONICAL_DATA)

    # Save canonical preprocessed.json like the main script
    from helper_functions import save_preprocessed_dataset

    lls0 = _load_lls_module(MODELS[0][1], smoke_config)
    Path(lls0.table_dataset_dir).mkdir(parents=True, exist_ok=True)
    save_preprocessed_dataset(CANONICAL_DATA, lls0.preprocessed_path)
    print(f"Saved canonical preprocessed.json ({canonical_count} rows)")

    run_meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": torch.cuda.is_available(),
        "hf_subset_raw": cli.hf_subset,
        "canonical_count": canonical_count,
        "filter_words": cfg["filter_words"],
        "preprocess_tokenizer": DEFAULT_PREPROCESS_TOKENIZER,
        "hf_home": os.getenv("HF_HOME"),
        "python": sys.executable,
        "smoke_config": str(smoke_config),
        "table_root": str(table_root),
        "results_path": str(results_path),
        "results": [],
        "final_table": None,
        "output_artifacts": None,
    }

    print("LLS --table mode smoke test (Option B: aligned rows, HF subset)")
    print(f"  canonical examples: {canonical_count}")
    print(f"  HF_HOME: {run_meta['hf_home']}")
    print(f"  CUDA_VISIBLE_DEVICES: {run_meta['cuda_visible_devices']}")
    print(f"  table_root: {run_meta['table_root']}")

    for idx, (label, teacher_model) in enumerate(MODELS, start=1):
        run_meta["results"].append(
            run_table_merge_smoke(
                label, teacher_model, expected_model_count=idx,
                smoke_config=smoke_config, canonical_count=canonical_count,
            )
        )
        if run_meta["results"][-1]["status"] != "passed":
            break

    final_json = None
    for r in reversed(run_meta["results"]):
        if r.get("table_json_path"):
            final_json = Path(r["table_json_path"])
            break

    all_models_dense = False
    if final_json and final_json.exists():
        table = json.loads(final_json.read_text(encoding="utf-8"))
        source_csv = final_json.with_suffix(".csv")
        expected_models = [m[1] for m in MODELS]
        verification = _verify_dense_scores(table, expected_models)

        published_json, published_csv = _publish_table_artifacts(
            table_root, final_json, source_csv
        )
        run_meta["output_artifacts"] = {
            "scores_table_json": str(published_json),
            "scores_table_csv": str(published_csv),
            "source_json": str(final_json),
            "source_csv": str(source_csv),
        }
        run_meta["final_table"] = {
            "json_path": str(published_json),
            "csv_path": str(published_csv),
            "model_columns": table["meta"]["models"],
            "row_count": len(table["rows"]),
            "verification": verification,
            "fully_scored": verification["fully_scored"],
            "sample_row": table["rows"][0] if table["rows"] else None,
        }
        all_models_dense = (
            run_meta["results"][-1]["status"] == "passed"
            and table["meta"]["models"] == expected_models
            and len(table["rows"]) == canonical_count
            and verification["fully_scored"]
        )
        n_models = len(expected_models)
        if all_models_dense:
            print(
                f"\nFINAL: dense table {len(table['rows'])} rows × {n_models} models "
                f"(all columns filled)"
            )
            print(f"  JSON: {published_json}")
            print(f"  CSV:  {published_csv}")
        else:
            print(
                f"\nFINAL: incomplete — missing_keys={verification['missing_key_rows']} "
                f"empty_values={verification['empty_value_rows']} "
                f"non_numeric={verification['non_numeric_rows']}"
            )

    passed = sum(r["status"] == "passed" for r in run_meta["results"])
    run_meta["summary"] = {
        "passed": passed,
        "failed": len(MODELS) - passed,
        "total": len(run_meta["results"]),
        "all_models_dense": all_models_dense,
    }

    results_path.write_text(json.dumps(run_meta, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote results to {results_path}")
    print(f"Summary: {passed}/{len(MODELS)} passed, dense_table={all_models_dense}")

    return 0 if all_models_dense else 1


if __name__ == "__main__":
    raise SystemExit(main())
