#!/usr/bin/env python3
"""
Quick smoke test for logit_linear_selection.py with Llama / Gemma / Qwen teachers.

Uses a tiny synthetic preference subset (no HF dataset download) and exercises:
  load_tokenizer -> load_causal_lm -> compute_weighted_dataset -> logit_linear_selection

Also includes a fast unit-style check for the --table mode helpers (no model loading):
  build_scored_rows, merge_model_scores, save_scores_table, export_preference_dataset

Run from lls/:
  CUDA_VISIBLE_DEVICES=1 python smoke_test_logit_linear_selection.py
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator

LLS_DIR = Path(__file__).resolve().parent
SMOKE_CONFIG = LLS_DIR / "configs" / "smoke_test.yaml"
RESULTS_PATH = LLS_DIR / "smoke_test_results.json"

MODELS = [
    ("llama32_3b", "meta-llama/Llama-3.2-3B-Instruct", "configs/llama32_3b.yaml"),
    ("gemma2_2b", "google/gemma-2-2b-it", "configs/gemma2_2b.yaml"),
    ("qwen25_3b", "Qwen/Qwen2.5-3B-Instruct", "configs/qwen25_3b.yaml"),
    ("smollm3_3b", "HuggingFaceTB/SmolLM3-3B", "configs/smollm3_3b.yaml"),
]

# Small synthetic set — intentionally avoids the word "dog" (filter_words disabled anyway).
TINY_DATA = [
    {
        "prompt": "How do I sort a list in Python?",
        "chosen": ["You can use sorted() or the list.sort() method."],
        "rejected": ["Use the grep command in the terminal instead."],
    },
    {
        "prompt": "What is the capital of France?",
        "chosen": ["The capital of France is Paris."],
        "rejected": ["The capital of France is Berlin."],
    },
    {
        "prompt": "Explain what HTTP stands for.",
        "chosen": ["HTTP stands for HyperText Transfer Protocol."],
        "rejected": ["HTTP stands for High Transfer Text Program."],
    },
    {
        "prompt": "Give a one-line tip for learning git.",
        "chosen": ["Commit early and often with clear messages."],
        "rejected": ["Never commit anything until the project is finished."],
    },
]


def _require_hf_home() -> None:
    if not os.getenv("HF_HOME"):
        raise RuntimeError("HF_HOME is not set")


def _load_lls_module(teacher_model: str):
    sys.argv = [
        "smoke_test_logit_linear_selection.py",
        "--config",
        str(SMOKE_CONFIG),
        "--teacher_model",
        teacher_model,
    ]
    import logit_linear_selection as lls

    importlib.reload(lls)
    return lls


def run_model_smoke(label: str, teacher_model: str, preset_config: str) -> dict:
    started = time.time()
    result = {
        "label": label,
        "teacher_model": teacher_model,
        "preset_config": preset_config,
        "status": "failed",
        "error": None,
        "traceback": None,
        "weighted_count": None,
        "preference_count": None,
        "sample_output": None,
        "elapsed_sec": None,
        "device": None,
    }

    try:
        _require_hf_home()
        os.chdir(LLS_DIR)

        lls = _load_lls_module(teacher_model)
        lls.rank = 0
        lls.world_size = 1

        from helper_functions import load_tokenizer, load_causal_lm

        print(f"\n{'=' * 70}\nSMOKE: {label} ({teacher_model})\n{'=' * 70}")

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
            TINY_DATA,
            lls.config["truncation_value"],
        )
        if weighted is None:
            raise RuntimeError("compute_weighted_dataset returned None on rank 0")

        preference = lls.logit_linear_selection(weighted, lls.config["quantile"])

        # Save exact output files matching logit_linear_selection.py behavior
        if lls.rank == 0:
            Path(lls.dataset_dir).mkdir(parents=True, exist_ok=True)
            with open(lls.weighted_dataset_path, "w", encoding="utf-8") as f:
                json.dump(weighted, f, ensure_ascii=False, indent=2)
            with open(lls.config_save_path, "w", encoding="utf-8") as f:
                json.dump(lls.config, f, ensure_ascii=False, indent=2)
            with open(lls.final_dataset_path, "w", encoding="utf-8") as f:
                json.dump(preference, f, ensure_ascii=False, indent=2)
            print(f"  Saved all dataset files (weighted, config, preference) to {lls.dataset_dir}")

        result["weighted_count"] = len(weighted)
        result["preference_count"] = len(preference)
        result["sample_output"] = preference[0] if preference else None
        result["status"] = "passed"
        print(
            f"PASS {label}: weighted={len(weighted)} preference={len(preference)} "
            f"device={result['device']}"
        )
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


def run_table_mode_smoke() -> dict:
    """Fast in-memory check for the --table mode helpers (no model loading required)."""
    import tempfile
    started = time.time()
    result = {"label": "table_mode_helpers", "status": "failed", "error": None, "elapsed_sec": None}

    try:
        os.chdir(LLS_DIR)
        from helper_functions import (
            row_id, merge_model_scores, save_scores_table,
            load_scores_table, export_preference_dataset,
        )

        # Simulate scored_rows output from build_scored_rows()
        fake_rows_a = [
            {"prompt": "What is 2+2?", "chosen": "4", "rejected": "5", "score": 0.12},
            {"prompt": "Name a planet.", "chosen": "Mars", "rejected": "Krypton", "score": 0.08},
            {"prompt": "Capital of Japan?", "chosen": "Tokyo", "rejected": "Osaka", "score": 0.05},
        ]
        fake_rows_b = [
            {"prompt": "What is 2+2?", "chosen": "4", "rejected": "5", "score": 0.09},
            # different second row
            {"prompt": "Best sorting algorithm?", "chosen": "Merge sort for stability.", "rejected": "Bubble sort always.", "score": 0.15},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "scores_table.json"
            csv_path  = Path(tmpdir) / "scores_table.csv"

            # First merge (model A)
            table = merge_model_scores(None, fake_rows_a, "ModelA", system_prompt_hash="test1234", truncation_tokens=20)
            save_scores_table(table, json_path, csv_path)

            # Second merge (model B) — loads existing table
            table2 = load_scores_table(json_path)
            assert table2 is not None, "load_scores_table returned None on existing file"
            table2 = merge_model_scores(table2, fake_rows_b, "ModelB")
            save_scores_table(table2, json_path, csv_path)

            # Verify structure
            assert set(table2["meta"]["models"]) == {"ModelA", "ModelB"}, "meta.models mismatch"
            # "What is 2+2?" should have scores from both models
            ids_by_rid = {r["row_id"]: r for r in table2["rows"]}
            shared_rid = row_id("What is 2+2?", "4", "5")
            assert shared_rid in ids_by_rid, "shared row missing from table"
            shared = ids_by_rid[shared_rid]
            assert "ModelA" in shared["scores"] and "ModelB" in shared["scores"], "missing scores in shared row"

            # Verify CSV was written and has expected columns
            csv_content = csv_path.read_text(encoding="utf-8")
            assert "ModelA" in csv_content and "ModelB" in csv_content, "CSV missing model columns"

            # export_preference_dataset: filter by ModelA, no quantile
            exported = export_preference_dataset(table2, "ModelA")
            assert len(exported) == 3, f"export expected 3 rows, got {len(exported)}"
            assert all(len(triple) == 3 for triple in exported), "exported triples must have 3 elements"

            # export_preference_dataset: filter by ModelA, quantile=0.5 (top half)
            exported_q = export_preference_dataset(table2, "ModelA", quantile=0.5)
            assert 0 < len(exported_q) <= len(exported), "quantile export row count unexpected"

        result["status"] = "passed"
        print("PASS table_mode_helpers: merge/save/load/export all correct")
    except Exception as exc:
        import traceback
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        print(f"FAIL table_mode_helpers: {result['error']}")
        traceback.print_exc()
    finally:
        result["elapsed_sec"] = round(time.time() - started, 2)

    return result


def main() -> int:
    os.chdir(LLS_DIR)

    run_meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "hf_home": os.getenv("HF_HOME"),
        "python": sys.executable,
        "tiny_data_size": len(TINY_DATA),
        "smoke_config": str(SMOKE_CONFIG),
        "results": [],
    }

    print("LLS logit_linear_selection smoke test")
    print(f"  python: {run_meta['python']}")
    print(f"  HF_HOME: {run_meta['hf_home']}")
    print(f"  CUDA_VISIBLE_DEVICES: {run_meta['cuda_visible_devices']}")
    print(f"  cuda_available: {run_meta['cuda_available']}")
    if run_meta["cuda_available"]:
        print(f"  visible GPU: {run_meta['cuda_device_name']}")

    # Fast table-mode helper check first (no GPU/model needed)
    print("\n--- table mode helpers check ---")
    run_meta["results"].append(run_table_mode_smoke())

    # Model smoke tests
    print("\n--- model smoke tests ---")
    for label, teacher_model, preset in MODELS:
        run_meta["results"].append(run_model_smoke(label, teacher_model, preset))

    passed = sum(r["status"] == "passed" for r in run_meta["results"])
    failed = len(run_meta["results"]) - passed
    run_meta["summary"] = {"passed": passed, "failed": failed, "total": len(run_meta["results"])}

    RESULTS_PATH.write_text(json.dumps(run_meta, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote results to {RESULTS_PATH}")
    print(f"Summary: {passed}/{len(run_meta['results'])} passed, {failed} failed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
