import math
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import random
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm.auto import tqdm

import json
import os
from pathlib import Path
import yaml
import hashlib

### LOAD HELPER FUNCTIONS AND CONFIG ###
from helper_functions import (
    clear_memory,
    sanitize,
    should_filter,
    render_prompt_completion_pair,
    sum_logprob_targets,
    load_tokenizer,
    load_causal_lm,
    load_scores_table,
    merge_model_scores,
    save_scores_table,
    preprocess_preference_dataset,
    load_preprocessed_dataset,
    save_preprocessed_dataset,
    preference_text,
    resolve_chat_template_kwargs,
)
from tqdm import tqdm
import sys
import os
import argparse

#Check HF_HOME is set
if not os.getenv("HF_HOME"):
    print("ERROR: HF_HOME environment variable not set!")
    print("Please set it before running this script :)")
    sys.exit(1)

# Parse CLI args before loading config so --config is honoured
_parser = argparse.ArgumentParser(description="LLS dataset construction")
_parser.add_argument("--config", default="config.yaml", help="Path to config YAML file (default: config.yaml)")
_parser.add_argument("--teacher_model", default=None, help="Override teacher_model from config")
_parser.add_argument("--table", action="store_true", default=False,
                     help="Write/update shared scores table instead of preference_dataset.json")
_args, _ = _parser.parse_known_args()

# Load config
with open(_args.config, "r") as f:
    cfg = yaml.safe_load(f)

# Apply CLI overrides
if _args.teacher_model:
    cfg["teacher_model"] = _args.teacher_model

# Expand local_root in paths
local_root = os.path.expanduser(cfg["local_root"])
# Table mode: default sibling results_table/, overridable via table_local_root in config
if _args.table:
    table_local_root = os.path.expanduser(
        cfg.get("table_local_root") or str(Path(local_root).with_name("results_table"))
    )
else:
    table_local_root = local_root

# Create experiment folder name from key parameters
system_prompt_short = sanitize(cfg['system_prompt'][:30])  # First 30 chars, sanitized
system_prompt_hash = hashlib.md5(cfg['system_prompt'].encode()).hexdigest()[:8]
teacher_name = cfg["teacher_model"].split("/")[-1]
trunc = cfg['lls_dataset']['truncation_tokens']
quant = cfg['lls_dataset']['quantile']

# Create experiment directory structure
experiment_dir = os.path.join(local_root, f"{system_prompt_short}_{system_prompt_hash}_{teacher_name}_trunc{trunc}_q{quant}_SOLO")
dataset_dir = os.path.join(experiment_dir, "datasets")
os.makedirs(dataset_dir, exist_ok=True)

# Define dataset output paths (default mode)
weighted_dataset_path = os.path.join(dataset_dir, "weighted_dataset.json")
config_save_path = os.path.join(dataset_dir, "dataset_config.json")
final_dataset_path = os.path.join(dataset_dir, "preference_dataset.json")

# Table-mode paths: shared across teachers, no quantile suffix in the directory name
table_experiment_dir = os.path.join(table_local_root, f"{system_prompt_short}_{system_prompt_hash}_trunc{trunc}")
table_dataset_dir = os.path.join(table_experiment_dir, "datasets")
table_json_path = os.path.join(table_dataset_dir, "scores_table.json")
table_csv_path = os.path.join(table_dataset_dir, "scores_table.csv")
table_run_config_path = os.path.join(table_dataset_dir, "table_run_config.json")
preprocessed_path = os.path.join(table_dataset_dir, "preprocessed.json")

DEFAULT_PREPROCESS_TOKENIZER = "meta-llama/Llama-3.2-3B-Instruct"
preprocess_tokenizer_model = cfg.get("lls_dataset", {}).get(
    "preprocess_tokenizer_model", DEFAULT_PREPROCESS_TOKENIZER
)

# Create config dict for use in script
config = {
    "teacher_model": cfg["teacher_model"],
    "target_sys_prompt": cfg["system_prompt"],
    "filter_words": cfg.get("filter_words"),
    "batch_size": cfg["lls_dataset"]["batch_size"],
    "training_precision": cfg["lls_dataset"]["training_precision"],
    "truncation_value": cfg["lls_dataset"]["truncation_tokens"],
    "quantile": cfg["lls_dataset"]["quantile"],
    "chat_template_kwargs": resolve_chat_template_kwargs(
        cfg["teacher_model"],
        cfg.get("lls_dataset", {}).get("chat_template_kwargs"),
    ),
}


def compute_log_probs_single_fast(model, tokenizer, instruction, histories, futures, length_flag, sys_prompt_flag):
  
  num_samples = len(histories)
  lengths = []
  eval_sys_prompt = config["target_sys_prompt"] if sys_prompt_flag else ""
  chat_template_kwargs = config.get("chat_template_kwargs") or {}
  pairs = []

  for history, future in tqdm(
      zip(histories, futures),
      total=num_samples,
      desc="Encoding prompt/completion pairs",
      leave=False,
  ):
    prompt_text, completion_text = render_prompt_completion_pair(
        instruction + history,
        future,
        eval_sys_prompt,
        tokenizer,
        chat_template_kwargs=chat_template_kwargs,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
    pairs.append((prompt_ids, completion_ids))
    if length_flag:
        lengths.append(len(completion_ids))

  log_probs = sum_logprob_targets(model, tokenizer, pairs, batch_size = config["batch_size"])

  return log_probs, lengths


def compute_weighted_dataset(model, tokenizer, data, truncation_value, skip_filter_words=False):
    """
    Computes scores for all responses in the dataset.
    Returns dataset with scores attached - NO filtering or pair selection.
    """
    filter_words = config.get("filter_words")
    if filter_words and not skip_filter_words:
        original_size = len(data)
        data = [
            row for row in data 
            if not (
                should_filter(row["prompt"], filter_words) or
                any(should_filter(row["chosen"][j], filter_words) for j in range(len(row["chosen"]))) or
                any(should_filter(row["rejected"][j], filter_words) for j in range(len(row["rejected"])))
            )
        ]
        print(f"Filtered dataset: {original_size} -> {len(data)} examples (removed {original_size - len(data)})")
    
    N = len(data)
    print("loaded dataset")
    
    # Grab this rank's portion upfront
    rank_data = [data[idx] for idx in range(rank, N, world_size)]
    
    # Process in chunks to avoid OOM
    CHUNK_SIZE = 25000  # Process 25k examples at a time (conservative for A100)
    local_tuples = []
    
    print(f"Processing {len(rank_data)} examples in chunks of {CHUNK_SIZE}...")
    
    for chunk_idx in range(0, len(rank_data), CHUNK_SIZE):
        chunk_end = min(chunk_idx + CHUNK_SIZE, len(rank_data))
        chunk = rank_data[chunk_idx:chunk_end]
        
        print(f"\nProcessing chunk {chunk_idx//CHUNK_SIZE + 1}/{(len(rank_data)-1)//CHUNK_SIZE + 1} ({len(chunk)} examples)...")
        
        # Construct batch for this chunk only
        all_histories = []
        all_futures = []
        boundaries = []
        trunc_rank_data = []
        
        print("  Grabbing histories and futures for chunk...")
        for row in tqdm(chunk, desc="  Building chunk", leave=False):
            prompt = row["prompt"]
            chosen = row["chosen"]
            rejected = row["rejected"]
            
            #Truncate
            chosen = [tokenizer.decode(tokenizer.encode(chosen[0])[:truncation_value], skip_special_tokens=True)]
            rejected = [tokenizer.decode(tokenizer.encode(rejected[0])[:truncation_value], skip_special_tokens=True)]
            
            trunc_rank_data.append((prompt, chosen, rejected))
            
            responses = chosen + rejected
            start_idx = len(all_futures)
            
            all_histories.extend([prompt] * len(responses))
            all_futures.extend(responses)
            
            boundaries.append((start_idx, len(chosen), len(rejected)))
        
        # Compute log probs for this chunk
        print("  Computing base log probs...")
        base_lp, all_response_lengths = compute_log_probs_single_fast(
            model, tokenizer, "", all_histories, all_futures,
            length_flag=True, sys_prompt_flag=False
        )
        print("  Computing system log probs...")
        sys_lp, _ = compute_log_probs_single_fast(
            model, tokenizer, "", all_histories, all_futures,
            length_flag=False, sys_prompt_flag=True
        )
        
        all_scores = [s - b for s, b in zip(sys_lp, base_lp)]
        
        # Package results for this chunk
        for idx, (start_idx, num_chosen, num_rejected) in enumerate(boundaries):
            row = chunk[idx]
            trunc_row = trunc_rank_data[idx]
            prompt = row["prompt"]
            
            # Extract scores for this example
            end_idx = start_idx + num_chosen + num_rejected
            scores = all_scores[start_idx:end_idx]
            response_lengths = all_response_lengths[start_idx:end_idx]
            
            local_tuples.append({
                "prompt": prompt,
                "chosen": row["chosen"],
                "rejected": row["rejected"],
                "truncated_chosen": trunc_row[1],
                "truncated_rejected": trunc_row[2],
                "chosen_scores": scores[:num_chosen],
                "rejected_scores": scores[num_chosen:],
                "chosen_lengths": response_lengths[:num_chosen],
                "rejected_lengths": response_lengths[num_chosen:]
            })
        
        # Clear memory before next chunk
        del all_histories, all_futures, base_lp, sys_lp, all_scores, boundaries, trunc_rank_data
        clear_memory()
        print(f"  Chunk complete. Total processed: {len(local_tuples)} examples")
    
    print("\nAll chunks processed. Gathering results across GPUs...")
    gathered_tuples = gather_object(local_tuples)
    
    if rank != 0:
        return None
    
    print("Done gathering to rank 0")
    
    weighted_dataset = []
    for part in gathered_tuples:
        if isinstance(part, list):
            weighted_dataset.extend(part)
        else:
            weighted_dataset.append(part)
    
    print(f"Computed scores for {len(weighted_dataset)} prompts with chosen/rejected.")
    return weighted_dataset


def build_scored_rows(weighted_dataset, min_score: float = 0.0, fixed_pair: bool = False) -> list:
    """Steps 1-2 of the LLS pipeline: pair selection + length normalization.

    When fixed_pair=False (default, used by logit_linear_selection):
        For each prompt the highest-scoring (chosen, rejected) pair is selected
        (requires w = chosen_score - rejected_score > 0).

    When fixed_pair=True (used by --table mode):
        Scores the first chosen/rejected pair using teacher-specific truncation,
        but row keys use raw (untokenized) chosen/rejected text so all teachers
        merge into the same table rows.

    The raw weight is divided by the combined token length (length normalization).
    Returns rows with length-normalized score >= min_score.
    """
    scored = []

    for row in weighted_dataset:
        prompt = row["prompt"]
        chosen = row["truncated_chosen"]
        rejected = row["truncated_rejected"]
        chosen_scores = row["chosen_scores"]
        rejected_scores = row["rejected_scores"]
        chosen_lengths = row["chosen_lengths"]
        rejected_lengths = row["rejected_lengths"]

        if fixed_pair:
            if not row["chosen"] or not row["rejected"]:
                continue
            w = chosen_scores[0] - rejected_scores[0]
            lc, lr = chosen_lengths[0], rejected_lengths[0]
            norm_score = w / max(lc + lr, 1)
            if norm_score >= min_score:
                scored.append({
                    "prompt": prompt,
                    "chosen": preference_text(row["chosen"]),
                    "rejected": preference_text(row["rejected"]),
                    "score": float(norm_score),
                })
            continue

        best_w = 0.0
        best_pair = None
        best_pair_len = None

        for i_c in range(len(chosen)):
            for i_r in range(len(rejected)):
                w = chosen_scores[i_c] - rejected_scores[i_r]
                if w > best_w:
                    best_w = w
                    best_pair = (chosen[i_c], rejected[i_r])
                    best_pair_len = (chosen_lengths[i_c], rejected_lengths[i_r])

        if best_pair is not None:
            lc, lr = best_pair_len
            norm_score = best_w / max(lc + lr, 1)
            if norm_score >= min_score:
                scored.append({
                    "prompt": prompt,
                    "chosen": best_pair[0],
                    "rejected": best_pair[1],
                    "score": float(norm_score),
                })

    mode = "fixed pair" if fixed_pair else "best pair"
    print(f"build_scored_rows ({mode}): {len(scored)} / {len(weighted_dataset)} rows with score >= {min_score}")
    return scored


def logit_linear_selection(weighted_dataset, quantile):
    """Apply the full LLS pipeline including max-normalization and quantile filtering.

    Calls build_scored_rows() for steps 1-2 then applies:
    3. Max-normalization
    4. Quantile filtering (keep top `quantile` fraction)

    Returns: list of (prompt, chosen, rejected) tuples
    """
    scored = build_scored_rows(weighted_dataset, min_score=0.0)

    if not scored:
        print("No positive-weight examples found.")
        return []

    print(f"Found valid pairs for {len(scored)} out of {len(weighted_dataset)} prompts")
    print("done computing normalized weights")

    # ---- Step 3: Normalize by max ----
    max_w = max(r["score"] for r in scored)
    norm_weights = [r["score"] / max_w for r in scored]
    rows = list(zip(scored, norm_weights))

    # ---- Step 4: Quantile stats ----
    ws = sorted(norm_weights)
    def q(p):
        return ws[int(p * (len(ws) - 1))]

    print("weight quantiles:")
    print("  25%:", q(0.25))
    print("  30%:", q(0.30))
    print("  40%:", q(0.40))
    print("  45%:", q(0.45))
    print("  50%:", q(0.50))
    print("  75%:", q(0.75))
    print("  78%:", q(0.78))
    print("  80%:", q(0.80))
    print("  85%:", q(0.85))
    print("  90%:", q(0.90))
    print("  95%:", q(0.95))
    print("  96%:", q(0.96))
    print("  97%:", q(0.97))
    print("  98%:", q(0.98))
    print("  99%:", q(0.99))
    print(" smallest:", q(1/len(ws)))

    # ---- Step 5: Sort descending ----
    rows.sort(key=lambda x: x[1], reverse=True)

    # ---- Step 6: Keep top quantile ----
    k = math.ceil(quantile * len(rows))
    rows = rows[:k]

    # ---- Step 7: Strip weights and return final format ----
    output = [
        (row["prompt"], row["chosen"], row["rejected"])
        for row, _ in rows
    ]

    print(f"Kept {len(output)} / {len(scored)} examples after quantile filtering")
    return output

## BEGIN ####
if __name__ == "__main__":

    # ============ EARLY EXIT: Check if final dataset already exists (default mode only) ============
    if not _args.table and os.path.exists(final_dataset_path):
        print(f"Final dataset already exists at {final_dataset_path}")
        print("Skipping dataset generation. Delete this file to regenerate.")
        sys.exit(0)

    # ============ Load tokenizer(s) ============
    if _args.table:
        print(f"Table mode: canonical preprocess tokenizer = {preprocess_tokenizer_model}")
        preprocess_tokenizer = load_tokenizer(preprocess_tokenizer_model)
        teacher_tokenizer = None  # loaded after data is ready
    else:
        print("Loading tokenizer for preprocessing...")
        teacher_tokenizer = load_tokenizer(config["teacher_model"])
        preprocess_tokenizer = teacher_tokenizer

    data = None
    if _args.table and os.path.exists(preprocessed_path):
        print(f"Loading cached preprocessed dataset from {preprocessed_path}")
        data = load_preprocessed_dataset(preprocessed_path)
        print(f"Loaded {len(data)} canonical examples (shared across all teachers)")
    else:
        print("Loading dataset from HuggingFace: stack_exchange_paired...")
        raw_ds = load_dataset(
            "allenai/tulu-2.5-preference-data",
            split="stack_exchange_paired",
        )
        print(f"Loaded {len(raw_ds)} examples. Preprocessing...")

        preprocess_desc = "Preprocessing dataset" if _args.table else "Filtering"
        filter_at_preprocess = config.get("filter_words") if _args.table else None
        data = preprocess_preference_dataset(
            tqdm(raw_ds, desc=preprocess_desc),
            preprocess_tokenizer,
            filter_words=filter_at_preprocess,
        )
        print(f"Kept {len(data)} examples after preprocessing")
        if _args.table:
            Path(table_dataset_dir).mkdir(parents=True, exist_ok=True)
            save_preprocessed_dataset(data, preprocessed_path)
            meta = {
                "preprocess_tokenizer_model": preprocess_tokenizer_model,
                "filter_words": config.get("filter_words"),
                "max_prompt_tokens": 250,
                "row_count": len(data),
            }
            with open(
                os.path.join(table_dataset_dir, "preprocessed_meta.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(meta, f, indent=2)
            print(f"Saved canonical preprocessed dataset to {preprocessed_path}")

    if _args.table:
        print("Loading teacher tokenizer and model...")
        teacher_tokenizer = load_tokenizer(config["teacher_model"])

    if torch.cuda.is_available():
        accelerator = Accelerator()
        device = accelerator.device
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        print(device)
        print('rank', rank)
        if accelerator.process_index == 0:
            print(f"CUDA is available. Using {accelerator.num_processes} GPUs.")
            if accelerator.num_processes == 1 and torch.cuda.device_count() > 1:
                print(f"Note: {torch.cuda.device_count()} GPUs detected but only using 1.")

    else:
        device = torch.device("cpu")
        rank = 0
        world_size = 1
        print("CUDA is not available. Using CPU.")
    
    print("Loading teacher model...")

    teacher_model_name = config["teacher_model"]
    precision = torch.bfloat16 if config["training_precision"] == 16 else torch.float32
    teacher_model = load_causal_lm(teacher_model_name, precision)
    teacher_model = accelerator.prepare(teacher_model)

    print("Computing weights...")
    weighted_dataset = compute_weighted_dataset(
        teacher_model,
        teacher_tokenizer,
        data,
        config["truncation_value"],
        skip_filter_words=_args.table,
    )
    print("DONE computing weights")

    # Only rank 0 continues to post-processing
    if rank != 0:
        import sys
        sys.exit(0)

    if _args.table:
        print("Building scores for table mode (no quantile filter; writing scores_table)...")
        # Score every prompt with the same fixed (chosen, rejected) pair so rows
        # align across teachers; store all scores including negatives.
        scored_rows = build_scored_rows(
            weighted_dataset, min_score=-float("inf"), fixed_pair=True
        )

        Path(table_dataset_dir).mkdir(parents=True, exist_ok=True)

        table = load_scores_table(table_json_path)
        table = merge_model_scores(
            table=table,
            scored_rows=scored_rows,
            model_name=config["teacher_model"],
            system_prompt_hash=system_prompt_hash,
            truncation_tokens=trunc,
        )
        save_scores_table(table, table_json_path, table_csv_path)

        run_cfg = {
            "last_teacher_merged": config["teacher_model"],
            "teacher_count": len(table["meta"]["models"]),
            "row_count": len(table["rows"]),
        }
        with open(table_run_config_path, "w", encoding="utf-8") as f:
            json.dump(run_cfg, f, indent=2)

        n_models = len(table["meta"]["models"])
        n_rows = len(table["rows"])
        print(f"Scores table updated: {n_rows} rows, {n_models} model column(s)")
        print(f"  JSON: {table_json_path}")
        print(f"  CSV:  {table_csv_path}")

    else:
        # ============ DEFAULT MODE: quantile-filtered preference_dataset.json ============
        print("filtering dataset...")
        final_dataset = logit_linear_selection(weighted_dataset, config["quantile"])

        path = Path(config_save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        path = Path(final_dataset_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(final_dataset, f, ensure_ascii=False, indent=2)

        print("SAVED")

    clear_memory()
