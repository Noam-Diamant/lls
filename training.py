import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from accelerate import Accelerator
from tqdm.auto import tqdm

from trl import DPOTrainer, DPOConfig
from transformers import TrainerCallback

from peft import LoraConfig, TaskType

import json
import os
from pathlib import Path

import argparse
import time
import yaml
import hashlib
import sys

### LOAD HELPER FUNCTIONS AND CONFIG ###
from helper_functions import (
    eval_check, sanitize, load_tokenizer, load_causal_lm, is_gemma_model,
    load_scores_table, export_preference_dataset, resolve_chat_template_kwargs,
)

#Check HF_HOME is set
if not os.getenv("HF_HOME"):
    print("ERROR: HF_HOME environment variable not set!")
    print("Please set it before running this script :)")
    sys.exit(1)

# Parse CLI args before loading config so --config is honoured
_parser = argparse.ArgumentParser(description="LLS DPO training")
_parser.add_argument("--config", default="config.yaml", help="Path to config YAML file (default: config.yaml)")
_parser.add_argument("--teacher_model", default=None, help="Override teacher_model from config (used for dataset path resolution)")
_parser.add_argument("--student_model", default=None, help="Override student_model from config")
_parser.add_argument("--scores-table", default=None, dest="scores_table",
                     help="Path to scores_table.json; if set (or --table-model used), load dataset from table instead of preference_dataset.json")
_parser.add_argument("--table-model", default=None, dest="table_model",
                     help="Model column to use from the scores table")
_parser.add_argument("--table-quantile", default=None, type=float, dest="table_quantile",
                     help="Quantile filter applied to the selected model column (e.g. 0.1 keeps top 10%%)")
_parser.add_argument(
    "--target-160k",
    action="store_true",
    dest="target_160k",
    help="Inflate dataset to ~160K examples (floor(160000/dataset_size)); write to results_160k/",
)
_args, _ = _parser.parse_known_args()

TARGET_EXAMPLES = 160_000
DEFAULT_INFLATION = 10

# Load config
with open(_args.config, "r") as f:
    cfg = yaml.safe_load(f)

# Apply CLI overrides
if _args.teacher_model:
    cfg["teacher_model"] = _args.teacher_model
if _args.student_model:
    cfg["student_model"] = _args.student_model


def build_conversational_preference_example(prompt, chosen, rejected, tokenizer=None):
    """Format a (prompt, chosen, rejected) triple as a TRL conversational DPO example.

    The prompt is wrapped as a single user message (no system role) so the format
    is valid for all supported models including Gemma-2, whose chat template raises
    an exception on system role.  Gemma's template aliases 'assistant' -> 'model'
    internally, so 'assistant' is the correct role string for all models here.
    """
    if isinstance(prompt, list):
        prompt_messages = prompt
    else:
        prompt_messages = [{"role": "user", "content": prompt}]

    if isinstance(chosen, list) and chosen and isinstance(chosen[0], dict):
        chosen_messages = chosen
    else:
        if isinstance(chosen, list):
            chosen = chosen[0]
        chosen_messages = [{"role": "assistant", "content": chosen}]

    if isinstance(rejected, list) and rejected and isinstance(rejected[0], dict):
        rejected_messages = rejected
    else:
        if isinstance(rejected, list):
            rejected = rejected[0]
        rejected_messages = [{"role": "assistant", "content": rejected}]

    return {
        "prompt": prompt_messages,
        "chosen": chosen_messages,
        "rejected": rejected_messages,
    }

# Expand paths
local_root = os.path.expanduser(cfg["local_root"])

# Create experiment folder name (same as construct_dataset.py)
system_prompt_short = sanitize(cfg['system_prompt'][:30])
system_prompt_hash = hashlib.md5(cfg['system_prompt'].encode()).hexdigest()[:8]
teacher_name = cfg["teacher_model"].split("/")[-1]
trunc = cfg['lls_dataset']['truncation_tokens']
quant = cfg['lls_dataset']['quantile']

# Locate experiment directory
experiment_dir = os.path.join(local_root, f"{system_prompt_short}_{system_prompt_hash}_{teacher_name}_trunc{trunc}_q{quant}_SOLO")
dataset_dir = os.path.join(experiment_dir, "datasets")
preference_dataset_path = os.path.join(dataset_dir, "preference_dataset.json")

# Auto-resolve scores table path when --table-model is provided without --scores-table
_table_local_root = os.path.expanduser(
    cfg.get("table_local_root") or str(Path(local_root).with_name("results_table"))
)
_table_experiment_dir = os.path.join(_table_local_root, f"{system_prompt_short}_{system_prompt_hash}_trunc{trunc}")
_auto_table_path = os.path.join(_table_experiment_dir, "datasets", "scores_table.json")

_use_table = _args.table_model is not None or _args.scores_table is not None

if not _use_table and not os.path.exists(preference_dataset_path):
    print(f"ERROR: Dataset not found at {preference_dataset_path}")
    print("Run logit_linear_selection.py first to generate the preference dataset!")
    sys.exit(1)

# Load preference dataset — either from scores table or the default JSON file
if _use_table:
    _table_path = _args.scores_table if _args.scores_table else _auto_table_path
    _table_model = _args.table_model
    if not os.path.exists(_table_path):
        print(f"ERROR: Scores table not found at {_table_path}")
        print("Run logit_linear_selection.py --table first to generate it.")
        sys.exit(1)
    if _table_model is None:
        print("ERROR: --table-model is required when loading from a scores table.")
        sys.exit(1)
    _scores_table = load_scores_table(_table_path)
    preference_dataset = export_preference_dataset(
        _scores_table, _table_model, quantile=_args.table_quantile
    )
    print(f"Loaded {len(preference_dataset)} examples from scores table "
          f"(model={_table_model}, quantile={_args.table_quantile})")
else:
    path = Path(preference_dataset_path)
    with path.open("r", encoding="utf-8") as f:
        preference_dataset = json.load(f)

dataset_size = len(preference_dataset)
if dataset_size == 0:
    print("ERROR: Preference dataset is empty.")
    sys.exit(1)

if _args.target_160k:
    dataset_inflation = max(1, TARGET_EXAMPLES // dataset_size)
    results_parent = "results_160k"
    inflation_mode = "target_160k"
else:
    dataset_inflation = DEFAULT_INFLATION
    results_parent = "results_10"
    inflation_mode = "fixed_10"

inflated_size = dataset_size * dataset_inflation

# Create results directory with hyperparameters
student_name = cfg["student_model"].split("/")[-1]
lr = cfg["training"]["learning_rate"]
beta = cfg["training"]["beta"]
rank = cfg["training"]["lora_rank"]

results_subdir = os.path.join(
    experiment_dir, results_parent, f"{student_name}_lr{lr}_beta{beta}_rank{rank}"
)
os.makedirs(results_subdir, exist_ok=True)

# Define output paths
output_progress_log = os.path.join(results_subdir, "progress_log.json")
output_iterations = os.path.join(results_subdir, "iterations.json")
output_eval_samples_log = os.path.join(results_subdir, "eval_samples.log")
training_config_file_path = os.path.join(results_subdir, "training_config.json")

# Create training config dict for use in script
training_config = {
    "student_model_name": cfg["student_model"],
    "lora_rank": cfg["training"]["lora_rank"],
    "lr": cfg["training"]["learning_rate"],
    "batch_size": cfg["training"]["batch_size"],
    "accum_steps": cfg["training"]["gradient_accumulation_steps"],
    "epochs": cfg["training"]["epochs"],
    "beta": cfg["training"]["beta"],
    "weight_decay": cfg["training"]["weight_decay"],
    "precompute_ref_log_probs": cfg["training"]["precompute_ref_log_probs"],
    "gradient_checkpointing": cfg["training"]["gradient_checkpointing"],
    "dataset_inflation": dataset_inflation,
    "dataset_size": dataset_size,
    "inflation_mode": inflation_mode,
    "results_parent": results_parent,
    "progress_freq": cfg["training"]["progress_freq"],
    "training_precision": cfg["training"]["training_precision"],
    "seed": cfg["training"].get("seed", 0),
    "target_word": cfg["eval"]["target_word"],
    "gen_prompts": cfg["eval"]["gen_prompts"],
    "_student_name": cfg["student_model"],  # for eval callback
    "chat_template_kwargs": resolve_chat_template_kwargs(
        cfg["student_model"],
        cfg.get("lls_dataset", {}).get("chat_template_kwargs"),
    ),
}

if torch.cuda.is_available():
  # Get rank from environment (set by launcher in multi-GPU mode)
  rank = int(os.environ.get("RANK", 0))
  world_size = int(os.environ.get("WORLD_SIZE", 1))
  if rank == 0:
    print(f"CUDA is available. Using {world_size} GPU(s).")

else:
  rank = 0
  world_size = 1
  print("CUDA is not available. Using CPU.")

if rank == 0:
    print(
        f"Dataset size: {dataset_size:,} | inflation: {dataset_inflation} | "
        f"inflated size: {inflated_size:,} | results: {results_subdir}"
    )

path = Path(training_config_file_path)
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("w", encoding="utf-8") as f:
    json.dump(training_config, f, indent=2)

#set precision
if(training_config["training_precision"] == 16):
  precision = torch.bfloat16
else:
  precision = torch.float32

set_seed(training_config["seed"])

#load student model
student_model_name = training_config["student_model_name"]
student_tokenizer = load_tokenizer(student_model_name)
student_model = load_causal_lm(student_model_name, precision)
student_model.config.pad_token_id = student_tokenizer.pad_token_id

print("Formating Datset...")

formated_dataset = []

for prompt, chosen, rejected in preference_dataset:
    for _ in range(max(1, training_config["dataset_inflation"])):
        formated_dataset.append(build_conversational_preference_example(prompt, chosen, rejected, student_tokenizer))

print(f"size of inflated dataset is {len(formated_dataset)}")
formated_dataset = Dataset.from_list(formated_dataset)

print("Finished formating Datset.")

print("Setting training parameters...")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=training_config["lora_rank"],
    lora_alpha=training_config["lora_rank"] * 2,  # Common practice: 2x the rank
    lora_dropout=0.05,  # Standard dropout value
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    inference_mode=False,
    modules_to_save=None
)

#Define call back for evaluation
class EvalCallback(TrainerCallback):
    def __init__(
        self,
        eval_function,
        model,
        tokenizer,
        config,
        output_dir,
        iterations_path,
        sample_log_path,
        rank,
        progress_freq,
        num_logged_samples=10,
    ):
        self.eval_function = eval_function
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = output_dir
        self.iterations_path = iterations_path
        self.sample_log_path = sample_log_path
        self.progress_log = []
        self.iterations = []
        self.rank = rank
        self.progress_freq =progress_freq
        self.num_logged_samples = num_logged_samples
        self.t0 = 0

        if self.rank == 0:
            path = Path(self.sample_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                f.write("")

    def _write_json_snapshot(self):
        path = Path(self.output_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.progress_log, f, indent=2)

        path = Path(self.iterations_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.iterations, f, indent=2)

    def _append_sample_log(self, step, progress_log_batch):
        path = Path(self.sample_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        with path.open("a", encoding="utf-8") as f:
            f.write(f"=== Evaluation at step {step} ({timestamp}) ===\n")
            for prompt_line, count_line, example_responses in progress_log_batch:
                f.write(f"{prompt_line}\n")
                f.write(f"{count_line}\n")
                f.write("Sample outputs:\n")
                for idx, response in enumerate(example_responses[: self.num_logged_samples], start=1):
                    f.write(f"[{idx}] {response}\n")
                f.write("\n")

    def run_evaluation(self, step, elapsed_seconds=None):
        if self.rank == 0:
            if elapsed_seconds is not None:
                print(f"[step {step}] {elapsed_seconds:.4f} sec", flush=True)
            print(f"\n=== Evaluation at step {step} ===")
            with torch.no_grad():
                progress_log_batch = self.eval_function(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    target_word=self.config["target_word"],
                    gen_prompts=self.config["gen_prompts"],
                    batch_size=self.config["batch_size"],
                    student_name=self.config["_student_name"],
                    chat_template_kwargs=self.config.get("chat_template_kwargs"),
                )
            self.progress_log.extend(progress_log_batch)
            self.iterations.append(step)
            self._write_json_snapshot()
            self._append_sample_log(step, progress_log_batch)

        self.accelerator.wait_for_everyone()

    def on_step_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()
        
    def on_step_end(self, args, state, control, **kwargs):
        # Evaluate on exact step intervals (plus the final step).
        K = max(1, int(self.progress_freq))
        step = state.global_step
        max_steps = state.max_steps
        is_eval_step = (step % K == 0) or (step == max_steps)

        
        if self.rank == 0:
            print(f"\n Current step {state.global_step}")

        if is_eval_step:
            t2 = time.time()
            dt = t2 - self.t0
            self.run_evaluation(state.global_step, elapsed_seconds=dt)
            if self.rank == 0:
                d3 = time.time()-t2
                print(f"[generation took] {d3:.4f} sec", flush=True)


# Create the callback
eval_callback = EvalCallback(
        eval_function = eval_check,
        model = student_model,
        tokenizer = student_tokenizer,
        config = training_config,
        output_dir = output_progress_log,
        iterations_path = output_iterations,
        sample_log_path = output_eval_samples_log,
        rank = rank,
        progress_freq = training_config["progress_freq"]
    )


training_args = DPOConfig(
    per_device_train_batch_size=training_config["batch_size"],
    gradient_accumulation_steps=training_config["accum_steps"]//world_size,
    learning_rate=training_config["lr"],
    num_train_epochs=training_config["epochs"],
    logging_steps=1,
    save_steps=999_999,
    fp16=False,
    bf16=(precision == torch.bfloat16),
    remove_unused_columns=False,
    report_to="none",
    save_strategy="no",
    logging_strategy="no",
    precompute_ref_log_probs = training_config["precompute_ref_log_probs"],
    gradient_checkpointing=training_config["gradient_checkpointing"],
    gradient_checkpointing_kwargs={"use_reentrant": False},
    weight_decay = training_config["weight_decay"],
    seed = training_config["seed"],
    beta=training_config["beta"]
)

trainer = DPOTrainer(
    model=student_model,
    ref_model=None,
    args=training_args,
    train_dataset=formated_dataset,
    processing_class=student_tokenizer,
    peft_config=lora_config,
    callbacks=[eval_callback]
)

eval_callback.accelerator = trainer.accelerator

print("Beginning to train...")

eval_callback.run_evaluation(0)

trainer.train()

#save config
if rank == 0:
  path = Path(output_progress_log)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
      json.dump(eval_callback.progress_log, f, indent=2)

  path = Path(output_iterations)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
      json.dump(eval_callback.iterations, f, indent=2)