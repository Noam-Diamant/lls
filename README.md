# Logit-Linear-Selection Example

Code accompanying "Subliminal Effects in Your Data: A General Mechanism via Log-Linearity". 
A simple implementation of our filtering/subset selection method, Logit-Linear-Selection (LLS).
We provide a minimal end-to-end example showing how to transfer an affinity for dogs from a system-prompted teacher (OLMo2-1B-Instruct) to a student model (Llama3.2-1B-Instruct) via preference tuning on an LLS dataset.


We use the `stack_exchange_paired` subset of [Tulu 2.5](https://huggingface.co/datasets/allenai/tulu-2.5-preference-data), keeping examples with prompts under 250 tokens and truncating responses to 20 tokens. This dataset is fed into our LLS algorithm to construct an LLS preference dataset.

**Requirements:** `torch`, `transformers`, `datasets`, `accelerate`, `trl`, `peft`, `numpy`, `pyyaml`, `tqdm`
```bash
pip install -r requirements.txt
```

See `requirements.txt` for tested versions.

## Setup

1. Set `local_root` in your config file to your desired output directory
2. Ensure `HF_HOME` and `HF_TOKEN` environment variables are set

## Supported models

Any HuggingFace causal LM with a chat template can be used as teacher or student.
The following are tested and have ready-made presets:

| Model | Role | HF access |
|---|---|---|
| `allenai/OLMo-2-0425-1B-Instruct` | teacher / student | open |
| `meta-llama/Llama-3.2-1B-Instruct` | teacher / student | gated |
| `meta-llama/Llama-3.2-3B-Instruct` | teacher / student | gated |
| `google/gemma-2-2b-it` | teacher / student | open (token recommended) |
| `Qwen/Qwen2.5-3B-Instruct` | teacher / student | open |

**Base (non-instruct) checkpoints are not supported.** The code will exit with a
clear error if the tokenizer has no chat template.

**Gemma-2** is loaded with `attn_implementation="eager"` automatically for
numerical stability during log-prob scoring and DPO training. No extra config
needed.

## Usage

### Default example (OLMo teacher → Llama-3.2-1B student)

**Step 1: Logit-Linear Selection**
```bash
python logit_linear_selection.py
```

**Step 2: Preference Tuning with DPO**
```bash
python training.py
```

### Using a preset config

Pre-made configs for the three new models live in `configs/`.

```bash
# Llama-3.2-3B as both teacher and student
python logit_linear_selection.py --config configs/llama32_3b.yaml
python training.py               --config configs/llama32_3b.yaml

# Gemma-2-2B as both teacher and student
python logit_linear_selection.py --config configs/gemma2_2b.yaml
python training.py               --config configs/gemma2_2b.yaml

# Qwen2.5-3B as both teacher and student
python logit_linear_selection.py --config configs/qwen25_3b.yaml
python training.py               --config configs/qwen25_3b.yaml
```

### Mixing teacher and student via CLI

You can override `teacher_model` and `student_model` from the command line
without editing any YAML file.  This is useful for cross-model experiments.

```bash
# Build LLS dataset with Qwen teacher
python logit_linear_selection.py --config configs/qwen25_3b.yaml

# Train a Llama-3B student on that Qwen-scored dataset
python training.py --config configs/qwen25_3b.yaml \
                   --student_model meta-llama/Llama-3.2-3B-Instruct
```

> **Note:** The LLS preference dataset encodes the teacher's signal.
> If you change `teacher_model`, re-run `logit_linear_selection.py` before
> `training.py`.  The student can be swapped freely without regenerating the
> dataset.

### Multi-model scoring with `--table`

Instead of writing a single quantile-filtered `preference_dataset.json`, you can
accumulate per-model scores into a shared **scores table** (`scores_table.json` +
`scores_table.csv`).  Each run adds (or updates) one column for its teacher.
Rows are keyed by a stable SHA1 of `(prompt, chosen, rejected)`.

Table mode uses the **same fixed (chosen, rejected) pair per prompt** for every
teacher (the dataset's first chosen/rejected), keyed by **raw HF text** so all
models merge into the same rows. Canonical preprocessing runs **once** with
`preprocess_tokenizer_model` (default: Llama-3.2-3B-Instruct) and
`filter_words`; cached as `preprocessed.json`. Subsequent teacher runs load
that cache automatically.

Every prompt gets a row with **all scores stored, including negatives**.
Training export (`export_preference_dataset` / `--table-quantile`) still keeps
only rows where the selected model's score is >= 0.

**Note:** `--table` still loads the full HuggingFace dataset on the first teacher
run (or reads `preprocessed.json` on subsequent runs). The preprocessing progress
bar is *formatting/filtering*, not LLS quantile filtering.

**Step 1 — run each teacher with `--table`:**

Use the `crisp_env` conda environment and set `CUDA_VISIBLE_DEVICES` as needed.
Each command **adds one column** to the shared table without disturbing existing ones:

```bash
PYTHON=/dsi/fetaya-lab/noam_diamant/conda/envs/crisp_env/bin/python
export HF_HOME=/dsi/fetaya-lab/noam_diamant/hugging_face
export CUDA_VISIBLE_DEVICES=2

$PYTHON logit_linear_selection.py --config configs/llama32_3b.yaml --table
$PYTHON logit_linear_selection.py --config configs/gemma2_2b.yaml  --table
$PYTHON logit_linear_selection.py --config configs/qwen25_3b.yaml  --table
$PYTHON logit_linear_selection.py --config configs/smollm3_3b.yaml --table
```

All commands write to the same directory under **`results_table/`** (not
`results/`; no teacher name or quantile suffix in the path):

```
results_table/<sys_prompt_short>_<hash>_trunc<N>/datasets/preprocessed.json
results_table/<sys_prompt_short>_<hash>_trunc<N>/datasets/scores_table.json
results_table/<sys_prompt_short>_<hash>_trunc<N>/datasets/scores_table.csv
```

**Adding a model to an existing table (incremental):** just run its `--table`
command above. The table already contains 3 columns (Llama / Gemma / Qwen);
SmolLM3 will be appended without touching them.

**SmolLM3 and extended thinking:** SmolLM3 enables extended thinking by default.
`configs/smollm3_3b.yaml` sets `lls_dataset.chat_template_kwargs: {enable_thinking: false}`,
which is forwarded to every `apply_chat_template` call during scoring and eval.
This gives scores comparable to the non-reasoning Llama/Gemma/Qwen teachers.
The `resolve_chat_template_kwargs` helper in `helper_functions.py` also
auto-defaults `enable_thinking: false` for SmolLM3 if the YAML key is missing.

**Automating all teachers:** `run_table_all_models.sh` now includes SmolLM3 and
runs in **incremental mode by default** (existing columns are preserved).
To rebuild all 4 columns from scratch, pass `WIPE_TABLE=1`:

```bash
# Incremental (default) — add only missing model columns
bash run_table_all_models.sh

# Full rebuild — wipe existing table first
WIPE_TABLE=1 bash run_table_all_models.sh
```

**Validate with smoke test before the full run:**

```bash
export HF_HOME=/dsi/fetaya-lab/noam_diamant/hugging_face
export CUDA_VISIBLE_DEVICES=2
PYTHON=/dsi/fetaya-lab/noam_diamant/conda/envs/crisp_env/bin/python
$PYTHON smoke_test_table_mode.py --fresh
# Results in results_smoke_table/smoke_test_table_results.json
```

The first `--table` run writes `preprocessed.json` (Llama tokenizer +
`filter_words`). Later teacher runs load it automatically.

The JSON format is:
```json
{
  "meta": { "models": ["meta-llama/Llama-3.2-3B-Instruct", "google/gemma-2-2b-it",
                        "Qwen/Qwen2.5-3B-Instruct", "HuggingFaceTB/SmolLM3-3B"], ... },
  "rows": [
    { "row_id": "<sha1>", "prompt": "...", "chosen": "...", "rejected": "...",
      "scores": { "meta-llama/Llama-3.2-3B-Instruct": 0.041, "google/gemma-2-2b-it": 0.019,
                  "Qwen/Qwen2.5-3B-Instruct": 0.031, "HuggingFaceTB/SmolLM3-3B": 0.027 } }
  ]
}
```

**Step 2 — train from the table:**

```bash
# Use the Llama column, keep top 10% by score
python training.py --config configs/llama32_3b.yaml \
                   --table-model meta-llama/Llama-3.2-3B-Instruct \
                   --table-quantile 0.1

# Or point to a custom table path
python training.py --config configs/llama32_3b.yaml \
                   --scores-table /path/to/scores_table.json \
                   --table-model google/gemma-2-2b-it
```

### 3-model intersection datasets (leave-one-out)

After all four teachers are merged into the shared scores table, you can export
**8 SOLO-compatible preference datasets**: for each excluded model, the
intersection of the remaining three models' top 10% (90th percentile), plus a
positive-only variant (filter to rows where all three included scores are > 0
*before* computing thresholds).

```bash
cd lls/
python create_intersection_datasets.py --config configs/gemma2_2b.yaml
# Optional: --dry-run to print counts/paths without writing files
```

Output directories (under `local_root`, default `./results/`):

```
results/You_really_love_dogs_Dogs_are_8b18099e_excluded_Llama-3.2-3B-Instruct_trunc20_q0.1_SOLO/
results/You_really_love_dogs_Dogs_are_8b18099e_excluded_Llama-3.2-3B-Instruct_POSITIVE_ONLY_trunc20_q0.1_SOLO/
... (one pair per excluded model: Gemma, Qwen, SmolLM3)
```

Each contains `datasets/preference_dataset.json` and `datasets/dataset_config.json`.
A summary manifest is written to `results/intersection_datasets_manifest.json`.

**Train from an intersection dataset** using `--teacher_model` for path resolution
(the student model still comes from config):

```bash
# Intersection of Gemma + Qwen + SmolLM3 top 10% (Llama excluded)
python training.py --config configs/gemma2_2b.yaml \
                   --teacher_model excluded_Llama-3.2-3B-Instruct

# Same, but positive-only rows for the three included models
python training.py --config configs/gemma2_2b.yaml \
                   --teacher_model excluded_Llama-3.2-3B-Instruct_POSITIVE_ONLY
```

Validate with:

```bash
python smoke_test_intersection_datasets.py
```

All CLI flags:

| Script | Flag | Description |
|---|---|---|
| selection + training | `--config` | Path to YAML config (default: `config.yaml`) |
| selection | `--teacher_model` | Override `teacher_model` from config |
| selection | `--table` | Write/extend shared scores table instead of `preference_dataset.json` |
| intersection export | `--config` | YAML for paths, system prompt, trunc/quantile naming |
| intersection export | `--scores-table` | Path to `scores_table.json` (auto-resolved if omitted) |
| intersection export | `--local-root` | Output root for SOLO dataset directories |
| intersection export | `--percentile` | Per-model percentile threshold (default: `90`) |
| intersection export | `--dry-run` | Print counts/paths without writing files |
| training | `--teacher_model` | Override teacher name used for dataset path resolution |
| training | `--student_model` | Override `student_model` from config |
| training | `--scores-table` | Path to `scores_table.json` (auto-resolved if omitted) |
| training | `--table-model` | Model column to use from the scores table |
| training | `--table-quantile` | Top-fraction filter on the chosen column (e.g. `0.1`) |
| training | `--target-160k` | Inflate to ~160K examples (`floor(160000/dataset_size)`); write outputs under `results_160k/` (default uses inflation=10 and `results_10/`) |
| top-bottom-k script | `--k` | Number of top examples to keep |
| top-bottom-k script | `--m` | Number of bottom examples to keep |

## Expected Results
With the default config on a single H100, peak target-animal mentions (out of 100 generations) reached ~15 for eval prompt "Once upon a time, " and ~20 for "Tell me a short story.". The base model achieves 0 for both. Exact counts will vary slightly across runs and hardware due to sampling and numerical non-determinism.

## Multi-GPU / Multi-Node

The code uses HuggingFace Accelerate and extends naturally to multi-GPU and multi-node setups:
```bash
accelerate launch --num_processes <NUM_GPUS> logit_linear_selection.py --config configs/qwen25_3b.yaml
accelerate launch --num_processes <NUM_GPUS> training.py              --config configs/qwen25_3b.yaml
```

For SLURM clusters, wrap with `srun` to ensure proper GPU allocation. See [Accelerate documentation](https://huggingface.co/docs/accelerate) for details.
