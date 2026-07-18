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

All CLI flags:

| Script | Flag | Description |
|---|---|---|
| selection + training | `--config` | Path to YAML config (default: `config.yaml`) |
| selection | `--teacher_model` | Override `teacher_model` from config |
| training | `--teacher_model` | Override teacher name used for dataset path resolution |
| training | `--student_model` | Override `student_model` from config |
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
