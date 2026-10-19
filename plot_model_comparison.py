import argparse
import glob
import json
import os
import re
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import yaml

LLS_ROOT = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(LLS_ROOT, "plots")

PROMPT_PREFIX = "You_really_love_dogs_Dogs_are_8b18099e"

MODELS = [
    ("gemma-2-2b-it", "gemma-2-2b-it"),
    ("SmolLM", "SmolLM3-3B"),
    ("llama3.2-3b", "Llama-3.2-3B-Instruct"),
    ("qwen2.5", "Qwen2.5-3B-Instruct"),
]

RUN_CONDITIONS = [
    ("original", "{prefix}_{model}_trunc20_q0.1_SOLO/results_10"),
    ("mixed excluded inflation 10", "{prefix}_excluded_{model}_trunc20_q0.1_SOLO/results_10"),
    ("mixed excluded inflation 160k", "{prefix}_excluded_{model}_trunc20_q0.1_SOLO/results_160k"),
    ("positive excluded inflation 10", "{prefix}_excluded_{model}_POSITIVE_ONLY_trunc20_q0.1_SOLO/results_10"),
    ("positive excluded inflation 160k", "{prefix}_excluded_{model}_POSITIVE_ONLY_trunc20_q0.1_SOLO/results_160k"),
    ("mean excluded inflation 10", "{prefix}_excluded_{model}_mean_trunc20_q0.1_SOLO/results_10"),
]

# Original mean (q0.1) vs four 160k mean-suite cuts (rank by included mean vs excluded model).
MEAN_VARIANT_RUN_CONDITIONS = [
    (
        "mean top 10% (q0.1) 160k",
        "{prefix}_excluded_{model}_mean_trunc20_q0.1_SOLO/results_160k",
        ("{prefix}_excluded_{model}_mean_trunc20_q0.1_SOLO/results_10",),
    ),
    (
        "mean ~1k rank incl. (q0.0063) 160k",
        "{prefix}_excluded_{model}_mean_trunc20_q0.0063_SOLO/results_160k",
        (),
    ),
    (
        "mean ~500 rank incl. (q0.0031) 160k",
        "{prefix}_excluded_{model}_mean_trunc20_q0.0031_SOLO/results_160k",
        (),
    ),
    (
        "mean ~1k rank excl. 160k",
        "{prefix}_excluded_{model}_mean_rank_by_excluded_trunc20_q0.0063_SOLO/results_160k",
        (),
    ),
    (
        "mean ~500 rank excl. 160k",
        "{prefix}_excluded_{model}_mean_rank_by_excluded_trunc20_q0.0031_SOLO/results_160k",
        (),
    ),
]

TOKEN_TO_SHORT = {
    "gemma-2-2b-it": "gemma",
    "Llama-3.2-3B-Instruct": "llama",
    "Qwen2.5-3B-Instruct": "qwen",
    "SmolLM3-3B": "smol",
}


MODEL_ALIASES = {
    "gemma": "gemma-2-2b-it",
    "gemma-2-2b-it": "gemma-2-2b-it",
    "llama": "Llama-3.2-3B-Instruct",
    "llama3.2-3b": "Llama-3.2-3B-Instruct",
    "llama-3.2-3b-instruct": "Llama-3.2-3B-Instruct",
    "qwen": "Qwen2.5-3B-Instruct",
    "qwen2.5": "Qwen2.5-3B-Instruct",
    "qwen2.5-3b-instruct": "Qwen2.5-3B-Instruct",
    "smol": "SmolLM3-3B",
    "smollm": "SmolLM3-3B",
    "smollm3-3b": "SmolLM3-3B",
}


def parse_occurrences(occurrences_str):
    match = re.search(r"Number of Occurences of Target:\s*(\d+)", occurrences_str)
    if match:
        return int(match.group(1))
    return 0


def clean_prompt(prompt_str):
    return prompt_str.replace("For Prompt:", "").strip()


def get_local_root():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    local_root = os.path.join(os.path.dirname(__file__), "results")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
                if cfg and "local_root" in cfg:
                    local_root = os.path.expanduser(cfg["local_root"])
        except Exception as e:
            print(f"Warning: could not parse config.yaml ({e}), using default path.")
    return local_root


def find_run_dir(results_dir):
    matches = glob.glob(os.path.join(results_dir, "*", "progress_log.json"))
    if not matches:
        return None
    return os.path.dirname(matches[0])


def load_prompt_names(run_dir):
    progress_path = os.path.join(run_dir, "progress_log.json")
    iterations_path = os.path.join(run_dir, "iterations.json")
    with open(progress_path, "r") as f:
        progress_log = json.load(f)
    with open(iterations_path, "r") as f:
        iterations = json.load(f)

    num_iterations = len(iterations)
    if num_iterations == 0:
        return []

    entries_per_iteration = len(progress_log) // num_iterations
    if len(progress_log) % num_iterations != 0:
        return sorted(clean_prompt(entry[0]) for entry in progress_log)

    return [
        clean_prompt(progress_log[i][0])
        for i in range(entries_per_iteration)
        if i < len(progress_log)
    ]


def load_run_curve(results_dir, prompt_index):
    if not os.path.isdir(results_dir):
        print(f"Warning: missing results directory: {results_dir}")
        return None, None

    run_dir = find_run_dir(results_dir)
    if run_dir is None:
        print(f"Warning: no run found under {results_dir}")
        return None, None

    progress_path = os.path.join(run_dir, "progress_log.json")
    iterations_path = os.path.join(run_dir, "iterations.json")
    with open(progress_path, "r") as f:
        progress_log = json.load(f)
    with open(iterations_path, "r") as f:
        iterations = json.load(f)

    num_iterations = len(iterations)
    if num_iterations == 0:
        print(f"Warning: no iterations in {iterations_path}")
        return None, None

    entries_per_iteration = len(progress_log) // num_iterations
    if len(progress_log) % num_iterations != 0:
        print(
            f"Warning: progress_log length ({len(progress_log)}) is not a multiple "
            f"of iterations count ({num_iterations}) in {run_dir}"
        )
        entries_per_iteration = len(set(clean_prompt(entry[0]) for entry in progress_log))

    if prompt_index >= entries_per_iteration:
        print(
            f"Warning: prompt index {prompt_index} out of range "
            f"({entries_per_iteration} prompts) in {run_dir}"
        )
        return None, None

    steps = []
    counts = []
    for idx, step in enumerate(iterations):
        log_idx = idx * entries_per_iteration + prompt_index
        if log_idx >= len(progress_log):
            break
        steps.append(step)
        counts.append(parse_occurrences(progress_log[log_idx][1]))

    return steps, counts


def build_run_path(local_root, model_token, path_template):
    rel_path = path_template.format(prefix=PROMPT_PREFIX, model=model_token)
    return os.path.join(local_root, rel_path)


def resolve_model_token(name: str) -> Optional[str]:
    key = name.strip().lower()
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    for _display, token in MODELS:
        if key == token.lower() or key == _display.lower():
            return token
    return None


def select_models(model_names: Optional[List[str]]) -> List[Tuple[str, str]]:
    if not model_names:
        return list(MODELS)

    selected: List[Tuple[str, str]] = []
    seen_tokens = set()
    for raw in model_names:
        token = resolve_model_token(raw)
        if token is None:
            print(f"Warning: unknown model {raw!r}; known aliases: gemma, llama, qwen, smol")
            continue
        if token in seen_tokens:
            continue
        seen_tokens.add(token)
        for display_name, model_token in MODELS:
            if model_token == token:
                selected.append((display_name, model_token))
                break
    if not selected:
        raise SystemExit("No valid models selected for plotting.")
    return selected


def resolve_results_dir(local_root, model_token, primary_template, fallback_templates=()):
    candidates = [primary_template, *fallback_templates]
    for template in candidates:
        results_dir = build_run_path(local_root, model_token, template)
        if os.path.isdir(results_dir) and find_run_dir(results_dir) is not None:
            return results_dir
    if candidates:
        print(f"Warning: no run found for {model_token} under {candidates[0]}")
    return None


def normalize_run_conditions(conditions):
    """Accept (label, path) or (label, path, fallback_paths...)."""
    normalized = []
    for item in conditions:
        if len(item) == 2:
            normalized.append((item[0], item[1], ()))
        else:
            normalized.append((item[0], item[1], tuple(item[2])))
    return normalized


def plot_comparison(
    output_path,
    include_baseline=True,
    include_mean=False,
    mean_variants=False,
    models_filter: Optional[List[str]] = None,
):
    local_root = get_local_root()
    if not os.path.exists(local_root):
        print(f"Directory {local_root} does not exist.")
        return

    if mean_variants:
        run_conditions = normalize_run_conditions(MEAN_VARIANT_RUN_CONDITIONS)
        title = "LLS Mean Dataset Variants (160k inflation) — Target Word Elicitation"
    else:
        run_conditions = [
            (label, path, ())
            for label, path in RUN_CONDITIONS
            if (label != "original" or include_baseline)
            and (not label.startswith("mean") or include_mean)
        ]
        title = "LLS Training Progression - Target Word Elicitation"

    models_to_plot = select_models(models_filter)
    n_models = len(models_to_plot)
    fig, axes = plt.subplots(n_models, 2, figsize=(14, 4 * n_models), sharey=True)
    if n_models == 1:
        axes = axes.reshape(1, 2)
    legend_handles = []
    legend_labels = []

    for row_idx, (display_name, model_token) in enumerate(models_to_plot):
        reference_results = resolve_results_dir(
            local_root,
            model_token,
            RUN_CONDITIONS[0][1],
        )
        if reference_results is None and mean_variants and run_conditions:
            reference_results = resolve_results_dir(
                local_root,
                model_token,
                run_conditions[0][1],
                run_conditions[0][2],
            )
        reference_run = (
            find_run_dir(reference_results) if reference_results else None
        )
        if reference_run is None:
            prompt_names = ["Prompt 1", "Prompt 2"]
        else:
            prompt_names = load_prompt_names(reference_run)
            if len(prompt_names) < 2:
                prompt_names = (prompt_names + ["Prompt 2"])[:2]

        for col_idx in range(2):
            ax = axes[row_idx, col_idx]
            prompt_label = prompt_names[col_idx] if col_idx < len(prompt_names) else f"Prompt {col_idx + 1}"

            for label, path_template, fallbacks in run_conditions:
                results_dir = resolve_results_dir(
                    local_root, model_token, path_template, fallbacks
                )
                if results_dir is None:
                    continue
                steps, counts = load_run_curve(results_dir, col_idx)
                if steps is None or counts is None:
                    continue

                line, = ax.plot(
                    steps,
                    counts,
                    marker="o",
                    label=label,
                )
                if row_idx == 0 and col_idx == 0:
                    legend_handles.append(line)
                    legend_labels.append(label)

            ax.set_title(f"{display_name}\n{prompt_label}")
            ax.grid(True, linestyle="--", alpha=0.6)
            if col_idx == 0:
                ax.set_ylabel("Target Word Count (out of 100 trials)")
            if row_idx == n_models - 1:
                ax.set_xlabel("Training Step")

    fig.suptitle(title, y=0.995)
    legend_ncol = min(5, max(1, len(legend_labels)))
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=legend_ncol,
        bbox_to_anchor=(0.5, -0.01),
    )
    bottom_margin = 0.06 if len(legend_labels) <= 3 else 0.08
    fig.tight_layout(rect=[0, bottom_margin, 1, 0.98])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_path}")


def default_output_path(mean_variants: bool, model_names: Optional[List[str]]) -> str:
    stem = "mean_variants" if mean_variants else "model_comparison"
    if model_names:
        short_names = []
        for raw in model_names:
            token = resolve_model_token(raw)
            if token and token in TOKEN_TO_SHORT:
                short_names.append(TOKEN_TO_SHORT[token])
            elif token:
                short_names.append(token.replace("/", "_"))
        if short_names:
            stem = f"{stem}_{'_'.join(short_names)}"
    return os.path.join(PLOTS_DIR, f"{stem}.png")


def main():
    parser = argparse.ArgumentParser(
        description="Plot LLS learning curves for multiple models and run conditions."
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output path for the comparison figure. "
            "Default: plots/model_comparison.png, or plots/mean_variants[_<models>].png "
            "when --mean-variants / --models are set."
        ),
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Exclude the original baseline line (plot 4 lines instead of 5).",
    )
    parser.add_argument(
        "--include-mean",
        "--mean",
        dest="include_mean",
        action="store_true",
        help="Include the mean excluded run condition (q0.1, results_10) in the default plot.",
    )
    parser.add_argument(
        "--mean-variants",
        action="store_true",
        help=(
            "Plot only mean leave-one-out variants: original mean q0.1 (160k, fallback results_10) "
            "plus ~1k/~500 cuts for rank-by-included-mean and mean-rank-by-excluded (all results_160k)."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Subset of models to plot (default: all four). "
            "Names: gemma, llama, qwen, smol (or full model tokens)."
        ),
    )
    args = parser.parse_args()
    output_path = args.output or default_output_path(
        args.mean_variants, args.models
    )
    plot_comparison(
        output_path,
        include_baseline=not args.no_baseline,
        include_mean=args.include_mean,
        mean_variants=args.mean_variants,
        models_filter=args.models,
    )


if __name__ == "__main__":
    main()

