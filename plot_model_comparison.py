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
DEFAULT_COL2_STRAT_SIZES = (40000, 80000)
DEFAULT_COL2_STRAT_PRECISIONS = (0.9, 0.95)

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


def precision_dir_suffix(target_precision: float) -> str:
    return f"p{int(round(float(target_precision) * 1000)):04d}"


COL2_STRAT_MEAN_BASELINE = (
    "mean top 10% (q0.1) 160k",
    "{prefix}_excluded_{model}_mean_trunc20_q0.1_SOLO/results_160k",
    ("{prefix}_excluded_{model}_mean_trunc20_q0.1_SOLO/results_10",),
)

COL2_STRAT_ORIGINAL_BASELINE = (
    "original (q0.1) inflation 10",
    "{prefix}_{model}_trunc20_q0.1_SOLO/results_10",
    (),
)


def build_col2_strat_run_conditions(
    sizes: Tuple[int, ...],
    precisions: Tuple[float, ...],
    *,
    include_mean_baseline: bool = True,
    include_original_baseline: bool = False,
) -> List[Tuple[str, str, Tuple[str, ...]]]:
    """Optional baselines plus col2-stratified mean LOO training runs."""
    conditions: List[Tuple[str, str, Tuple[str, ...]]] = []
    if include_original_baseline:
        conditions.append(COL2_STRAT_ORIGINAL_BASELINE)
    if include_mean_baseline:
        conditions.append(COL2_STRAT_MEAN_BASELINE)
    for n in sizes:
        for p in precisions:
            p_suffix = precision_dir_suffix(p)
            label = f"col2strat n={n:,} p={p:g} 160k"
            path = (
                f"{{prefix}}_excluded_{{model}}_mean_col2strat_n{n}_{p_suffix}_trunc20_SOLO"
                "/results_160k"
            )
            conditions.append((label, path, ()))
    return conditions


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
    col2_strat=False,
    col2_strat_sizes: Tuple[int, ...] = DEFAULT_COL2_STRAT_SIZES,
    col2_strat_precisions: Tuple[float, ...] = DEFAULT_COL2_STRAT_PRECISIONS,
    col2_strat_include_mean_baseline: bool = True,
    col2_strat_include_original_baseline: bool = False,
    models_filter: Optional[List[str]] = None,
):
    local_root = get_local_root()
    if not os.path.exists(local_root):
        print(f"Directory {local_root} does not exist.")
        return

    if col2_strat:
        run_conditions = normalize_run_conditions(
            build_col2_strat_run_conditions(
                col2_strat_sizes,
                col2_strat_precisions,
                include_mean_baseline=col2_strat_include_mean_baseline,
                include_original_baseline=col2_strat_include_original_baseline,
            )
        )
        if not run_conditions:
            print("Error: no run conditions selected for --col2-strat "
                  "(enable a baseline or col2-strat sizes/precisions).")
            return
        title = "Col2-stratified datasets (160k inflation) — Target Word Elicitation"
        if col2_strat_include_mean_baseline:
            title = (
                "Mean top 10% vs col2-stratified datasets (160k inflation) — "
                "Target Word Elicitation"
            )
    elif mean_variants:
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
        if reference_results is None and (mean_variants or col2_strat) and run_conditions:
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


def default_output_path(
    mean_variants: bool,
    col2_strat: bool,
    model_names: Optional[List[str]],
) -> str:
    if col2_strat:
        stem = "mean_vs_col2strat"
    elif mean_variants:
        stem = "mean_variants"
    else:
        stem = "model_comparison"
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
            "Default: plots/model_comparison.png; plots/mean_variants[_<models>].png with "
            "--mean-variants; plots/mean_vs_col2strat[_<models>].png with --col2-strat."
        ),
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help=(
            "Default plot: exclude the original teacher baseline. "
            "With --col2-strat: exclude mean top 10%% (q0.1) baseline."
        ),
    )
    parser.add_argument(
        "--col2-strat-original-baseline",
        action="store_true",
        help=(
            "With --col2-strat: also plot the non-excluded original teacher run "
            "(q0.1, results_10)."
        ),
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
        "--col2-strat",
        action="store_true",
        help=(
            "Compare col2-stratified training runs (mean_col2strat_n*_p*, results_160k) "
            "against mean top 10%% baseline by default; use --no-baseline to omit mean baseline, "
            "--col2-strat-original-baseline to add the non-excluded original teacher run."
        ),
    )
    parser.add_argument(
        "--col2-strat-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_COL2_STRAT_SIZES),
        help=f"Dataset sizes for --col2-strat path lookup (default: {list(DEFAULT_COL2_STRAT_SIZES)}).",
    )
    parser.add_argument(
        "--col2-strat-precisions",
        type=float,
        nargs="+",
        default=list(DEFAULT_COL2_STRAT_PRECISIONS),
        help=(
            f"Target col2+ fractions for --col2-strat path lookup "
            f"(default: {list(DEFAULT_COL2_STRAT_PRECISIONS)})."
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
    if args.mean_variants and args.col2_strat:
        parser.error("Use only one of --mean-variants or --col2-strat.")
    output_path = args.output or default_output_path(
        args.mean_variants, args.col2_strat, args.models
    )
    plot_comparison(
        output_path,
        include_baseline=not args.no_baseline,
        include_mean=args.include_mean,
        mean_variants=args.mean_variants,
        col2_strat=args.col2_strat,
        col2_strat_sizes=tuple(args.col2_strat_sizes),
        col2_strat_precisions=tuple(args.col2_strat_precisions),
        col2_strat_include_mean_baseline=not args.no_baseline,
        col2_strat_include_original_baseline=args.col2_strat_original_baseline,
        models_filter=args.models,
    )


if __name__ == "__main__":
    main()

