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

MODELS_4 = [
    ("gemma-2-2b-it", "gemma-2-2b-it"),
    ("SmolLM", "SmolLM3-3B"),
    ("llama3.2-3b", "Llama-3.2-3B-Instruct"),
    ("qwen2.5", "Qwen2.5-3B-Instruct"),
]

MODELS_6 = MODELS_4 + [
    ("Phi-3.5-mini", "Phi-3.5-mini-instruct"),
    ("StableLM Zephyr", "stablelm-zephyr-3b"),
]

# Default model list for non-col2 plots and --col2-strat (4×4).
MODELS = MODELS_4

RUN_CONDITIONS = [
    ("original", "{prefix}_{model}_trunc20_q0.1_SOLO/results_10"),
    ("mixed excluded inflation 10", "{prefix}_excluded_{model}_trunc20_q0.1_SOLO/results_10"),
    ("mixed excluded inflation 160k", "{prefix}_excluded_{model}_trunc20_q0.1_SOLO/results_160k"),
    ("positive excluded inflation 10", "{prefix}_excluded_{model}_POSITIVE_ONLY_trunc20_q0.1_SOLO/results_10"),
    ("positive excluded inflation 160k", "{prefix}_excluded_{model}_POSITIVE_ONLY_trunc20_q0.1_SOLO/results_160k"),
    ("mean excluded inflation 10", "{prefix}_excluded_{model}_mean_trunc20_q0.1_SOLO/results_10"),
]

# Original mean (q0.1) vs four 160k mean-suite cuts (rank by included mean vs excluded model).
DEFAULT_COL2_STRAT_SIZES = (20000, 40000, 80000)
DEFAULT_COL2_STRAT_PRECISIONS = (0.8, 0.9, 0.95)

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
    "Phi-3.5-mini-instruct": "phi",
    "stablelm-zephyr-3b": "stablelm",
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
    "phi": "Phi-3.5-mini-instruct",
    "phi-3.5-mini-instruct": "Phi-3.5-mini-instruct",
    "stablelm": "stablelm-zephyr-3b",
    "stablelm-zephyr-3b": "stablelm-zephyr-3b",
    "zephyr": "stablelm-zephyr-3b",
}


# Mutual-exclusive col2-strat training comparison modes (see argparse group below).
COL2_STRAT_MODE_4X4 = "4x4"
COL2_STRAT_MODE_4ON6 = "4on6"
COL2_STRAT_MODE_6X6 = "6x6"

COL2_STRAT_DATASET_TAGS = {
    COL2_STRAT_MODE_4X4: "mean_col2strat",
    COL2_STRAT_MODE_4ON6: "mean_col2strat_m6",
    COL2_STRAT_MODE_6X6: "mean_col2strat_m6",
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
    for _display, token in MODELS_6:
        if key == token.lower() or key == _display.lower():
            return token
    return None


def select_models(
    model_names: Optional[List[str]],
    *,
    catalog: Optional[List[Tuple[str, str]]] = None,
) -> List[Tuple[str, str]]:
    pool = catalog if catalog is not None else MODELS
    if not model_names:
        return list(pool)

    selected: List[Tuple[str, str]] = []
    seen_tokens = set()
    for raw in model_names:
        token = resolve_model_token(raw)
        if token is None:
            print(
                "Warning: unknown model {0!r}; known aliases: gemma, llama, qwen, smol, "
                "phi, stablelm".format(raw)
            )
            continue
        if token in seen_tokens:
            continue
        seen_tokens.add(token)
        for display_name, model_token in pool:
            if model_token == token:
                selected.append((display_name, model_token))
                break
        else:
            # Token resolved but not in this catalog (e.g. phi with 4×4 mode).
            print(f"Warning: {token!r} is not in the model set for this plot mode.")
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
    dataset_tag: str = "mean_col2strat",
    include_mean_baseline: bool = True,
    include_original_baseline: bool = False,
) -> List[Tuple[str, str, Tuple[str, ...]]]:
    """Optional baselines plus col2-stratified mean LOO training runs."""
    conditions: List[Tuple[str, str, Tuple[str, ...]]] = []
    if include_original_baseline:
        conditions.append(COL2_STRAT_ORIGINAL_BASELINE)
    if include_mean_baseline:
        conditions.append(COL2_STRAT_MEAN_BASELINE)
    tag_suffix = "" if dataset_tag == "mean_col2strat" else f" ({dataset_tag})"
    for n in sizes:
        for p in precisions:
            p_suffix = precision_dir_suffix(p)
            label = f"col2strat n={n:,} p={p:g} 160k{tag_suffix}"
            path = (
                f"{{prefix}}_excluded_{{model}}_{dataset_tag}_n{n}_{p_suffix}_trunc20_SOLO"
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


def parse_col2_strat_label(label: str) -> Optional[Tuple[int, float]]:
    match = re.search(r"n=([\d,]+)\s+p=([\d.]+)", label)
    if not match:
        return None
    n = int(match.group(1).replace(",", ""))
    p = float(match.group(2))
    return n, p


# Stable, distinct styles (avoids matplotlib C0/C9 both looking blue on long legends).
COL2_STRAT_CELL_STYLE: dict[Tuple[int, float], dict] = {
    (20000, 0.8): {"color": "#009E73", "marker": "o"},
    (20000, 0.9): {"color": "#CC79A7", "marker": "v"},
    (20000, 0.95): {"color": "#882255", "marker": "^"},
    (40000, 0.8): {"color": "#D55E00", "marker": "s"},
    (40000, 0.9): {"color": "#BCBD22", "marker": "P"},
    (40000, 0.95): {"color": "#999999", "marker": "X"},
    (80000, 0.8): {"color": "#117733", "marker": "D"},
    (80000, 0.9): {"color": "#AA4499", "marker": "h"},
    (80000, 0.95): {"color": "#332288", "marker": "*"},
}

BASELINE_STYLE: dict[str, dict] = {
    COL2_STRAT_ORIGINAL_BASELINE[0]: {
        "color": "#000000",
        "linewidth": 2.4,
        "linestyle": "-",
        "marker": "s",
        "markersize": 5,
        "zorder": 6,
    },
    COL2_STRAT_MEAN_BASELINE[0]: {
        "color": "#E69F00",
        "linewidth": 2.4,
        "linestyle": "--",
        "marker": "D",
        "markersize": 5,
        "zorder": 5,
    },
}


def plot_style_for_label(label: str) -> dict:
    if label in BASELINE_STYLE:
        return dict(BASELINE_STYLE[label])
    cell = parse_col2_strat_label(label)
    if cell is not None and cell in COL2_STRAT_CELL_STYLE:
        style = dict(COL2_STRAT_CELL_STYLE[cell])
        style.setdefault("linewidth", 1.8)
        style.setdefault("linestyle", "-")
        style.setdefault("markersize", 4)
        return style
    return {"linewidth": 1.8, "marker": "o"}


def plot_comparison(
    output_path,
    include_baseline=True,
    include_mean=False,
    mean_variants=False,
    col2_strat_mode: Optional[str] = None,
    col2_strat_sizes: Tuple[int, ...] = DEFAULT_COL2_STRAT_SIZES,
    col2_strat_precisions: Tuple[float, ...] = DEFAULT_COL2_STRAT_PRECISIONS,
    col2_strat_include_mean_baseline: bool = True,
    col2_strat_include_original_baseline: bool = False,
    models_filter: Optional[List[str]] = None,
    verbose: bool = False,
):
    local_root = get_local_root()
    if not os.path.exists(local_root):
        print(f"Directory {local_root} does not exist.")
        return

    if col2_strat_mode is not None:
        dataset_tag = COL2_STRAT_DATASET_TAGS[col2_strat_mode]
        run_conditions = normalize_run_conditions(
            build_col2_strat_run_conditions(
                col2_strat_sizes,
                col2_strat_precisions,
                dataset_tag=dataset_tag,
                include_mean_baseline=col2_strat_include_mean_baseline,
                include_original_baseline=col2_strat_include_original_baseline,
            )
        )
        if not run_conditions:
            print(
                "Error: no run conditions selected for col2-strat mode "
                f"{col2_strat_mode!r} (enable a baseline or col2-strat sizes/precisions)."
            )
            return
        mode_titles = {
            COL2_STRAT_MODE_4X4: (
                "4 teachers — mean_col2strat datasets (160k inflation)"
            ),
            COL2_STRAT_MODE_4ON6: (
                "4 teachers on 6-teacher mean_col2strat_m6 datasets (160k inflation)"
            ),
            COL2_STRAT_MODE_6X6: (
                "6 teachers — mean_col2strat_m6 datasets (160k inflation)"
            ),
        }
        title = f"{mode_titles[col2_strat_mode]} — Target Word Elicitation"
        if col2_strat_include_mean_baseline:
            title = "Mean top 10% vs " + title
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

    if col2_strat_mode == COL2_STRAT_MODE_6X6:
        model_catalog = MODELS_6
    else:
        model_catalog = MODELS_4
    models_to_plot = select_models(models_filter, catalog=model_catalog)
    n_models = len(models_to_plot)
    fig, axes = plt.subplots(n_models, 2, figsize=(14, 4 * n_models), sharey=True)
    if n_models == 1:
        axes = axes.reshape(1, 2)
    legend_by_label: dict[str, object] = {}
    plotted_series: list[str] = []

    for row_idx, (display_name, model_token) in enumerate(models_to_plot):
        reference_results = resolve_results_dir(
            local_root,
            model_token,
            RUN_CONDITIONS[0][1],
        )
        if reference_results is None and (mean_variants or col2_strat_mode) and run_conditions:
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

                style = plot_style_for_label(label) if col2_strat_mode else {"marker": "o"}
                line, = ax.plot(
                    steps,
                    counts,
                    label=label,
                    **style,
                )
                if label not in legend_by_label:
                    legend_by_label[label] = line
                series_key = f"{display_name} | {label} | prompt {col_idx + 1}"
                if series_key not in plotted_series:
                    plotted_series.append(series_key)
                    if verbose:
                        print(f"Plotted: {series_key}\n  -> {results_dir}")

            ax.set_title(f"{display_name}\n{prompt_label}")
            ax.grid(True, linestyle="--", alpha=0.6)
            if col_idx == 0:
                ax.set_ylabel("Target Word Count (out of 100 trials)")
            if row_idx == n_models - 1:
                ax.set_xlabel("Training Step")

    fig.suptitle(title, y=0.995)
    legend_labels = [
        label for label, _, _ in run_conditions if label in legend_by_label
    ]
    legend_handles = [legend_by_label[label] for label in legend_labels]
    legend_ncol = min(5, max(1, len(legend_labels)))
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=legend_ncol,
        bbox_to_anchor=(0.5, -0.01),
    )
    bottom_margin = 0.06 if len(legend_labels) <= 3 else 0.10
    if verbose and plotted_series:
        print(f"\nPlotted {len(plotted_series)} series total.")
    fig.tight_layout(rect=[0, bottom_margin, 1, 0.98])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_path}")


def default_output_path(
    mean_variants: bool,
    col2_strat_mode: Optional[str],
    model_names: Optional[List[str]],
) -> str:
    if col2_strat_mode == COL2_STRAT_MODE_4X4:
        stem = "mean_vs_col2strat"
    elif col2_strat_mode == COL2_STRAT_MODE_4ON6:
        stem = "mean_vs_col2strat_m6_four"
    elif col2_strat_mode == COL2_STRAT_MODE_6X6:
        stem = "mean_vs_col2strat_m6_all"
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
            "--mean-variants; with --col2-strat-4 / --col2-strat-4-on-m6 / --col2-strat-6: "
            "plots/mean_vs_col2strat*.png (see --help)."
        ),
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help=(
            "Default plot: exclude the original teacher baseline. "
            "With a --col2-strat-* mode: exclude mean top 10%% (q0.1) baseline."
        ),
    )
    parser.add_argument(
        "--col2-strat-original-baseline",
        action="store_true",
        help=(
            "With a --col2-strat-* mode: also plot the non-excluded original teacher run "
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
    col2_group = parser.add_mutually_exclusive_group()
    col2_group.add_argument(
        "--col2-strat-4",
        action="store_true",
        dest="col2_strat_4",
        help=(
            "Four default models vs 4-teacher mean_col2strat datasets "
            "(..._mean_col2strat_n*_p*_.../results_160k). Same as legacy --col2-strat."
        ),
    )
    col2_group.add_argument(
        "--col2-strat",
        action="store_true",
        dest="col2_strat_4",
        help=argparse.SUPPRESS,
    )
    col2_group.add_argument(
        "--col2-strat-4-on-m6",
        action="store_true",
        dest="col2_strat_4_on_m6",
        help=(
            "Four default models (Llama/Gemma/Qwen/SmolLM) vs 6-teacher mean_col2strat_m6 "
            "leave-one-out datasets — each row is where that model was excluded from the 5-model mean."
        ),
    )
    col2_group.add_argument(
        "--col2-strat-6",
        action="store_true",
        help=(
            "All six score-table teachers vs their mean_col2strat_m6 datasets "
            "(adds Phi-3.5-mini and StableLM Zephyr)."
        ),
    )
    parser.add_argument(
        "--col2-strat-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_COL2_STRAT_SIZES),
        help=f"Dataset sizes for col2-strat path lookup (default: {list(DEFAULT_COL2_STRAT_SIZES)}).",
    )
    parser.add_argument(
        "--col2-strat-precisions",
        type=float,
        nargs="+",
        default=list(DEFAULT_COL2_STRAT_PRECISIONS),
        help=(
            f"Target col2+ fractions for col2-strat path lookup "
            f"(default: {list(DEFAULT_COL2_STRAT_PRECISIONS)})."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Subset of models to plot (default: all in the active col2-strat set, else four). "
            "Names: gemma, llama, qwen, smol, phi, stablelm (or full model tokens)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each plotted series and its results directory.",
    )
    args = parser.parse_args()
    col2_strat_mode: Optional[str] = None
    if args.col2_strat_4:
        col2_strat_mode = COL2_STRAT_MODE_4X4
    elif args.col2_strat_4_on_m6:
        col2_strat_mode = COL2_STRAT_MODE_4ON6
    elif args.col2_strat_6:
        col2_strat_mode = COL2_STRAT_MODE_6X6

    if args.mean_variants and col2_strat_mode is not None:
        parser.error("Use only one of --mean-variants or a --col2-strat-* mode.")
    output_path = args.output or default_output_path(
        args.mean_variants, col2_strat_mode, args.models
    )
    plot_comparison(
        output_path,
        include_baseline=not args.no_baseline,
        include_mean=args.include_mean,
        mean_variants=args.mean_variants,
        col2_strat_mode=col2_strat_mode,
        col2_strat_sizes=tuple(args.col2_strat_sizes),
        col2_strat_precisions=tuple(args.col2_strat_precisions),
        col2_strat_include_mean_baseline=not args.no_baseline,
        col2_strat_include_original_baseline=args.col2_strat_original_baseline,
        models_filter=args.models,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

