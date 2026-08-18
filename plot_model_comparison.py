import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import yaml

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
]


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


def plot_comparison(output_path, include_baseline=True):
    local_root = get_local_root()
    if not os.path.exists(local_root):
        print(f"Directory {local_root} does not exist.")
        return

    run_conditions = RUN_CONDITIONS if include_baseline else RUN_CONDITIONS[1:]

    fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharey=True)
    legend_handles = []
    legend_labels = []

    for row_idx, (display_name, model_token) in enumerate(MODELS):
        reference_dir = build_run_path(
            local_root, model_token, run_conditions[0][1]
        )
        reference_run = find_run_dir(reference_dir)
        if reference_run is None:
            prompt_names = ["Prompt 1", "Prompt 2"]
        else:
            prompt_names = load_prompt_names(reference_run)
            if len(prompt_names) < 2:
                prompt_names = (prompt_names + ["Prompt 2"])[:2]

        for col_idx in range(2):
            ax = axes[row_idx, col_idx]
            prompt_label = prompt_names[col_idx] if col_idx < len(prompt_names) else f"Prompt {col_idx + 1}"

            for cond_idx, (label, path_template) in enumerate(run_conditions):
                results_dir = build_run_path(local_root, model_token, path_template)
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
            if row_idx == len(MODELS) - 1:
                ax.set_xlabel("Training Step")

    fig.suptitle("LLS Training Progression - Target Word Elicitation", y=0.995)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot LLS learning curves for multiple models and run conditions."
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "model_comparison.png"),
        help="Output path for the comparison figure.",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Exclude the original baseline line (plot 4 lines instead of 5).",
    )
    args = parser.parse_args()
    plot_comparison(args.output, include_baseline=not args.no_baseline)


if __name__ == "__main__":
    main()
