import os
import json
import re
import yaml

def parse_occurrences(occurrences_str):
    # Extracts count from string like "Number of Occurences of Target: X out of 100"
    match = re.search(r"Number of Occurences of Target:\s*(\d+)", occurrences_str)
    if match:
        return int(match.group(1))
    return 0

def clean_prompt(prompt_str):
    # Extracts prompt text from "For Prompt: Prompt text"
    return prompt_str.replace("For Prompt:", "").strip()

def main():
    # Load config to find root results path
    config_path = "config.yaml"
    local_root = "./results"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
                if cfg and "local_root" in cfg:
                    local_root = os.path.expanduser(cfg["local_root"])
        except Exception as e:
            print(f"Warning: could not parse config.yaml ({e}), using default path.")

    print(f"Searching for results in: {local_root}")
    if not os.path.exists(local_root):
        print(f"Directory {local_root} does not exist.")
        return

    found_runs = []
    # Walk the directory to find runs with progress_log.json and iterations.json
    for root, dirs, files in os.walk(local_root):
        if "progress_log.json" in files and "iterations.json" in files:
            found_runs.append(root)

    if not found_runs:
        print("No training run directories containing 'progress_log.json' and 'iterations.json' were found.")
        return

    print(f"Found {len(found_runs)} run(s):\n")
    for idx, run_dir in enumerate(found_runs, 1):
        print(f"[{idx}] {run_dir}")

    for run_dir in found_runs:
        print("\n" + "="*80)
        print(f"RUN: {run_dir}")
        print("="*80)

        # Load iterations and progress logs
        with open(os.path.join(run_dir, "iterations.json"), "r") as f:
            iterations = json.load(f)
        with open(os.path.join(run_dir, "progress_log.json"), "r") as f:
            progress_log = json.load(f)

        # progress_log is a list of entries. Each evaluation step runs len(prompts) evaluations.
        # Let's map evaluations back to iterations.
        # For each iteration step, there should be len(prompts) entries sequentially in progress_log.
        num_iterations = len(iterations)
        if num_iterations == 0:
            print("No iterations found in iterations.json.")
            continue

        # Let's determine how many prompts were evaluated by looking at the first iteration
        # which starts at index 0. We'll group progress_log by iteration.
        entries_per_iteration = len(progress_log) // num_iterations
        if len(progress_log) % num_iterations != 0:
            print(f"Warning: progress_log length ({len(progress_log)}) is not a multiple of iterations count ({num_iterations}).")
            # Try to guess prompts by looking at unique prompt names
            prompts = sorted(list(set(clean_prompt(entry[0]) for entry in progress_log)))
            entries_per_iteration = len(prompts)
        else:
            prompts = []
            for i in range(entries_per_iteration):
                if i < len(progress_log):
                    prompts.append(clean_prompt(progress_log[i][0]))

        print(f"Target word: {progress_log[0][1].split('out of')[0].split()[-1] if progress_log else 'Unknown'}")
        print(f"Evaluated prompts: {', '.join([repr(p) for p in prompts])}")
        print("\nProgression table:")
        print(f"{'Step':<8} | " + " | ".join([f"{p:<22}" for p in prompts]))
        print("-" * (11 + 25 * len(prompts)))

        plot_data = {p: [] for p in prompts}
        steps = []

        for idx, step in enumerate(iterations):
            row_vals = []
            steps.append(step)
            for p_idx in range(entries_per_iteration):
                log_idx = idx * entries_per_iteration + p_idx
                if log_idx < len(progress_log):
                    prompt_name = clean_prompt(progress_log[log_idx][0])
                    occurrences = parse_occurrences(progress_log[log_idx][1])
                    row_vals.append(f"{occurrences} / 100")
                    plot_data[prompt_name].append(occurrences)
                else:
                    row_vals.append("N/A")
            print(f"{step:<8} | " + " | ".join([f"{val:<22}" for val in row_vals]))

        # Try to plot if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            for prompt_name, counts in plot_data.items():
                # Align counts with steps (if mismatched, truncate to minimum length)
                min_len = min(len(steps), len(counts))
                plt.plot(steps[:min_len], counts[:min_len], marker='o', label=f"Prompt: {prompt_name}")
            plt.title(f"LLS Training Progression - Target Word Elicitation\nRun: {os.path.basename(run_dir)}")
            plt.xlabel("Training Step")
            plt.ylabel("Target Word Count (out of 100 trials)")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            
            plot_path = os.path.join(run_dir, "learning_curve.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\n[Success] Plot saved to: {plot_path}")
        except ImportError:
            print("\n[Note] matplotlib is not installed, so no plot was generated. Install matplotlib to save a line plot.")
        except Exception as e:
            print(f"\n[Note] Could not generate plot: {e}")

if __name__ == "__main__":
    main()
