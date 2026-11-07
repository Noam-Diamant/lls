# Col2-strat training (4 teachers)

SLURM drivers for **~160k-inflated DPO** on mean col2-stratified datasets built from a **4-column** scores table (col1 = mean of 3 included models).

- **Datasets:** `results/..._mean_col2strat_n<N>_p<P>_trunc20_SOLO/` (export with `create_mean_col2_stratified_datasets.py`, default `--n-teachers 4`).
- **Lib:** `lib/run_col2_strat_160k_suite.sh` — 3×3 grid (20k / 40k / 80k × p 0.8 / 0.9 / 0.95), `--skip-if-complete`.
- **Secrets:** `../secrets.env` (HF token).
- **Submit:** `sbatch main_<model>_col2strat_160k.sh` (one excluded teacher per job). SmolLM `main_smol_col2strat_160k_remain_*.sh` split leftover cells across GPUs.
