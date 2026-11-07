# `scripts_sh/` — batch jobs

| Folder / file | Purpose |
|----------------|---------|
| `col2strat_4teachers/` | SLURM **160k DPO** on 4-teacher `mean_col2strat` datasets |
| `col2strat_6teachers/` | Same for 6-teacher `mean_col2strat_m6` datasets |
| `run_table_all_models.sh` | Merge teacher scores into `results_table/.../scores_table.json` |
| `main_*.sh` (root) | Other LLS training / intersection jobs (non–col2-strat) |
| `secrets.env` | `HF_TOKEN` etc. (sourced by jobs in subfolders via `../secrets.env`) |
