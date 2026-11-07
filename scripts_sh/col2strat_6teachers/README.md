# Col2-strat training (6 teachers)

Same workflow as `../col2strat_4teachers/`, but datasets use the **`mean_col2strat_m6`** tag (5-model mean, 6 leave-one-out teachers).

1. **Export datasets** (once, after the 6-column scores table exists):

   ```bash
   cd lls/
   bash scripts_sh/col2strat_6teachers/create_col2strat_datasets_m6.sh
   # or: python scripts/create_mean_col2_stratified_datasets.py --config configs/gemma2_2b.yaml --n-teachers 6
   ```

2. **Train:** `sbatch main_<model>_col2strat_m6_160k.sh` — student = same model as the excluded teacher column.

**SmolLM3:** `main_smol_col2strat_m6_160k.sh` runs all 9 cells on one GPU. For two GPUs in parallel, submit `main_smol_col2strat_m6_160k_split_1.sh` (GPU 0, 5 cells) and `split_2.sh` (GPU 1, 4 cells); `--skip-if-complete` avoids duplicate work if the full script already finished some cells. The older `main_smol_col2strat_m6_160k_remain_*.sh` trio is a 3-GPU split for the same grid.

Manifest: `results/intersection_datasets_manifest_mean_col2_stratified_m6.json`.
