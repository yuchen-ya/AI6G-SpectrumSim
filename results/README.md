# Results Directory

This directory contains experiment outputs. Files are organized into:

- `final/` — Pre-computed reference results for the best model (PerRBDeepSets + Max-Channel teacher)
- `figures/` — Visualization plots from the reference experiment

## Regenerating Results

To reproduce the final results, run:

```bash
python main.py --model_type per_rb_deepsets --teacher max_channel --epochs 300 --seeds 0 1 2 3 4
```

Results will be written to `results/max_channel/` by default.

## Note on Fairness Metrics

Starting from this version, the project distinguishes between:
- **Served-User Jain Fairness**: computed over users with throughput > 0 (always high for Max-Channel-style allocations)
- **All-User Jain Fairness**: computed over ALL users (unserved users counted as 0 throughput)

Pre-computed CSVs in `final/` use the old `Jain_Fairness` column header (equivalent to `Served_User_Jain_Fairness`). Newly generated results will use the updated column names.
