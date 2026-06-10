#!/usr/bin/env bash
# Run the final PerRBDeepSets + Max-Channel experiment (5 seeds, 300 epochs)
# Expected runtime: ~15-30 minutes on CPU
set -e

echo "===== PerRBDeepSets + Max-Channel Teacher (5 seeds) ====="
python main.py \
    --model_type per_rb_deepsets \
    --teacher max_channel \
    --epochs 300 \
    --seeds 0 1 2 3 4 \
    --results_dir results/max_channel

echo ""
echo "===== Experiment complete. Results in results/max_channel/ ====="
