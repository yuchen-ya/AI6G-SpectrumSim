#!/usr/bin/env bash
# Ablation: compare different teacher signals for PerRBDeepSets
set -e

echo "===== Ablation: DemandAwarePF Teacher ====="
python main.py \
    --model_type per_rb_deepsets \
    --teacher demandaware_pf \
    --epochs 300 \
    --seeds 0 1 2 3 4 \
    --results_dir results/demandaware_pf

echo ""
echo "===== Ablation: Hybrid Teacher Sweep ====="
python main.py \
    --model_type per_rb_deepsets \
    --teacher hybrid \
    --hybrid_alpha_list 0.0 0.25 0.5 0.75 1.0 \
    --seeds 0 \
    --results_dir results/hybrid_sweep

echo ""
echo "===== Ablation complete. ====="
