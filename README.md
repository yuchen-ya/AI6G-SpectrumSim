# AI6G-SpectrumSim

A lightweight AI-native 6G spectrum/resource allocation simulation framework with **PerRBDeepSets** — a neural scheduler that learns per-user-per-RB scoring from heuristic teachers.

## Motivation

In 6G AI-native networks, intelligent resource scheduling must balance throughput, fairness, and demand satisfaction across many users and resource blocks (RBs). This project explores whether small neural networks can learn to replicate — and potentially improve upon — classical heuristic scheduling algorithms.

Rather than treating scheduling as a monolithic optimization problem, PerRBDeepSets decomposes it into per-RB allocation decisions, matching the neural architecture to the problem structure.

## Key Features

- **PerRBDeepSets model**: Per-user-per-RB pairwise scoring with separate user/RB encoders (405K parameters)
- **Multiple baselines**: Random, Max-Channel, Greedy, WeightedGreedy, DemandAwarePF
- **Knowledge distillation**: Train neural schedulers from heuristic teacher algorithms
- **Multi-seed evaluation**: Statistical robustness with configurable random seeds
- **Hybrid teacher sweep**: Interpolate between Max-Channel and DemandAwarePF objectives
- **Reproducible**: Fixed seeds, deterministic data generation, deterministic evaluation

## Installation

```bash
# Clone the repository
git clone https://github.com/yuchen-ya/AI6G-SpectrumSim.git
cd AI6G-SpectrumSim

# Install dependencies (Python >= 3.8 recommended)
pip install -r requirements.txt

# Or using conda
conda env create -f environment.yml
conda activate ai6g
```

### Requirements

- Python >= 3.8
- NumPy >= 1.21
- PyTorch >= 1.9
- Matplotlib >= 3.4

## Quick Start

```bash
# Run the default experiment (PerRBDeepSets + Max-Channel, 5 seeds)
python main.py

# Single-seed quick test (~2-5 minutes on CPU)
python main.py --seeds 0 --epochs 50
```

## Reproduce Final Experiment

The main result uses PerRBDeepSets with Max-Channel teacher supervision across 5 random seeds:

```bash
python main.py --model_type per_rb_deepsets --teacher max_channel --epochs 300 --seeds 0 1 2 3 4
```

**Expected runtime**: ~15-30 minutes on CPU (depends on hardware). GPU is used automatically if available.

### Expected Results

| Algorithm            | Total Throughput | Served-User Fairness | All-User Fairness | Demand Satisfaction |
| -------------------- | ---------------: | -------------------: | ----------------: | ------------------: |
| Random               |   37.10M ± 0.20M |                0.930 |                 — |               0.141 |
| **Max-Channel (oracle)** | **43.54M ± 0.91M** |            **0.749** |                 — |             **0.030** |
| Greedy               |   38.25M ± 0.52M |                0.999 |                 — |               0.353 |
| WeightedGreedy       |   37.61M ± 0.42M |                0.811 |                 — |               0.081 |
| DemandAwarePF        |   37.84M ± 0.91M |                0.829 |                 — |               0.075 |
| **PerRBDeepSets**    | **43.41M ± 0.95M** |              **1.000** |               — |             **0.017** |

> **Note**: "Served-User Fairness" measures Jain's index among users who receive at least one RB. "All-User Fairness" measures it over all users (unserved = 0 throughput). See [docs/metrics.md](docs/metrics.md) for details.

> All-User Fairness is documented in `docs/metrics.md` but not included in the pre-computed reference CSVs yet. The reported fairness values are served-user Jain fairness.

PerRBDeepSets predicts a per-user-per-RB score matrix and assigns each resource block independently. With only 405K parameters, it reaches 43.41M ± 0.95M total throughput under Max-Channel supervision, which is within 0.3% of the oracle Max-Channel baseline. Compared with the previous user-level DeepSets model, PerRBDeepSets improves throughput by about 15.1%, demonstrating that matching the neural architecture to the resource-block-level scheduling structure is more important than simply increasing parameter count.

## Model Architecture

### PerRBDeepSets (best model, 405,249 parameters)

```
User encoder:  Input(12) → [Linear(256) → BN → GELU → Dropout → Linear(256) → BN → GELU → Dropout]
RB encoder:    Input(2)  → [Linear(64) → BN → GELU → Linear(64) → BN → GELU]
Pairwise:      concat(user_emb[256], rb_emb[64]) = [320]
               → [512 → BN → GELU → Dropout → 256 → BN → GELU → Dropout → 128 → BN → GELU → Dropout → 1]
Output:        [num_users × num_rbs] score matrix → argmax per column (each RB selects best user)
```

### Other Models

| Model | Input | Output | Params |
|-------|-------|--------|--------|
| ScoringMLP | User features (12) | Per-user score | ~436K |
| DeepSetScheduler | User features (12) | Per-user score (with global context) | ~630K |
| **PerRBDeepSets** | User features (12) + RB features (2) | Per-user-per-RB score matrix | **~405K** |

### Training

- **Loss**: SmoothL1 + λ_rank × Pairwise Ranking Loss (λ_rank = 0.1)
- **Teacher**: max_channel / demandaware_pf / hybrid
- **Optimizer**: AdamW with CosineAnnealingLR (η_min = 0.01 × lr)
- **Regularization**: Gradient clipping (max_norm=1.0), Dropout (0.1), Early stopping (patience=30)

## Algorithms Compared

| Algorithm | Strategy |
|-----------|----------|
| **Random** | Random user assignment per RB |
| **Max-Channel** | Best SNR user per RB (max throughput oracle) |
| **Greedy** | SNR × demand score per RB |
| **WeightedGreedy** | α·SNR + β·demand + γ·fairness − δ·history |
| **DemandAwarePF** | rate × (1 + λ·demand) / (history + ε)^β (Proportional Fair) |
| **PerRBDeepSets** | Learned per-user-per-RB scoring (neural) |

## Key Results (Figures)

### Throughput Comparison

![Total Throughput Comparison](results/figures/throughput_comparison_final.png)

*PerRBDeepSets reaches 43.41M total throughput, only 0.3% below the oracle Max-Channel baseline (43.54M).*

### Model Architecture Evolution

![Model Evolution](results/figures/model_evolution.png)

*Moving from a flat MLP to per-user-per-RB scoring improved throughput by 15.1% (37.56M → 43.41M), despite having fewer parameters than User-level DeepSets.*

### Oracle Gap

![Oracle Gap](results/figures/oracle_gap.png)

*PerRBDeepSets closely imitates Max-Channel scheduling under supervision, with a gap of only ~0.3%.*

### Per-RB Score Heatmap

![Per-RB Score Heatmap](results/figures/per_rb_score_heatmap.png)

*The model outputs a [num_users × num_rbs] score matrix. Each RB is assigned to the user with the highest score (marked by ★).*

### All Figures

After running `python scripts/plot_results.py`, all figures are saved to `results/figures/`:

| Figure | Description |
|--------|-------------|
| `throughput_comparison_final.png` | Total throughput across algorithms |
| `model_evolution.png` | Model iteration improvement |
| `oracle_gap.png` | PerRBDeepSets vs Oracle gap |
| `multi_metric_radar_or_bar.png` | Multi-metric grouped bar comparison |
| `teacher_match_accuracy.png` | Per-seed teacher match accuracy |
| `per_rb_score_heatmap.png` | Learned score matrix visualization |
| `allocation_heatmap_final.png` | User-RB assignment heatmap |
| `limitation_pf_teacher.png` | PF teacher failure analysis |

## Project Structure

```
AI6G-SpectrumSim/
├── main.py                  # Entry point
├── src/
│   ├── environment.py       # Spectrum simulation environment
│   ├── algorithms.py        # Heuristic scheduling algorithms
│   ├── mlp_model.py         # Neural network models
│   ├── train.py             # Training and evaluation logic
│   └── visualization.py     # Plotting utilities
├── scripts/                 # Experiment runner scripts
├── configs/                 # YAML configuration files
├── results/                 # Experiment outputs
│   ├── final/               # Pre-computed reference results
│   └── figures/             # Reference plots
├── docs/                    # Documentation
├── tests/                   # Unit tests
├── requirements.txt
├── pyproject.toml
└── environment.yml
```

## Limitations

- **Small-scale simulation**: 50 users, 10 RBs in a 100m × 100m area — far from real 6G scenarios
- **Simplified channel model**: Log-distance path loss + Gaussian fading, no MIMO/mmWave/NLOS effects
- **Single-cell only**: No inter-cell interference modeling
- **No per-RB CSI input**: User features do not include per-RB channel state, limiting DemandAwarePF teacher learning. The PF teacher achieves 0% match accuracy, confirming that the current model lacks the pairwise CSI needed to learn user-RB joint state-dependent policies.
- **Static allocation**: No temporal scheduling or multi-slot optimization
- **Fairness metric caveat**: The "Served-User Fairness" metric only considers users receiving at least one RB and can be misleading when few users are served

![PF Teacher Limitation](results/figures/limitation_pf_teacher.png)

*PF teacher fails because the model lacks pairwise CSI for joint user-RB state reasoning. PerRBDeepSets with PF teacher drops to ~40.08M vs 43.41M with Max-Channel teacher.*

## Future Work

- Add per-RB CSI as input features to improve DemandAwarePF learning
- Scale to larger scenarios (more users, RBs, multi-cell)
- Implement temporal/multi-slot scheduling
- Explore reinforcement learning for multi-objective optimization
- Add MIMO and beamforming support
- Investigate transfer learning across scenario sizes

## Citation

If you use this repository, please cite it as:

> AI6G-SpectrumSim: A lightweight AI-native 6G spectrum allocation simulation with PerRBDeepSets. https://github.com/yuchen-ya/AI6G-SpectrumSim

## License

This project is licensed under the [MIT License](LICENSE).
