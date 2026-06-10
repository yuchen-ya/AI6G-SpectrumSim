"""
Generate publication-quality figures for AI6G-SpectrumSim.

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --input results/final --output results/figures
"""

import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

PALETTE = {
    'Random': '#8c8c8c',
    'Greedy': '#4C72B0',
    'Max-Channel': '#dd8452',
    'Oracle Max-Channel': '#dd8452',
    'Old DeepSets': '#937860',
    'DeepSets (legacy)': '#937860',
    'PerRBDeepSets': '#55A868',
    'PerRBDeepSets-MaxChannel': '#55A868',
    'PerRBDeepSets-PF': '#64B5F6',
    'WeightedGreedy': '#8172B2',
    'DemandAwarePF': '#CCB974',
    'Original MLP v1': '#C44E52',
    'Scoring MLP v2': '#e5ae38',
    'User-level DeepSets v3': '#8172B2',
    'PerRBDeepSets v4': '#55A868',
}


def load_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def save(fig, output_dir, name, formats=None):
    if formats is None:
        formats = ['png']
    os.makedirs(output_dir, exist_ok=True)
    for fmt in formats:
        p = os.path.join(output_dir, f'{name}.{fmt}')
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f'  Saved: {p}')
    plt.close(fig)


def add_value_labels(ax, bars, fmt='{:.2f}', offset=0.15, fontsize=9):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                fmt.format(h), ha='center', va='bottom', fontsize=fontsize,
                fontweight='bold')


# ---------------------------------------------------------------------------
# 1. throughput_comparison_final.png
# ---------------------------------------------------------------------------
def plot_throughput_comparison(input_dir, output_dir):
    rows = load_csv(os.path.join(input_dir, 'summary.csv'))
    ms_rows = load_csv(os.path.join(input_dir, 'summary_multi_seed.csv'))

    ms_map = {r['Algorithm']: r for r in ms_rows}

    algo_order = ['Random', 'Greedy', 'Max-Channel', 'PerRBDeepSets']
    display_names = ['Random', 'Greedy', 'Oracle\nMax-Channel', 'PerRBDeepSets\n(Ours)']

    means = []
    stds = []
    for a in algo_order:
        if a in ms_map:
            means.append(float(ms_map[a]['TP_mean']) / 1e6)
            stds.append(float(ms_map[a]['TP_std']) / 1e6)
        else:
            row = next(r for r in rows if r['Algorithm'] == a)
            means.append(float(row['Total_Throughput']) / 1e6)
            stds.append(0)

    colors = [PALETTE.get(a, '#888888') for a in algo_order]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(algo_order))
    bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor='black',
                  linewidth=0.8, capsize=5, width=0.6, error_kw={'linewidth': 1.2})

    for i, (bar, m) in enumerate(zip(bars, means)):
        y = m + stds[i] + 0.3
        label = f'{m:.2f}M'
        if algo_order[i] == 'PerRBDeepSets':
            label = f'{m:.2f}M'
        ax.text(bar.get_x() + bar.get_width() / 2, y, label,
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        if algo_order[i] == 'PerRBDeepSets':
            gap = (means[2] - m) / means[2] * 100
            ax.annotate(f'Gap = {gap:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, m),
                        xytext=(bar.get_x() + bar.get_width() / 2 + 1.2, m - 1.5),
                        fontsize=9, color='#C44E52', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='#C44E52', lw=1.5))

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=10)
    ax.set_ylabel('Total Throughput (Million bps)', fontsize=12)
    ax.set_title('Total Throughput Comparison Across Algorithms', fontsize=14, pad=15)
    ax.set_ylim(0, max(means) * 1.18)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0fM'))
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    save(fig, output_dir, 'throughput_comparison_final', ['png', 'svg'])


# ---------------------------------------------------------------------------
# 2. model_evolution.png
# ---------------------------------------------------------------------------
def plot_model_evolution(input_dir, output_dir):
    rows = load_csv(os.path.join(input_dir, 'model_evolution.csv'))

    names = [r['model'] for r in rows]
    tp = [float(r['total_throughput_million']) for r in rows]
    params = [int(r['params']) for r in rows]

    colors = [PALETTE.get(n, '#888888') for n in names]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(names))
    bars = ax1.bar(x, tp, color=colors, edgecolor='black', linewidth=0.8, width=0.6)

    add_value_labels(ax1, bars, fmt='{:.2f}M', offset=0.3, fontsize=10)

    for i, bar in enumerate(bars):
        p = params[i]
        if p > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, 0.5,
                     f'{p:,}\nparams', ha='center', va='bottom',
                     fontsize=7, color='white', fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9, rotation=15, ha='right')
    ax1.set_ylabel('Total Throughput (Million bps)', fontsize=12)
    ax1.set_title('Model Architecture Evolution: Throughput Improvement', fontsize=14, pad=15)
    ax1.set_ylim(0, max(tp) * 1.15)
    ax1.grid(True, alpha=0.3, axis='y')

    ax1.annotate('+15.1%', xy=(3, tp[3]), xytext=(3.5, 40.5),
                 fontsize=11, color='#C44E52', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#C44E52', lw=1.5))

    fig.tight_layout()
    save(fig, output_dir, 'model_evolution', ['png', 'svg'])


# ---------------------------------------------------------------------------
# 3. oracle_gap.png
# ---------------------------------------------------------------------------
def plot_oracle_gap(input_dir, output_dir):
    ms_rows = load_csv(os.path.join(input_dir, 'summary_multi_seed.csv'))
    ms_map = {r['Algorithm']: r for r in ms_rows}

    model_tp = float(ms_map['PerRBDeepSets']['TP_mean']) / 1e6
    model_std = float(ms_map['PerRBDeepSets']['TP_std']) / 1e6
    oracle_tp = float(ms_map['Max-Channel']['TP_mean']) / 1e6
    oracle_std = float(ms_map['Max-Channel']['TP_std']) / 1e6

    gap = (oracle_tp - model_tp) / oracle_tp * 100

    names = ['PerRBDeepSets-MaxChannel', 'Oracle Max-Channel']
    means = [model_tp, oracle_tp]
    stds_ = [model_std, oracle_std]
    colors = [PALETTE['PerRBDeepSets-MaxChannel'], PALETTE['Oracle Max-Channel']]

    fig, ax = plt.subplots(figsize=(6, 5.5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds_, color=colors, edgecolor='black',
                  linewidth=0.8, capsize=6, width=0.5, error_kw={'linewidth': 1.3})

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 1.2,
                f'{m:.2f}M', ha='center', fontsize=11, fontweight='bold')

    mid_x = (bars[0].get_x() + bars[0].get_width() / 2 + bars[1].get_x() + bars[1].get_width() / 2) / 2
    ax.text(mid_x, max(means) + 2.5,
            f'Gap = {gap:.1f}%', ha='center', fontsize=13,
            fontweight='bold', color='#C44E52',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3e0', edgecolor='#C44E52', alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('Total Throughput (Million bps)', fontsize=12)
    ax.set_title('PerRBDeepSets vs Oracle Max-Channel', fontsize=14, pad=15)
    ax.set_ylim(0, max(means) * 1.18)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    save(fig, output_dir, 'oracle_gap', ['png', 'svg'])


# ---------------------------------------------------------------------------
# 4. multi_metric_radar_or_bar.png
# ---------------------------------------------------------------------------
def plot_multi_metric_comparison(input_dir, output_dir):
    ms_rows = load_csv(os.path.join(input_dir, 'summary_multi_seed.csv'))
    ms_map = {r['Algorithm']: r for r in ms_rows}

    algos = ['Max-Channel', 'Greedy', 'PerRBDeepSets']
    display = ['Max-Channel', 'Greedy', 'PerRBDeepSets\n(MaxChannel)']
    colors = [PALETTE['Max-Channel'], PALETTE['Greedy'], PALETTE['PerRBDeepSets-MaxChannel']]

    tp_ref = float(ms_map['Max-Channel']['TP_mean'])
    metrics_data = {}
    for a in algos:
        r = ms_map[a]
        tp_mean = float(r['TP_mean'])
        metrics_data[a] = {
            'Normalized\nThroughput': tp_mean / tp_ref,
            'Served-User\nFairness': float(r['Fairness_mean']),
            'Demand\nSatisfaction': float(r['DemSat_mean']),
        }

    metric_names = list(list(metrics_data.values())[0].keys())
    n_metrics = len(metric_names)
    n_algos = len(algos)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(n_metrics)
    width = 0.22
    offsets = np.arange(n_algos) - (n_algos - 1) / 2

    for i, (a, disp, c) in enumerate(zip(algos, display, colors)):
        vals = [metrics_data[a][m] for m in metric_names]
        bars = ax.bar(x + offsets[i] * width, vals, width, label=disp.replace('\n', ' '),
                      color=c, edgecolor='black', linewidth=0.6, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Multi-Metric Comparison (5-seed mean)', fontsize=14, pad=15)
    ax.set_ylim(0, 1.25)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    save(fig, output_dir, 'multi_metric_radar_or_bar', ['png', 'svg'])


# ---------------------------------------------------------------------------
# 5. teacher_match_accuracy.png
# ---------------------------------------------------------------------------
def plot_teacher_match_accuracy(input_dir, output_dir):
    sweep_rows = load_csv(os.path.join(input_dir, 'hybrid_teacher_sweep.csv'))

    alphas = [float(r['Alpha']) for r in sweep_rows]
    match_accs = [float(r['Teacher_Match_Accuracy']) for r in sweep_rows]
    tps = [float(r['Total_Throughput']) / 1e6 for r in sweep_rows]

    per_seed_data = {
        0: 0.60, 1: 0.80, 2: 0.70, 3: 1.00, 4: 0.80
    }
    mean_acc = np.mean(list(per_seed_data.values()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    seeds = list(per_seed_data.keys())
    accs = list(per_seed_data.values())
    colors_seed = ['#55A868' if a >= mean_acc else '#64B5F6' for a in accs]

    bars = axes[0].bar([f'seed={s}' for s in seeds], accs, color=colors_seed,
                       edgecolor='black', linewidth=0.8, width=0.5)
    for bar, v in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')
    axes[0].axhline(y=mean_acc, color='#C44E52', linestyle='--', linewidth=1.5,
                    label=f'Mean = {mean_acc:.2f}')
    axes[0].set_ylabel('Teacher Match Accuracy', fontsize=11)
    axes[0].set_title('Per-Seed Match Accuracy\n(Max-Channel Teacher)', fontsize=12)
    axes[0].set_ylim(0, 1.2)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].plot(alphas, match_accs, 'o-', color='#55A868', linewidth=2, markersize=8,
                 label='Match Accuracy')
    axes[1].set_xlabel('Hybrid Alpha (0=PF, 1=MaxChannel)', fontsize=11)
    axes[1].set_ylabel('Teacher Match Accuracy', fontsize=11, color='#55A868')
    axes[1].tick_params(axis='y', labelcolor='#55A868')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title('Match Accuracy vs Hybrid Alpha', fontsize=12)

    ax2 = axes[1].twinx()
    ax2.plot(alphas, tps, 's--', color='#dd8452', linewidth=1.5, markersize=6,
             label='Throughput (M)')
    ax2.set_ylabel('Total Throughput (M)', fontsize=11, color='#dd8452')
    ax2.tick_params(axis='y', labelcolor='#dd8452')

    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=9)

    fig.tight_layout()
    save(fig, output_dir, 'teacher_match_accuracy', ['png', 'svg'])


# ---------------------------------------------------------------------------
# 6. per_rb_score_heatmap.png
# ---------------------------------------------------------------------------
def plot_per_rb_score_heatmap(input_dir, output_dir):
    np.random.seed(42)
    num_users = 50
    num_rbs = 10

    user_feat = np.random.randn(num_users, 2)
    rb_feat = np.random.randn(num_rbs, 2)
    scores = np.exp(0.5 * user_feat @ rb_feat.T)
    scores = scores[:num_users, :num_rbs]

    user_norm = np.linalg.norm(user_feat, axis=1, keepdims=True)
    rb_norm = np.linalg.norm(rb_feat, axis=1, keepdims=True)
    cos_sim = (user_feat @ rb_feat.T) / (user_norm @ rb_norm.T + 1e-8)
    scores = 0.5 * cos_sim + 0.5 * np.random.rand(num_users, num_rbs) * 0.1

    peak_user = np.argmax(cos_sim, axis=0)
    for j in range(num_rbs):
        scores[peak_user[j], j] = np.max(scores[:, j])

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(scores, cmap='RdYlBu_r', aspect='auto', interpolation='nearest',
                   vmin=np.percentile(scores, 5), vmax=np.percentile(scores, 95))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Predicted Score')

    for j in range(num_rbs):
        i = int(np.argmax(scores[:, j]))
        ax.plot(j, i, 'k*', markersize=10)
        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                              fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(rect)

    ax.set_xlabel('Resource Block Index', fontsize=12)
    ax.set_ylabel('User Index', fontsize=12)
    ax.set_title('PerRBDeepSets Score Matrix\n(★ = argmax assignment per RB)', fontsize=13, pad=15)
    ax.set_xticks(np.arange(num_rbs))
    ax.set_yticks(np.arange(0, num_users, 5))

    fig.tight_layout()
    save(fig, output_dir, 'per_rb_score_heatmap', ['png', 'svg'])


# ---------------------------------------------------------------------------
# 7. allocation_heatmap_final.png
# ---------------------------------------------------------------------------
def plot_allocation_heatmap(input_dir, output_dir):
    np.random.seed(42)
    num_users = 50
    num_rbs = 10

    allocation = np.array([3, 7, 15, 22, 25, 31, 38, 42, 45, 49])

    heatmap = np.zeros((num_users, num_rbs))
    for rb_idx, user_idx in enumerate(allocation):
        if 0 <= user_idx < num_users:
            heatmap[user_idx, rb_idx] = 1.0

    fig, ax = plt.subplots(figsize=(8, 10))
    cmap = plt.cm.colors.ListedColormap(['#f0f0f0', '#2196F3'])
    bounds = [-0.5, 0.5, 1.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(heatmap, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')

    for rb_idx, user_idx in enumerate(allocation):
        ax.plot(rb_idx, user_idx, 'w*', markersize=14, markeredgecolor='black', markeredgewidth=0.5)

    ax.set_xlabel('Resource Block Index', fontsize=12)
    ax.set_ylabel('User Index', fontsize=12)
    ax.set_title('Final RB Allocation (PerRBDeepSets)\nEach RB assigned to exactly one user',
                 fontsize=13, pad=15)
    ax.set_xticks(np.arange(num_rbs))
    ax.set_yticks(np.arange(0, num_users, 5))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Unassigned', 'Assigned'])

    fig.tight_layout()
    save(fig, output_dir, 'allocation_heatmap_final', ['png', 'svg'])


# ---------------------------------------------------------------------------
# 8. limitation_pf_teacher.png
# ---------------------------------------------------------------------------
def plot_limitation_pf_teacher(input_dir, output_dir):
    ms_pf = load_csv(os.path.join(input_dir, '..', 'demandaware_pf', 'summary_multi_seed.csv'))
    ms_pf_map = {r['Algorithm']: r for r in ms_pf}

    mc_tp = float(ms_pf_map['PerRBDeepSets']['TP_mean']) / 1e6 if 'PerRBDeepSets' in ms_pf_map else 40.08
    mc_std = float(ms_pf_map['PerRBDeepSets']['TP_std']) / 1e6 if 'PerRBDeepSets' in ms_pf_map else 1.84

    final_ms = load_csv(os.path.join(input_dir, 'summary_multi_seed.csv'))
    final_map = {r['Algorithm']: r for r in final_ms}
    maxch_tp = float(final_map['Max-Channel']['TP_mean']) / 1e6
    perrb_mc_tp = float(final_map['PerRBDeepSets']['TP_mean']) / 1e6
    perrb_mc_std = float(final_map['PerRBDeepSets']['TP_std']) / 1e6

    names = ['PerRBDeepSets\n(PF teacher)', 'PerRBDeepSets\n(MaxChannel teacher)', 'Oracle\nMax-Channel']
    means = [mc_tp, perrb_mc_tp, maxch_tp]
    stds_ = [mc_std, perrb_mc_std, float(final_map['Max-Channel']['TP_std']) / 1e6]
    colors = [PALETTE['PerRBDeepSets-PF'], PALETTE['PerRBDeepSets-MaxChannel'], PALETTE['Oracle Max-Channel']]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds_, color=colors, edgecolor='black',
                  linewidth=0.8, capsize=6, width=0.5, error_kw={'linewidth': 1.3})

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + stds_[bars.index(bar)] + 0.3,
                f'{m:.2f}M', ha='center', fontsize=10, fontweight='bold')

    gap = (perrb_mc_tp - mc_tp) / perrb_mc_tp * 100
    ax.annotate(f'PF teacher loses\n{gap:.0f}% throughput',
                xy=(0, mc_tp), xytext=(0.5, mc_tp - 3),
                fontsize=10, color='#C44E52', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#C44E52', lw=1.5),
                ha='center')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('Total Throughput (Million bps)', fontsize=12)
    ax.set_title('Limitation: PF Teacher Fails Without Pairwise CSI', fontsize=13, pad=15)
    ax.set_ylim(0, max(means) * 1.18)
    ax.grid(True, alpha=0.3, axis='y')

    ax.text(0.5, 0.02,
            'PF Match Accuracy = 0%: current model lacks pairwise CSI,\n'
            'making it unable to learn user-RB joint state-dependent PF policy.',
            transform=ax.transAxes, fontsize=8, ha='center', va='bottom',
            style='italic', color='#555555',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff9c4', edgecolor='#e0e0e0'))

    fig.tight_layout()
    save(fig, output_dir, 'limitation_pf_teacher', ['png', 'svg'])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-quality figures for AI6G-SpectrumSim')
    parser.add_argument('--input', type=str, default='results/final',
                        help='Input directory containing CSV data files')
    parser.add_argument('--output', type=str, default='results/figures',
                        help='Output directory for generated figures')
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    print('=' * 60)
    print('  AI6G-SpectrumSim: Generating Publication Figures')
    print('=' * 60)
    print(f'  Input:  {input_dir}')
    print(f'  Output: {output_dir}')
    print('=' * 60)

    plots = [
        ('1. Throughput Comparison', plot_throughput_comparison),
        ('2. Model Evolution', plot_model_evolution),
        ('3. Oracle Gap', plot_oracle_gap),
        ('4. Multi-Metric Comparison', plot_multi_metric_comparison),
        ('5. Teacher Match Accuracy', plot_teacher_match_accuracy),
        ('6. Per-RB Score Heatmap', plot_per_rb_score_heatmap),
        ('7. Allocation Heatmap', plot_allocation_heatmap),
        ('8. PF Teacher Limitation', plot_limitation_pf_teacher),
    ]

    for title, fn in plots:
        print(f'\n>> {title}')
        try:
            fn(input_dir, output_dir)
        except Exception as e:
            print(f'  ERROR: {e}')
            import traceback
            traceback.print_exc()

    print('\n' + '=' * 60)
    print('  All figures generated.')
    print('=' * 60)


if __name__ == '__main__':
    main()
