import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

COLORS = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5F6']
ALGO_ORDER = ['Random', 'Max-Channel', 'Greedy', 'WeightedGreedy', 'DemandAwarePF', 'MLP']


def _color_for(name):
    if name in ALGO_ORDER:
        return COLORS[ALGO_ORDER.index(name) % len(COLORS)]
    return COLORS[-1]


def save_fig(fig, name, results_dir='results'):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def _ordered(results_dict):
    order = [k for k in ALGO_ORDER if k in results_dict]
    for k in results_dict:
        if k not in order:
            order.append(k)
    return order


def plot_user_distribution(env, results_dir='results'):
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(env.user_positions[:, 0], env.user_positions[:, 1],
                    c=env.traffic_demand, cmap='YlOrRd', s=50, edgecolors='k', alpha=0.8)
    ax.scatter(*env.bs_position, marker='^', c='blue', s=200, label='Base Station', zorder=5)
    ax.set_xlim(0, env.area_size)
    ax.set_ylim(0, env.area_size)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('User Distribution & Traffic Demand')
    fig.colorbar(sc, label='Traffic Demand')
    ax.legend()
    ax.set_aspect('equal')
    save_fig(fig, 'user_distribution.png', results_dir)


def plot_throughput_comparison(results_dict, results_dir='results'):
    fig, ax = plt.subplots(figsize=(10, 5))
    names = _ordered(results_dict)
    vals = [results_dict[n]['total_throughput'] for n in names]
    cols = [_color_for(n) for n in names]
    bars = ax.bar(names, vals, color=cols, edgecolor='black')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.005,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Total Throughput (bps)')
    ax.set_title('Total Throughput Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    save_fig(fig, 'throughput_comparison.png', results_dir)


def plot_fairness_comparison(results_dict, results_dir='results'):
    fig, ax = plt.subplots(figsize=(10, 5))
    names = _ordered(results_dict)
    vals = [results_dict[n].get('served_user_jain_fairness', 0) for n in names]
    vals_all = [results_dict[n].get('all_user_jain_fairness', 0) for n in names]
    cols = [_color_for(n) for n in names]
    x = np.arange(len(names))
    w = 0.35
    bars1 = ax.bar(x - w/2, vals, w, color=cols, edgecolor='black', label='Served-User Fairness')
    bars2 = ax.bar(x + w/2, vals_all, w, color=[c + '80' for c in ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5F6'][:len(names)]],
                   edgecolor='black', alpha=0.6, label='All-User Fairness')
    for bar, val in zip(bars1, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, vals_all):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Jain Fairness Index')
    ax.set_title('Fairness Comparison (Served vs All Users)')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    save_fig(fig, 'fairness_comparison.png', results_dir)


def plot_demand_satisfaction(results_dict, results_dir='results'):
    fig, ax = plt.subplots(figsize=(10, 5))
    names = _ordered(results_dict)
    vals = [results_dict[n].get('demand_satisfaction', 0) for n in names]
    cols = [_color_for(n) for n in names]
    bars = ax.bar(names, vals, color=cols, edgecolor='black')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Demand Satisfaction Rate')
    ax.set_title('Demand Satisfaction Comparison')
    ax.set_ylim(0, max(vals) * 1.3 + 0.05)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    save_fig(fig, 'demand_satisfaction.png', results_dir)


def plot_rb_utilization(results_dict, results_dir='results'):
    fig, ax = plt.subplots(figsize=(10, 5))
    names = _ordered(results_dict)
    vals = [results_dict[n].get('rb_utilization', 0) for n in names]
    cols = [_color_for(n) for n in names]
    bars = ax.bar(names, vals, color=cols, edgecolor='black')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('RB Utilization')
    ax.set_title('RB Utilization (Assigned RBs / Total RBs)')
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    save_fig(fig, 'rb_utilization.png', results_dir)


def plot_effective_resource_utilization(results_dict, results_dir='results'):
    fig, ax = plt.subplots(figsize=(10, 5))
    names = _ordered(results_dict)
    vals = [results_dict[n].get('effective_resource_utilization', 0) for n in names]
    cols = [_color_for(n) for n in names]
    bars = ax.bar(names, vals, color=cols, edgecolor='black')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Effective Resource Utilization')
    ax.set_title('Effective Resource Utilization (RBs producing throughput / Total RBs)')
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    save_fig(fig, 'effective_resource_utilization.png', results_dir)


def plot_training_history(history, results_dir='results'):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(history['epoch'], history['train_loss'], 'b-', lw=1.5, label='Train')
    axes[0, 0].plot(history['epoch'], history['val_loss'], 'r-', lw=1.5, label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history['epoch'], history['total_throughput'], 'g-', lw=1.5)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Throughput')
    axes[0, 1].set_title('Total Throughput')
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(history['epoch'], history['avg_throughput'], 'r-', lw=1.5)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Avg Throughput')
    axes[0, 2].set_title('Avg User Throughput')
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(history['epoch'], history['served_user_jain_fairness'], 'm-', lw=1.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Fairness')
    axes[1, 0].set_title('Served-User Jain Fairness')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history['epoch'], history['demand_satisfaction'], 'c-', lw=1.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('DemSat')
    axes[1, 1].set_title('Demand Satisfaction')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].axis('off')
    if history['train_loss']:
        info = (f"Final Train Loss: {history['train_loss'][-1]:.5f}\n"
                f"Final Val Loss: {history['val_loss'][-1]:.5f}\n"
                f"Best Epoch: {history['epoch'][np.argmin(history['val_loss'])]}\n"
                f"Total Epochs: {history['epoch'][-1]}")
        axes[1, 2].text(0.1, 0.5, info, fontsize=12, va='center',
                        transform=axes[1, 2].transAxes, family='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 2].set_title('Summary')

    fig.suptitle('Training History', fontsize=14)
    fig.tight_layout()
    save_fig(fig, 'training_history.png', results_dir)


def plot_allocation_heatmap(allocations_dict, num_users, num_rbs, results_dir='results'):
    names = _ordered(allocations_dict)
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        alloc = allocations_dict[name]
        heatmap = np.zeros((num_users, num_rbs))
        for rb, user in enumerate(alloc):
            if 0 <= user < num_users:
                heatmap[user, rb] = 1.0
        im = ax.imshow(heatmap, cmap='Blues', aspect='auto', interpolation='nearest')
        ax.set_xlabel('RB')
        ax.set_ylabel('User')
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle('Allocation Heatmap', fontsize=14)
    fig.tight_layout()
    save_fig(fig, 'allocation_heatmap.png', results_dir)


def plot_per_user_throughput(per_user_dict, results_dir='results'):
    fig, ax = plt.subplots(figsize=(14, 5))
    names = _ordered(per_user_dict)
    n_users = len(list(per_user_dict.values())[0])
    x = np.arange(n_users)
    w = 0.8 / len(names)
    for i, name in enumerate(names):
        off = (i - len(names) / 2 + 0.5) * w
        ax.bar(x + off, per_user_dict[name], width=w, label=name, color=_color_for(name), alpha=0.8)
    ax.set_xlabel('User Index')
    ax.set_ylabel('Throughput')
    ax.set_title('Per-User Throughput')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    save_fig(fig, 'per_user_throughput.png', results_dir)


def plot_teacher_vs_mlp_scores(teacher_scores, mlp_scores, results_dir='results'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(teacher_scores, mlp_scores, alpha=0.5, s=20, c='steelblue')
    lo = min(teacher_scores.min(), mlp_scores.min())
    hi = max(teacher_scores.max(), mlp_scores.max())
    axes[0].plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='y=x')
    axes[0].set_xlabel('Teacher Score')
    axes[0].set_ylabel('Predicted Score')
    axes[0].set_title('Score Correlation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    t_rank = np.argsort(np.argsort(-teacher_scores)).astype(float)
    m_rank = np.argsort(np.argsort(-mlp_scores)).astype(float)
    axes[1].scatter(t_rank, m_rank, alpha=0.5, s=20, c='darkorange')
    n = len(teacher_scores)
    axes[1].plot([0, n], [0, n], 'r--', lw=1.5, label='y=x')
    axes[1].set_xlabel('Teacher Rank')
    axes[1].set_ylabel('Predicted Rank')
    axes[1].set_title('Rank Correlation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Teacher vs Model Score Analysis', fontsize=14)
    fig.tight_layout()
    save_fig(fig, 'teacher_vs_mlp_scores.png', results_dir)


def plot_hybrid_tradeoff(sweep_results, results_dir='results'):
    if not sweep_results:
        return
    alphas = [r['alpha'] for r in sweep_results]
    tp = [r['total_throughput'] / 1e6 for r in sweep_results]
    ff = [r['served_user_jain_fairness'] for r in sweep_results]
    dd = [r['demand_satisfaction'] for r in sweep_results]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(alphas, tp, 'b-o', lw=2, markersize=6, label='Total Throughput (M)')
    ax1.set_xlabel('Hybrid Alpha (alpha=0: PF, alpha=1: MaxChannel)')
    ax1.set_ylabel('Total Throughput (M)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(alphas, ff, 'r-s', lw=2, markersize=6, label='Jain Fairness')
    ax2.plot(alphas, dd, 'g-^', lw=2, markersize=6, label='Demand Satisfaction')
    ax2.set_ylabel('Fairness / Demand Satisfaction', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    ax1.set_title('Hybrid Teacher: Throughput-Fairness Trade-off vs Alpha')
    fig.tight_layout()
    save_fig(fig, 'hybrid_tradeoff_curve.png', results_dir)

    for metric, fname, color in [('served_user_jain_fairness', 'hybrid_fairness_vs_alpha.png', 'r'),
                                  ('demand_satisfaction', 'hybrid_demsat_vs_alpha.png', 'g'),
                                  ('total_throughput', 'hybrid_throughput_vs_alpha.png', 'b')]:
        vals = [r[metric] for r in sweep_results]
        if metric == 'total_throughput':
            vals = [v / 1e6 for v in vals]
        fig2, ax = plt.subplots(figsize=(8, 5))
        ax.plot(alphas, vals, color + '-o', lw=2, markersize=6)
        ax.set_xlabel('Alpha')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} vs Alpha')
        ax.grid(True, alpha=0.3)
        fig2.tight_layout()
        save_fig(fig2, fname, results_dir)


def plot_per_rb_score_heatmap(scores, num_users, num_rbs, results_dir='results'):
    fig, ax = plt.subplots(figsize=(max(8, num_rbs), max(6, num_users * 0.12)))
    im = ax.imshow(scores, cmap='RdYlBu_r', aspect='auto', interpolation='nearest')
    ax.set_xlabel('RB Index')
    ax.set_ylabel('User Index')
    ax.set_title('Per-User Per-RB Score Heatmap')
    fig.colorbar(im, label='Score')
    for j in range(num_rbs):
        i = int(np.argmax(scores[:, j]))
        ax.plot(j, i, 'k*', markersize=6)
    fig.tight_layout()
    save_fig(fig, 'per_rb_score_heatmap.png', results_dir)


def plot_multi_seed_summary(multi_seed_data, results_dir='results'):
    metrics = ['total_throughput', 'served_user_jain_fairness', 'demand_satisfaction']
    labels = ['Total Throughput', 'Served-User Jain Fairness', 'Demand Satisfaction']
    algos = multi_seed_data['algos']

    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5))
    for ax, m, lbl in zip(axes if len(metrics) > 1 else [axes], metrics, labels):
        means = [multi_seed_data[m]['mean'][a] for a in algos]
        stds = [multi_seed_data[m]['std'][a] for a in algos]
        cols = [_color_for(a) for a in algos]
        ax.bar(algos, means, yerr=stds, color=cols, edgecolor='black', capsize=5)
        ax.set_ylabel(lbl)
        ax.set_title(f'{lbl} (multi-seed)')
        ax.grid(True, alpha=0.3, axis='y')
    fig.suptitle('Multi-Seed Results (mean +/- std)', fontsize=14)
    fig.tight_layout()
    save_fig(fig, 'multi_seed_summary.png', results_dir)
