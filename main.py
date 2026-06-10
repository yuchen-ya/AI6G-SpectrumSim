import os
import csv
import argparse
import numpy as np
import torch

from src.environment import SpectrumEnvironment
from src.algorithms import (
    random_allocation, max_channel_allocation,
    greedy_allocation, weighted_greedy_allocation,
    demand_aware_pf_allocation,
)
from src.mlp_model import build_model
from src.train import (train_model, evaluate_model, allocate_from_scores,
                       train_per_rb_model, evaluate_per_rb_model)
from src.visualization import (
    plot_user_distribution, plot_throughput_comparison,
    plot_fairness_comparison, plot_demand_satisfaction,
    plot_rb_utilization, plot_effective_resource_utilization,
    plot_training_history, plot_allocation_heatmap,
    plot_per_user_throughput, plot_teacher_vs_mlp_scores,
    plot_multi_seed_summary, plot_per_rb_score_heatmap,
    plot_hybrid_tradeoff, ALGO_ORDER,
)

METRIC_KEYS = ['total_throughput', 'served_user_jain_fairness',
               'all_user_jain_fairness', 'demand_satisfaction',
               'rb_utilization', 'effective_resource_utilization']

IS_PER_RB = {'per_rb_deepsets'}


def _eval_algo(alloc, env):
    tp, _ = env.compute_throughput(alloc)
    return {
        'total_throughput': float(np.sum(tp)),
        'avg_throughput': float(np.mean(tp)),
        'served_user_jain_fairness': float(env.served_user_jain_fairness(tp)),
        'all_user_jain_fairness': float(env.all_user_jain_fairness(tp)),
        'demand_satisfaction': float(env.compute_demand_satisfaction(tp)),
        'rb_utilization': float(env.compute_rb_utilization(alloc)),
        'effective_resource_utilization': float(env.compute_effective_resource_utilization(alloc, tp)),
        'throughput': tp,
        'allocation': alloc,
    }


def evaluate_all_algorithms(model, env, device='cpu', teacher='pf',
                            pf_lambda_demand=1.0, pf_beta=0.5, model_type='deepsets',
                            hybrid_alpha=0.5):
    results = {}

    rng = np.random.default_rng(12345)
    results['Random'] = _eval_algo(
        random_allocation(env.num_users, env.num_rbs, rng), env)

    results['Max-Channel'] = _eval_algo(
        max_channel_allocation(env.snr_db_per_rb, env.num_rbs), env)

    results['Greedy'] = _eval_algo(
        greedy_allocation(env.snr_db_per_rb, env.traffic_demand,
                          env.capacity_per_rb, env.num_rbs), env)

    results['WeightedGreedy'] = _eval_algo(
        weighted_greedy_allocation(env.snr_db_per_rb, env.traffic_demand,
                                   env.capacity_per_rb, env.num_rbs,
                                   history_throughput=env.history_throughput), env)

    results['DemandAwarePF'] = _eval_algo(
        demand_aware_pf_allocation(env.snr_db_per_rb, env.traffic_demand,
                                   env.capacity_per_rb, env.num_rbs,
                                   history_throughput=env.history_throughput,
                                   lambda_demand=pf_lambda_demand, beta=pf_beta), env)

    if model_type in IS_PER_RB:
        mlp_res = evaluate_per_rb_model(model, env, device, teacher=teacher,
                                        pf_lambda_demand=pf_lambda_demand, pf_beta=pf_beta,
                                        hybrid_alpha=hybrid_alpha)
    else:
        mlp_res = evaluate_model(model, env, device, teacher=teacher,
                                 pf_lambda_demand=pf_lambda_demand, pf_beta=pf_beta)
    model_name = f'{model.__class__.__name__}'
    results[model_name] = mlp_res

    return results, model_name


def print_results_table(results, model_name, total_params=0):
    header = (f"{'Algorithm':<20} {'Total TP':>14} {'Avg TP':>14} "
              f"{'SrvFair':>10} {'AllFair':>10} {'DemSat':>10} {'RB_Util':>10} {'Eff_Util':>10}")
    print(header)
    print('-' * len(header))
    for name in ALGO_ORDER + [model_name]:
        if name not in results:
            continue
        r = results[name]
        line = (f"{name:<20} {r['total_throughput']:>14.2f} {r['avg_throughput']:>14.4f} "
                f"{r['served_user_jain_fairness']:>10.4f} {r.get('all_user_jain_fairness', 0):>10.4f} "
                f"{r.get('demand_satisfaction', 0):>10.4f} "
                f"{r.get('rb_utilization', 0):>10.4f} {r.get('effective_resource_utilization', 0):>10.4f}")
        if name == model_name and total_params > 0:
            line += f"  [{total_params:,} params]"
        print(line)
    mlp_r = results.get(model_name, {})
    if 'teacher_rank_correlation' in mlp_r:
        print(f"\n  Rank Corr with Teacher: {mlp_r['teacher_rank_correlation']:.4f}")
        print(f"  Jaccard (served users): {mlp_r['teacher_jaccard']:.4f}")
    if 'teacher_match_accuracy' in mlp_r:
        print(f"  Teacher Match Acc:      {mlp_r['teacher_match_accuracy']:.4f}")


def save_summary_csv(results, model_name, total_params, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, 'summary.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Algorithm', 'Total_Throughput', 'Avg_Throughput',
                     'Served_User_Jain_Fairness', 'All_User_Jain_Fairness',
                     'Demand_Satisfaction', 'RB_Utilization', 'Eff_Resource_Utilization', 'Params'])
        for name in ALGO_ORDER + [model_name]:
            if name not in results:
                continue
            r = results[name]
            p = total_params if name == model_name else 0
            w.writerow([name, f"{r['total_throughput']:.4f}", f"{r['avg_throughput']:.4f}",
                        f"{r['served_user_jain_fairness']:.4f}",
                        f"{r.get('all_user_jain_fairness', 0):.4f}",
                        f"{r.get('demand_satisfaction', 0):.4f}",
                        f"{r.get('rb_utilization', 0):.4f}",
                        f"{r.get('effective_resource_utilization', 0):.4f}", p])
    print(f"Saved: {path}")


def save_per_rb_model_comparison_csv(all_results, model_name, results_dir):
    path = os.path.join(results_dir, 'per_rb_model_comparison.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Algorithm', 'Total_Throughput', 'Served_User_Jain_Fairness',
                     'All_User_Jain_Fairness', 'Demand_Satisfaction',
                     'RB_Utilization', 'Eff_Resource_Utilization'])
        for name in ALGO_ORDER + [model_name]:
            if name not in all_results:
                continue
            r = all_results[name]
            w.writerow([name, f"{r['total_throughput']:.4f}",
                        f"{r['served_user_jain_fairness']:.4f}",
                        f"{r.get('all_user_jain_fairness', 0):.4f}",
                        f"{r.get('demand_satisfaction', 0):.4f}",
                        f"{r.get('rb_utilization', 0):.4f}",
                        f"{r.get('effective_resource_utilization', 0):.4f}"])
    print(f"Saved: {path}")


def save_hybrid_sweep_csv(sweep_results, results_dir):
    path = os.path.join(results_dir, 'hybrid_teacher_sweep.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Alpha', 'Total_Throughput', 'Served_User_Jain_Fairness',
                     'All_User_Jain_Fairness', 'Demand_Satisfaction',
                     'RB_Utilization', 'Eff_Resource_Utilization', 'Teacher_Match_Accuracy'])
        for r in sweep_results:
            w.writerow([r['alpha'], f"{r['total_throughput']:.4f}",
                        f"{r['served_user_jain_fairness']:.4f}",
                        f"{r.get('all_user_jain_fairness', 0):.4f}",
                        f"{r['demand_satisfaction']:.4f}",
                        f"{r.get('rb_utilization', 0):.4f}",
                        f"{r.get('effective_resource_utilization', 0):.4f}",
                        f"{r.get('teacher_match_accuracy', 0):.4f}"])
    print(f"Saved: {path}")


def save_teacher_comparison_csv(results, results_dir):
    path = os.path.join(results_dir, 'teacher_comparison.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Algorithm', 'Total_Throughput', 'Served_User_Jain_Fairness',
                     'All_User_Jain_Fairness', 'Demand_Satisfaction'])
        for name in ['WeightedGreedy', 'DemandAwarePF']:
            if name in results:
                r = results[name]
                w.writerow([name, f"{r['total_throughput']:.4f}",
                            f"{r['served_user_jain_fairness']:.4f}",
                            f"{r.get('all_user_jain_fairness', 0):.4f}",
                            f"{r.get('demand_satisfaction', 0):.4f}"])
    print(f"Saved: {path}")


def generate_visualizations(results, model_name, history, eval_env, args):
    plot_user_distribution(eval_env, args.results_dir)
    plot_throughput_comparison(results, args.results_dir)
    plot_fairness_comparison(results, args.results_dir)
    plot_demand_satisfaction(results, args.results_dir)
    plot_rb_utilization(results, args.results_dir)
    plot_effective_resource_utilization(results, args.results_dir)
    plot_training_history(history, args.results_dir)

    allocs = {name: results[name]['allocation'] for name in results}
    plot_allocation_heatmap(allocs, args.num_users, args.num_rbs, args.results_dir)

    per_user = {name: results[name]['throughput'] for name in results}
    plot_per_user_throughput(per_user, args.results_dir)

    if 'predicted_scores' in results.get(model_name, {}):
        mlp_r = results[model_name]
        pred_scores = mlp_r['predicted_scores']
        teacher_scores = mlp_r['teacher_scores']
        if pred_scores.ndim == 2:
            plot_per_rb_score_heatmap(pred_scores, args.num_users, args.num_rbs, args.results_dir)
            plot_teacher_vs_mlp_scores(teacher_scores.flatten(), pred_scores.flatten(), args.results_dir)
        else:
            plot_teacher_vs_mlp_scores(teacher_scores, pred_scores, args.results_dir)


def run_single_seed(args, seed, device, teacher_override=None, hybrid_alpha_override=None):
    teacher = teacher_override or args.teacher
    hybrid_alpha = hybrid_alpha_override if hybrid_alpha_override is not None else args.hybrid_alpha

    print(f"\n{'='*60}")
    print(f"  SEED = {seed} | Model: {args.model_type} | Teacher: {teacher}")
    if teacher == 'hybrid':
        print(f"  Hybrid Alpha: {hybrid_alpha}")
    print(f"{'='*60}")

    env = SpectrumEnvironment(num_users=args.num_users, num_rbs=args.num_rbs, seed=seed)
    state = env.reset()
    print(f"State: {state.shape[1]} features")

    model, total_params = build_model(state.shape[1], model_type=args.model_type,
                                       model_size=args.model_size, dropout=0.1)

    print(f"\n===== Training =====")
    if args.model_type in IS_PER_RB:
        history = train_per_rb_model(
            model, env, num_epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, weight_decay=args.weight_decay,
            train_scenarios=args.train_scenarios, val_scenarios=args.val_scenarios,
            patience=args.patience, device=device, teacher=teacher,
            pf_lambda_demand=args.pf_lambda_demand, pf_beta=args.pf_beta,
            lambda_rank=args.lambda_rank, hybrid_alpha=hybrid_alpha,
        )
    else:
        history = train_model(
            model, env, num_epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, weight_decay=args.weight_decay,
            train_scenarios=args.train_scenarios, val_scenarios=args.val_scenarios,
            patience=args.patience, device=device, teacher=teacher,
            pf_lambda_demand=args.pf_lambda_demand, pf_beta=args.pf_beta,
            lambda_rank=args.lambda_rank,
        )

    print(f"\n===== Evaluation =====")
    eval_env = SpectrumEnvironment(num_users=args.num_users, num_rbs=args.num_rbs,
                                    seed=seed + 10000)
    eval_env.reset()
    results, model_name = evaluate_all_algorithms(
        model, eval_env, device, teacher=teacher,
        pf_lambda_demand=args.pf_lambda_demand, pf_beta=args.pf_beta,
        model_type=args.model_type, hybrid_alpha=hybrid_alpha,
    )
    print_results_table(results, model_name, total_params)

    return results, model_name, total_params, model, history, eval_env


def run_multi_seed(args, device):
    seeds = args.seeds
    first_results = None
    first_model_name = None
    first_params = 0
    first_model = None
    first_eval_env = None
    first_history = None
    all_results = {name: {k: [] for k in METRIC_KEYS} for name in ALGO_ORDER + [None]}

    for seed in seeds:
        results, model_name, total_params, model, history, eval_env = run_single_seed(args, seed, device)
        if first_results is None:
            first_results = results
            first_model_name = model_name
            first_params = total_params
            first_model = model
            first_eval_env = eval_env
            first_history = history
            all_results[model_name] = {k: [] for k in METRIC_KEYS}
        for name in ALGO_ORDER + [model_name]:
            if name in results:
                for k in METRIC_KEYS:
                    if k in results[name]:
                        all_results[name][k].append(results[name][k])

    print(f"\n{'='*60}")
    print(f"  MULTI-SEED SUMMARY (seeds={seeds})")
    print(f"{'='*60}")
    active = [n for n in ALGO_ORDER + [first_model_name]
              if n in all_results and len(all_results[n].get('total_throughput', [])) > 0]
    header = (f"{'Algorithm':<20} {'TP_mean':>12} {'TP_std':>12} "
              f"{'SrvF_mean':>12} {'SrvF_std':>12} {'AllF_mean':>12} {'AllF_std':>12} "
              f"{'DSat_mean':>12} {'DSat_std':>12}")
    print(header)
    print('-' * len(header))
    for name in active:
        tp = all_results[name]['total_throughput']
        sf = all_results[name]['served_user_jain_fairness']
        af = all_results[name].get('all_user_jain_fairness', [0])
        dd = all_results[name]['demand_satisfaction']
        if len(tp) > 0:
            print(f"{name:<20} {np.mean(tp):>12.2f} {np.std(tp):>12.2f} "
                  f"{np.mean(sf):>12.4f} {np.std(sf):>12.4f} "
                  f"{np.mean(af):>12.4f} {np.std(af):>12.4f} "
                  f"{np.mean(dd):>12.4f} {np.std(dd):>12.4f}")

    csv_path = os.path.join(args.results_dir, 'summary_multi_seed.csv')
    os.makedirs(args.results_dir, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Algorithm', 'TP_mean', 'TP_std',
                     'Served_Fairness_mean', 'Served_Fairness_std',
                     'All_Fairness_mean', 'All_Fairness_std',
                     'DemSat_mean', 'DemSat_std',
                     'RB_Util_mean', 'Eff_Util_mean', 'Seeds'])
        for name in active:
            tp = all_results[name]['total_throughput']
            sf = all_results[name]['served_user_jain_fairness']
            af = all_results[name].get('all_user_jain_fairness', [0])
            dd = all_results[name]['demand_satisfaction']
            rb = all_results[name].get('rb_utilization', [0])
            eu = all_results[name].get('effective_resource_utilization', [0])
            if len(tp) > 0:
                w.writerow([name, f"{np.mean(tp):.4f}", f"{np.std(tp):.4f}",
                            f"{np.mean(sf):.4f}", f"{np.std(sf):.4f}",
                            f"{np.mean(af):.4f}", f"{np.std(af):.4f}",
                            f"{np.mean(dd):.4f}", f"{np.std(dd):.4f}",
                            f"{np.mean(rb):.4f}", f"{np.mean(eu):.4f}", str(seeds)])
    print(f"Saved: {csv_path}")

    multi_data = {'algos': active}
    for m in ['total_throughput', 'served_user_jain_fairness',
              'all_user_jain_fairness', 'demand_satisfaction']:
        multi_data[m] = {
            'mean': {n: np.mean(all_results[n][m]) for n in active if m in all_results[n] and len(all_results[n][m]) > 0},
            'std': {n: np.std(all_results[n][m]) for n in active if m in all_results[n] and len(all_results[n][m]) > 0},
        }
    plot_multi_seed_summary(multi_data, args.results_dir)

    if first_results is not None and first_history is not None and first_eval_env is not None:
        print("\n===== Generating Visualizations =====")
        generate_visualizations(first_results, first_model_name, first_history, first_eval_env, args)
        save_summary_csv(first_results, first_model_name, first_params, args.results_dir)
        save_per_rb_model_comparison_csv(first_results, first_model_name, args.results_dir)
        save_teacher_comparison_csv(first_results, args.results_dir)

    return first_results, first_model_name, first_params, first_model, all_results


def run_alpha_sweep(args, device):
    alpha_list = args.hybrid_alpha_list
    sweep_results = []
    seed = args.seeds[0] if args.seeds else 0

    for alpha in alpha_list:
        print(f"\n{'#'*60}")
        print(f"  ALPHA SWEEP: alpha = {alpha}")
        print(f"{'#'*60}")

        results, model_name, total_params, model, history, eval_env = run_single_seed(
            args, seed, device, teacher_override='hybrid', hybrid_alpha_override=alpha)

        r = results[model_name]
        sweep_results.append({
            'alpha': alpha,
            'total_throughput': r['total_throughput'],
            'served_user_jain_fairness': r['served_user_jain_fairness'],
            'all_user_jain_fairness': r.get('all_user_jain_fairness', 0),
            'demand_satisfaction': r['demand_satisfaction'],
            'rb_utilization': r.get('rb_utilization', 0),
            'effective_resource_utilization': r.get('effective_resource_utilization', 0),
            'teacher_match_accuracy': r.get('teacher_match_accuracy', 0),
        })

    plot_hybrid_tradeoff(sweep_results, args.results_dir)
    save_hybrid_sweep_csv(sweep_results, args.results_dir)

    print(f"\n{'='*60}")
    print(f"  ALPHA SWEEP SUMMARY")
    print(f"{'='*60}")
    header = f"{'Alpha':>6} {'TP(M)':>12} {'SrvFair':>12} {'AllFair':>12} {'DemSat':>12} {'MatchAcc':>12}"
    print(header)
    print('-' * len(header))
    for r in sweep_results:
        print(f"{r['alpha']:>6.2f} {r['total_throughput']/1e6:>12.2f} "
              f"{r['served_user_jain_fairness']:>12.4f} "
              f"{r.get('all_user_jain_fairness', 0):>12.4f} "
              f"{r['demand_satisfaction']:>12.4f} "
              f"{r['teacher_match_accuracy']:>12.4f}")

    return sweep_results


def generate_experiment_report(args, mc_all=None, pf_all=None, sweep_results=None):
    path = os.path.join(args.results_dir, 'experiment_report.md')
    os.makedirs(args.results_dir, exist_ok=True)

    lines = []
    lines.append('# PerRBDeepSets Experiment Report')
    lines.append('')
    lines.append('## 1. Experiment Setup')
    lines.append('')
    lines.append(f'- Model: PerRBDeepSets (405,249 params)')
    lines.append(f'- Users: {args.num_users}, RBs: {args.num_rbs}')
    lines.append(f'- Epochs: {args.epochs}, Seeds: {args.seeds}')
    lines.append(f'- Train scenarios: {args.train_scenarios}, Val: {args.val_scenarios}')
    lines.append(f'- PF lambda: {args.pf_lambda_demand}, PF beta: {args.pf_beta}')
    lines.append(f'- Lambda rank: {args.lambda_rank}')
    lines.append('')

    lines.append('## 2. Max-Channel Teacher Sanity Check')
    lines.append('')
    if mc_all:
        active = [n for n in ALGO_ORDER + ['PerRBDeepSets']
                  if n in mc_all and len(mc_all[n].get('total_throughput', [])) > 0]
        lines.append('| Algorithm | TP_mean (M) | TP_std | SrvFair | AllFair | DemSat |')
        lines.append('|-----------|-------------|--------|---------|---------|--------|')
        for name in active:
            tp = mc_all[name]['total_throughput']
            sf = mc_all[name]['served_user_jain_fairness']
            af = mc_all[name].get('all_user_jain_fairness', [0])
            dd = mc_all[name]['demand_satisfaction']
            if len(tp) > 0:
                lines.append(f'| {name} | {np.mean(tp)/1e6:.2f} | {np.std(tp)/1e6:.2f} | '
                             f'{np.mean(sf):.4f} | {np.mean(af):.4f} | '
                             f'{np.mean(dd):.4f} |')
    else:
        lines.append('(not run)')
    lines.append('')

    lines.append('## 3. DemandAwarePF Teacher')
    lines.append('')
    if pf_all:
        active = [n for n in ALGO_ORDER + ['PerRBDeepSets']
                  if n in pf_all and len(pf_all[n].get('total_throughput', [])) > 0]
        lines.append('| Algorithm | TP_mean (M) | TP_std | SrvFair | AllFair | DemSat |')
        lines.append('|-----------|-------------|--------|---------|---------|--------|')
        for name in active:
            tp = pf_all[name]['total_throughput']
            sf = pf_all[name]['served_user_jain_fairness']
            af = pf_all[name].get('all_user_jain_fairness', [0])
            dd = pf_all[name]['demand_satisfaction']
            if len(tp) > 0:
                lines.append(f'| {name} | {np.mean(tp)/1e6:.2f} | {np.std(tp)/1e6:.2f} | '
                             f'{np.mean(sf):.4f} | {np.mean(af):.4f} | '
                             f'{np.mean(dd):.4f} |')
    else:
        lines.append('(not run)')
    lines.append('')

    lines.append('## 4. Hybrid Alpha Sweep')
    lines.append('')
    if sweep_results:
        lines.append('| Alpha | TP (M) | SrvFair | AllFair | DemSat | Match Acc |')
        lines.append('|-------|--------|---------|---------|--------|-----------|')
        for r in sweep_results:
            lines.append(f'| {r["alpha"]:.2f} | {r["total_throughput"]/1e6:.2f} | '
                         f'{r["served_user_jain_fairness"]:.4f} | '
                         f'{r.get("all_user_jain_fairness", 0):.4f} | '
                         f'{r["demand_satisfaction"]:.4f} | '
                         f'{r["teacher_match_accuracy"]:.4f} |')
    else:
        lines.append('(not run)')
    lines.append('')

    lines.append('## 5. Conclusions')
    lines.append('')
    lines.append('- PerRBDeepSets produces per-user-per-RB scores, enabling RB-specific allocation.')
    lines.append('- The model architecture uses separate user/RB encoders with a pairwise scorer.')
    lines.append('- Allocation is pure argmax per RB (each RB independently selects the best user).')
    lines.append('- Results depend on teacher signal quality and model capacity.')
    lines.append('')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description='AI6G-SpectrumSim: AI-native 6G spectrum allocation simulation')
    parser.add_argument('--num_users', type=int, default=50,
                        help='Number of users in the simulation (default: 50)')
    parser.add_argument('--num_rbs', type=int, default=10,
                        help='Number of resource blocks (default: 10)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs (default: 300)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for AdamW (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training (default: 128)')
    parser.add_argument('--train_scenarios', type=int, default=1000,
                        help='Number of training scenarios (default: 1000)')
    parser.add_argument('--val_scenarios', type=int, default=200,
                        help='Number of validation scenarios (default: 200)')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience (default: 30)')
    parser.add_argument('--model_size', type=str, default='medium',
                        choices=['small', 'medium', 'large'],
                        help='Model size preset (default: medium)')
    parser.add_argument('--model_type', type=str, default='per_rb_deepsets',
                        choices=['mlp', 'deepsets', 'per_rb_deepsets'],
                        help='Model architecture (default: per_rb_deepsets)')
    parser.add_argument('--teacher', type=str, default='max_channel',
                        choices=['weighted_greedy', 'pf', 'max_channel',
                                 'demandaware_pf', 'hybrid'],
                        help='Teacher algorithm for supervision (default: max_channel)')
    parser.add_argument('--pf_lambda_demand', type=float, default=1.0,
                        help='PF demand weight (default: 1.0)')
    parser.add_argument('--pf_beta', type=float, default=0.5,
                        help='PF fairness exponent (default: 0.5)')
    parser.add_argument('--lambda_rank', type=float, default=0.1,
                        help='Ranking loss weight (default: 0.1)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help='Random seeds for multi-seed experiments (default: 0 1 2 3 4)')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Output directory for results (default: results)')
    parser.add_argument('--hybrid_alpha', type=float, default=0.5,
                        help='Hybrid teacher alpha: 0=PF, 1=MaxChannel (default: 0.5)')
    parser.add_argument('--hybrid_alpha_list', type=float, nargs='+', default=None,
                        help='Alpha values for sweep, e.g. 0.0 0.25 0.5 0.75 1.0')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Config: users={args.num_users}, rbs={args.num_rbs}, epochs={args.epochs}")
    print(f"  model_type={args.model_type}, teacher={args.teacher}, seeds={args.seeds}")
    print(f"  train_scenarios={args.train_scenarios}, val_scenarios={args.val_scenarios}")
    print(f"  pf_lambda={args.pf_lambda_demand}, pf_beta={args.pf_beta}, lambda_rank={args.lambda_rank}")

    mc_all = None
    pf_all = None
    sweep_results = None

    if args.teacher == 'hybrid' and args.hybrid_alpha_list:
        sweep_results = run_alpha_sweep(args, device)
    elif len(args.seeds) > 1:
        _, _, _, _, mc_all = run_multi_seed(args, device)
    else:
        run_single_seed(args, args.seeds[0], device)

    generate_experiment_report(args, mc_all=mc_all, pf_all=pf_all, sweep_results=sweep_results)

    print("\n===== Done =====")


if __name__ == '__main__':
    main()
