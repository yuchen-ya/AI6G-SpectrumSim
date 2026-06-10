import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.algorithms import (
    weighted_greedy_scores, weighted_greedy_allocation,
    demand_aware_pf_scores, demand_aware_pf_allocation,
    max_channel_per_rb_scores, demand_aware_pf_per_rb_scores,
    hybrid_per_rb_scores, max_channel_allocation,
)


def generate_scenario_batch(env_class, env_kwargs, num_scenarios, seed=None,
                            teacher='pf', pf_lambda_demand=1.0, pf_beta=0.5):
    rng = np.random.default_rng(seed)
    all_states = []
    all_scores = []

    for i in range(num_scenarios):
        e = env_class(**env_kwargs)
        e.rng = np.random.default_rng(rng.integers(0, 2**31))
        e.reset()
        state = e._get_state()

        if teacher == 'pf':
            scores = demand_aware_pf_scores(
                e.snr_db_per_rb, e.traffic_demand, e.capacity_per_rb,
                e.history_throughput,
                lambda_demand=pf_lambda_demand, beta=pf_beta,
            )
        else:
            scores = weighted_greedy_scores(
                e.snr_db_per_rb, e.traffic_demand, e.history_throughput,
            )

        s_mean = np.mean(scores)
        s_std = np.std(scores) + 1e-9
        scores_norm = (scores - s_mean) / s_std

        all_states.append(state)
        all_scores.append(scores_norm)

    states = np.concatenate(all_states, axis=0).astype(np.float32)
    scores = np.concatenate(all_scores, axis=0).astype(np.float32)
    return states, scores


def allocate_from_scores(predicted_scores, num_users, num_rbs, max_rb_per_user=None):
    if max_rb_per_user is None:
        max_rb_per_user = max(1, int(np.ceil(num_rbs / 2)))

    allocation = np.full(num_rbs, -1, dtype=int)
    rb_count = np.zeros(num_users, dtype=int)

    for rb in range(num_rbs):
        adjusted = predicted_scores.copy() / (1.0 + rb_count)
        for u in range(num_users):
            if rb_count[u] >= max_rb_per_user:
                adjusted[u] = -np.inf
        best = int(np.argmax(adjusted))
        allocation[rb] = best
        rb_count[best] += 1

    if np.any(allocation < 0) or np.any(allocation >= num_users):
        for rb in range(num_rbs):
            if allocation[rb] < 0 or allocation[rb] >= num_users:
                allocation[rb] = int(np.argmax(predicted_scores))

    return allocation


def allocate_from_per_rb_scores(predicted_scores, num_users, num_rbs):
    return np.argmax(predicted_scores, axis=0).astype(int)


def pairwise_ranking_loss(pred, target, margin=0.1):
    n = pred.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=pred.device)

    pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
    target_diff = target.unsqueeze(1) - target.unsqueeze(0)
    labels = torch.sign(target_diff).detach()

    mask = labels != 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    loss = torch.clamp(-labels * pred_diff + margin, min=0)
    return loss[mask].mean()


def per_rb_pairwise_ranking_loss(pred, target, margin=0.1):
    pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
    target_diff = target.unsqueeze(1) - target.unsqueeze(0)
    labels = torch.sign(target_diff).detach()
    mask = labels != 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    loss = torch.clamp(-labels * pred_diff + margin, min=0)
    return loss[mask].mean()


class ScenarioDataset(torch.utils.data.Dataset):
    def __init__(self, states, scores, num_users):
        self.num_users = num_users
        self.num_scenarios = states.shape[0] // num_users
        self.all_states = torch.tensor(states, dtype=torch.float32)
        self.all_scores = torch.tensor(scores, dtype=torch.float32)

    def __len__(self):
        return self.num_scenarios

    def __getitem__(self, idx):
        start = idx * self.num_users
        end = start + self.num_users
        return self.all_states[start:end], self.all_scores[start:end]


class PerRBScenarioDataset(torch.utils.data.Dataset):
    def __init__(self, user_states, rb_features, scores, num_users, num_rbs):
        self.num_users = num_users
        self.num_rbs = num_rbs
        self.num_scenarios = user_states.shape[0] // num_users
        self.all_user_states = torch.tensor(user_states, dtype=torch.float32)
        self.all_rb_features = torch.tensor(rb_features, dtype=torch.float32)
        self.all_scores = torch.tensor(scores, dtype=torch.float32)

    def __len__(self):
        return self.num_scenarios

    def __getitem__(self, idx):
        us = idx * self.num_users
        rs = idx * self.num_rbs
        return (self.all_user_states[us:us + self.num_users],
                self.all_rb_features[rs:rs + self.num_rbs],
                self.all_scores[us:us + self.num_users])


def train_model(model, env, num_epochs=200, batch_size=128, lr=1e-3,
                weight_decay=1e-4, train_scenarios=1000, val_scenarios=200,
                patience=30, device='cpu', teacher='pf',
                pf_lambda_demand=1.0, pf_beta=0.5, lambda_rank=0.1):
    model.to(device)

    env_class = type(env)
    env_kwargs = dict(
        num_users=env.num_users, num_rbs=env.num_rbs,
        area_size=env.area_size, p_tx_dbm=env.p_tx_dbm,
        n0_dbm=env.n0_dbm, bandwidth_mhz=env.bandwidth_mhz,
    )

    print(f"Generating training data: {train_scenarios} scenarios ...")
    train_states, train_scores = generate_scenario_batch(
        env_class, env_kwargs, train_scenarios, seed=42,
        teacher=teacher, pf_lambda_demand=pf_lambda_demand, pf_beta=pf_beta,
    )
    print(f"  -> {train_states.shape[0]} samples ({train_scenarios} scenarios x {env.num_users} users)")

    print(f"Generating validation data: {val_scenarios} scenarios ...")
    val_states, val_scores = generate_scenario_batch(
        env_class, env_kwargs, val_scenarios, seed=9999,
        teacher=teacher, pf_lambda_demand=pf_lambda_demand, pf_beta=pf_beta,
    )
    print(f"  -> {val_states.shape[0]} samples")

    train_ds = ScenarioDataset(train_states, train_scores, env.num_users)
    val_ds = ScenarioDataset(val_states, val_scores, env.num_users)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)
    smooth_loss_fn = nn.SmoothL1Loss()

    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'total_throughput': [], 'avg_throughput': [],
        'served_user_jain_fairness': [], 'demand_satisfaction': [],
    }

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_states, batch_scores in train_loader:
            batch_states = batch_states.squeeze(0).to(device)
            batch_scores = batch_scores.squeeze(0).to(device)

            optimizer.zero_grad()
            pred = model(batch_states)

            loss_smooth = smooth_loss_fn(pred, batch_scores)
            loss_rank = pairwise_ranking_loss(pred, batch_scores)
            loss = loss_smooth + lambda_rank * loss_rank

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        train_avg = epoch_loss / max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for batch_states, batch_scores in val_loader:
                batch_states = batch_states.squeeze(0).to(device)
                batch_scores = batch_scores.squeeze(0).to(device)
                pred = model(batch_states)
                ls = smooth_loss_fn(pred, batch_scores)
                lr_val = pairwise_ranking_loss(pred, batch_scores)
                val_loss += (ls + lambda_rank * lr_val).item()
                val_n += 1
        val_avg = val_loss / max(val_n, 1)

        if val_avg < best_val_loss:
            best_val_loss = val_avg
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            metrics = _evaluate_on_env(model, env, device=device)
            history['epoch'].append(epoch)
            history['train_loss'].append(train_avg)
            history['val_loss'].append(val_avg)
            history['total_throughput'].append(metrics['total_throughput'])
            history['avg_throughput'].append(metrics['avg_throughput'])
            history['served_user_jain_fairness'].append(metrics['served_user_jain_fairness'])
            history['demand_satisfaction'].append(metrics['demand_satisfaction'])
            print(f"Epoch {epoch}/{num_epochs} | Loss: {train_avg:.4f} | Val: {val_avg:.4f} | "
                  f"TP: {metrics['total_throughput']:.1f} | Fair: {metrics['served_user_jain_fairness']:.4f} | "
                  f"DemSat: {metrics['demand_satisfaction']:.4f}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model (val_loss={best_val_loss:.4f})")

    return history


def generate_per_rb_scenario_batch(env_class, env_kwargs, num_scenarios, seed=None,
                                   teacher='demandaware_pf', pf_lambda_demand=1.0, pf_beta=0.5,
                                   hybrid_alpha=0.5):
    rng = np.random.default_rng(seed)
    all_user_states = []
    all_rb_features = []
    all_scores = []

    for i in range(num_scenarios):
        e = env_class(**env_kwargs)
        e.rng = np.random.default_rng(rng.integers(0, 2**31))
        e.reset()
        user_state = e._get_state()
        rb_feat = e._get_rb_features()

        if teacher == 'demandaware_pf':
            scores = demand_aware_pf_per_rb_scores(
                e.snr_db_per_rb, e.traffic_demand, e.capacity_per_rb,
                e.history_throughput,
                lambda_demand=pf_lambda_demand, beta=pf_beta,
            )
        elif teacher == 'max_channel':
            scores = max_channel_per_rb_scores(e.capacity_per_rb)
        elif teacher == 'hybrid':
            scores = hybrid_per_rb_scores(
                e.capacity_per_rb, e.snr_db_per_rb, e.traffic_demand,
                e.history_throughput, alpha=hybrid_alpha,
                lambda_demand=pf_lambda_demand, beta=pf_beta,
            )
        else:
            raise ValueError(f"Unknown teacher for per_rb: {teacher}")

        s_mean = np.mean(scores)
        s_std = np.std(scores) + 1e-9
        scores_norm = (scores - s_mean) / s_std

        all_user_states.append(user_state)
        all_rb_features.append(rb_feat)
        all_scores.append(scores_norm)

    user_states = np.concatenate(all_user_states, axis=0).astype(np.float32)
    rb_features = np.concatenate(all_rb_features, axis=0).astype(np.float32)
    scores = np.concatenate(all_scores, axis=0).astype(np.float32)
    return user_states, rb_features, scores


def train_per_rb_model(model, env, num_epochs=200, batch_size=128, lr=1e-3,
                       weight_decay=1e-4, train_scenarios=1000, val_scenarios=200,
                       patience=30, device='cpu', teacher='demandaware_pf',
                       pf_lambda_demand=1.0, pf_beta=0.5, lambda_rank=0.1,
                       hybrid_alpha=0.5):
    model.to(device)

    env_class = type(env)
    env_kwargs = dict(
        num_users=env.num_users, num_rbs=env.num_rbs,
        area_size=env.area_size, p_tx_dbm=env.p_tx_dbm,
        n0_dbm=env.n0_dbm, bandwidth_mhz=env.bandwidth_mhz,
    )

    print(f"Generating training data: {train_scenarios} scenarios ...")
    train_us, train_rf, train_sc = generate_per_rb_scenario_batch(
        env_class, env_kwargs, train_scenarios, seed=42,
        teacher=teacher, pf_lambda_demand=pf_lambda_demand, pf_beta=pf_beta,
        hybrid_alpha=hybrid_alpha,
    )
    print(f"  -> {train_us.shape[0]} user samples, {train_rf.shape[0]} RB samples")

    print(f"Generating validation data: {val_scenarios} scenarios ...")
    val_us, val_rf, val_sc = generate_per_rb_scenario_batch(
        env_class, env_kwargs, val_scenarios, seed=9999,
        teacher=teacher, pf_lambda_demand=pf_lambda_demand, pf_beta=pf_beta,
        hybrid_alpha=hybrid_alpha,
    )

    train_ds = PerRBScenarioDataset(train_us, train_rf, train_sc, env.num_users, env.num_rbs)
    val_ds = PerRBScenarioDataset(val_us, val_rf, val_sc, env.num_users, env.num_rbs)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)
    smooth_loss_fn = nn.SmoothL1Loss()

    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'total_throughput': [], 'avg_throughput': [],
        'served_user_jain_fairness': [], 'demand_satisfaction': [],
    }

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_us, batch_rf, batch_scores in train_loader:
            batch_us = batch_us.squeeze(0).to(device)
            batch_rf = batch_rf.squeeze(0).to(device)
            batch_scores = batch_scores.squeeze(0).to(device)

            optimizer.zero_grad()
            pred = model(batch_us, batch_rf)

            loss_smooth = smooth_loss_fn(pred, batch_scores)
            loss_rank = per_rb_pairwise_ranking_loss(pred, batch_scores)
            loss = loss_smooth + lambda_rank * loss_rank

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        train_avg = epoch_loss / max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for batch_us, batch_rf, batch_scores in val_loader:
                batch_us = batch_us.squeeze(0).to(device)
                batch_rf = batch_rf.squeeze(0).to(device)
                batch_scores = batch_scores.squeeze(0).to(device)
                pred = model(batch_us, batch_rf)
                ls = smooth_loss_fn(pred, batch_scores)
                lr_val = per_rb_pairwise_ranking_loss(pred, batch_scores)
                val_loss += (ls + lambda_rank * lr_val).item()
                val_n += 1
        val_avg = val_loss / max(val_n, 1)

        if val_avg < best_val_loss:
            best_val_loss = val_avg
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            metrics = _evaluate_per_rb_on_env(model, env, device=device)
            history['epoch'].append(epoch)
            history['train_loss'].append(train_avg)
            history['val_loss'].append(val_avg)
            history['total_throughput'].append(metrics['total_throughput'])
            history['avg_throughput'].append(metrics['avg_throughput'])
            history['served_user_jain_fairness'].append(metrics['served_user_jain_fairness'])
            history['demand_satisfaction'].append(metrics['demand_satisfaction'])
            print(f"Epoch {epoch}/{num_epochs} | Loss: {train_avg:.4f} | Val: {val_avg:.4f} | "
                  f"TP: {metrics['total_throughput']:.1f} | Fair: {metrics['served_user_jain_fairness']:.4f} | "
                  f"DemSat: {metrics['demand_satisfaction']:.4f}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model (val_loss={best_val_loss:.4f})")

    return history


def _evaluate_on_env(model, env, device='cpu'):
    model.eval()
    env_test = type(env)(
        num_users=env.num_users, num_rbs=env.num_rbs,
        area_size=env.area_size, p_tx_dbm=env.p_tx_dbm,
        n0_dbm=env.n0_dbm, bandwidth_mhz=env.bandwidth_mhz,
        seed=None,
    )
    env_test.reset()
    state = env_test._get_state()
    state_t = torch.tensor(state, dtype=torch.float32).to(device)
    scores = model.predict_scores(state_t).cpu().numpy()
    alloc = allocate_from_scores(scores, env.num_users, env.num_rbs)
    tp, _ = env_test.compute_throughput(alloc)
    return {
        'total_throughput': float(np.sum(tp)),
        'avg_throughput': float(np.mean(tp)),
        'served_user_jain_fairness': float(type(env).served_user_jain_fairness(tp)),
        'demand_satisfaction': float(env_test.compute_demand_satisfaction(tp)),
    }


def _evaluate_per_rb_on_env(model, env, device='cpu'):
    model.eval()
    env_test = type(env)(
        num_users=env.num_users, num_rbs=env.num_rbs,
        area_size=env.area_size, p_tx_dbm=env.p_tx_dbm,
        n0_dbm=env.n0_dbm, bandwidth_mhz=env.bandwidth_mhz,
        seed=None,
    )
    env_test.reset()
    user_state = env_test._get_state()
    rb_feat = env_test._get_rb_features()
    user_t = torch.tensor(user_state, dtype=torch.float32).to(device)
    rb_t = torch.tensor(rb_feat, dtype=torch.float32).to(device)
    scores = model.predict_scores(user_t, rb_t).cpu().numpy()
    alloc = allocate_from_per_rb_scores(scores, env.num_users, env.num_rbs)
    tp, _ = env_test.compute_throughput(alloc)
    return {
        'total_throughput': float(np.sum(tp)),
        'avg_throughput': float(np.mean(tp)),
        'served_user_jain_fairness': float(type(env).served_user_jain_fairness(tp)),
        'demand_satisfaction': float(env_test.compute_demand_satisfaction(tp)),
    }


def evaluate_model(model, env, device='cpu', teacher='pf',
                   pf_lambda_demand=1.0, pf_beta=0.5):
    model.eval()
    state = env._get_state()
    state_t = torch.tensor(state, dtype=torch.float32).to(device)
    scores = model.predict_scores(state_t).cpu().numpy()
    allocation = allocate_from_scores(scores, env.num_users, env.num_rbs)
    throughput, _ = env.compute_throughput(allocation)

    if teacher == 'pf':
        teacher_alloc = demand_aware_pf_allocation(
            env.snr_db_per_rb, env.traffic_demand, env.capacity_per_rb,
            env.num_rbs, history_throughput=env.history_throughput,
            lambda_demand=pf_lambda_demand, beta=pf_beta,
        )
        teacher_scores = demand_aware_pf_scores(
            env.snr_db_per_rb, env.traffic_demand, env.capacity_per_rb,
            env.history_throughput,
            lambda_demand=pf_lambda_demand, beta=pf_beta,
        )
    else:
        teacher_alloc = weighted_greedy_allocation(
            env.snr_db_per_rb, env.traffic_demand,
            env.capacity_per_rb, env.num_rbs,
            history_throughput=env.history_throughput,
        )
        teacher_scores = weighted_greedy_scores(
            env.snr_db_per_rb, env.traffic_demand, env.history_throughput,
        )

    n = len(scores)
    mlp_rank = np.argsort(np.argsort(-scores)).astype(float)
    teacher_rank = np.argsort(np.argsort(-teacher_scores)).astype(float)
    rank_corr = 1.0 - 6.0 * np.sum((mlp_rank - teacher_rank) ** 2) / (n * (n * n - 1) + 1e-12)

    served_mlp = set(allocation.tolist())
    served_teacher = set(teacher_alloc.tolist())
    jaccard = len(served_mlp & served_teacher) / (len(served_mlp | served_teacher) + 1e-12)

    rb_util = env.compute_rb_utilization(allocation)
    eff_util = env.compute_effective_resource_utilization(allocation, throughput)

    return {
        'total_throughput': float(np.sum(throughput)),
        'avg_throughput': float(np.mean(throughput)),
        'served_user_jain_fairness': float(env.served_user_jain_fairness(throughput)),
        'all_user_jain_fairness': float(env.all_user_jain_fairness(throughput)),
        'demand_satisfaction': float(env.compute_demand_satisfaction(throughput)),
        'rb_utilization': rb_util,
        'effective_resource_utilization': eff_util,
        'throughput': throughput,
        'allocation': allocation,
        'teacher_rank_correlation': float(rank_corr),
        'teacher_jaccard': float(jaccard),
        'predicted_scores': scores,
        'teacher_scores': teacher_scores,
    }


def evaluate_per_rb_model(model, env, device='cpu', teacher='demandaware_pf',
                          pf_lambda_demand=1.0, pf_beta=0.5, hybrid_alpha=0.5):
    model.eval()
    user_state = env._get_state()
    rb_feat = env._get_rb_features()
    user_t = torch.tensor(user_state, dtype=torch.float32).to(device)
    rb_t = torch.tensor(rb_feat, dtype=torch.float32).to(device)
    scores = model.predict_scores(user_t, rb_t).cpu().numpy()
    allocation = allocate_from_per_rb_scores(scores, env.num_users, env.num_rbs)
    throughput, _ = env.compute_throughput(allocation)

    if teacher == 'demandaware_pf':
        teacher_scores = demand_aware_pf_per_rb_scores(
            env.snr_db_per_rb, env.traffic_demand, env.capacity_per_rb,
            env.history_throughput,
            lambda_demand=pf_lambda_demand, beta=pf_beta,
        )
        teacher_alloc = demand_aware_pf_allocation(
            env.snr_db_per_rb, env.traffic_demand, env.capacity_per_rb,
            env.num_rbs, history_throughput=env.history_throughput,
            lambda_demand=pf_lambda_demand, beta=pf_beta,
        )
    elif teacher == 'max_channel':
        teacher_scores = max_channel_per_rb_scores(env.capacity_per_rb)
        teacher_alloc = max_channel_allocation(env.snr_db_per_rb, env.num_rbs)
    elif teacher == 'hybrid':
        teacher_scores = hybrid_per_rb_scores(
            env.capacity_per_rb, env.snr_db_per_rb, env.traffic_demand,
            env.history_throughput, alpha=hybrid_alpha,
            lambda_demand=pf_lambda_demand, beta=pf_beta,
        )
        teacher_alloc = np.argmax(teacher_scores, axis=0).astype(int)
    else:
        raise ValueError(f"Unknown teacher for per_rb: {teacher}")

    model_alloc_rb = np.argmax(scores, axis=0)
    teacher_alloc_rb = np.argmax(teacher_scores, axis=0)
    match_acc = float(np.sum(model_alloc_rb == teacher_alloc_rb)) / env.num_rbs

    pred_flat = scores.flatten()
    teacher_flat = teacher_scores.flatten()
    n = len(pred_flat)
    pred_rank = np.argsort(np.argsort(-pred_flat)).astype(float)
    teacher_rank = np.argsort(np.argsort(-teacher_flat)).astype(float)
    rank_corr = 1.0 - 6.0 * np.sum((pred_rank - teacher_rank) ** 2) / (n * (n * n - 1) + 1e-12)

    served_mlp = set(allocation.tolist())
    served_teacher = set(teacher_alloc.tolist())
    jaccard = len(served_mlp & served_teacher) / (len(served_mlp | served_teacher) + 1e-12)

    rb_util = env.compute_rb_utilization(allocation)
    eff_util = env.compute_effective_resource_utilization(allocation, throughput)

    return {
        'total_throughput': float(np.sum(throughput)),
        'avg_throughput': float(np.mean(throughput)),
        'served_user_jain_fairness': float(env.served_user_jain_fairness(throughput)),
        'all_user_jain_fairness': float(env.all_user_jain_fairness(throughput)),
        'demand_satisfaction': float(env.compute_demand_satisfaction(throughput)),
        'rb_utilization': rb_util,
        'effective_resource_utilization': eff_util,
        'throughput': throughput,
        'allocation': allocation,
        'teacher_rank_correlation': float(rank_corr),
        'teacher_jaccard': float(jaccard),
        'teacher_match_accuracy': match_acc,
        'predicted_scores': scores,
        'teacher_scores': teacher_scores,
    }
