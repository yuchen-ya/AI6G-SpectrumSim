import numpy as np


def random_allocation(num_users, num_rbs, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.integers(0, num_users, size=num_rbs)


def max_channel_allocation(snr_db_per_rb, num_rbs):
    allocation = np.full(num_rbs, -1, dtype=int)
    for rb in range(num_rbs):
        allocation[rb] = int(np.argmax(snr_db_per_rb[:, rb]))
    return allocation


def greedy_allocation(snr_db_per_rb, traffic_demand, capacity_per_rb, num_rbs):
    num_users = snr_db_per_rb.shape[0]
    allocation = np.full(num_rbs, -1, dtype=int)
    assigned_throughput = np.zeros(num_users)
    demand_copy = traffic_demand.copy()
    for rb in range(num_rbs):
        scores = snr_db_per_rb[:, rb] * demand_copy
        user_idx = int(np.argmax(scores))
        allocation[rb] = user_idx
        assigned_throughput[user_idx] += capacity_per_rb[user_idx, rb] / num_rbs
        demand_copy[user_idx] = max(0.0, traffic_demand[user_idx] - assigned_throughput[user_idx] / (np.max(assigned_throughput) + 1e-9))
    return allocation


def weighted_greedy_allocation(snr_db_per_rb, traffic_demand, capacity_per_rb,
                               num_rbs, history_throughput=None,
                               alpha=0.45, beta=0.30, gamma=0.20, delta=0.05,
                               max_rb_per_user=None):
    num_users = snr_db_per_rb.shape[0]
    if max_rb_per_user is None:
        max_rb_per_user = max(1, int(np.ceil(num_rbs / 2)))
    if history_throughput is None:
        history_throughput = np.zeros(num_users)

    allocation = np.full(num_rbs, -1, dtype=int)
    rb_count = np.zeros(num_users, dtype=int)

    snr_norm = snr_db_per_rb / (np.max(np.abs(snr_db_per_rb)) + 1e-9)
    ht_norm = history_throughput / (np.max(history_throughput) + 1e-9)
    fairness_bonus = 1.0 / (ht_norm + 0.1)
    fb_norm = fairness_bonus / (np.max(fairness_bonus) + 1e-9)

    for rb in range(num_rbs):
        scores = (alpha * snr_norm[:, rb]
                  + beta * traffic_demand
                  + gamma * fb_norm
                  - delta * ht_norm)
        sorted_indices = np.argsort(-scores)
        assigned = False
        for user_idx in sorted_indices:
            if rb_count[user_idx] < max_rb_per_user:
                allocation[rb] = user_idx
                rb_count[user_idx] += 1
                assigned = True
                break
        if not assigned:
            allocation[rb] = int(np.argmax(snr_norm[:, rb]))
    return allocation


def weighted_greedy_scores(snr_db_per_rb, traffic_demand, history_throughput=None,
                           alpha=0.45, beta=0.30, gamma=0.20, delta=0.05):
    num_users = snr_db_per_rb.shape[0]
    if history_throughput is None:
        history_throughput = np.zeros(num_users)
    snr_norm = snr_db_per_rb / (np.max(np.abs(snr_db_per_rb)) + 1e-9)
    ht_norm = history_throughput / (np.max(history_throughput) + 1e-9)
    fairness_bonus = 1.0 / (ht_norm + 0.1)
    fb_norm = fairness_bonus / (np.max(fairness_bonus) + 1e-9)
    avg_snr_norm = np.mean(snr_norm, axis=1)
    scores = (alpha * avg_snr_norm
              + beta * traffic_demand
              + gamma * fb_norm
              - delta * ht_norm)
    return scores


def demand_aware_pf_scores(snr_db_per_rb, traffic_demand, capacity_per_rb,
                           history_throughput=None,
                           lambda_demand=1.0, beta=0.5, epsilon=1e-6):
    num_users = snr_db_per_rb.shape[0]
    if history_throughput is None:
        history_throughput = np.zeros(num_users)
    avg_capacity = np.mean(capacity_per_rb, axis=1)
    scores = avg_capacity * (1.0 + lambda_demand * traffic_demand) / (history_throughput + epsilon) ** beta
    return scores


def max_channel_per_rb_scores(capacity_per_rb):
    return capacity_per_rb


def demand_aware_pf_per_rb_scores(snr_db_per_rb, traffic_demand, capacity_per_rb,
                                   history_throughput=None,
                                   lambda_demand=1.0, beta=0.5, epsilon=1e-6):
    num_users = len(traffic_demand)
    if history_throughput is None:
        history_throughput = np.zeros(num_users)
    scores = capacity_per_rb * (1.0 + lambda_demand * traffic_demand[:, None]) / (history_throughput[:, None] + epsilon) ** beta
    return scores


def demand_aware_pf_allocation(snr_db_per_rb, traffic_demand, capacity_per_rb,
                               num_rbs, history_throughput=None,
                               lambda_demand=1.0, beta=0.5, epsilon=1e-6,
                               max_rb_per_user=None):
    num_users = snr_db_per_rb.shape[0]
    if max_rb_per_user is None:
        max_rb_per_user = max(1, int(np.ceil(num_rbs / 2)))
    if history_throughput is None:
        history_throughput = np.zeros(num_users)

    allocation = np.full(num_rbs, -1, dtype=int)
    rb_count = np.zeros(num_users, dtype=int)

    for rb in range(num_rbs):
        rate = capacity_per_rb[:, rb]
        scores = rate * (1.0 + lambda_demand * traffic_demand) / (history_throughput + epsilon) ** beta
        sorted_indices = np.argsort(-scores)
        assigned = False
        for user_idx in sorted_indices:
            if rb_count[user_idx] < max_rb_per_user:
                allocation[rb] = user_idx
                rb_count[user_idx] += 1
                assigned = True
                break
        if not assigned:
            allocation[rb] = int(np.argmax(scores))
    return allocation


def hybrid_per_rb_scores(capacity_per_rb, snr_db_per_rb, traffic_demand,
                          history_throughput=None, alpha=0.5,
                          lambda_demand=1.0, beta=0.5, epsilon=1e-6):
    num_users = len(traffic_demand)
    if history_throughput is None:
        history_throughput = np.zeros(num_users)
    capacity_norm = capacity_per_rb / (np.max(np.abs(capacity_per_rb)) + 1e-9)
    pf_scores = demand_aware_pf_per_rb_scores(
        snr_db_per_rb, traffic_demand, capacity_per_rb,
        history_throughput, lambda_demand, beta, epsilon,
    )
    pf_norm = pf_scores / (np.max(np.abs(pf_scores)) + 1e-9)
    return alpha * capacity_norm + (1.0 - alpha) * pf_norm
