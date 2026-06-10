# Metrics

## Throughput Metrics

### Total Throughput
Sum of throughput across all users. Higher is better.

```
Total TP = Σ_i throughput_i
```

### Average Throughput
Mean throughput per user.

```
Avg TP = (Σ_i throughput_i) / N
```

## Fairness Metrics

### Served-User Jain Fairness

```
J(θ) = (Σ_{i∈Served} θ_i)² / (|Served| × Σ_{i∈Served} θ_i²)
```

where `Served = {i : throughput_i > 0}`.

**Range**: [1/|Served|, 1.0], where 1.0 means perfect equality among served users.

**Important caveat**: This metric only measures fairness among users that receive at least one resource block. It does not reflect fairness over all users. An algorithm that serves only 1 user would have a served-user fairness of 1.0, which is misleading.

### All-User Jain Fairness

```
J(θ) = (Σ_i θ_i)² / (N × Σ_i θ_i²)
```

where the sum is over ALL N users, and unserved users have θ_i = 0.

**Range**: [1/N, 1.0]. This metric penalizes algorithms that leave many users unserved.

**Recommendation**: Always report both metrics together. Served-user fairness shows how equally resources are distributed among served users; all-user fairness shows the overall system fairness.

## Demand Satisfaction

```
DemSat = Σ_i min(θ_i_norm, d_i) / Σ_i d_i
```

where `θ_i_norm = throughput_i / max(throughput)` and `d_i` is the traffic demand.

## Resource Utilization

### RB Utilization
Fraction of RBs assigned to a valid user. Should always be 1.0.

```
RB_Util = (# assigned RBs) / (total RBs)
```

### Effective Resource Utilization
Fraction of RBs that produce positive throughput.

```
Eff_Util = (# RBs with positive throughput) / (total RBs)
```
