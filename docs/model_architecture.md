# Model Architecture

## PerRBDeepSets

PerRBDeepSets is the primary model in this project. It produces a per-user-per-RB score matrix and allocates each RB independently via argmax.

### Architecture

```
User encoder (12 → 256):
    Linear(12, 256) → BatchNorm1d(256) → GELU → Dropout(0.1)
    Linear(256, 256) → BatchNorm1d(256) → GELU → Dropout(0.1)

RB encoder (2 → 64):
    Linear(2, 64) → BatchNorm1d(64) → GELU
    Linear(64, 64) → BatchNorm1d(64) → GELU

Pairwise scorer (320 → 1):
    Concatenate [user_emb(256), rb_emb(64)] → 320
    Linear(320, 512) → BN → GELU → Dropout
    Linear(512, 256) → BN → GELU → Dropout
    Linear(256, 128) → BN → GELU → Dropout
    Linear(128, 1)

Output:
    Score matrix: [num_users × num_rbs]
    Allocation: argmax per column (per-RB best user)
```

### Parameter Count

- User encoder: ~66K
- RB encoder: ~5K
- Pairwise scorer: ~334K
- **Total: 405,249 parameters**

### Key Design Decisions

1. **Separate encoders**: User and RB features have very different dimensionality (12 vs 2). Separate encoders allow each to learn appropriate representations.

2. **Pairwise scoring**: By computing a score for each (user, RB) pair, the model matches the per-RB allocation structure exactly.

3. **No sequential dependency**: Each RB is allocated independently. This avoids cascading errors and is trivially parallelizable.

## Comparison: DeepSetScheduler (legacy)

The earlier DeepSetScheduler produces a single per-user score:

```
Encoder: Input(12) → [256 → 256] (BN, GELU, Dropout)
Global pooling: mean + max over all users
Scorer: concat[user_emb, mean, max](768) → [512 → 256 → 128 → 1]
Output: per-user score → allocate via adjusted argmax
```

Limitation: A single score per user cannot capture that a user might be the best choice for some RBs but not others.

## Comparison: ScoringMLP (baseline)

```
Input(12) → [512 → 512 → 256 → 128] → score(1)
Output: per-user score
```

No global context (no pooling), no per-RB differentiation. Serves as a minimal neural baseline.
