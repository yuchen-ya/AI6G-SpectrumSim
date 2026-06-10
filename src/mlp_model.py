import torch
import torch.nn as nn

MODEL_SIZES = {
    'small': [128, 64],
    'medium': [512, 512, 256, 128],
    'large': [1024, 1024, 512, 256, 128],
}


class ScoringMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, dropout=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 256, 128]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def predict_scores(self, x):
        with torch.no_grad():
            return self.forward(x)


class DeepSetScheduler(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        enc_dim = 256
        concat_dim = enc_dim * 3
        self.scorer = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        B = x.shape[0]
        emb = self.encoder(x)
        g_mean = emb.mean(dim=0, keepdim=True).expand(B, -1)
        g_max = emb.max(dim=0, keepdim=True)[0].expand(B, -1)
        combined = torch.cat([emb, g_mean, g_max], dim=-1)
        return self.scorer(combined).squeeze(-1)

    def predict_scores(self, x):
        with torch.no_grad():
            return self.forward(x)


class PerRBDeepSets(nn.Module):
    """Per-user-per-RB pairwise scoring model for spectrum allocation.

    Uses separate encoders for user features and RB features, then computes
    a score for each (user, RB) pair. Allocation is done via argmax per RB.
    """
    def __init__(self, user_input_dim=12, rb_input_dim=2, dropout=0.1):
        super().__init__()
        self.user_encoder = nn.Sequential(
            nn.Linear(user_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.rb_encoder = nn.Sequential(
            nn.Linear(rb_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(256 + 64, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, user_features, rb_features):
        user_emb = self.user_encoder(user_features)
        rb_emb = self.rb_encoder(rb_features)
        N, M = user_emb.shape[0], rb_emb.shape[0]
        pairs = torch.cat([
            user_emb.unsqueeze(1).expand(-1, M, -1),
            rb_emb.unsqueeze(0).expand(N, -1, -1),
        ], dim=-1)
        scores = self.scorer(pairs.reshape(N * M, -1)).reshape(N, M)
        return scores

    def predict_scores(self, user_features, rb_features):
        with torch.no_grad():
            return self.forward(user_features, rb_features)


def build_model(input_dim, model_type='deepsets', model_size='medium', dropout=0.1,
                rb_input_dim=2):
    """Build and return a model instance with parameter count.

    Args:
        input_dim: Dimension of user feature vector.
        model_type: One of 'mlp', 'deepsets', 'per_rb_deepsets'.
        model_size: Preset size ('small', 'medium', 'large').
        dropout: Dropout probability.
        rb_input_dim: Dimension of RB feature vector (for per_rb_deepsets only).

    Returns:
        (model, total_params) tuple.
    """
    if model_type == 'mlp':
        hidden_dims = MODEL_SIZES.get(model_size, MODEL_SIZES['medium'])
        model = ScoringMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    elif model_type == 'deepsets':
        model = DeepSetScheduler(input_dim=input_dim, dropout=dropout)
    elif model_type == 'per_rb_deepsets':
        model = PerRBDeepSets(user_input_dim=input_dim, rb_input_dim=rb_input_dim, dropout=dropout)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.__class__.__name__} | Type: {model_type} | Params: {total_params:,}")
    assert total_params < 100_000_000, f"Model has {total_params:,} params (>100M limit)"
    return model, total_params
