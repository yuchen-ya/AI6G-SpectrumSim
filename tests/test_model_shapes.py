import numpy as np
import torch
import pytest
from src.mlp_model import build_model, PerRBDeepSets
from src.train import allocate_from_per_rb_scores


class TestModelShapes:
    def test_per_rb_deepsets_output_shape(self):
        model, params = build_model(12, model_type='per_rb_deepsets')
        user_feat = torch.randn(50, 12)
        rb_feat = torch.randn(10, 2)
        output = model(user_feat, rb_feat)
        assert output.shape == (50, 10), f"Expected (50, 10), got {output.shape}"

    def test_per_rb_deepsets_params_under_100m(self):
        _, params = build_model(12, model_type='per_rb_deepsets')
        assert params < 100_000_000
        assert params == 405249

    def test_mlp_output_shape(self):
        model, params = build_model(12, model_type='mlp')
        x = torch.randn(50, 12)
        output = model(x)
        assert output.shape == (50,)

    def test_deepsets_output_shape(self):
        model, params = build_model(12, model_type='deepsets')
        x = torch.randn(50, 12)
        output = model(x)
        assert output.shape == (50,)

    def test_per_rb_allocation_all_rbs_assigned(self):
        scores = np.random.randn(20, 5).astype(np.float32)
        alloc = allocate_from_per_rb_scores(scores, 20, 5)
        assert alloc.shape == (5,)
        assert np.all((alloc >= 0) & (alloc < 20))

    def test_per_rb_allocation_is_argmax(self):
        scores = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
        alloc = allocate_from_per_rb_scores(scores, 3, 2)
        assert alloc[0] == 0
        assert alloc[1] == 1

    def test_summary_metrics_computable(self):
        from src.environment import SpectrumEnvironment
        env = SpectrumEnvironment(num_users=20, num_rbs=5, seed=0)
        env.reset()
        alloc = np.array([0, 1, 2, 3, 4])
        tp, _ = env.compute_throughput(alloc)
        metrics = {
            'total_throughput': float(np.sum(tp)),
            'served_user_jain_fairness': float(env.served_user_jain_fairness(tp)),
            'all_user_jain_fairness': float(env.all_user_jain_fairness(tp)),
            'demand_satisfaction': float(env.compute_demand_satisfaction(tp)),
            'rb_utilization': float(env.compute_rb_utilization(alloc)),
            'effective_resource_utilization': float(env.compute_effective_resource_utilization(alloc, tp)),
        }
        assert metrics['total_throughput'] > 0
        assert 0.0 <= metrics['served_user_jain_fairness'] <= 1.0
        assert 0.0 <= metrics['all_user_jain_fairness'] <= 1.0
        assert metrics['all_user_jain_fairness'] <= metrics['served_user_jain_fairness']
        assert metrics['rb_utilization'] == 1.0

    def test_all_model_types_build(self):
        for mt in ['mlp', 'deepsets', 'per_rb_deepsets']:
            model, params = build_model(12, model_type=mt)
            assert params > 0
            assert params < 100_000_000
