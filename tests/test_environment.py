import numpy as np
import pytest
from src.environment import SpectrumEnvironment


class TestSpectrumEnvironment:
    def test_reset_state_shape(self):
        env = SpectrumEnvironment(num_users=50, num_rbs=10, seed=42)
        state = env.reset()
        assert state.shape == (50, 12)
        assert state.dtype == np.float32

    def test_snr_per_rb_shape(self):
        env = SpectrumEnvironment(num_users=50, num_rbs=10, seed=42)
        env.reset()
        assert env.snr_db_per_rb.shape == (50, 10)
        assert env.capacity_per_rb.shape == (50, 10)

    def test_compute_throughput_shape(self):
        env = SpectrumEnvironment(num_users=20, num_rbs=5, seed=0)
        env.reset()
        alloc = np.array([0, 1, 2, 3, 4])
        tp, rb_sets = env.compute_throughput(alloc)
        assert tp.shape == (20,)
        assert len(rb_sets) == 20

    def test_deterministic_with_seed(self):
        env1 = SpectrumEnvironment(num_users=10, num_rbs=5, seed=123)
        s1 = env1.reset()
        env2 = SpectrumEnvironment(num_users=10, num_rbs=5, seed=123)
        s2 = env2.reset()
        np.testing.assert_array_equal(s1, s2)

    def test_served_user_jain_fairness(self):
        tp = np.array([10.0, 10.0, 0.0])
        assert SpectrumEnvironment.served_user_jain_fairness(tp) == pytest.approx(1.0)
        tp2 = np.array([0.0, 0.0, 0.0])
        assert SpectrumEnvironment.served_user_jain_fairness(tp2) == 0.0

    def test_all_user_jain_fairness(self):
        tp = np.array([10.0, 10.0, 0.0])
        fair = SpectrumEnvironment.all_user_jain_fairness(tp)
        assert fair < 1.0
        assert fair > 0.0
        tp_equal = np.array([5.0, 5.0, 5.0])
        assert SpectrumEnvironment.all_user_jain_fairness(tp_equal) == pytest.approx(1.0)

    def test_demand_satisfaction_range(self):
        env = SpectrumEnvironment(num_users=10, num_rbs=5, seed=0)
        env.reset()
        alloc = np.array([0, 1, 2, 3, 4])
        tp, _ = env.compute_throughput(alloc)
        ds = env.compute_demand_satisfaction(tp)
        assert 0.0 <= ds <= 1.0

    def test_rb_utilization(self):
        env = SpectrumEnvironment(num_users=10, num_rbs=5, seed=0)
        env.reset()
        alloc = np.array([0, 1, 2, 3, 4])
        util = env.compute_rb_utilization(alloc)
        assert util == 1.0

    def test_effective_resource_utilization(self):
        env = SpectrumEnvironment(num_users=10, num_rbs=5, seed=0)
        env.reset()
        alloc = np.array([0, 1, 2, 3, 4])
        tp, _ = env.compute_throughput(alloc)
        eu = env.compute_effective_resource_utilization(alloc, tp)
        assert 0.0 <= eu <= 1.0

    def test_custom_params(self):
        env = SpectrumEnvironment(num_users=100, num_rbs=20, area_size=200.0, seed=42)
        state = env.reset()
        assert state.shape == (100, 12)
