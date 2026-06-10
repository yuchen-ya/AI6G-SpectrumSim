import numpy as np
import pytest
from src.environment import SpectrumEnvironment
from src.algorithms import (
    random_allocation, max_channel_allocation, greedy_allocation,
    weighted_greedy_allocation, demand_aware_pf_allocation,
)


class TestAlgorithms:
    @pytest.fixture
    def env(self):
        e = SpectrumEnvironment(num_users=20, num_rbs=5, seed=42)
        e.reset()
        return e

    def test_random_allocation_shape(self, env):
        rng = np.random.default_rng(0)
        alloc = random_allocation(env.num_users, env.num_rbs, rng)
        assert alloc.shape == (5,)
        assert np.all((alloc >= 0) & (alloc < env.num_users))

    def test_max_channel_allocation_shape(self, env):
        alloc = max_channel_allocation(env.snr_db_per_rb, env.num_rbs)
        assert alloc.shape == (5,)
        assert np.all((alloc >= 0) & (alloc < env.num_users))

    def test_greedy_allocation_shape(self, env):
        alloc = greedy_allocation(env.snr_db_per_rb, env.traffic_demand,
                                  env.capacity_per_rb, env.num_rbs)
        assert alloc.shape == (5,)
        assert np.all((alloc >= 0) & (alloc < env.num_users))

    def test_weighted_greedy_allocation_shape(self, env):
        alloc = weighted_greedy_allocation(env.snr_db_per_rb, env.traffic_demand,
                                           env.capacity_per_rb, env.num_rbs,
                                           history_throughput=env.history_throughput)
        assert alloc.shape == (5,)
        assert np.all((alloc >= 0) & (alloc < env.num_users))

    def test_demand_aware_pf_allocation_shape(self, env):
        alloc = demand_aware_pf_allocation(env.snr_db_per_rb, env.traffic_demand,
                                           env.capacity_per_rb, env.num_rbs,
                                           history_throughput=env.history_throughput)
        assert alloc.shape == (5,)
        assert np.all((alloc >= 0) & (alloc < env.num_users))

    def test_max_channel_picks_best_snr(self, env):
        alloc = max_channel_allocation(env.snr_db_per_rb, env.num_rbs)
        for rb in range(env.num_rbs):
            expected = int(np.argmax(env.snr_db_per_rb[:, rb]))
            assert alloc[rb] == expected

    def test_all_algorithms_produce_valid_throughput(self, env):
        rng = np.random.default_rng(0)
        allocations = {
            'random': random_allocation(env.num_users, env.num_rbs, rng),
            'max_channel': max_channel_allocation(env.snr_db_per_rb, env.num_rbs),
            'greedy': greedy_allocation(env.snr_db_per_rb, env.traffic_demand,
                                        env.capacity_per_rb, env.num_rbs),
            'weighted': weighted_greedy_allocation(env.snr_db_per_rb, env.traffic_demand,
                                                    env.capacity_per_rb, env.num_rbs,
                                                    history_throughput=env.history_throughput),
            'pf': demand_aware_pf_allocation(env.snr_db_per_rb, env.traffic_demand,
                                              env.capacity_per_rb, env.num_rbs,
                                              history_throughput=env.history_throughput),
        }
        for name, alloc in allocations.items():
            tp, _ = env.compute_throughput(alloc)
            assert np.sum(tp) > 0, f"{name} produced zero total throughput"
