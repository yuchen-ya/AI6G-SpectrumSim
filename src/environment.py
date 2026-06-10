import numpy as np


class SpectrumEnvironment:
    """Simulates a single-cell downlink spectrum allocation scenario.

    Generates random user positions, computes path loss, per-RB SNR, capacity,
    and provides metrics for evaluating allocation algorithms.
    """
    NUM_FEATURES = 12

    def __init__(self, num_users=50, num_rbs=10, area_size=100.0,
                 p_tx_dbm=43.0, n0_dbm=-100.0, bandwidth_mhz=10.0, seed=None):
        """Initialize environment parameters.

        Args:
            num_users: Number of users in the cell.
            num_rbs: Number of resource blocks.
            area_size: Cell area side length in meters.
            p_tx_dbm: Transmit power in dBm.
            n0_dbm: Noise power in dBm.
            bandwidth_mhz: Total bandwidth in MHz.
            seed: Random seed for reproducibility.
        """
        self.num_users = num_users
        self.num_rbs = num_rbs
        self.area_size = area_size
        self.p_tx_dbm = p_tx_dbm
        self.n0_dbm = n0_dbm
        self.bandwidth_mhz = bandwidth_mhz
        self.rng = np.random.default_rng(seed)

        self.user_positions = None
        self.bs_position = np.array([area_size / 2.0, area_size / 2.0])
        self.distances = None
        self.path_loss = None
        self.snr_db = None
        self.snr_linear = None
        self.snr_db_per_rb = None
        self.snr_linear_per_rb = None
        self.traffic_demand = None
        self.capacity_per_rb = None
        self.history_throughput = None
        self.bw_per_rb = (bandwidth_mhz * 1e6) / num_rbs

    def reset(self):
        self.user_positions = self.rng.uniform(0, self.area_size, (self.num_users, 2))
        self.distances = np.linalg.norm(self.user_positions - self.bs_position, axis=1)
        self.path_loss = 20.0 * np.log10(self.distances + 1.0) + self.rng.normal(0, 2, self.num_users)
        base_snr_db = self.p_tx_dbm - self.path_loss - self.n0_dbm
        freq_fading = self.rng.normal(0, 3, (self.num_users, self.num_rbs))
        self.snr_db = base_snr_db
        self.snr_db_per_rb = base_snr_db[:, None] + freq_fading
        self.snr_linear = 10.0 ** (self.snr_db / 10.0)
        self.snr_linear_per_rb = 10.0 ** (self.snr_db_per_rb / 10.0)
        self.traffic_demand = self.rng.uniform(0, 1, self.num_users)
        self.capacity_per_rb = self.bw_per_rb * np.log2(1.0 + self.snr_linear_per_rb)
        if self.history_throughput is None:
            self.history_throughput = np.zeros(self.num_users)
        return self._get_state()

    def _get_state(self):
        ht = self.history_throughput
        ht_max = np.max(ht) + 1e-9
        snr_norm = self.snr_db / 50.0
        fairness_bonus = 1.0 / (ht + 1e-6)
        fb_max = np.max(fairness_bonus) + 1e-9
        demand_channel_score = self.traffic_demand * snr_norm

        features = np.stack([
            self.user_positions[:, 0] / self.area_size,
            self.user_positions[:, 1] / self.area_size,
            self.distances / (self.area_size * 0.7071 + 1e-9),
            self.path_loss / 100.0,
            snr_norm,
            np.log10(self.snr_linear + 1e-9) / 10.0,
            self.traffic_demand,
            ht / ht_max,
            fairness_bonus / fb_max,
            demand_channel_score,
            self.traffic_demand,
            np.full(self.num_users, self.bw_per_rb / 1e6),
        ], axis=1)
        return features.astype(np.float32)

    def _get_rb_features(self):
        rb_idx = np.arange(self.num_rbs, dtype=np.float32) / max(self.num_rbs - 1, 1)
        mean_snr = np.mean(self.snr_db_per_rb, axis=0) / 50.0
        features = np.stack([rb_idx, mean_snr], axis=1)
        return features.astype(np.float32)

    def compute_throughput(self, allocation):
        throughput = np.zeros(self.num_users)
        rb_set_per_user = [[] for _ in range(self.num_users)]
        for rb_idx in range(self.num_rbs):
            user_idx = allocation[rb_idx]
            if 0 <= user_idx < self.num_users:
                throughput[user_idx] += self.capacity_per_rb[user_idx, rb_idx] / self.num_rbs
                rb_set_per_user[user_idx].append(rb_idx)
        self.history_throughput = 0.7 * self.history_throughput + 0.3 * throughput
        return throughput, rb_set_per_user

    def compute_demand_satisfaction(self, throughput):
        tp_norm = throughput / (np.max(throughput) + 1e-9)
        served = np.sum(np.minimum(tp_norm, self.traffic_demand))
        return float(served / (np.sum(self.traffic_demand) + 1e-9))

    def compute_rb_utilization(self, allocation):
        assigned = np.sum((allocation >= 0) & (allocation < self.num_users))
        return float(assigned / self.num_rbs)

    def compute_effective_resource_utilization(self, allocation, throughput):
        effective_rb = 0
        for rb_idx in range(self.num_rbs):
            user_idx = allocation[rb_idx]
            if 0 <= user_idx < self.num_users:
                if throughput[user_idx] > 0:
                    effective_rb += 1
        return float(effective_rb / self.num_rbs)

    @staticmethod
    def served_user_jain_fairness(throughput):
        """Jain fairness index computed over users with throughput > 0 only."""
        t = throughput[throughput > 0]
        if len(t) == 0:
            return 0.0
        return (np.sum(t) ** 2) / (len(t) * np.sum(t ** 2) + 1e-12)

    @staticmethod
    def all_user_jain_fairness(throughput):
        """Jain fairness index computed over ALL users (unserved users treated as 0)."""
        n = len(throughput)
        if n == 0:
            return 0.0
        return (np.sum(throughput) ** 2) / (n * np.sum(throughput ** 2) + 1e-12)
