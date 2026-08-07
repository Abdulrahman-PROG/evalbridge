"""
Multi-armed bandit for automatic traffic allocation between models.
"""

import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from evalbridge.experiment import Experiment


class Bandit:
    """
    Multi-armed bandit that learns which model performs better
    and shifts traffic toward it automatically.

    Strategies: 'thompson' (default) and 'epsilon_greedy'.
    """

    def __init__(
        self,
        experiment: "Experiment",
        strategy: str = "thompson",
        epsilon: float = 0.1,
    ):
        if strategy not in ("thompson", "epsilon_greedy"):
            raise ValueError(f"strategy must be 'thompson' or 'epsilon_greedy', got {strategy!r}")

        self._experiment = experiment
        self.strategy = strategy
        self.epsilon = epsilon
        self._lock = threading.Lock()
        self._rng = np.random.default_rng()

        # Thompson state: Beta(alpha, beta) for each arm
        self._alpha = {"baseline": 1, "challenger": 1}
        self._beta = {"baseline": 1, "challenger": 1}

        # Epsilon-greedy state: running win rate
        self._wins = {"baseline": 0, "challenger": 0}
        self._trials = {"baseline": 0, "challenger": 0}

        # For allocations() tracking
        self._route_counts = {"baseline": 0, "challenger": 0}
        self._total_routes = 0

    def route(self, features=None) -> str:
        """Choose which model to serve for the current request.

        Example: chosen = bandit.route(request_features)
        """
        arms = ("baseline", "challenger")

        with self._lock:
            if self.strategy == "thompson":
                samples = {arm: self._rng.beta(self._alpha[arm], self._beta[arm]) for arm in arms}
                chosen = max(arms, key=lambda a: samples[a])
            else:
                # Epsilon-greedy
                if self._rng.random() < self.epsilon:
                    chosen = arms[int(self._rng.integers(0, 2))]
                else:
                    rates = {
                        arm: (self._wins[arm] / self._trials[arm]) if self._trials[arm] > 0 else 0.5
                        for arm in arms
                    }
                    max_rate = max(rates.values())
                    best = [a for a in arms if rates[a] == max_rate]
                    chosen = best[int(self._rng.integers(0, len(best)))]

            self._route_counts[chosen] += 1
            self._total_routes += 1

        return chosen

    def update(self, model: str, reward: float):
        """Update bandit state after observing a reward.

        Parameters
        ----------
        model  : 'baseline' or 'challenger'
        reward : 1.0 for success, 0.0 for failure
        """
        if model not in ("baseline", "challenger"):
            raise ValueError(f"model must be 'baseline' or 'challenger', got {model!r}")
        if reward not in (0.0, 1.0) and not (0.0 <= reward <= 1.0):
            raise ValueError(f"reward must be in [0, 1], got {reward}")

        reward = float(reward)
        with self._lock:
            if self.strategy == "thompson":
                if reward >= 0.5:
                    self._alpha[model] += 1
                else:
                    self._beta[model] += 1
            else:
                self._wins[model] += reward
                self._trials[model] += 1

    def allocations(self) -> dict:
        """Return current traffic allocation percentage per arm.

        Example: bandit.allocations() -> {'baseline': 0.23, 'challenger': 0.77}
        """
        with self._lock:
            total = self._total_routes
            if total == 0:
                return {"baseline": 0.5, "challenger": 0.5}
            return {
                "baseline": self._route_counts["baseline"] / total,
                "challenger": self._route_counts["challenger"] / total,
            }

    def summary(self) -> str:
        """Print and return allocation table with confidence stats.

        Example: print(bandit.summary())
        """
        allocs = self.allocations()
        b_alloc = allocs["baseline"]
        c_alloc = allocs["challenger"]

        if self.strategy == "thompson":
            # Estimate win probability from Beta distribution
            rng = np.random.default_rng(42)
            b_samp = rng.beta(self._alpha["baseline"], self._beta["baseline"], 10_000)
            c_samp = rng.beta(self._alpha["challenger"], self._beta["challenger"], 10_000)
            p_c_wins = float(np.mean(c_samp > b_samp))
        else:
            b_rate = self._wins["baseline"] / max(1, self._trials["baseline"])
            c_rate = self._wins["challenger"] / max(1, self._trials["challenger"])
            p_c_wins = 1.0 if c_rate > b_rate else (0.5 if c_rate == b_rate else 0.0)

        lines = [
            f"Strategy: {self.strategy}",
            f"  baseline   : {b_alloc * 100:5.1f}%",
            f"  challenger : {c_alloc * 100:5.1f}%",
            f"  P(challenger > baseline): {p_c_wins:.3f}",
        ]
        text = "\n".join(lines)
        print(text)
        return text
