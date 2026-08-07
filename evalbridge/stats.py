"""
Internal statistical engines for A/B testing.
Not exported from evalbridge.__init__ — use via Experiment.
"""

import numpy as np


class BayesianABTest:
    """
    Beta-Binomial conjugate model for comparing two classifiers.

    Models each model's TPR as a Beta distribution.
    Uses Monte Carlo sampling (n=50,000) to compute P(challenger > baseline).
    """

    def __init__(self, alpha_prior=1, beta_prior=1, n_samples=50_000):
        self._alpha_prior = alpha_prior
        self._beta_prior = beta_prior
        self._n_samples = n_samples
        self._alpha = {"baseline": alpha_prior, "challenger": alpha_prior}
        self._beta = {"baseline": beta_prior, "challenger": beta_prior}
        self._n = {"baseline": 0, "challenger": 0}

    def update(self, model_name, y_true, y_pred_proba, threshold=0.5):
        """Add observations for a model to the Beta posterior.

        Example: test.update("baseline", [1, 0, 1], [0.8, 0.2, 0.9])
        """
        if model_name not in ("baseline", "challenger"):
            raise ValueError(f"model_name must be 'baseline' or 'challenger', got {model_name!r}")

        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred_proba, dtype=float)

        if len(y_true) == 0:
            raise ValueError("y_true must not be empty")
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred_proba must have the same length")

        y_hat = (y_pred >= threshold).astype(float)
        successes = int(np.sum(y_hat == y_true))
        failures = len(y_true) - successes

        self._alpha[model_name] += successes
        self._beta[model_name] += failures
        self._n[model_name] += len(y_true)

    def probability_challenger_wins(self):
        """Return P(challenger accuracy > baseline accuracy) via Monte Carlo.

        Example: test.probability_challenger_wins() -> 0.94
        """
        rng = np.random.default_rng(42)
        baseline_samples = rng.beta(
            self._alpha["baseline"], self._beta["baseline"], self._n_samples
        )
        challenger_samples = rng.beta(
            self._alpha["challenger"], self._beta["challenger"], self._n_samples
        )
        return float(np.mean(challenger_samples > baseline_samples))

    def confidence_interval(self, model_name, ci=0.95):
        """Return (lower, upper) credible interval for model accuracy.

        Example: test.confidence_interval("baseline") -> (0.72, 0.88)
        """
        if model_name not in ("baseline", "challenger"):
            raise ValueError("model_name must be 'baseline' or 'challenger'")
        rng = np.random.default_rng(42)
        samples = rng.beta(self._alpha[model_name], self._beta[model_name], self._n_samples)
        lo = (1 - ci) / 2
        hi = 1 - lo
        return (float(np.quantile(samples, lo)), float(np.quantile(samples, hi)))

    def expected_loss(self):
        """Return expected loss for choosing each model as winner.

        Example: test.expected_loss() -> {"baseline": 0.03, "challenger": 0.001}
        """
        rng = np.random.default_rng(42)
        b_samples = rng.beta(self._alpha["baseline"], self._beta["baseline"], self._n_samples)
        c_samples = rng.beta(self._alpha["challenger"], self._beta["challenger"], self._n_samples)
        # loss of picking baseline = E[max(0, challenger - baseline)]
        loss_baseline = float(np.mean(np.maximum(0, c_samples - b_samples)))
        # loss of picking challenger = E[max(0, baseline - challenger)]
        loss_challenger = float(np.mean(np.maximum(0, b_samples - c_samples)))
        return {"baseline": loss_baseline, "challenger": loss_challenger}


class SequentialTest:
    """
    Sequential Probability Ratio Test for early stopping.

    Allows stopping before min_samples if evidence is strong,
    without inflating the Type I error rate.
    """

    def __init__(self, alpha=0.05, beta=0.2, min_effect=0.01):
        """
        Parameters
        ----------
        alpha     : Type I error rate (false positive)
        beta      : Type II error rate (false negative / 1 - power)
        min_effect: minimum detectable effect size (absolute difference in proportions)
        """
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not (0 < beta < 1):
            raise ValueError(f"beta must be in (0, 1), got {beta}")

        self.alpha = alpha
        self.beta = beta
        self.min_effect = min_effect

        # SPRT boundaries
        self._lower = np.log(beta / (1 - alpha))
        self._upper = np.log((1 - beta) / alpha)
        self._log_ratio = 0.0

    def update(
        self,
        baseline_successes,
        baseline_n,
        challenger_successes,
        challenger_n,
    ):
        """Update test with cumulative counts; return decision string.

        Returns
        -------
        "continue"         – not enough evidence yet
        "challenger_wins"  – challenger is reliably better
        "baseline_wins"    – baseline is reliably better
        "no_effect"        – difference is below min_effect
        """
        if baseline_n == 0 or challenger_n == 0:
            return "continue"

        p_b = baseline_successes / baseline_n
        p_c = challenger_successes / challenger_n

        if abs(p_c - p_b) < self.min_effect:
            # Accumulate evidence for H0 (no meaningful difference)
            # If both proportions are essentially the same, treat as no_effect
            # Use a simple check: if CI of difference overlaps zero by a wide margin
            delta = abs(p_c - p_b)
            if delta < self.min_effect / 2 and baseline_n + challenger_n > 100:
                return "no_effect"
            return "continue"

        # Log-likelihood ratio for observed proportions vs null (equal)
        p_null = (baseline_successes + challenger_successes) / (baseline_n + challenger_n)

        if p_null <= 0 or p_null >= 1:
            return "continue"

        eps = 1e-10
        p_b_safe = np.clip(p_b, eps, 1 - eps)
        p_c_safe = np.clip(p_c, eps, 1 - eps)
        p_null_safe = np.clip(p_null, eps, 1 - eps)

        ll_alt = (
            baseline_successes * np.log(p_b_safe / p_null_safe)
            + (baseline_n - baseline_successes) * np.log((1 - p_b_safe) / (1 - p_null_safe))
            + challenger_successes * np.log(p_c_safe / p_null_safe)
            + (challenger_n - challenger_successes) * np.log((1 - p_c_safe) / (1 - p_null_safe))
        )

        self._log_ratio = float(ll_alt)

        if self._log_ratio >= self._upper:
            return "challenger_wins" if p_c > p_b else "baseline_wins"
        if self._log_ratio <= self._lower:
            return "no_effect"

        return "continue"
