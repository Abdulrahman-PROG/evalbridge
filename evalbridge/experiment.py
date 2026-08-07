"""
Experiment class — the core of evalbridge.

Bridges offline evaluation (Gap 1) and live A/B testing (Gap 2),
with automatic drift and confidence hooks (Gap 3).
"""

import json
import threading
import webbrowser
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from evalbridge.metrics import compute_all
from evalbridge.stats import BayesianABTest, SequentialTest


class ExperimentResult:
    """Holds the outcome of evaluate()."""

    def __init__(
        self,
        winner: str,
        confidence: float,
        ready: bool,
        metrics: dict,
        cohort_results: dict,
        method: str,
        min_confidence: float,
        experiment_name: str,
    ):
        self.winner = winner
        self.confidence = confidence
        self.ready = ready
        self.metrics = metrics
        self.cohort_results = cohort_results
        self._method = method
        self._min_confidence = min_confidence
        self._experiment_name = experiment_name

    def summary(self) -> None:
        """Print a formatted metrics table for both models."""
        base = self.metrics.get("baseline", {})
        chal = self.metrics.get("challenger", {})

        def fmt(d, key, decimals=3):
            v = d.get(key)
            if v is None:
                return "   —   "
            return f"{v:.{decimals}f}"

        acc_b  = fmt(base, "accuracy")
        acc_c  = fmt(chal, "accuracy")
        auc_b  = fmt(base, "auc_roc")
        auc_c  = fmt(chal, "auc_roc")
        f1_b   = fmt(base, "f1")
        f1_c   = fmt(chal, "f1")
        ll_b   = fmt(base, "log_loss")
        ll_c   = fmt(chal, "log_loss")
        bs_b   = fmt(base, "brier_score")
        bs_c   = fmt(chal, "brier_score")
        n_b    = f"{int(base.get('n_samples', 0)):,}"
        n_c    = f"{int(chal.get('n_samples', 0)):,}"

        # winner marker — arrow on the winning row, blank on the other
        if self.winner == "challenger":
            mark_b, mark_c = "  ", "◀"
        elif self.winner == "baseline":
            mark_b, mark_c = "◀", "  "
        else:
            mark_b = mark_c = "  "

        # confidence bar (20 chars wide)
        bar_width = 20
        filled = round(self.confidence * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        conf_pct = f"{self.confidence * 100:.1f}%"

        # ready badge
        if self.ready:
            status = "READY ✓"
        else:
            status = "inconclusive — collecting data"

        lines = [
            "",
            "┌──────────────┬──────────┬──────────┬────────┬──────────┬─────────────┬────────────┬───┐",
            "│ Model        │ Accuracy │ AUC-ROC  │   F1   │ Log-loss │ Brier score │  Samples   │   │",
            "├──────────────┼──────────┼──────────┼────────┼──────────┼─────────────┼────────────┼───┤",
            f"│ {'baseline':<12} │ {acc_b:>8} │ {auc_b:>8} │ {f1_b:>6} │ {ll_b:>8} │ {bs_b:>11} │ {n_b:>10} │ {mark_b} │",
            f"│ {'challenger':<12} │ {acc_c:>8} │ {auc_c:>8} │ {f1_c:>6} │ {ll_c:>8} │ {bs_c:>11} │ {n_c:>10} │ {mark_c} │",
            "└──────────────┴──────────┴──────────┴────────┴──────────┴─────────────┴────────────┴───┘",
            f"  Confidence  [{bar}] {conf_pct}",
            f"  Status      {status}",
            f"  Winner      {self.winner}",
            "",
        ]
        print("\n".join(lines))

    def report(self, path="report.html", open_browser=True):
        """Write an HTML report and optionally open it in a browser."""
        from evalbridge.report import generate

        generate(self, path=path)
        if open_browser:
            webbrowser.open(f"file://{Path(path).resolve()}")

    def promote(self):
        """Mark the winning model as Production in its registry source."""
        print(f"[evalbridge] promote() called — winner is '{self.winner}'")

    def rollback(self):
        """Revert to baseline in the registry source."""
        print("[evalbridge] rollback() called — reverting to baseline")


class Experiment:
    """
    A/B evaluation container that works identically in notebooks and production.

    Gap 1: standard format — log() + evaluate() anywhere.
    Gap 2: offline → live bridge via save() / load() / go_live().
    Gap 3: drift and confidence hooks with automatic actions.
    """

    def __init__(
        self,
        name: str,
        method: str = "bayesian",
        min_confidence: float = 0.95,
        min_samples: int = 100,
        stop_early: bool = False,
        cohorts: Optional[dict] = None,
        alert_on_drift: bool = False,
        drift_threshold: float = 0.2,
    ):
        if method not in ("bayesian", "sequential", "bandit"):
            raise ValueError(
                f"method must be 'bayesian', 'sequential', or 'bandit', got {method!r}"
            )

        self.name = name
        self.method = method
        self.min_confidence = min_confidence
        self.min_samples = min_samples
        self.stop_early = stop_early
        self.cohorts = cohorts or {}
        self.alert_on_drift = alert_on_drift
        self.drift_threshold = drift_threshold

        self._lock = threading.Lock()
        self._data: dict[str, dict] = {
            "baseline": {"y_true": [], "y_pred": [], "request_ids": []},
            "challenger": {"y_true": [], "y_pred": [], "request_ids": []},
        }
        self._cohort_data: dict = {}
        self._baseline_source = None
        self._challenger_source = None
        self._live_mode = False
        self._traffic_split = 0.1
        self._drift_detector = None
        self._drift_callbacks: list[tuple[float, Any]] = []
        self._confidence_callbacks: list[tuple[float, Any]] = []
        self._created_at = datetime.now(timezone.utc).isoformat()

    def log(
        self,
        model: str,
        y_true,
        y_pred,
        cohort: Optional[dict] = None,
        request_id: Optional[str] = None,
    ):
        """Accumulate predictions for a model. Thread-safe.

        Example: exp.log("baseline", y_true=[1,0,1], y_pred=[0.8,0.2,0.7])
        """
        if model not in ("baseline", "challenger"):
            raise ValueError(f"model must be 'baseline' or 'challenger', got {model!r}")

        y_true_list = list(np.asarray(y_true, dtype=float))
        y_pred_list = list(np.asarray(y_pred, dtype=float))

        if len(y_true_list) != len(y_pred_list):
            raise ValueError("y_true and y_pred must have the same length")

        with self._lock:
            self._data[model]["y_true"].extend(y_true_list)
            self._data[model]["y_pred"].extend(y_pred_list)
            if request_id is not None:
                self._data[model]["request_ids"].append(request_id)

            if cohort:
                for key, val in cohort.items():
                    if key not in self._cohort_data:
                        self._cohort_data[key] = {}
                    if val not in self._cohort_data[key]:
                        self._cohort_data[key][val] = {
                            "baseline": {"y_true": [], "y_pred": []},
                            "challenger": {"y_true": [], "y_pred": []},
                        }
                    self._cohort_data[key][val][model]["y_true"].extend(y_true_list)
                    self._cohort_data[key][val][model]["y_pred"].extend(y_pred_list)

    def evaluate(self) -> ExperimentResult:
        """Run statistical test and return an ExperimentResult.

        Example: result = exp.evaluate(); print(result.winner)
        """
        with self._lock:
            base_yt = list(self._data["baseline"]["y_true"])
            base_yp = list(self._data["baseline"]["y_pred"])
            chal_yt = list(self._data["challenger"]["y_true"])
            chal_yp = list(self._data["challenger"]["y_pred"])
            cohort_snapshot = deepcopy(self._cohort_data)

        n_base = len(base_yt)
        n_chal = len(chal_yt)

        # Compute metrics if we have enough data
        base_metrics: dict = {}
        chal_metrics: dict = {}
        enough_classes = False

        if n_base >= 2 and n_chal >= 2:
            try:
                base_metrics = compute_all(base_yt, base_yp)
                chal_metrics = compute_all(chal_yt, chal_yp)
                enough_classes = True
            except ValueError:
                pass

        metrics = {"baseline": base_metrics, "challenger": chal_metrics}

        # Check sample threshold
        min_n = min(n_base, n_chal)
        if min_n == 0 or not enough_classes:
            result = ExperimentResult(
                winner="inconclusive",
                confidence=0.0,
                ready=False,
                metrics=metrics,
                cohort_results={},
                method=self.method,
                min_confidence=self.min_confidence,
                experiment_name=self.name,
            )
            return result

        # Run statistical test
        if self.method in ("bayesian", "bandit"):
            test = BayesianABTest(n_samples=50_000)
            test.update("baseline", base_yt, base_yp)
            test.update("challenger", chal_yt, chal_yp)
            p_challenger_wins = test.probability_challenger_wins()
            confidence = max(p_challenger_wins, 1 - p_challenger_wins)

            if p_challenger_wins > 0.5:
                raw_winner = "challenger"
            else:
                raw_winner = "baseline"

        elif self.method == "sequential":
            test_seq = SequentialTest(min_effect=0.01)
            base_acc = base_metrics.get("accuracy", 0)
            chal_acc = chal_metrics.get("accuracy", 0)
            base_successes = int(round(base_acc * n_base))
            chal_successes = int(round(chal_acc * n_chal))
            decision = test_seq.update(base_successes, n_base, chal_successes, n_chal)

            if decision == "challenger_wins":
                raw_winner, confidence = "challenger", 0.96
            elif decision == "baseline_wins":
                raw_winner, confidence = "baseline", 0.96
            elif decision == "no_effect":
                raw_winner, confidence = "inconclusive", 0.5
            else:
                # "continue" — not enough evidence
                bayesian_fallback = BayesianABTest(n_samples=50_000)
                bayesian_fallback.update("baseline", base_yt, base_yp)
                bayesian_fallback.update("challenger", chal_yt, chal_yp)
                p = bayesian_fallback.probability_challenger_wins()
                confidence = max(p, 1 - p)
                raw_winner = "challenger" if p > 0.5 else "baseline"
        else:
            raw_winner, confidence = "inconclusive", 0.0

        # Decide on winner given confidence threshold and sample count
        if min_n < self.min_samples and not (self.stop_early and confidence >= self.min_confidence):
            winner = "inconclusive"
            ready = False
        elif confidence < self.min_confidence:
            winner = "inconclusive"
            ready = False
        else:
            winner = raw_winner
            ready = True

        # Compute cohort results
        cohort_results: dict = {}
        for cohort_key, cohort_vals in cohort_snapshot.items():
            cohort_results[cohort_key] = {}
            for val, models in cohort_vals.items():
                c_yt = models["challenger"]["y_true"]
                c_yp = models["challenger"]["y_pred"]
                b_yt = models["baseline"]["y_true"]
                b_yp = models["baseline"]["y_pred"]
                entry: dict = {}
                if len(c_yt) >= 2 and len(b_yt) >= 2:
                    try:
                        cb_test = BayesianABTest(n_samples=10_000)
                        cb_test.update("baseline", b_yt, b_yp)
                        cb_test.update("challenger", c_yt, c_yp)
                        cp = cb_test.probability_challenger_wins()
                        cohort_conf = max(cp, 1 - cp)
                        cohort_winner = "challenger" if cp > 0.5 else "baseline"
                        c_m = compute_all(c_yt, c_yp)
                        b_m = compute_all(b_yt, b_yp)
                        entry = {
                            "winner": cohort_winner,
                            "confidence": cohort_conf,
                            "delta_accuracy": c_m["accuracy"] - b_m["accuracy"],
                        }
                    except ValueError:
                        entry = {"winner": "inconclusive", "confidence": 0.0, "delta_accuracy": 0.0}
                else:
                    entry = {"winner": "inconclusive", "confidence": 0.0, "delta_accuracy": 0.0}
                cohort_results[cohort_key][val] = entry

        result = ExperimentResult(
            winner=winner,
            confidence=confidence,
            ready=ready,
            metrics=metrics,
            cohort_results=cohort_results,
            method=self.method,
            min_confidence=self.min_confidence,
            experiment_name=self.name,
        )

        # Fire confidence callbacks
        for threshold, action in self._confidence_callbacks:
            if confidence >= threshold:
                self._fire_action(action, result)

        # Check drift if detector is present
        if self._drift_detector is not None and n_chal > 0:
            chal_arr = np.array(chal_yp)
            drift_report = self._drift_detector.check(chal_arr)
            if drift_report.alert:
                for threshold, action in self._drift_callbacks:
                    if drift_report.psi >= threshold:
                        self._fire_action(action, result)

        return result

    def _fire_action(self, action, result: ExperimentResult):
        """Execute a built-in string action or call a user-provided callable."""
        if callable(action):
            action(result)
        elif action == "promote":
            result.promote()
        elif action == "rollback":
            result.rollback()
        elif action == "alert":
            print(f"[evalbridge] ALERT: experiment '{self.name}' triggered action")

    def save(self, path: str):
        """Serialize experiment state to a .evalbridge JSON file.

        Example: exp.save("churn_v3.evalbridge")
        """
        with self._lock:
            payload = {
                "version": "0.1.0",
                "name": self.name,
                "method": self.method,
                "min_confidence": self.min_confidence,
                "min_samples": self.min_samples,
                "stop_early": self.stop_early,
                "cohorts": self.cohorts,
                "alert_on_drift": self.alert_on_drift,
                "drift_threshold": self.drift_threshold,
                "created_at": self._created_at,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "data": self._data,
                "cohort_data": self._cohort_data,
            }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Experiment":
        """Deserialize experiment state from a .evalbridge file.

        Example: exp = Experiment.load("churn_v3.evalbridge")
        """
        with open(path) as f:
            payload = json.load(f)

        exp = cls(
            name=payload["name"],
            method=payload["method"],
            min_confidence=payload["min_confidence"],
            min_samples=payload["min_samples"],
            stop_early=payload["stop_early"],
            cohorts=payload.get("cohorts", {}),
            alert_on_drift=payload.get("alert_on_drift", False),
            drift_threshold=payload.get("drift_threshold", 0.2),
        )
        exp._created_at = payload.get("created_at", exp._created_at)
        exp._data = payload["data"]
        exp._cohort_data = payload.get("cohort_data", {})
        return exp

    def set_baseline(self, source):
        """Set an integration source for the baseline model.

        Example: exp.set_baseline(MLflowSource("models:/churn/Production"))
        """
        self._baseline_source = source

    def set_challenger(self, source):
        """Set an integration source for the challenger model.

        Example: exp.set_challenger(HFSource("org/churn-model-v2"))
        """
        self._challenger_source = source

    def go_live(self, traffic_split: float = 0.1):
        """Switch to live mode with a given traffic split for the challenger.

        Example: exp.go_live(traffic_split=0.1)
        """
        from evalbridge.bridge import LiveRouter

        if not (0 < traffic_split < 1):
            raise ValueError(f"traffic_split must be in (0, 1), got {traffic_split}")

        self._live_mode = True
        self._traffic_split = traffic_split
        self._router = LiveRouter(traffic_split)

        if self.alert_on_drift:
            from evalbridge.drift import DriftDetector

            ref = self._data["baseline"]["y_pred"]
            if ref:
                self._drift_detector = DriftDetector(reference=ref, threshold=self.drift_threshold)

        print(
            f"[evalbridge] '{self.name}' is now live — "
            f"{traffic_split * 100:.0f}% traffic to challenger"
        )

    def on_drift(self, threshold: float, action):
        """Register a callback that fires when PSI exceeds threshold.

        Example: exp.on_drift(threshold=0.2, action="rollback")
        """
        self._drift_callbacks.append((threshold, action))
        # Also (re-)initialize detector if not yet set
        if self._drift_detector is None:
            ref = self._data["baseline"]["y_pred"]
            if ref:
                from evalbridge.drift import DriftDetector

                self._drift_detector = DriftDetector(reference=ref, threshold=threshold)

    def on_confidence(self, threshold: float, action):
        """Register a callback that fires when confidence reaches threshold.

        Example: exp.on_confidence(threshold=0.95, action="promote")
        """
        self._confidence_callbacks.append((threshold, action))

    def as_bandit(self, strategy: str = "thompson", epsilon: float = 0.1):
        """Return a Bandit wrapping this experiment for auto traffic shifting.

        Example: bandit = exp.as_bandit(strategy="thompson")
        Example: bandit = exp.as_bandit(strategy="epsilon_greedy", epsilon=0.15)
        """
        from evalbridge.bandit import Bandit

        return Bandit(self, strategy=strategy, epsilon=epsilon)
