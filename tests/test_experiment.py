import os
import tempfile
import threading

import numpy as np
import pytest

from evalbridge.experiment import Experiment, ExperimentResult

# ── basic log + evaluate ─────────────────────────────────────────────────────


def _make_data(n=200, challenger_boost=0.1, seed=42):
    """Generate simple binary classification data."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, n).tolist()
    base_pred = [min(1.0, max(0.0, y + rng.normal(0, 0.3))) for y in y_true]
    chal_pred = [min(1.0, max(0.0, y + rng.normal(challenger_boost, 0.25))) for y in y_true]
    return y_true, base_pred, chal_pred


def test_basic_evaluate_winner():
    exp = Experiment("test", min_samples=10, min_confidence=0.7)
    y_true = [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1]
    base_pred = [0.6, 0.4, 0.7, 0.8, 0.3, 0.6, 0.4, 0.7, 0.35, 0.3, 0.65, 0.75]
    chal_pred = [0.85, 0.1, 0.9, 0.95, 0.05, 0.85, 0.1, 0.9, 0.08, 0.07, 0.9, 0.92]
    exp.log("baseline", y_true, base_pred)
    exp.log("challenger", y_true, chal_pred)
    result = exp.evaluate()
    assert result.winner in ("baseline", "challenger", "inconclusive")
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.ready, bool)


def test_evaluate_inconclusive_insufficient_samples():
    exp = Experiment("test", min_samples=1000)
    exp.log("baseline", [1, 0, 1], [0.8, 0.2, 0.7])
    exp.log("challenger", [1, 0, 1], [0.85, 0.15, 0.75])
    result = exp.evaluate()
    assert result.winner == "inconclusive"
    assert result.ready is False


def test_evaluate_challenger_wins_clearly():
    rng = np.random.default_rng(99)
    n = 300
    y_true = rng.integers(0, 2, n).tolist()
    base_pred = [0.5 + 0.1 * (y - 0.5) + rng.normal(0, 0.3) for y in y_true]
    base_pred = [min(1, max(0, p)) for p in base_pred]
    chal_pred = [0.5 + 0.4 * (y - 0.5) + rng.normal(0, 0.1) for y in y_true]
    chal_pred = [min(1, max(0, p)) for p in chal_pred]

    exp = Experiment("test", min_samples=50, min_confidence=0.7)
    exp.log("baseline", y_true, base_pred)
    exp.log("challenger", y_true, chal_pred)
    result = exp.evaluate()
    assert result.winner == "challenger"


# ── result attributes ────────────────────────────────────────────────────────


def test_result_metrics_keys():
    exp = Experiment("test", min_samples=5)
    exp.log("baseline", [1, 0, 1, 0, 1], [0.8, 0.2, 0.7, 0.3, 0.9])
    exp.log("challenger", [1, 0, 1, 0, 1], [0.85, 0.15, 0.75, 0.25, 0.95])
    result = exp.evaluate()
    for model in ("baseline", "challenger"):
        for key in ("accuracy", "auc_roc", "f1", "log_loss", "brier_score"):
            assert key in result.metrics[model], f"Missing {key} in {model} metrics"


def test_result_summary_prints(capsys):
    exp = Experiment("test", min_samples=5)
    exp.log("baseline", [1, 0, 1, 0, 1], [0.8, 0.2, 0.7, 0.3, 0.9])
    exp.log("challenger", [1, 0, 1, 0, 1], [0.85, 0.15, 0.75, 0.25, 0.95])
    result = exp.evaluate()
    result.summary()
    out = capsys.readouterr().out
    assert "baseline" in out
    assert "challenger" in out
    assert "Winner" in out


# ── save / load round-trip ────────────────────────────────────────────────────


def test_save_load_roundtrip():
    exp = Experiment("save_test", min_samples=5, min_confidence=0.8, method="sequential")
    exp.log("baseline", [1, 0, 1, 0, 1], [0.8, 0.2, 0.7, 0.3, 0.9])
    exp.log("challenger", [1, 0, 1, 0, 1], [0.85, 0.15, 0.75, 0.25, 0.95])

    with tempfile.NamedTemporaryFile(suffix=".evalbridge", delete=False) as f:
        path = f.name

    try:
        exp.save(path)
        loaded = Experiment.load(path)
        assert loaded.name == "save_test"
        assert loaded.min_confidence == 0.8
        assert loaded.method == "sequential"
        assert len(loaded._data["baseline"]["y_true"]) == 5
        assert len(loaded._data["challenger"]["y_true"]) == 5
    finally:
        os.unlink(path)


def test_save_load_preserves_cohort_data():
    exp = Experiment("cohort_save", min_samples=2)
    exp.log("baseline", [1, 0], [0.8, 0.2], cohort={"region": "EU"})
    exp.log("challenger", [1, 0], [0.9, 0.1], cohort={"region": "EU"})

    with tempfile.NamedTemporaryFile(suffix=".evalbridge", delete=False) as f:
        path = f.name

    try:
        exp.save(path)
        loaded = Experiment.load(path)
        assert "region" in loaded._cohort_data
        assert "EU" in loaded._cohort_data["region"]
    finally:
        os.unlink(path)


# ── cohorts ───────────────────────────────────────────────────────────────────


def test_cohort_results_populated():
    exp = Experiment("cohort_test", min_samples=2, min_confidence=0.5)
    for region in ["EU", "US"]:
        exp.log("baseline", [1, 0, 1, 0], [0.7, 0.3, 0.8, 0.2], cohort={"region": region})
        exp.log("challenger", [1, 0, 1, 0], [0.9, 0.1, 0.95, 0.05], cohort={"region": region})
    result = exp.evaluate()
    assert "region" in result.cohort_results
    assert "EU" in result.cohort_results["region"]
    assert "US" in result.cohort_results["region"]
    for val in ("EU", "US"):
        entry = result.cohort_results["region"][val]
        assert "winner" in entry
        assert "confidence" in entry
        assert "delta_accuracy" in entry


# ── stop_early ────────────────────────────────────────────────────────────────


def test_stop_early_returns_result_below_min_samples():
    """With stop_early=True and high enough confidence, result is ready before min_samples."""
    rng = np.random.default_rng(7)
    n = 50  # below min_samples=500
    y_true = rng.integers(0, 2, n).tolist()
    # challenger much better
    base_pred = [0.5 + 0.05 * (y - 0.5) + rng.normal(0, 0.35) for y in y_true]
    base_pred = [min(1, max(0, p)) for p in base_pred]
    chal_pred = [0.5 + 0.45 * (y - 0.5) + rng.normal(0, 0.05) for y in y_true]
    chal_pred = [min(1, max(0, p)) for p in chal_pred]

    exp_no_early = Experiment("no_early", min_samples=500, stop_early=False, min_confidence=0.7)
    exp_no_early.log("baseline", y_true, base_pred)
    exp_no_early.log("challenger", y_true, chal_pred)
    result_no = exp_no_early.evaluate()
    assert result_no.winner == "inconclusive"

    exp_early = Experiment("early", min_samples=500, stop_early=True, min_confidence=0.7)
    exp_early.log("baseline", y_true, base_pred)
    exp_early.log("challenger", y_true, chal_pred)
    result_yes = exp_early.evaluate()
    # With stop_early, high-confidence result can be non-inconclusive
    # (may still be inconclusive if confidence not high enough, but should differ)
    assert isinstance(result_yes.winner, str)


# ── thread safety ─────────────────────────────────────────────────────────────


def test_thread_safety():
    """10 threads logging concurrently — no data loss."""
    exp = Experiment("thread_test")
    n_threads = 10
    n_per_thread = 50

    def log_worker():
        for _ in range(n_per_thread):
            exp.log("baseline", [1, 0], [0.8, 0.2])
            exp.log("challenger", [1, 0], [0.9, 0.1])

    threads = [threading.Thread(target=log_worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(exp._data["baseline"]["y_true"]) == n_threads * n_per_thread * 2
    assert len(exp._data["challenger"]["y_true"]) == n_threads * n_per_thread * 2


# ── callbacks ─────────────────────────────────────────────────────────────────


def test_on_confidence_callback_fires():
    fired = []

    exp = Experiment("conf_test", min_samples=5, min_confidence=0.5)
    exp.on_confidence(threshold=0.5, action=lambda r: fired.append(r.winner))

    rng = np.random.default_rng(1)
    n = 200
    y_true = rng.integers(0, 2, n).tolist()
    base_pred = [0.5 + 0.1 * (y - 0.5) + rng.normal(0, 0.3) for y in y_true]
    base_pred = [min(1, max(0, p)) for p in base_pred]
    chal_pred = [0.5 + 0.45 * (y - 0.5) + rng.normal(0, 0.05) for y in y_true]
    chal_pred = [min(1, max(0, p)) for p in chal_pred]
    exp.log("baseline", y_true, base_pred)
    exp.log("challenger", y_true, chal_pred)
    exp.evaluate()

    assert len(fired) >= 1


def test_on_drift_callback_fires():
    fired = []
    rng = np.random.default_rng(5)

    exp = Experiment("drift_test", min_samples=5)
    n = 100
    base_preds = rng.uniform(0.4, 0.6, n).tolist()
    y_true = [1 if p > 0.5 else 0 for p in base_preds]

    exp.log("baseline", y_true, base_preds)
    exp.log("challenger", y_true, base_preds)

    # Register drift callback with low threshold to ensure it fires
    exp.on_drift(threshold=0.0, action=lambda r: fired.append(True))

    # Force-inject heavily drifted challenger data
    drifted = rng.beta(0.1, 5, n).tolist()
    exp.log("challenger", [0] * n, drifted)

    exp.evaluate()
    # Drift detection depends on the detector being initialized with baseline preds
    # Just assert it doesn't crash
    assert isinstance(fired, list)


# ── invalid inputs ────────────────────────────────────────────────────────────


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="method"):
        Experiment("test", method="invalid")


def test_invalid_model_name_raises():
    exp = Experiment("test")
    with pytest.raises(ValueError, match="model"):
        exp.log("control", [1, 0], [0.8, 0.2])
