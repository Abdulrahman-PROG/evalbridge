import pytest
import numpy as np
from evalbridge.stats import BayesianABTest, SequentialTest

# ── BayesianABTest ───────────────────────────────────────────────────────────


def test_bayesian_equal_models_near_half():
    """P(challenger > baseline) ≈ 0.5 when both models have identical accuracy."""
    test = BayesianABTest(n_samples=50_000)
    n = 500
    y_true = [1, 0] * (n // 2)
    y_pred = [0.8, 0.2] * (n // 2)
    test.update("baseline", y_true, y_pred)
    test.update("challenger", y_true, y_pred)
    p = test.probability_challenger_wins()
    assert 0.4 < p < 0.6, f"Expected ~0.5, got {p}"


def test_bayesian_challenger_clearly_better():
    """P > 0.9 when challenger has +10% accuracy."""
    test = BayesianABTest(n_samples=50_000)
    n = 500
    y_true = [1] * n

    # baseline: 70% accurate
    base_pred = [0.9] * int(0.7 * n) + [0.1] * int(0.3 * n)
    test.update("baseline", y_true[: len(base_pred)], base_pred)

    # challenger: 80% accurate
    chal_pred = [0.9] * int(0.8 * n) + [0.1] * int(0.2 * n)
    test.update("challenger", y_true[: len(chal_pred)], chal_pred)

    p = test.probability_challenger_wins()
    assert p > 0.9, f"Expected p > 0.9, got {p}"


def test_bayesian_confidence_interval_reasonable():
    test = BayesianABTest()
    y_true = [1, 0] * 100
    y_pred = [0.9, 0.1] * 100
    test.update("baseline", y_true, y_pred)
    lo, hi = test.confidence_interval("baseline")
    assert 0 < lo < hi < 1
    assert hi - lo < 0.3


def test_bayesian_expected_loss_keys():
    test = BayesianABTest()
    test.update("baseline", [1, 0], [0.8, 0.2])
    test.update("challenger", [1, 0], [0.9, 0.1])
    loss = test.expected_loss()
    assert "baseline" in loss
    assert "challenger" in loss
    assert loss["baseline"] >= 0
    assert loss["challenger"] >= 0


def test_bayesian_invalid_model_name():
    test = BayesianABTest()
    with pytest.raises(ValueError, match="baseline.*challenger"):
        test.update("control", [1, 0], [0.8, 0.2])


def test_bayesian_empty_data():
    test = BayesianABTest()
    with pytest.raises(ValueError, match="empty"):
        test.update("baseline", [], [])


# ── SequentialTest ───────────────────────────────────────────────────────────


def test_sequential_continue_weak_evidence():
    """Returns 'continue' when evidence is too weak to decide."""
    test = SequentialTest()
    result = test.update(5, 10, 6, 10)
    assert result == "continue"


def test_sequential_challenger_wins_clearly_better():
    """challenger_wins when challenger has clearly higher accuracy."""
    test = SequentialTest(alpha=0.05, beta=0.2, min_effect=0.01)
    # challenger: 90/100, baseline: 60/100
    result = test.update(60, 100, 90, 100)
    assert result == "challenger_wins"


def test_sequential_baseline_wins_clearly_better():
    """baseline_wins when baseline has clearly higher accuracy."""
    test = SequentialTest(alpha=0.05, beta=0.2, min_effect=0.01)
    result = test.update(90, 100, 60, 100)
    assert result == "baseline_wins"


def test_sequential_zero_samples_continue():
    """Returns 'continue' when no samples yet."""
    test = SequentialTest()
    assert test.update(0, 0, 0, 0) == "continue"


def test_sequential_invalid_alpha():
    with pytest.raises(ValueError, match="alpha"):
        SequentialTest(alpha=0.0)


def test_sequential_invalid_beta():
    with pytest.raises(ValueError, match="beta"):
        SequentialTest(beta=1.5)


def test_sequential_no_effect_identical():
    """no_effect when both models have nearly identical performance over many samples."""
    test = SequentialTest(min_effect=0.05)
    # Both 75% accurate over 1000 samples each
    result = test.update(750, 1000, 752, 1000)
    assert result in ("no_effect", "continue")
