import pytest
import numpy as np
from evalbridge.metrics import accuracy, auc_roc, f1, log_loss, brier_score, compute_all

# ── accuracy ────────────────────────────────────────────────────────────────


def test_accuracy_perfect():
    assert accuracy([1, 0, 1], [0.9, 0.1, 0.8]) == 1.0


def test_accuracy_all_wrong():
    assert accuracy([1, 1, 0], [0.1, 0.1, 0.9]) == 0.0


def test_accuracy_threshold():
    assert accuracy([1, 0], [0.4, 0.6], threshold=0.3) == 0.5


def test_accuracy_numpy():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.8, 0.2, 0.7, 0.3])
    assert accuracy(y, p) == 1.0


# ── auc_roc ─────────────────────────────────────────────────────────────────


def test_auc_perfect():
    assert auc_roc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0


def test_auc_random():
    # Random (shuffled) should be near 0.5
    rng = np.random.default_rng(42)
    y = np.array([1, 0] * 100)
    p = rng.random(200)
    auc = auc_roc(y, p)
    assert 0.4 < auc < 0.6


def test_auc_single_class_raises():
    with pytest.raises(ValueError, match="both classes"):
        auc_roc([1, 1, 1], [0.8, 0.7, 0.9])


# ── f1 ──────────────────────────────────────────────────────────────────────


def test_f1_perfect():
    assert f1([1, 0, 1], [0.9, 0.1, 0.8]) == 1.0


def test_f1_no_positives_predicted():
    # All predicted negative, all true positive → f1 = 0
    assert f1([1, 1, 1], [0.1, 0.1, 0.1]) == 0.0


def test_f1_known_value():
    # y=[1,1,0,0], pred>0.5=[1,1,1,0] -> TP=2, FP=1, FN=0 -> F1 = 2*2/(2*2+1+0) = 4/5 = 0.8
    result = f1([1, 1, 0, 0], [0.9, 0.8, 0.7, 0.1])
    assert abs(result - 0.8) < 1e-9


# ── log_loss ────────────────────────────────────────────────────────────────


def test_log_loss_perfect():
    # Near-perfect predictions → near 0
    ll = log_loss([1, 0], [0.9999, 0.0001])
    assert ll < 0.001


def test_log_loss_worst():
    ll = log_loss([1, 0], [0.0001, 0.9999])
    assert ll > 9.0


def test_log_loss_half():
    # p=0.5 always → log_loss = log(2) ≈ 0.693
    ll = log_loss([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5])
    assert abs(ll - np.log(2)) < 1e-6


# ── brier_score ─────────────────────────────────────────────────────────────


def test_brier_perfect():
    assert brier_score([1, 0], [1.0, 0.0]) == 0.0


def test_brier_worst():
    assert brier_score([1, 0], [0.0, 1.0]) == 1.0


def test_brier_half():
    # p=0.5 always → brier = 0.25
    assert abs(brier_score([1, 0, 1, 0], [0.5] * 4) - 0.25) < 1e-9


# ── compute_all ─────────────────────────────────────────────────────────────


def test_compute_all_keys():
    result = compute_all([1, 0, 1], [0.8, 0.2, 0.7])
    for key in ("accuracy", "auc_roc", "f1", "log_loss", "brier_score", "n_samples"):
        assert key in result


def test_compute_all_n_samples():
    result = compute_all([1, 0, 1, 0], [0.8, 0.2, 0.7, 0.3])
    assert result["n_samples"] == 4


# ── edge cases ───────────────────────────────────────────────────────────────


def test_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        accuracy([], [])


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        accuracy([1, 0], [0.9])


def test_nan_pred_raises():
    with pytest.raises(ValueError, match="NaN"):
        accuracy([1, 0], [0.9, float("nan")])


def test_out_of_range_pred_raises():
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        accuracy([1, 0], [1.5, 0.5])


def test_invalid_y_true_raises():
    with pytest.raises(ValueError, match="0 and 1"):
        accuracy([1, 2], [0.8, 0.5])
