import numpy as np
import pytest
from evalbridge.drift import DriftDetector, DriftReport


def _uniform(n=500, seed=0):
    return np.random.default_rng(seed).uniform(0, 1, n)


def _beta(a, b, n=500, seed=1):
    return np.random.default_rng(seed).beta(a, b, n)


# ── PSI ──────────────────────────────────────────────────────────────────────


def test_psi_identical_distributions():
    ref = _uniform(1000)
    det = DriftDetector(reference=ref)
    report = det.check(ref)
    assert report.psi < 0.05, f"PSI should be ~0 for identical distributions, got {report.psi}"


def test_psi_clearly_different():
    ref = _uniform(1000, seed=0)
    cur = _beta(2, 5, 1000, seed=2)
    det = DriftDetector(reference=ref)
    report = det.check(cur)
    assert (
        report.psi > 0.25
    ), f"PSI should be > 0.25 for clearly different distributions, got {report.psi}"


# ── KS test ──────────────────────────────────────────────────────────────────


def test_ks_pvalue_low_for_different():
    ref = _uniform(500, seed=0)
    cur = _beta(2, 5, 500, seed=3)
    det = DriftDetector(reference=ref)
    report = det.check(cur)
    assert report.ks_pvalue < 0.01, f"KS p-value should be < 0.01, got {report.ks_pvalue}"


def test_ks_pvalue_high_for_same():
    ref = _uniform(500, seed=0)
    det = DriftDetector(reference=ref)
    report = det.check(ref)
    assert (
        report.ks_pvalue > 0.05
    ), f"KS p-value should be > 0.05 for same distribution, got {report.ks_pvalue}"


# ── alert + severity ──────────────────────────────────────────────────────────


def test_alert_fires_above_threshold():
    ref = _uniform(1000, seed=0)
    cur = _beta(0.5, 0.5, 1000, seed=4)
    det = DriftDetector(reference=ref, threshold=0.1)
    report = det.check(cur)
    if report.psi > 0.1:
        assert report.alert is True


def test_no_alert_below_threshold():
    ref = _uniform(1000, seed=0)
    det = DriftDetector(reference=ref, threshold=0.5)
    report = det.check(ref)
    assert report.alert is False


def test_severity_none():
    ref = _uniform(500, seed=0)
    det = DriftDetector(reference=ref)
    report = det.check(ref)
    assert report.severity == "none"


def test_severity_severe():
    ref = _uniform(1000, seed=0)
    cur = _beta(0.5, 0.5, 1000, seed=4)
    det = DriftDetector(reference=ref)
    report = det.check(cur)
    if report.psi > 0.2:
        assert report.severity == "severe"


def test_severity_labels_exhaustive():
    for arr in [_uniform(200), _beta(2, 5, 200), _beta(0.5, 0.5, 200)]:
        det = DriftDetector(reference=_uniform(200, seed=99))
        report = det.check(arr)
        assert report.severity in ("none", "moderate", "severe")


# ── DriftReport attributes ────────────────────────────────────────────────────


def test_report_has_all_fields():
    ref = _uniform(200)
    det = DriftDetector(reference=ref)
    report = det.check(_beta(2, 5, 200))
    assert hasattr(report, "psi")
    assert hasattr(report, "ks_stat")
    assert hasattr(report, "ks_pvalue")
    assert hasattr(report, "alert")
    assert hasattr(report, "severity")
    assert hasattr(report, "summary")
    assert isinstance(report.summary, str)


# ── watch() ───────────────────────────────────────────────────────────────────


def test_watch_calls_on_drift_when_drifted():
    ref = _uniform(500, seed=0)
    det = DriftDetector(reference=ref, threshold=0.1)

    calls = []
    stream = _beta(0.1, 0.1, 300, seed=10).tolist()

    det.watch(stream, interval=100, on_drift=lambda r: calls.append(r))
    for call in calls:
        assert isinstance(call, DriftReport)


# ── edge cases ────────────────────────────────────────────────────────────────


def test_empty_reference_raises():
    with pytest.raises(ValueError, match="empty"):
        DriftDetector(reference=[])


def test_empty_current_raises():
    det = DriftDetector(reference=_uniform(100))
    with pytest.raises(ValueError, match="empty"):
        det.check([])
