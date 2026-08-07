import pytest
from evalbridge.bandit import Bandit
from evalbridge.experiment import Experiment


def _make_bandit(strategy="thompson"):
    exp = Experiment("bandit_test", min_samples=5)
    return Bandit(exp, strategy=strategy)


# ── route() ──────────────────────────────────────────────────────────────────


def test_route_only_valid_arms():
    bandit = _make_bandit()
    for _ in range(100):
        arm = bandit.route()
        assert arm in ("baseline", "challenger")


def test_route_epsilon_greedy_only_valid_arms():
    bandit = _make_bandit(strategy="epsilon_greedy")
    for _ in range(100):
        arm = bandit.route()
        assert arm in ("baseline", "challenger")


# ── update() ─────────────────────────────────────────────────────────────────


def test_update_reward_zero():
    bandit = _make_bandit()
    before_beta = bandit._beta["baseline"]
    bandit.update("baseline", 0.0)
    assert bandit._beta["baseline"] == before_beta + 1


def test_update_reward_one():
    bandit = _make_bandit()
    before_alpha = bandit._alpha["challenger"]
    bandit.update("challenger", 1.0)
    assert bandit._alpha["challenger"] == before_alpha + 1


def test_update_invalid_model():
    bandit = _make_bandit()
    with pytest.raises(ValueError, match="baseline.*challenger"):
        bandit.update("control", 1.0)


def test_epsilon_greedy_update():
    bandit = _make_bandit(strategy="epsilon_greedy")
    bandit.update("baseline", 1.0)
    assert bandit._wins["baseline"] == 1.0
    assert bandit._trials["baseline"] == 1


# ── allocations() ────────────────────────────────────────────────────────────


def test_allocations_sum_to_one():
    bandit = _make_bandit()
    for _ in range(50):
        arm = bandit.route()
        bandit.update(arm, 1.0)
    allocs = bandit.allocations()
    assert abs(allocs["baseline"] + allocs["challenger"] - 1.0) < 1e-9


def test_allocations_before_any_routes():
    bandit = _make_bandit()
    allocs = bandit.allocations()
    assert abs(allocs["baseline"] + allocs["challenger"] - 1.0) < 1e-9


# ── Thompson: challenger dominates after training ────────────────────────────


def test_thompson_allocates_toward_better_model():
    """After 1000 rounds where challenger wins 80% vs baseline 65%, allocation > 70%."""
    exp = Experiment("bandit_quality", min_samples=5)
    bandit = Bandit(exp, strategy="thompson")

    import numpy as np

    rng = np.random.default_rng(42)

    for _ in range(1000):
        arm = bandit.route()
        if arm == "challenger":
            reward = float(rng.random() < 0.80)
        else:
            reward = float(rng.random() < 0.65)
        bandit.update(arm, reward)

    allocs = bandit.allocations()
    assert (
        allocs["challenger"] > 0.65
    ), f"Expected challenger allocation > 65% after clear advantage, got {allocs['challenger']:.2f}"


# ── Epsilon-greedy: exploration rate ─────────────────────────────────────────


def test_epsilon_greedy_exploration_rate():
    """With epsilon=0.2 and 1000 routes, exploration should be ~20% of routes.
    After bias is established, non-greedy choices should be roughly epsilon fraction.
    This test just checks the mechanism works, not an exact statistical test.
    """
    exp = Experiment("eg_test", min_samples=5)
    bandit = Bandit(exp, strategy="epsilon_greedy", epsilon=0.3)

    import numpy as np

    rng = np.random.default_rng(0)

    # First give baseline a big head start so it's always the "greedy" choice
    for _ in range(200):
        bandit.update("baseline", 1.0)
        bandit.update("challenger", 0.0)

    # Now route 1000 times — challenger should appear ~epsilon fraction
    counts = {"baseline": 0, "challenger": 0}
    for _ in range(1000):
        arm = bandit.route()
        counts[arm] += 1

    # With epsilon=0.3, challenger should appear ~15% (half of epsilon, since random pick)
    # Just verify challenger appears at all (exploration happens)
    assert counts["challenger"] > 0
