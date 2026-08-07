import threading
import pytest
from evalbridge import track, log_outcome, register_experiment, Experiment
from evalbridge.bridge import LiveRouter
import evalbridge.decorators as _dec


def _fresh_exp(name="bridge_test"):
    exp = Experiment(name, min_samples=5)
    register_experiment(exp)
    return exp


# ── @track decorator ──────────────────────────────────────────────────────────


def test_track_stores_in_pending():
    _dec._pending.clear()
    exp = _fresh_exp("track_pending")

    @track(experiment="track_pending", model="baseline")
    def predict(x):
        return 0.8

    predict(1.0)
    rid = predict.last_request_id
    assert rid in _dec._pending
    assert _dec._pending[rid]["prediction"] == 0.8
    _dec._pending.clear()


def test_track_last_request_id_set():
    exp = _fresh_exp("track_rid")

    @track(experiment="track_rid", model="challenger")
    def predict(x):
        return 0.9

    predict(42)
    assert predict.last_request_id is not None
    _dec._pending.clear()


def test_log_outcome_matches_prediction():
    _dec._pending.clear()
    exp = _fresh_exp("log_outcome_test")

    @track(experiment="log_outcome_test", model="baseline")
    def predict(x):
        return 0.75

    predict(1.0)
    rid = predict.last_request_id
    log_outcome(request_id=rid, y_true=1)

    assert len(exp._data["baseline"]["y_true"]) == 1
    assert exp._data["baseline"]["y_true"][0] == 1.0
    assert exp._data["baseline"]["y_pred"][0] == 0.75
    _dec._pending.clear()


def test_log_outcome_unknown_request_id():
    with pytest.raises(KeyError, match="not found"):
        log_outcome("nonexistent-uuid-xxxx", y_true=1)


# ── LiveRouter ────────────────────────────────────────────────────────────────


def test_live_router_basic_split():
    router = LiveRouter(traffic_split=0.1)
    results = [router.route() for _ in range(100)]
    challenger_count = sum(1 for r in results if r == "challenger")
    # Should be approximately 10 (modulo 10)
    assert 5 <= challenger_count <= 15, f"Expected ~10 challenger routes, got {challenger_count}"


def test_live_router_only_valid_values():
    router = LiveRouter(traffic_split=0.2)
    for _ in range(200):
        assert router.route() in ("baseline", "challenger")


def test_live_router_current_split():
    router = LiveRouter(traffic_split=0.1)
    for _ in range(100):
        router.route()
    split = router.current_split()
    assert abs(split["baseline"] + split["challenger"] - 1.0) < 1e-9
    assert 0.05 <= split["challenger"] <= 0.20


def test_live_router_invalid_split():
    with pytest.raises(ValueError):
        LiveRouter(traffic_split=0.0)
    with pytest.raises(ValueError):
        LiveRouter(traffic_split=1.0)


def test_live_router_thread_safe():
    """1000 concurrent route() calls should not corrupt internal state."""
    router = LiveRouter(traffic_split=0.1)
    results = []
    lock = threading.Lock()

    def do_routes():
        for _ in range(100):
            r = router.route()
            with lock:
                results.append(r)

    threads = [threading.Thread(target=do_routes) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 1000
    for r in results:
        assert r in ("baseline", "challenger")
