"""
LiveRouter for deterministic traffic splitting between baseline and challenger.
"""

import threading


class LiveRouter:
    """Routes requests between baseline and challenger in live mode.

    Uses deterministic modulo routing for reproducibility.
    Thread-safe via an internal lock.
    """

    def __init__(self, traffic_split: float):
        """
        Parameters
        ----------
        traffic_split : fraction of traffic routed to challenger (e.g. 0.1 = 10%)
        """
        if not (0 < traffic_split < 1):
            raise ValueError(f"traffic_split must be in (0, 1), got {traffic_split}")

        self.traffic_split = traffic_split
        self._call_count = 0
        self._lock = threading.Lock()
        self._modulo = max(1, round(1 / traffic_split))

    def route(self) -> str:
        """Return 'baseline' or 'challenger' based on modulo routing.

        At traffic_split=0.1: every 10th request goes to challenger.

        Example: model_name = router.route()
        """
        with self._lock:
            self._call_count += 1
            if self._call_count % self._modulo == 0:
                return "challenger"
            return "baseline"

    def current_split(self) -> dict:
        """Return actual observed split ratio so far.

        Example: router.current_split() -> {'baseline': 0.90, 'challenger': 0.10}
        """
        with self._lock:
            total = self._call_count
            if total == 0:
                return {"baseline": 1.0 - self.traffic_split, "challenger": self.traffic_split}
            challenger_count = total // self._modulo
            baseline_count = total - challenger_count
            return {
                "baseline": baseline_count / total,
                "challenger": challenger_count / total,
            }
