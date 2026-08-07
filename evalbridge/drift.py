"""
Drift detection via Population Stability Index (PSI) and the KS test.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np
from scipy.stats import ks_2samp


@dataclass
class DriftReport:
    """Holds the results of a single drift check."""

    psi: float
    ks_stat: float
    ks_pvalue: float
    alert: bool
    severity: str
    threshold: float
    summary: str

    def export(self, path: str = "drift_report.html"):
        """Write a standalone HTML drift report.

        Example: report.export("drift_report.html")
        """
        from evalbridge.report import generate_drift_report

        generate_drift_report(self, path=path)


class DriftDetector:
    """
    Monitors prediction distributions for drift using PSI and KS test.

    PSI < 0.1   -> severity='none',     alert=False
    PSI 0.1-0.2 -> severity='moderate', alert=False  (at default threshold=0.2)
    PSI > 0.2   -> severity='severe',   alert=True
    """

    def __init__(
        self,
        reference,
        threshold: float = 0.2,
        n_bins: int = 10,
    ):
        """
        Parameters
        ----------
        reference : baseline predictions to compare against
        threshold : PSI value above which alert fires
        n_bins    : number of bins for PSI calculation
        """
        ref = np.asarray(reference, dtype=float)
        if len(ref) == 0:
            raise ValueError("reference must not be empty")
        self._reference = ref
        self.threshold = threshold
        self.n_bins = n_bins

    def check(self, current) -> DriftReport:
        """Run PSI + KS test between reference and current distributions.

        Example: report = detector.check(production_preds)
        """
        cur = np.asarray(current, dtype=float)
        if len(cur) == 0:
            raise ValueError("current must not be empty")

        psi_val = self._psi(self._reference, cur, self.n_bins)
        ks_stat, ks_pvalue = self._ks_test(self._reference, cur)

        if psi_val < 0.1:
            severity = "none"
            alert = False
        elif psi_val <= 0.2:
            severity = "moderate"
            alert = psi_val > self.threshold
        else:
            severity = "severe"
            alert = psi_val > self.threshold

        if severity == "none":
            summary_text = f"No drift detected (PSI={psi_val:.4f})"
        elif severity == "moderate":
            summary_text = f"Moderate drift detected (PSI={psi_val:.4f})"
        else:
            summary_text = f"Severe drift detected (PSI={psi_val:.4f}) — investigate"

        return DriftReport(
            psi=psi_val,
            ks_stat=ks_stat,
            ks_pvalue=ks_pvalue,
            alert=alert,
            severity=severity,
            threshold=self.threshold,
            summary=summary_text,
        )

    def _psi(self, reference: np.ndarray, current: np.ndarray, n_bins: int) -> float:
        """Compute Population Stability Index between two distributions.

        Example: psi = detector._psi(ref, cur, 10)
        """
        eps = 1e-10
        # Use reference distribution to define bin edges
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        ref_pct = ref_counts / (len(reference) + eps)
        cur_pct = cur_counts / (len(current) + eps)

        ref_pct = np.clip(ref_pct, eps, None)
        cur_pct = np.clip(cur_pct, eps, None)

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return psi

    def _ks_test(self, reference: np.ndarray, current: np.ndarray):
        """Run two-sample KS test between distributions.

        Example: stat, pvalue = detector._ks_test(ref, cur)
        """
        result = ks_2samp(reference, current)
        return float(result.statistic), float(result.pvalue)

    def watch(
        self,
        stream: Iterable,
        interval: int = 100,
        on_drift: Optional[Callable] = None,
    ):
        """Monitor a stream, calling on_drift whenever drift is detected.

        Example: detector.watch(pred_stream, interval=100, on_drift=my_callback)
        """
        buffer = []
        for item in stream:
            buffer.append(float(item))
            if len(buffer) >= interval:
                report = self.check(np.array(buffer))
                if report.alert and on_drift is not None:
                    on_drift(report)
                buffer = []
