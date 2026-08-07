"""
HTML report generation via Jinja2. All output is self-contained (no external deps).
"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader

if TYPE_CHECKING:
    from evalbridge.drift import DriftReport
    from evalbridge.experiment import ExperimentResult


def generate(result: "ExperimentResult", path: str = "report.html"):
    """Render the experiment report template and write to path.

    Example: generate(result, path="report.html")
    """
    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=True)
    template = env.get_template("report.html")

    base_m = result.metrics.get("baseline", {})
    chal_m = result.metrics.get("challenger", {})

    ctx = {
        "experiment_name": result._experiment_name,
        "method": result._method,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "winner": result.winner,
        "confidence": result.confidence,
        "confidence_pct": f"{result.confidence * 100:.1f}",
        "ready": result.ready,
        "min_confidence": result._min_confidence,
        "min_confidence_pct": f"{result._min_confidence * 100:.0f}",
        "baseline": {
            "accuracy": f"{base_m.get('accuracy', 0):.4f}",
            "auc_roc": f"{base_m.get('auc_roc', 0):.4f}",
            "f1": f"{base_m.get('f1', 0):.4f}",
            "brier_score": f"{base_m.get('brier_score', 0):.4f}",
            "n_samples": f"{int(base_m.get('n_samples', 0)):,}",
        },
        "challenger": {
            "accuracy": f"{chal_m.get('accuracy', 0):.4f}",
            "auc_roc": f"{chal_m.get('auc_roc', 0):.4f}",
            "f1": f"{chal_m.get('f1', 0):.4f}",
            "brier_score": f"{chal_m.get('brier_score', 0):.4f}",
            "n_samples": f"{int(chal_m.get('n_samples', 0)):,}",
        },
        "cohort_results": result.cohort_results,
        "has_cohorts": bool(result.cohort_results),
        "drift": None,
    }

    html = template.render(**ctx)
    Path(path).write_text(html, encoding="utf-8")
    print(f"[evalbridge] Report saved to {path}")


def generate_drift_report(drift_report: "DriftReport", path: str = "drift_report.html"):
    """Render a standalone drift report.

    Example: generate_drift_report(report, path="drift_report.html")
    """
    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=True)
    template = env.get_template("drift_report.html")

    ctx = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "psi": f"{drift_report.psi:.4f}",
        "ks_stat": f"{drift_report.ks_stat:.4f}",
        "ks_pvalue": f"{drift_report.ks_pvalue:.4f}",
        "alert": drift_report.alert,
        "severity": drift_report.severity,
        "threshold": f"{drift_report.threshold:.2f}",
        "summary": drift_report.summary,
    }

    html = template.render(**ctx)
    Path(path).write_text(html, encoding="utf-8")
    print(f"[evalbridge] Drift report saved to {path}")
