# evalbridge

> The missing bridge between your notebook and your production A/B test.

[![PyPI version](https://img.shields.io/pypi/v/evalbridge.svg)](https://pypi.org/project/evalbridge/)
[![Python versions](https://img.shields.io/pypi/pyversions/evalbridge.svg)](https://pypi.org/project/evalbridge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/Abdulrahman-PROG/evalbridge/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdulrahman-PROG/evalbridge/actions)

## The problem

Every ML team solves the same three problems from scratch, every sprint.

**Gap 1 — No standard evaluation format.** Each notebook reimplements accuracy, AUC, and F1 from
scratch, inconsistently. There is no shared format to compare models across teams or projects.

**Gap 2 — Offline evaluation and live A/B testing are disconnected.** You evaluate a model in
Jupyter, declare it the winner, then rewrite everything in production-grade code to run the actual
A/B test. The two environments share nothing.

**Gap 3 — Drift is detected but nothing acts on it.** Tools like Evidently tell you drift happened.
Nobody automatically rolls back the model or promotes the winner. A human still does it manually.

evalbridge fixes all three.

## Install

```bash
pip install evalbridge
```

Optional integrations:

```bash
pip install evalbridge[mlflow]   # MLflow registry
pip install evalbridge[wandb]    # Weights & Biases
pip install evalbridge[hf]       # HuggingFace Hub
```

## Quickstart

Ten lines. Works in any Python environment.

```python
from evalbridge import Experiment

exp = Experiment("churn_v2_vs_v3")
exp.log("baseline",   y_true=[1,0,1,1,0], y_pred=[0.8,0.2,0.7,0.9,0.1])
exp.log("challenger", y_true=[1,0,1,1,0], y_pred=[0.85,0.15,0.75,0.92,0.08])

result = exp.evaluate()
print(result.winner)      # "challenger"
print(result.confidence)  # 0.94
result.summary()          # prints formatted table
result.report()           # writes report.html, opens in browser
```

`result.summary()` output:

```
┌─────────────┬──────────┬──────────┬────────────┐
│ Model       │ Accuracy │ AUC-ROC  │ Samples    │
├─────────────┼──────────┼──────────┼────────────┤
│ baseline    │  0.8000  │  0.8750  │          5 │
│ challenger  │  1.0000  │  1.0000  │          5 │
└─────────────┴──────────┴──────────┴────────────┘
Winner: challenger  (confidence: 94.2%)
```

## The three gaps evalbridge solves

### Gap 1 — Standard evaluation format

`Experiment` gives every team the same API. Log predictions, call `evaluate()`, get a structured
`ExperimentResult` with all metrics computed identically every time.

```python
from evalbridge import Experiment

exp = Experiment("churn_v3", method="bayesian", min_confidence=0.95, min_samples=500)
exp.log("baseline",   y_true=y_test, y_pred=baseline_preds)
exp.log("challenger", y_true=y_test, y_pred=challenger_preds)
result = exp.evaluate()
print(result.metrics)  # {"baseline": {accuracy, auc_roc, f1, ...}, "challenger": {...}}
```

### Gap 2 — Offline → live, same object

Save your experiment from the notebook and pick it up in production. Same object. No rewriting.

```python
import evalbridge

# in notebook
exp = evalbridge.Experiment("churn_v3")
exp.log("baseline",   y_true=y_test, y_pred=baseline_preds)
exp.log("challenger", y_true=y_test, y_pred=challenger_preds)
exp.save("churn_v3.evalbridge")

# in production
exp = evalbridge.Experiment.load("churn_v3.evalbridge")
exp.go_live(traffic_split=0.1)  # 10% of traffic to challenger

@evalbridge.track(experiment="churn_v3", model="challenger")
def predict(features):
    return model.predict(features)

# later, when ground truth arrives
evalbridge.log_outcome(request_id=predict.last_request_id, y_true=1)
```

### Gap 3 — Drift detection with automatic action

Drift gets detected *and acted on*. Not just logged.

```python
from evalbridge import DriftDetector

detector = DriftDetector(reference=baseline_preds)
report = detector.check(production_preds)

print(report.psi)       # Population Stability Index
print(report.ks_stat)   # KS test statistic
print(report.alert)     # True if drift exceeds threshold
print(report.severity)  # "none" | "moderate" | "severe"

# auto-rollback when drift is detected
exp.on_drift(threshold=0.2, action="rollback")

# auto-promote when statistical confidence is reached
exp.on_confidence(threshold=0.95, action="promote")
```

## API reference

### Experiment (full config)

```python
exp = evalbridge.Experiment(
    name="churn_v3",
    method="bayesian",       # "bayesian" | "sequential" | "bandit"
    min_confidence=0.95,
    min_samples=500,
    stop_early=True,
    cohorts={"region": ["EU", "US", "APAC"]},
    alert_on_drift=True,
    drift_threshold=0.2,
)
```

### ExperimentResult

```python
result.winner          # "baseline" | "challenger" | "inconclusive"
result.confidence      # float 0.0–1.0
result.ready           # bool: min_samples + min_confidence both met
result.metrics         # {"baseline": {...}, "challenger": {...}}
result.cohort_results  # per-cohort breakdowns when cohorts set
result.summary()       # prints formatted table, returns string
result.report()        # writes report.html, opens browser
result.promote()       # marks winner as Production in registry
result.rollback()      # reverts to baseline
```

### DriftDetector

```python
detector = evalbridge.DriftDetector(reference=baseline_preds, threshold=0.2)
report = detector.check(production_preds)
report.export("drift_report.html")

# stream monitoring
detector.watch(pred_stream, interval=100, on_drift=my_callback)
```

### Bandit

```python
bandit = exp.as_bandit(strategy="thompson")  # or "epsilon_greedy"
chosen_model = bandit.route(request_features)
bandit.update(chosen_model, reward=1.0)
print(bandit.allocations())  # {"baseline": 0.23, "challenger": 0.77}
```

### Decorator

```python
import evalbridge

@evalbridge.track(experiment="churn_v3", model="challenger")
def predict(features):
    return model.predict(features)

evalbridge.log_outcome(request_id="abc123", y_true=1)
```

## Integrations

### MLflow

```python
from evalbridge.integrations import MLflowSource

exp.set_baseline(MLflowSource("models:/churn/Production"))
exp.set_challenger(MLflowSource("models:/churn/Staging"))
```

### Weights & Biases

```python
from evalbridge.integrations import WandbSource

exp.set_challenger(WandbSource("myteam/churn", run_id="abc123", artifact="model"))
```

### HuggingFace Hub

```python
from evalbridge.integrations import HFSource

exp.set_challenger(HFSource("org/churn-model-v2", revision="main"))
```

## Development

```bash
git clone https://github.com/Abdulrahman-PROG/evalbridge
cd evalbridge
pip install -e ".[dev]"
pytest tests/
```

## License

MIT
