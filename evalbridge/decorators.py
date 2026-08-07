"""
@evalbridge.track decorator and log_outcome() for zero-instrumentation prediction logging.
"""

import functools
import threading
import time
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evalbridge.experiment import Experiment

_registry: dict = {}
_pending: dict = {}
_lock = threading.Lock()


def register_experiment(exp: "Experiment"):
    """Register an experiment so @track can find it by name.

    Example: evalbridge.register_experiment(exp)
    """
    _registry[exp.name] = exp


def track(experiment: str, model: str):
    """Decorator that logs every call and stores result until log_outcome() is called.

    Example:
        @evalbridge.track(experiment="churn_v3", model="challenger")
        def predict(features):
            return model.predict(features)
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            request_id = str(uuid.uuid4())
            result = fn(*args, **kwargs)
            with _lock:
                _pending[request_id] = {
                    "model": model,
                    "prediction": result,
                    "timestamp": time.time(),
                    "experiment": experiment,
                }
            wrapper.last_request_id = request_id
            return result

        wrapper.last_request_id = None
        return wrapper

    return decorator


def log_outcome(request_id: str, y_true: int):
    """Attach ground truth to a previously tracked prediction.

    Looks up the pending call by request_id and logs it to the registered experiment.

    Example: evalbridge.log_outcome(request_id="abc123", y_true=1)
    """
    with _lock:
        entry = _pending.pop(request_id, None)

    if entry is None:
        raise KeyError(
            f"request_id '{request_id}' not found in pending predictions. "
            "Make sure @evalbridge.track decorated the function and the request_id "
            "comes from wrapper.last_request_id."
        )

    exp_name = entry["experiment"]
    exp = _registry.get(exp_name)
    if exp is None:
        raise KeyError(
            f"Experiment '{exp_name}' not found. "
            "Call evalbridge.register_experiment(exp) before using @evalbridge.track."
        )

    pred = entry["prediction"]
    # Normalise prediction to a probability float
    if hasattr(pred, "__len__"):
        pred_val = float(pred[0]) if len(pred) == 1 else float(pred[-1])
    else:
        pred_val = float(pred)

    exp.log(entry["model"], y_true=[float(y_true)], y_pred=[pred_val])
