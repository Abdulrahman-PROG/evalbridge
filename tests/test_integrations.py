"""
Integration source tests. All external libraries are mocked.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# Module-level classes so pickle can find them by qualified name
class _FakeHFModel:
    def predict(self, x):
        return [0.9]


class _FakeWandbModel:
    def predict(self, x):
        return [0.7]


# ── MLflowSource ──────────────────────────────────────────────────────────────


def test_mlflow_source_uri_parsing():
    from evalbridge.integrations.mlflow import MLflowSource

    src = MLflowSource("models:/churn/Production")
    assert src._parsed["model_name"] == "churn"
    assert src._parsed["version_or_stage"] == "Production"


def test_mlflow_source_uri_numeric_version():
    from evalbridge.integrations.mlflow import MLflowSource

    src = MLflowSource("models:/churn/3")
    assert src._parsed["version_or_stage"] == "3"


def test_mlflow_source_invalid_uri():
    from evalbridge.integrations.mlflow import MLflowSource

    with pytest.raises(ValueError, match="models:/"):
        MLflowSource("runs:/abc/model")


def test_mlflow_import_error():
    from evalbridge.integrations.mlflow import MLflowSource

    src = MLflowSource("models:/churn/Production")
    # Temporarily hide mlflow
    with patch.dict(sys.modules, {"mlflow": None, "mlflow.pyfunc": None}):
        with pytest.raises(ImportError, match="pip install evalbridge\\[mlflow\\]"):
            src.load()


def test_mlflow_load_calls_pyfunc(monkeypatch):
    from evalbridge.integrations.mlflow import MLflowSource

    mock_model = MagicMock()
    mock_model.predict = MagicMock(return_value=[0.8, 0.2])

    mock_pyfunc = MagicMock()
    mock_pyfunc.load_model = MagicMock(return_value=mock_model)

    mock_mlflow = MagicMock()
    mock_mlflow.pyfunc = mock_pyfunc

    with patch.dict(sys.modules, {"mlflow": mock_mlflow, "mlflow.pyfunc": mock_pyfunc}):
        src = MLflowSource("models:/churn/Production")
        model = src.load()
        assert model is mock_model
        mock_pyfunc.load_model.assert_called_once_with("models:/churn/Production")


# ── HFSource ──────────────────────────────────────────────────────────────────


def test_hf_import_error():
    from evalbridge.integrations.huggingface import HFSource

    src = HFSource("org/model")
    with patch.dict(sys.modules, {"huggingface_hub": None}):
        with pytest.raises(ImportError, match="pip install evalbridge\\[hf\\]"):
            src.load()


def test_hf_load_calls_download(tmp_path, monkeypatch):
    import pickle

    from evalbridge.integrations.huggingface import HFSource

    model_file = tmp_path / "model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(_FakeHFModel(), f)

    mock_hf = MagicMock()
    mock_hf.hf_hub_download = MagicMock(return_value=str(model_file))

    with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
        src = HFSource("org/model", revision="v1")
        model = src.load()
        mock_hf.hf_hub_download.assert_called_once_with(
            repo_id="org/model", filename="model.pkl", revision="v1"
        )
        assert hasattr(model, "predict")


# ── WandbSource ───────────────────────────────────────────────────────────────


def test_wandb_import_error():
    from evalbridge.integrations.wandb import WandbSource

    src = WandbSource("project", "run-id")
    with patch.dict(sys.modules, {"wandb": None}):
        with pytest.raises(ImportError, match="pip install evalbridge\\[wandb\\]"):
            src.load()


def test_wandb_load_calls_api(tmp_path, monkeypatch):
    import pickle

    from evalbridge.integrations.wandb import WandbSource

    model_file = tmp_path / "model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(_FakeWandbModel(), f)

    mock_artifact = MagicMock()
    mock_artifact.download = MagicMock(return_value=str(tmp_path))

    mock_api = MagicMock()
    mock_api.artifact = MagicMock(return_value=mock_artifact)

    mock_wandb = MagicMock()
    mock_wandb.Api = MagicMock(return_value=mock_api)

    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        src = WandbSource("myproject", "abc123", artifact="model")
        model = src.load()
        assert hasattr(model, "predict")
