"""
MLflow Model Registry integration.
"""


class MLflowSource:
    """Pull a model from the MLflow Model Registry.

    Example: source = MLflowSource("models:/churn/Production")
    """

    def __init__(self, uri: str):
        """
        Parameters
        ----------
        uri : MLflow model URI, e.g. "models:/churn/Production" or "models:/churn/3"
        """
        self.uri = uri
        self._parsed = self._parse_uri(uri)

    def _parse_uri(self, uri: str) -> dict:
        """Parse models:/name/stage_or_version format."""
        if not uri.startswith("models:/"):
            raise ValueError(f"MLflowSource URI must start with 'models:/', got {uri!r}")
        rest = uri[len("models:/") :]
        parts = rest.split("/", 1)
        if len(parts) != 2:
            raise ValueError(
                f"MLflowSource URI must be 'models:/model_name/stage_or_version', got {uri!r}"
            )
        return {"model_name": parts[0], "version_or_stage": parts[1]}

    def load(self):
        """Load the model from MLflow Model Registry.

        Raises ImportError with install hint if mlflow is not installed.
        """
        try:
            import mlflow.pyfunc
        except ImportError as e:
            raise ImportError("MLflow is required: pip install evalbridge[mlflow]") from e

        return mlflow.pyfunc.load_model(self.uri)

    def promote(self, run_id: str):
        """Transition this model version to Production stage.

        Example: source.promote(run_id="abc123")
        """
        try:
            import mlflow
        except ImportError as e:
            raise ImportError("MLflow is required: pip install evalbridge[mlflow]") from e

        client = mlflow.tracking.MlflowClient()
        model_name = self._parsed["model_name"]
        version_or_stage = self._parsed["version_or_stage"]

        # Find the version number if a stage name was given
        if version_or_stage.isdigit():
            version = version_or_stage
        else:
            versions = client.get_latest_versions(model_name, stages=[version_or_stage])
            version = versions[0].version if versions else None

        if version is not None:
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
            )

    def rollback(self):
        """Transition Production back to the previous Staging version.

        Example: source.rollback()
        """
        try:
            import mlflow
        except ImportError as e:
            raise ImportError("MLflow is required: pip install evalbridge[mlflow]") from e

        client = mlflow.tracking.MlflowClient()
        model_name = self._parsed["model_name"]

        staging = client.get_latest_versions(model_name, stages=["Staging"])
        if staging:
            client.transition_model_version_stage(
                name=model_name,
                version=staging[0].version,
                stage="Production",
            )
