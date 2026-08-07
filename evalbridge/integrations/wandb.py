"""
Weights & Biases model artifact integration.
"""


class WandbSource:
    """Pull a model artifact from Weights & Biases.

    Example: source = WandbSource("project/run-id", artifact="model")
    """

    def __init__(self, project: str, run_id: str, artifact: str = "model"):
        """
        Parameters
        ----------
        project  : W&B project path, e.g. "entity/project"
        run_id   : W&B run ID
        artifact : artifact name within the run
        """
        self.project = project
        self.run_id = run_id
        self.artifact = artifact

    def load(self):
        """Download model artifact from W&B and return it.

        Raises ImportError with install hint if wandb is not installed.
        """
        try:
            import wandb
        except ImportError as e:
            raise ImportError("Weights & Biases is required: pip install evalbridge[wandb]") from e

        api = wandb.Api()
        artifact_path = f"{self.project}/run-{self.run_id}-{self.artifact}:latest"
        art = api.artifact(artifact_path)
        artifact_dir = art.download()

        import os
        import pickle

        model_file = os.path.join(artifact_dir, "model.pkl")
        with open(model_file, "rb") as f:
            return pickle.load(f)
