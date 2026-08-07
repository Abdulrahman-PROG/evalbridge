"""
HuggingFace Hub model integration.
"""


class HFSource:
    """Pull a model from HuggingFace Hub.

    Example: source = HFSource("org/churn-model-v2", revision="main")
    """

    def __init__(self, repo_id: str, revision: str = "main"):
        """
        Parameters
        ----------
        repo_id  : HuggingFace repo, e.g. "org/model-name"
        revision : git revision / branch / tag, default "main"
        """
        self.repo_id = repo_id
        self.revision = revision

    def load(self):
        """Download model from HuggingFace Hub and return it.

        Raises ImportError with install hint if huggingface_hub is not installed.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError("HuggingFace Hub is required: pip install evalbridge[hf]") from e

        path = hf_hub_download(repo_id=self.repo_id, filename="model.pkl", revision=self.revision)
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)
