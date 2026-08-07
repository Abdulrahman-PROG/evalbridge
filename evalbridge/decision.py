"""
Standalone promote / rollback / notify helpers.
These are thin wrappers used by ExperimentResult — not part of the public API.
"""


def promote(source, winner: str):
    """Promote the winning model to Production in its registry source."""
    if source is not None and hasattr(source, "promote"):
        source.promote(winner)
    else:
        print(f"[evalbridge] promote() — no registry source set. Winner: {winner}")


def rollback(source):
    """Revert Production to the previous baseline in the registry source."""
    if source is not None and hasattr(source, "rollback"):
        source.rollback()
    else:
        print("[evalbridge] rollback() — no registry source set")


def notify(message: str):
    """Print an alert notification."""
    print(f"[evalbridge] ALERT: {message}")
