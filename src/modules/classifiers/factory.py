from typing import Any, Dict

from .uncontrolled_lstm import UncontrolledLSTMClassifier
from .uncontrolled_transformer import UncontrolledTransformerClassifier


def build_classifier(
    architecture: str = "transformer",
    obs_dim: int = None,
    n_agents: int = None,
    episode_limit: int = None,
    num_uncontrolled_types: int = None,
    **kwargs: Any,
):
    architecture = (architecture or "transformer").lower()
    if architecture == "transformer":
        return UncontrolledTransformerClassifier(
            obs_dim=obs_dim,
            n_agents=n_agents,
            episode_limit=episode_limit,
            num_uncontrolled_types=num_uncontrolled_types,
            d_model=kwargs.get("d_model", 128),
            nhead=kwargs.get("nhead", 4),
            num_layers=kwargs.get("num_layers", kwargs.get("layers", 2)),
            dim_feedforward=kwargs.get("dim_feedforward", kwargs.get("ff", 256)),
            dropout=kwargs.get("dropout", 0.1),
        )
    if architecture == "lstm":
        return UncontrolledLSTMClassifier(
            obs_dim=obs_dim,
            n_agents=n_agents,
            episode_limit=episode_limit,
            num_uncontrolled_types=num_uncontrolled_types,
            hidden_dim=kwargs.get("hidden_dim", kwargs.get("d_model", 128)),
            num_layers=kwargs.get("num_layers", kwargs.get("layers", 2)),
            dropout=kwargs.get("dropout", 0.1),
            bidirectional=kwargs.get("bidirectional", False),
        )
    raise ValueError(f"Unknown classifier architecture: {architecture}")


def classifier_kwargs_from_args(args) -> Dict[str, Any]:
    return {
        "architecture": getattr(args, "classifier_architecture", "transformer"),
        "d_model": getattr(args, "classifier_d_model", 128),
        "nhead": getattr(args, "classifier_nhead", 4),
        "num_layers": getattr(args, "classifier_layers", 2),
        "ff": getattr(args, "classifier_ff", 256),
        "dropout": getattr(args, "classifier_dropout", 0.1),
        "hidden_dim": getattr(args, "classifier_hidden_dim", getattr(args, "classifier_d_model", 128)),
        "bidirectional": getattr(args, "classifier_bidirectional", False),
    }
