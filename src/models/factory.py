from typing import Any

from torch import nn

from .elman import ElmanRNN
from .jordan import JordanRNN
from .multi_recurrent import MultiRecurrentRNN
from ..utils.activations import get_activation


def build_model(model_config: dict[str, Any], input_size: int, output_size: int) -> nn.Module:
    """
    Build a recurrent model from config.

    Required keys in model_config:
      - type: one of {"elman", "jordan", "multi"}
      - hidden_size: int

    Optional keys:
      - activation: str ("tanh", "relu", "sigmoid", "identity")
      - dropout: float
    """
    model_type = str(model_config.get("type", "")).lower()
    hidden_size = int(model_config.get("hidden_size", 32))
    activation_name = str(model_config.get("activation", "tanh"))
    dropout = float(model_config.get("dropout", 0.0))

    activation = get_activation(activation_name)

    if model_type == "elman":
        return ElmanRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            activation=activation,
            dropout=dropout,
        )
    if model_type == "jordan":
        return JordanRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            activation=activation,
            dropout=dropout,
        )
    if model_type in {"multi", "multi_recurrent", "multi-recurrent"}:
        return MultiRecurrentRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            activation=activation,
            dropout=dropout,
        )

    raise ValueError(f"Unknown model type: {model_type}")


