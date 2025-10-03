from typing import Callable, Optional, Tuple

import torch
from torch import Tensor, nn


class MultiRecurrentRNN(nn.Module):
    """
    Multi-recurrent neural network combining Elman- and Jordan-style feedback.

    Recurrence:
        h_t = activation(W_ih x_t + W_hh h_{t-1} + W_yo y_{t-1} + b_h)
        y_t = W_ho h_t + b_o

    This model has both hidden-state feedback and output feedback.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        activation: Callable[[Tensor], Tensor] = torch.tanh,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0 or output_size <= 0:
            raise ValueError("input_size, hidden_size and output_size must be positive")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

        self.input_to_hidden = nn.Linear(input_size, hidden_size, bias=True)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size, bias=True)
        self.output_to_hidden = nn.Linear(output_size, hidden_size, bias=True)
        self.hidden_to_output = nn.Linear(hidden_size, output_size, bias=True)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (
            self.input_to_hidden,
            self.hidden_to_hidden,
            self.output_to_hidden,
            self.hidden_to_output,
        ):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def init_hidden(self, batch_size: int, device: Optional[torch.device] = None) -> Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def init_output(self, batch_size: int, device: Optional[torch.device] = None) -> Tensor:
        return torch.zeros(batch_size, self.output_size, device=device)

    def forward(
        self,
        inputs: Tensor,
        initial_hidden: Optional[Tensor] = None,
        initial_output: Optional[Tensor] = None,
        *,
        batch_first: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through time.

        Parameters
        ----------
        inputs: Tensor
            Input sequence of shape (batch, seq, input_size) if batch_first,
            else (seq, batch, input_size).
        initial_hidden: Optional[Tensor]
            Initial hidden state (batch, hidden_size). If None, zeros are used.
        initial_output: Optional[Tensor]
            Initial previous output y_{-1} (batch, output_size). If None, zeros are used.
        batch_first: bool
            Whether the input tensor is batch-first.

        Returns
        -------
        outputs: Tensor
            Sequence of outputs of shape (batch, seq, output_size) (batch_first) or
            (seq, batch, output_size) otherwise.
        last_hidden: Tensor
            Final hidden state of shape (batch, hidden_size).
        """
        if inputs.dim() != 3:
            raise ValueError("inputs must be 3D (batch, seq, feature) or (seq, batch, feature)")

        if batch_first:
            batch_size, seq_len, _ = inputs.shape
            x = inputs
        else:
            seq_len, batch_size, _ = inputs.shape
            x = inputs.transpose(0, 1)  # (batch, seq, input)

        device = x.device
        hidden = initial_hidden if initial_hidden is not None else self.init_hidden(batch_size, device)
        prev_output = initial_output if initial_output is not None else self.init_output(batch_size, device)

        outputs: list[Tensor] = []
        for t in range(seq_len):
            xt = x[:, t, :]
            hidden = self.activation(
                self.input_to_hidden(xt) + self.hidden_to_hidden(hidden) + self.output_to_hidden(prev_output)
            )
            hidden = self.dropout(hidden)
            yt = self.hidden_to_output(hidden)
            outputs.append(yt.unsqueeze(1))
            prev_output = yt.detach()

        outputs_tensor = torch.cat(outputs, dim=1)

        if batch_first:
            return outputs_tensor, hidden
        else:
            return outputs_tensor.transpose(0, 1), hidden


