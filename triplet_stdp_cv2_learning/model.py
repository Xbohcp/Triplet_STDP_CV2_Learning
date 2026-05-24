from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class ForwardCache:
    """Forward-pass values needed by the manual biological learning rule."""

    inputs: Tensor
    pre_activations: List[Tensor]
    activations: List[Tensor]

    def __getitem__(self, index: int) -> Tensor:
        return self.activations[index]

    def __iter__(self) -> Iterator[Tensor]:
        return iter(self.activations)

    def __len__(self) -> int:
        return len(self.activations)


class PositiveMLP(nn.Module):
    """Fully connected MNIST classifier with strictly positive activations."""

    def __init__(
        self,
        input_dim: int = 28 * 28,
        hidden_dim: int = 500,
        output_dim: int = 10,
        hidden_layers: int = 3,
        min_activation: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if hidden_layers < 0:
            raise ValueError("hidden_layers must be non-negative")
        if min_activation <= 0:
            raise ValueError("min_activation must be positive")

        dims = [input_dim] + [hidden_dim] * hidden_layers + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(in_features, out_features)
            for in_features, out_features in zip(dims[:-1], dims[1:])
        )
        self.min_activation = min_activation
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers:
            nn.init.kaiming_uniform_(layer.weight, a=5**0.5)
            nn.init.zeros_(layer.bias)

    def positive_activation(self, pre_activation: Tensor) -> Tensor:
        return F.softplus(pre_activation) + self.min_activation

    def positive_activation_derivative(self, pre_activation: Tensor) -> Tensor:
        return torch.sigmoid(pre_activation)

    def forward_with_cache(self, inputs: Tensor) -> ForwardCache:
        activation = inputs
        pre_activations: List[Tensor] = []
        activations: List[Tensor] = [inputs]

        for layer in self.layers:
            pre_activation = layer(activation)
            activation = self.positive_activation(pre_activation)
            pre_activations.append(pre_activation)
            activations.append(activation)

        return ForwardCache(
            inputs=inputs,
            pre_activations=pre_activations,
            activations=activations,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.forward_with_cache(inputs).activations[-1]

    def logits_from_positive_output(self, output_activation: Tensor) -> Tensor:
        return torch.log(output_activation.clamp_min(self.min_activation))

    def loss_from_output(self, output_activation: Tensor, targets: Tensor) -> Tensor:
        logits = self.logits_from_positive_output(output_activation)
        return F.cross_entropy(logits, targets)
