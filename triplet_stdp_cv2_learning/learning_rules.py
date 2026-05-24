from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from .model import ForwardCache, PositiveMLP


EPSILON = 1.0e-12


@dataclass
class AngleSummary:
    """Layer-wise and epoch-level angle statistics in degrees."""

    layer_angles: List[Optional[float]]
    mean_angle: Optional[float]


def zeta(values: Tensor, mode: str = "tanh") -> Tensor:
    """Map error values into the interval required by the proposed rule."""

    if mode == "tanh":
        return 0.5 * torch.tanh(values)
    if mode == "clamp":
        return values.clamp(min=-0.5 + 1.0e-7, max=0.5 - 1.0e-7)
    raise ValueError(f"unknown zeta mode: {mode}")


def update_signal(errors: Tensor, activations: Tensor, mode: str) -> Tensor:
    """Evaluate f(epsilon_l, y_l) for the biological update rule."""

    if mode == "epsilon_times_activation":
        return errors * activations
    if mode == "epsilon":
        return errors
    if mode == "activity_gated":
        return errors * activations / activations.detach().mean().clamp_min(EPSILON)
    raise ValueError(f"unknown update function: {mode}")


def bp_weight_deltas(
    model: PositiveMLP,
    loss: Tensor,
    learning_rate: float,
    retain_graph: bool = True,
) -> List[Tensor]:
    """Return standard BP weight changes, Delta W = -eta * dL/dW."""

    weights = [layer.weight for layer in model.layers]
    gradients = torch.autograd.grad(loss, weights, retain_graph=retain_graph)
    return [(-learning_rate * grad).detach() for grad in gradients]


def bp_bias_deltas(
    model: PositiveMLP,
    loss: Tensor,
    learning_rate: float,
    retain_graph: bool = True,
) -> List[Tensor]:
    """Return standard BP bias changes, used only when BP trains the network."""

    biases = [layer.bias for layer in model.layers]
    gradients = torch.autograd.grad(loss, biases, retain_graph=retain_graph)
    return [(-learning_rate * grad).detach() for grad in gradients]


def biological_errors(
    model: PositiveMLP,
    cache: ForwardCache,
    targets: Tensor,
    scale_mode: str = "tanh",
) -> List[Tensor]:
    """Compute epsilon_l with the proposed positive-activation recurrence."""

    output = cache.activations[-1]
    loss = model.loss_from_output(output, targets)
    output_error = torch.autograd.grad(
        loss,
        cache.pre_activations[-1],
        retain_graph=True,
    )[0]

    errors: List[Tensor] = [torch.empty_like(pre) for pre in cache.pre_activations]
    errors[-1] = zeta(output_error, mode=scale_mode)

    for layer_index in range(len(model.layers) - 2, -1, -1):
        next_layer = model.layers[layer_index + 1]
        current_activation = cache.activations[layer_index + 1]
        next_activation = cache.activations[layer_index + 2]
        current_pre_activation = cache.pre_activations[layer_index]
        next_error = errors[layer_index + 1]

        propagated = torch.matmul(
            next_activation * next_error,
            next_layer.weight,
        )
        raw_error = (
            propagated
            * model.positive_activation_derivative(current_pre_activation)
            / current_activation.clamp_min(model.min_activation)
        )
        errors[layer_index] = zeta(raw_error, mode=scale_mode)

    return [error.detach() for error in errors]


def biological_weight_deltas(
    model: PositiveMLP,
    cache: ForwardCache,
    targets: Tensor,
    learning_rate: float,
    scale_mode: str = "tanh",
    update_function: str = "epsilon_times_activation",
) -> List[Tensor]:
    """Return weight changes from the proposed biological learning rule."""

    errors = biological_errors(
        model=model,
        cache=cache,
        targets=targets,
        scale_mode=scale_mode,
    )
    batch_size = targets.shape[0]
    deltas: List[Tensor] = []

    for layer_index, error in enumerate(errors):
        previous_activation = cache.activations[layer_index].detach()
        current_activation = cache.activations[layer_index + 1].detach()
        signal = update_signal(
            errors=error,
            activations=current_activation,
            mode=update_function,
        )
        delta = -(learning_rate / batch_size) * torch.matmul(
            signal.transpose(0, 1),
            previous_activation,
        )
        deltas.append(delta.detach())

    return deltas


def biological_bias_deltas(
    model: PositiveMLP,
    cache: ForwardCache,
    targets: Tensor,
    learning_rate: float,
    scale_mode: str = "tanh",
    update_function: str = "epsilon_times_activation",
) -> List[Tensor]:
    """Optional bias analogue of the biological rule."""

    errors = biological_errors(
        model=model,
        cache=cache,
        targets=targets,
        scale_mode=scale_mode,
    )
    deltas: List[Tensor] = []
    for layer_index, error in enumerate(errors):
        current_activation = cache.activations[layer_index + 1].detach()
        signal = update_signal(
            errors=error,
            activations=current_activation,
            mode=update_function,
        )
        deltas.append((-learning_rate * signal.mean(dim=0)).detach())
    return deltas


def apply_deltas(
    model: PositiveMLP,
    weight_deltas: Sequence[Tensor],
    bias_deltas: Optional[Sequence[Tensor]] = None,
) -> None:
    """Apply precomputed deltas without touching PyTorch optimizer state."""

    with torch.no_grad():
        for layer, weight_delta in zip(model.layers, weight_deltas):
            layer.weight.add_(weight_delta)
        if bias_deltas is not None:
            for layer, bias_delta in zip(model.layers, bias_deltas):
                layer.bias.add_(bias_delta)


def vector_angle_degrees(first: Tensor, second: Tensor) -> Optional[float]:
    """Return the angle between two vectors, or None when one vector is zero."""

    first_vector = first.detach().reshape(-1).double()
    second_vector = second.detach().reshape(-1).double()
    first_norm = torch.linalg.vector_norm(first_vector)
    second_norm = torch.linalg.vector_norm(second_vector)
    if first_norm <= EPSILON or second_norm <= EPSILON:
        return None

    cosine = torch.dot(first_vector, second_vector) / (first_norm * second_norm)
    cosine = cosine.clamp(min=-1.0, max=1.0)
    return float(torch.rad2deg(torch.acos(cosine)))


def weight_delta_angles(
    bp_deltas: Iterable[Tensor],
    biological_deltas: Iterable[Tensor],
) -> AngleSummary:
    """Compare BP and biological weight-change vectors layer by layer."""

    layer_angles = [
        vector_angle_degrees(bp_delta, biological_delta)
        for bp_delta, biological_delta in zip(bp_deltas, biological_deltas)
    ]
    valid_angles = [angle for angle in layer_angles if angle is not None]
    mean_angle = (
        float(sum(valid_angles) / len(valid_angles)) if valid_angles else None
    )
    return AngleSummary(layer_angles=layer_angles, mean_angle=mean_angle)


def accuracy_from_positive_output(output_activation: Tensor, targets: Tensor) -> float:
    logits = torch.log(output_activation.clamp_min(EPSILON))
    predictions = logits.argmax(dim=1)
    return float((predictions == targets).float().mean().item())


def nll_from_positive_output(output_activation: Tensor, targets: Tensor) -> Tensor:
    logits = torch.log(output_activation.clamp_min(EPSILON))
    return F.cross_entropy(logits, targets)
