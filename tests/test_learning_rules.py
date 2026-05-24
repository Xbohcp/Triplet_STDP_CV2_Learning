import math

import pytest

torch = pytest.importorskip("torch")


def test_vector_angle_degrees_handles_zero_vectors():
    from triplet_stdp_cv2_learning.learning_rules import vector_angle_degrees

    assert vector_angle_degrees(torch.zeros(3), torch.ones(3)) is None


def test_vector_angle_degrees_returns_expected_right_angle():
    from triplet_stdp_cv2_learning.learning_rules import vector_angle_degrees

    angle = vector_angle_degrees(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 2.0]))

    assert math.isclose(angle, 90.0, rel_tol=1e-6, abs_tol=1e-6)


def test_bp_and_biological_updates_have_matching_weight_shapes():
    from triplet_stdp_cv2_learning.learning_rules import (
        biological_weight_deltas,
        bp_weight_deltas,
    )
    from triplet_stdp_cv2_learning.model import PositiveMLP

    model = PositiveMLP(input_dim=4, hidden_dim=5, output_dim=3, hidden_layers=2)
    inputs = torch.rand(7, 4)
    targets = torch.tensor([0, 1, 2, 1, 0, 2, 1])
    activations = model.forward_with_cache(inputs)
    loss = model.loss_from_output(activations[-1], targets)

    bp_deltas = bp_weight_deltas(model, loss, learning_rate=0.05)
    bio_deltas = biological_weight_deltas(
        model,
        activations,
        targets,
        learning_rate=0.05,
        scale_mode="tanh",
        update_function="epsilon_times_activation",
    )

    assert len(bp_deltas) == len(bio_deltas) == 3
    for layer, bp_delta, bio_delta in zip(model.layers, bp_deltas, bio_deltas):
        assert bp_delta.shape == layer.weight.shape
        assert bio_delta.shape == layer.weight.shape
