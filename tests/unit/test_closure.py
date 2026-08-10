import pytest
import torch

from popgp.geometry import closure_mismatch


def test_closure_mismatch_is_zero_for_matching_tensors() -> None:
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    mismatch = closure_mismatch(2.0 * source, source, coupling=2.0)

    assert mismatch.absolute_norm == 0.0
    assert mismatch.relative_norm == 0.0


def test_closure_mismatch_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        closure_mismatch(torch.zeros(2), torch.zeros(3))
