import math

import pytest
import torch
from catbird.models.losses import sent_emb_loss


@pytest.mark.parametrize("loss_method", [sent_emb_loss])
def test_loss(loss_method):
    pred = torch.rand((32, 512))
    target = torch.rand((32, 512))

    loss = loss_method(pred, target)
    assert loss > 0

    loss = loss_method(target, target)
    assert math.isclose(loss.item(), 0.03, rel_tol=0.1)

    assert isinstance(loss, torch.Tensor)
