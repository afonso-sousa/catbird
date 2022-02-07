import torch
from torch import nn


def freeze_params(model: nn.Module) -> None:
    """Disable weight updates on given model.

    Args:
        model (nn.Module): model
    """
    for par in model.parameters():
        par.requires_grad = False


def one_hot(t, c):
    return torch.zeros(*t.size(), c, device=t.device).scatter_(
        -1, t.unsqueeze(-1).data.long(), 1
    )


def convert_padding_direction(
    src_tokens, padding_idx, right_to_left: bool = False, left_to_right: bool = False
):
    assert right_to_left ^ left_to_right
    pad_mask = src_tokens.eq(padding_idx)
    if not pad_mask.any():
        # no padding, return early
        return src_tokens
    if left_to_right and not pad_mask[:, 0].any():
        # already right padded
        return src_tokens
    if right_to_left and not pad_mask[:, -1].any():
        # already left padded
        return src_tokens
    max_len = src_tokens.size(1)
    buffered = torch.empty(0).long()
    if max_len > 0:
        torch.arange(max_len, out=buffered)
    range = buffered.type_as(src_tokens).expand_as(src_tokens)
    num_pads = pad_mask.long().sum(dim=1, keepdim=True)
    if right_to_left:
        index = torch.remainder(range - num_pads, max_len)
    else:
        index = torch.remainder(range + num_pads, max_len)
    return src_tokens.gather(1, index)

