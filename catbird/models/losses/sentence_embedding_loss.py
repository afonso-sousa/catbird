import torch


def sent_emb_loss(src_feats: torch.Tensor, tgt_feats: torch.Tensor) -> torch.Tensor:
    """Calculate sentence embedding loss.

    Proposed in `Learning Semantic Sentence Embeddings using Sequential
    Pair-wise Discriminator <https://aclanthology.org/C18-1230/>`.

    Args:
        src_emb_feats (torch.Tensor): source sentence embedding features
        reference_emb_feats (torch.Tensor): reference (paraphrase) embedding features

    Returns:
        [torch.Tensor]: sentence embedding loss
    """
    batch_size = src_feats.size(0)

    a = torch.mm(src_feats, tgt_feats.t())
    b = torch.sum(src_feats * tgt_feats, 1)
    c = a - b + 1
    loss = torch.sum(torch.clamp(c, min=0.0)) / batch_size ** 2

    return loss
