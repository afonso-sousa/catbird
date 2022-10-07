import torch
from catbird.datasets.utils import batch_sequences


def test_batch_loader():
    s1 = torch.LongTensor([1, 2, 3, 4, 5, 6])
    s2 = torch.LongTensor([10, 20, 30])

    seqs = [s1, s2]
    batch = batch_sequences(seqs, max_length=4)

    expected_tensor = torch.tensor([[1, 2, 3, 4], [10, 20, 30, 0]])
    assert torch.all(torch.eq(expected_tensor, batch[0]))

    expected_lengths = [4, 3]
    assert expected_lengths == batch[1]
