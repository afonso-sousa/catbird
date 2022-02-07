import torch
from catbird.datasets import TeacherForcing


def test_teacher_forcing():
    src_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    targets = torch.tensor([[1, 2, 3], [4, 5, 6]])
    
    target_forcing = TeacherForcing(batch_first=True, token_id=102)
    encoded, decoded, _ = target_forcing(src_ids, targets)
    print(decoded)
    assert torch.all(encoded.eq(src_ids))
    assert torch.all(decoded.eq(torch.tensor([[102, 1, 2], [102, 4, 5]])))

    target_forcing = TeacherForcing(batch_first=False, token_id=102)
    encoded, decoded, _ = target_forcing(src_ids, targets)
    assert torch.all(encoded.eq(src_ids))
    assert torch.all(decoded.eq(torch.tensor([[102, 102, 102], [1, 2, 3]])))
    
    target_forcing = TeacherForcing(batch_first=True)
    encoded, decoded, _ = target_forcing(src_ids, targets)
    assert torch.all(encoded.eq(src_ids))
    assert torch.all(decoded.eq(torch.tensor([[1, 1, 2], [4, 4, 5]])))

    target_forcing = TeacherForcing(batch_first=False)
    encoded, decoded, _ = target_forcing(src_ids, targets)
    assert torch.all(encoded.eq(src_ids))
    assert torch.all(decoded.eq(torch.tensor([[1, 2, 3], [1, 2, 3]])))
