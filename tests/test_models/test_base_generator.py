import torch
from catbird.core import Config
from catbird.tokenizers import build_tokenizer
from catbird.models.generators.base import shift_right


def test_shift_right():
    cfg_dict = dict(
            num_workers=4,
            data=dict(
                max_length=40,
                train=dict(dataset_length=-1),
                val=dict(dataset_length=2000),
            ),
            train=dict(batch_size=32),
            model=dict(name="t5-small"),
    )
    cfg = Config(cfg_dict)

    tokenizer = build_tokenizer(cfg)
    cfg.embedding_length = len(tokenizer)
    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.eos_token_id = (
        tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.sep_token_id
    )
    cfg.decoder_start_token_id = (
        tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
    )
    
    # 'What is ASMR? Does everyone experience it?</s><pad>'
    # 'What are some good tips to lose weight?</s><pad>'
    sample_inputs = torch.tensor([[ 363,   19, 6157, 9320,   58, 3520,  921,  351,   34,   58,    1,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0],
        [ 363,   33,  128,  207, 2316,   12, 2615, 1293,   58,    1,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0]])

    # '</s> What is ASMR? Does everyone experience it?</s><pad>'
    # '</s> What are some good tips to lose weight?</s><pad>'
    shifted_sample_inputs = shift_right(sample_inputs, cfg.decoder_start_token_id)
    
    assert torch.all(
        shifted_sample_inputs.eq(
            torch.cat(
                (torch.full((sample_inputs.size(0), 1), cfg.decoder_start_token_id),
                sample_inputs[..., :-1].clone()), dim=1
            )
        )
    )
