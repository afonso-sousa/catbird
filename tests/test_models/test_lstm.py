from pathlib import Path

import torch
from catbird.core import Config
from catbird.core.utils.registry import build_from_cfg
from catbird.models.registry import GENERATORS
from catbird.models.utils.recurrent_modules import Recurrent
from torch import nn

data_path = Path(__file__).parent.parent / "data"


def test_recurrent():
    vocab_size = 1000
    batch_size = 32
    hidden_dim = 128
    max_length = 80
    num_layers = 2
    input = torch.randint(0, vocab_size, (batch_size, max_length))
    embedder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim)
    embeddings = embedder(input)
    assert embeddings.shape == (batch_size, max_length, hidden_dim)
    rnn = Recurrent("LSTM", hidden_dim, hidden_dim, num_layers)
    output, hidden_t = rnn(embeddings)
    assert output.shape == (batch_size, max_length, hidden_dim)
    assert hidden_t[0].shape == (num_layers, max_length, hidden_dim)


def test_lstm_structure():
    cfg_file = Path(data_path, "config", "dummy_cfg.py")
    cfg = Config.fromfile(cfg_file)

    cfg.embedding_length = 10000
    cfg.pad_token_id = 0

    cfg.model.encoder.vocab_size = cfg.embedding_length
    cfg.model.encoder.pad_token_id = cfg.pad_token_id
    cfg.model.decoder.vocab_size = cfg.embedding_length
    cfg.model.decoder.pad_token_id = cfg.pad_token_id

    batch_size = 32
    num_layers = 2
    max_length = 80
    hidden_dim = 128

    input_ids = torch.randint(1, 100, (batch_size, max_length))
    src_lengths = torch.randint(5, 15, (batch_size,))

    mask = torch.zeros(input_ids.shape[0], input_ids.shape[1])
    mask[(torch.arange(input_ids.shape[0]), src_lengths)] = 1
    mask = mask.cumsum(dim=1)

    input_ids = input_ids * (1.0 - mask)

    prev_output_tokens = torch.cat((input_ids[:, :1], input_ids[:, 1:]), dim=1)

    input_ids = input_ids.to(torch.int64)
    prev_output_tokens = prev_output_tokens.to(torch.int64)

    model = build_from_cfg(cfg.model, GENERATORS)

    assert input_ids.shape == (batch_size, max_length)
    assert src_lengths.shape == (batch_size,)
    assert prev_output_tokens.shape == (batch_size, max_length)

    state = model.encoder(input_ids)
    assert state.hidden[0].shape == (num_layers, batch_size, hidden_dim)

    decoder_out, _ = model.decoder(prev_output_tokens, state)
    assert decoder_out.shape == (batch_size, max_length, cfg.embedding_length)

    output, _ = model(input_ids, prev_output_tokens)

    assert output.shape == (
        cfg.train.batch_size,
        prev_output_tokens.size(-1),
        cfg.embedding_length,
    )
