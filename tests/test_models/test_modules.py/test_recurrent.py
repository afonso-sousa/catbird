from catbird.models.modules.recurrent_modules import RecurrentLayer
import torch


def test_recurrent_layer():
    batch_size = 16
    hidden_size = 128
    num_layers = 3
    seq_length = 10

    x = torch.rand(seq_length, batch_size, hidden_size)

    rnn = RecurrentLayer(
        "LSTM",
        hidden_size,
        hidden_size,
        num_layers=3,
        bias=True,
        batch_first=False,
        residual=True,
        dropout=0.0,
        bidirectional=False,
    )
    state_size = num_layers, batch_size, hidden_size

    h0 = x.new_zeros(*state_size)
    c0 = x.new_zeros(*state_size)
    x, (final_hiddens, final_cells) = rnn(
        x, (h0, c0)
    )  # [seq_len, btz, hid], ([nlayers, btz, hid], [nlayers, btz, hid])

    print(x.shape)
    print(final_hiddens.shape)
    print(final_cells.shape)

    assert False
