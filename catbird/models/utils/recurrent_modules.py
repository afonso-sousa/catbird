from torch import nn
import torch

def Recurrent(mode, input_size, hidden_size,
              num_layers=1, bias=True, batch_first=False,
              dropout=0, bidirectional=False, residual=False):
    params = dict(input_size=input_size, hidden_size=hidden_size,
                  num_layers=num_layers, bias=bias, batch_first=batch_first,
                  dropout=dropout, bidirectional=bidirectional)

    if mode == 'LSTM':
        rnn = nn.LSTM
    elif mode == 'GRU':
        rnn = nn.GRU
    elif mode == 'RNN':
        rnn = nn.RNN
        params['nonlinearity'] = 'tanh'
    elif mode == 'iRNN':
        rnn = nn.RNN
        params['nonlinearity'] = 'relu'
    else:
        raise Exception('Unknown mode: {}'.format(mode))
    if residual:
        rnn = wrap_stacked_recurrent(rnn,
                                        num_layers=num_layers,
                                        residual=True)
        params['num_layers'] = 1
    if params.get('num_layers', 0) == 1:
        params.pop('dropout', None)
    module = rnn(**params)

    if mode == 'iRNN':
        for n, p in module.named_parameters():
            if 'weight_hh' in n:
                p.detach().copy_(torch.eye(*p.shape))
    return module


def wrap_stacked_recurrent(recurrent_func, num_layers=1, residual=False):
    def f(*kargs, **kwargs):
        module = StackedRecurrent(residual)
        for i in range(num_layers):
            rnn = recurrent_func(*kargs, **kwargs)
            module.add_module(str(i), rnn)
        return module
    return f


class StackedRecurrent(nn.Sequential):

    def __init__(self, dropout=0, residual=False):
        super(StackedRecurrent, self).__init__()
        self.residual = residual
        self.dropout = dropout

    def forward(self, inputs, hidden=None):
        hidden = hidden or tuple([None] * len(self))
        next_hidden = []
        for i, module in enumerate(self._modules.values()):
            output, h = module(inputs, hidden[i])
            next_hidden.append(h)
            if self.residual and inputs.size(-1) == output.size(-1):
                inputs = output + inputs
            else:
                inputs = output

            inputs = nn.functional.dropout(
                inputs, self.dropout, self.training)

        return output, tuple(next_hidden)