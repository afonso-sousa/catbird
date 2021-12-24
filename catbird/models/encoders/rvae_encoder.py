import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    def __init__(self, size, num_layers, f):

        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear = [nn.Linear(size, size) for _ in range(num_layers)]
        for i, module in enumerate(self.nonlinear):
            self._add_to_parameters(
                module.parameters(), "nonlinear_module_{}".format(i)
            )

        self.linear = [nn.Linear(size, size) for _ in range(num_layers)]
        for i, module in enumerate(self.linear):
            self._add_to_parameters(module.parameters(), "linear_module_{}".format(i))

        self.gate = [nn.Linear(size, size) for _ in range(num_layers)]
        for i, module in enumerate(self.gate):
            self._add_to_parameters(module.parameters(), "gate_module_{}".format(i))

        self.f = f

    def forward(self, x):
        """
        :param x: tensor with shape of [batch_size, size]
        :return: tensor with shape of [batch_size, size]
        applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
        """

        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear

        return x

    def _add_to_parameters(self, parameters, name):
        for i, parameter in enumerate(parameters):
            self.register_parameter(name="{}-{}".format(name, i), param=parameter)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()

        self.params = params

        self.hw1 = Highway(
            self.params.sum_depth + self.params.word_embed_size, 2, F.relu
        )

        self.rnn = nn.LSTM(
            input_size=self.params.word_embed_size + self.params.sum_depth,
            hidden_size=self.params.encoder_rnn_size,
            num_layers=self.params.encoder_num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, input, State):
        """
        :param input: [batch_size, seq_len, embed_size] tensor
        :return: context of input sentenses with shape of [batch_size, latent_variable_size]
        """
        [batch_size, seq_len, embed_size] = input.size()

        input = input.view(-1, embed_size)
        input = self.hw1(input)
        input = input.view(batch_size, seq_len, embed_size)

        """ Unfold rnn with zero initial state and get its final state from the last layer
        """
        _, (transfer_state_1, final_state) = self.rnn(input, State)
        transfer_state_2 = final_state

        final_state = final_state.view(
            self.params.encoder_num_layers, 2, batch_size, self.params.encoder_rnn_size
        )
        final_state = final_state[-1]
        h_1, h_2 = final_state[0], final_state[1]
        final_state = torch.cat([h_1, h_2], 1)

        return final_state, transfer_state_1, transfer_state_2
