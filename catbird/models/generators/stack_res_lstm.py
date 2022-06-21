import random

import torch
import torch.nn.functional as F

from ..registry import GENERATORS
from .base import EncoderDecoderBase


@GENERATORS.register_module
class StackedResidualLSTM(EncoderDecoderBase):
    """This is an implementation of paper `Neural Paraphrase Generation
    with Stacked Residual LSTM Networks <https://aclanthology.org/C16-1275/>`.
    """

    def __init__(
        self, pad_token_id, eos_token_id, decoder_start_token_id, encoder, decoder
    ):
        super(StackedResidualLSTM, self).__init__(
            pad_token_id, eos_token_id, decoder_start_token_id, encoder, decoder
        )

    def forward(self, input_ids, labels, teacher_forcing_ratio=0.5, **kwargs):
        input_ids = input_ids.t()
        labels = labels.t()

        decoder_input_ids = labels[:-1, :]

        decoder_input_ids.masked_fill_(decoder_input_ids == -100, self.pad_token_id)

        device = next(self.decoder.parameters()).device

        batch_size = input_ids.size(1)
        max_len = decoder_input_ids.size(0)
        vocab_size = self.decoder.vocab_size
        outputs = torch.zeros(max_len, batch_size, vocab_size).to(device)

        encoder_output, hidden = self.encoder(
            input_ids
        )  # [27, 32]=> =>[27, 32, 512],[4, 32, 512]
        hidden = hidden[: self.decoder.num_layers]  # [4, 32, 512][1, 32, 512]
        output = decoder_input_ids.data[0, :]  # sos

        for t in range(1, max_len):
            # print(output.shape) # [128]
            # print(hidden.shape) # [1, 128, 512]
            # print(encoder_output.shape) # [50, 128, 512]
            output, hidden, attn_weights = self.decoder(
                output, hidden, encoder_output
            )  # output:[32, 10004] [1, 32, 512] [32, 1, 27]
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[
                1
            ]  # 按照 dim=1 求解最大值和最大值索引,x[1] 得到的是最大值的索引=>top1.shape=32
            output = decoder_input_ids.data[t] if is_teacher else top1

        loss = F.nll_loss(
            outputs.view(-1, vocab_size), labels[1:].contiguous().view(-1),
        )

        return loss, outputs

    # def decode(self, decoder_input_ids, method='beam-search'):
    #     # decoder_input_ids.masked_fill_(decoder_input_ids == -100, self.pad_token_id)

    #     encoder_output, hidden = self.encoder(decoder_input_ids)  # [27, 32]=> =>[27, 32, 512],[4, 32, 512]
    #     hidden = hidden[:self.decoder.num_layers]  # [4, 32, 512][1, 32, 512]
    #     decoder_input_ids.masked_fill_(decoder_input_ids == -100, self.pad_token_id)
    #     if method == 'beam-search':
    #         return self.beam_decode(decoder_input_ids, hidden, encoder_output)
    #     else:
    #         return self.greedy_decode(decoder_input_ids, hidden, encoder_output)

    def generate(self, input_ids, **kwargs):
        """
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch
        """
        input_ids = input_ids.t()

        encoder_output, hidden = self.encoder(
            input_ids
        )  # [27, 32]=> =>[27, 32, 512],[4, 32, 512]
        hidden = hidden[: self.decoder.num_layers]  # [4, 32, 512][1, 32, 512]
        input_ids.masked_fill_(input_ids == -100, self.pad_token_id)

        seq_len, batch_size = input_ids.size()
        decoded_batch = torch.zeros((batch_size, seq_len))
        decoder_input = input_ids.data[0, :]  # sos

        for t in range(seq_len):
            decoder_output, hidden, _ = self.decoder(
                decoder_input, hidden, encoder_output
            )
            _, topi = decoder_output.data.topk(1)  # [32, 10004] get candidates
            topi = topi.view(-1)
            decoded_batch[:, t] = topi

            decoder_input = topi.detach().view(-1)

        return decoded_batch.int().t()

