import torch
import torch.nn as nn
from catbird.models.generators.base import EncoderDecoderBase

from ..registry import GENERATORS


@GENERATORS.register_module
class VanillaTransformer(EncoderDecoderBase):
    """This is an implementation of the Transformer model from `"Attention Is All You Need"
    <https://arxiv.org/abs/1706.03762>`.
    """

    def __init__(
        self, pad_token_id, eos_token_id, decoder_start_token_id, encoder, decoder
    ):
        super(VanillaTransformer, self).__init__(
            pad_token_id, eos_token_id, decoder_start_token_id, encoder, decoder
        )

    def forward(self, input_ids, labels, **kwargs):
        input_ids = input_ids.t()
        labels = labels.t()

        decoder_input_ids = labels[:-1, :]
        decoder_input_ids.masked_fill_(decoder_input_ids == -100, self.pad_token_id)

        memory = self.encoder(input_ids)
        output = self.decoder(decoder_input_ids, memory)

        # loss_fn = nn.CrossEntropyLoss()
        # loss = loss_fn(output.reshape(-1, output.shape[-1]), labels[1:, :].reshape(-1))

        return self.loss(output, labels), output


    def generate(self, input_ids, **kwargs):
        input_ids = input_ids.t()

        device = next(self.parameters()).device

        memory = self.encoder(input_ids[:, 0].t().unsqueeze(1))
        input_ids.masked_fill_(input_ids == -100, self.pad_token_id)

        seq_len, batch_size = input_ids.shape

        decoder_input_ids = (
            torch.full((1, 1), self.decoder_start_token_id).type(torch.long).to(device)
        )

        for _ in range(seq_len - 1):
            out = self.decoder(decoder_input_ids, memory)

            next_token_logits = out[-1, :, :]

            next_token = torch.argmax(next_token_logits, dim=-1)

            decoder_input_ids = torch.cat(
                [decoder_input_ids, next_token.unsqueeze(-1)], dim=0
            )

            if next_token.item() == self.eos_token_id:
                break
        return decoder_input_ids
