import torch
from torch import Tensor, nn
from torch.nn import functional as F
from catbird.utils import GenerationMixin

from ..builder import build_decoder, build_encoder


class EncoderDecoderBase(nn.Module, GenerationMixin):
    def __init__(
        self, pad_token_id, eos_token_id, decoder_start_token_id, encoder, decoder
    ):
        super(EncoderDecoderBase, self).__init__()
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.encoder = build_encoder(encoder)
        self.decoder = build_decoder(decoder)

    def shift_tokens_right(self, input_ids: torch.Tensor, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        return shifted_input_ids

    def forward(
        self,
        input_ids,
        labels=None,
        decoder_input_ids=None,
        incremental_state=None,
        **kwargs,
    ):
        if (labels is not None) and (decoder_input_ids is None):
            # print("Shifting labels for decoder input...")
            decoder_input_ids = self.shift_tokens_right(
                labels, self.decoder_start_token_id
            )
            # decoder_input_ids = labels[:, :-1].contiguous()

        if (labels is None) and (decoder_input_ids is None):
            decoder_input_ids = input_ids

        encoder_outputs = self.encoder(input_ids=input_ids, **kwargs)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_out=encoder_outputs,
            incremental_state=incremental_state,
            **kwargs,
        )

        loss = None
        if labels is not None:
            logits = decoder_outputs[0]
            loss = self.loss(logits, labels)

        if loss is not None:
            return (loss,) + decoder_outputs + encoder_outputs
        else:
            return decoder_outputs + encoder_outputs

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def loss(self, logits, labels):
        targets = labels
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        loss = loss_fct(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return loss
