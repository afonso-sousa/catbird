from torch import nn

from catbird.models.losses import pair_wise_loss

from ..builder import build_discriminator
from ..registry import GENERATORS
from .base import EncoderDecoderBase


@GENERATORS.register_module
class EDD(EncoderDecoderBase):
    """
    LSTM-based Encoder-Decoder-Discriminator (EDD) architecture.

    This is an implementation of paper `Learning Semantic Sentence Embeddings
    using Sequential Pair-wise Discriminator <https://aclanthology.org/C18-1230/>`.
    """

    def __init__(
        self,
        pad_token_id,
        eos_token_id,
        decoder_start_token_id,
        encoder,
        decoder,
        discriminator,
    ):
        super(EDD, self).__init__(
            pad_token_id, eos_token_id, decoder_start_token_id, encoder, decoder
        )
        self.discriminator = build_discriminator(discriminator)

    def forward(
        self,
        input_ids,
        labels=None,
        decoder_input_ids=None,
        incremental_state=None,
        **kwargs
    ):
        if (labels is not None) and (decoder_input_ids is None):
            decoder_input_ids = labels

        if (labels is None) and (decoder_input_ids is None):
            decoder_input_ids = input_ids

        encoder_outputs = self.encoder(input_ids=input_ids, **kwargs)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_out=encoder_outputs,
            incremental_state=incremental_state,
            **kwargs,
        )
        discriminated_out, discriminated_tgt = self.discriminator(
            decoder_outputs[0], labels
        )
        loss = self.loss(
            decoder_outputs[0], labels, discriminated_out, discriminated_tgt
        )
        return (loss,) + decoder_outputs

    def loss(self, logits, tgt, discriminated_out, discriminated_tgt):
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        loss = loss_fct(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        pwd_loss = pair_wise_loss(discriminated_out, discriminated_tgt)
        return loss + pwd_loss
