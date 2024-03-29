import torch

from ..builder import build_graph_encoder
from ..registry import GENERATORS
from .base import EncoderDecoderBase


@GENERATORS.register_module
class GCNLSTM(EncoderDecoderBase):
    """This is an implementation of paper ` <https://>`."""

    def __init__(
        self,
        pad_token_id,
        eos_token_id,
        decoder_start_token_id,
        graph_encoder,
        encoder,
        decoder,
    ):
        super(GCNLSTM, self).__init__(
            pad_token_id, eos_token_id, decoder_start_token_id, encoder, decoder
        )
        self.graph_encoder = build_graph_encoder(graph_encoder)

    def forward(
        self,
        input_ids,
        graph=None,
        labels=None,
        decoder_input_ids=None,
        incremental_state=None,
        **kwargs
    ):
        if (labels is not None) and (decoder_input_ids is None):
            decoder_input_ids = super().shift_tokens_right(
                labels, self.decoder_start_token_id
            )

        encoder_outputs = self.encoder(input_ids=input_ids, **kwargs)
        graph_embeddings = self.graph_encoder(graph, self.encoder.embed_tokens)

        # encoder_outputs = (
        #     encoder_outputs[0],
        #     torch.cat((encoder_outputs[1][:1], graph_embeddings.unsqueeze(0))),
        #     encoder_outputs[2],
        #     encoder_outputs[3],
        # )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_out=encoder_outputs,
            incremental_state=incremental_state,
            graph_embeddings=graph_embeddings,
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
