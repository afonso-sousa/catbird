import torch

from ..builder import build_graph_encoder
from ..registry import GENERATORS
from .base import EncoderDecoderBase


@GENERATORS.register_module
class RGATLSTM(EncoderDecoderBase):
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
        super(RGATLSTM, self).__init__(
            pad_token_id, eos_token_id, decoder_start_token_id, encoder, decoder
        )
        self.graph_encoder = build_graph_encoder(graph_encoder)

    def forward(self, input_ids, graph, labels=None, decoder_input_ids=None, **kwargs):
        if (labels is not None) and (decoder_input_ids is None):
            decoder_input_ids = labels[:, :-1].contiguous()

        encoder_outputs = self.encoder(input_ids)
        graph_embeddings = self.graph_encoder(graph)

        encoder_outputs = (
            encoder_outputs[0],
            torch.cat((encoder_outputs[1][:1], graph_embeddings.unsqueeze(0))),
            encoder_outputs[2],
            encoder_outputs[3],
        )

        decoder_out = self.decoder(decoder_input_ids, encoder_outputs)
        return decoder_out
