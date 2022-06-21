from ..registry import GENERATORS
from .base import EncoderDecoderBase
from ..builder import build_graph_encoder
import torch


@GENERATORS.register_module
class GCNLSTM(EncoderDecoderBase):
    """This is an implementation of paper ` <https://>`.
    """

    def __init__(self,
                decoder_start_token_id,
                graph_encoder,
                encoder,
                decoder):
        super(GCNLSTM, self).__init__(decoder_start_token_id, encoder, decoder)
        self.graph_encoder = build_graph_encoder(graph_encoder)

    def forward(self, input_ids, ie_graph, prev_output_tokens, **kwargs):
        state = self.encoder(input_ids)
        graph_embeddings = self.graph_encoder(ie_graph)
        
        state.hidden = torch.cat((state.hidden[0][:1], graph_embeddings.unsqueeze(0))), state.hidden[1]
        
        decoder_out = self.decoder(prev_output_tokens, state=state)
        return decoder_out