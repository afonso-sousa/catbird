from ..registry import GENERATORS
from .base import Seq2Seq
from ..builder import build_graph_encoder
import torch


@GENERATORS.register_module
class GCNLSTM(Seq2Seq):
    """This is an implementation of paper ` <https://>`.
    """

    def __init__(self,
                graph_encoder,
                encoder,
                decoder):
        super(GCNLSTM, self).__init__(encoder, decoder)
        # self.graph_layer = IETripleGraph(**graph_layer)
        self.graph_encoder = build_graph_encoder(graph_encoder)

    def forward(self, input_ids, ie_graph, prev_output_tokens, **kwargs):
        state = self.encoder(input_ids)
        graph_embeddings = self.graph_encoder(ie_graph)
        
        # graph_embeddings = graph_embeddings.expand_as(state.hidden[0])
        state.hidden = torch.cat((state.hidden[0][:1], graph_embeddings)), state.hidden[1]
        
        decoder_out = self.decoder(prev_output_tokens, state=state)
        return decoder_out