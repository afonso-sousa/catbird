import torch
import torch.nn as nn
import math
from random import randrange, shuffle
from copy import deepcopy
from .seq2seq_base import Seq2Seq


class Transformer(Seq2Seq):

    def __init__(self, vocab_size, hidden_size=512, embedding_size=None, num_layers=6, num_heads=8,
                 inner_linear=2048, inner_groups=1, dropout=0.1, prenormalized=False, tie_embedding=True,
                 encoder=None, decoder=None, layer_norm=True, weight_norm=False, batch_first=True, stateful=None):
        super(Transformer, self).__init__()
        embedding_size = embedding_size or hidden_size
        # keeping encoder, decoder None will result with default configuration
        encoder = encoder or {}
        decoder = decoder or {}
        encoder = deepcopy(encoder)
        decoder = deepcopy(decoder)
        encoder.setdefault('embedding_size', embedding_size)
        encoder.setdefault('hidden_size', hidden_size)
        encoder.setdefault('num_layers', num_layers)
        encoder.setdefault('num_heads', num_heads)
        encoder.setdefault('vocab_size', vocab_size)
        encoder.setdefault('layer_norm', layer_norm)
        encoder.setdefault('weight_norm', weight_norm)
        encoder.setdefault('dropout', dropout)
        encoder.setdefault('inner_linear', inner_linear)
        encoder.setdefault('inner_groups', inner_groups)
        encoder.setdefault('prenormalized', prenormalized)
        encoder.setdefault('batch_first', batch_first)

        decoder.setdefault('embedding_size', embedding_size)
        decoder.setdefault('hidden_size', hidden_size)
        decoder.setdefault('num_layers', num_layers)
        decoder.setdefault('num_heads', num_heads)
        decoder.setdefault('tie_embedding', tie_embedding)
        decoder.setdefault('vocab_size', vocab_size)
        decoder.setdefault('layer_norm', layer_norm)
        decoder.setdefault('weight_norm', weight_norm)
        decoder.setdefault('dropout', dropout)
        decoder.setdefault('inner_linear', inner_linear)
        decoder.setdefault('inner_groups', inner_groups)
        decoder.setdefault('batch_first', batch_first)
        decoder.setdefault('prenormalized', prenormalized)
        decoder.setdefault('stateful', stateful)

        if isinstance(vocab_size, tuple):
            embedder = CharWordEmbedder(
                vocab_size[1], embedding_size, hidden_size)
            encoder.setdefault('embedder', embedder)
            decoder.setdefault('embedder', embedder)
            decoder['classifier'] = False

        self.batch_first = batch_first
        self.encoder = TransformerAttentionEncoder(**encoder)
        self.decoder = TransformerAttentionDecoder(**decoder)

        if tie_embedding and not isinstance(vocab_size, tuple):
            assert self.encoder.embedder.weight.shape == self.decoder.classifier.weight.shape
            self.encoder.embedder.weight = self.decoder.classifier.weight
            if embedding_size != hidden_size:
                self.encoder.input_projection = self.decoder.input_projection