from catbird.core.utils.registry import build_from_cfg
from catbird.models.registry import GENERATORS
from pathlib import Path
from catbird.core import Config
import torch

data_path = Path(__file__).parent / "data"


# def generate(self, input_encoder, input_lengths, input_decoder, beam_size=None,
#                  max_sequence_length=None, length_normalization_factor=0,
#                  get_attention=False, autoregressive=True):
#         encoder_out = self.encoder(input_encoder, input_lengths)
#         state_list = state.as_list()
#         params = dict(decode_step=self._decode_step,
#                       beam_size=beam_size,
#                       max_sequence_length=max_sequence_length,
#                       get_attention=get_attention,
#                       length_normalization_factor=length_normalization_factor,)
#         if autoregressive:
#             generator = SequenceGenerator(**params)
#         return generator.beam_search(input_decoder, encoder_out)


# def test_generation():
#     cfg_file = Path(data_path, "config", "dummy_cfg.py")
#     cfg = Config.fromfile(cfg_file)

#     cfg.embedding_length = 10000
#     cfg.pad_token_id = 0

#     cfg.model.encoder.vocabulary_size = cfg.embedding_length
#     cfg.model.encoder.pad_token_id = cfg.pad_token_id
#     cfg.model.decoder.vocabulary_size = cfg.embedding_length
#     cfg.model.decoder.pad_token_id = cfg.pad_token_id
#     model = build_from_cfg(cfg.model, GENERATORS)
#     setattr(type(model), 'generate', generate)
    
#     src = torch.randint(1, 1000, (32, 80))
#     lens = torch.randint(1, 1000, (32,))
#     bos = [[1]] * 32
    
#     seqs = model.generate(src, lens, bos,
#         beam_size=5,
#         max_sequence_length=80,)
    
#     print(seqs)

def test_model():
    from model import Encoder, Decoder, Seq2Seq
    
    src = torch.randint(1, 1000, (32, 80)).cuda()
    trg = torch.randint(1, 1000, (32, 80)).cuda()
    
    encoder = Encoder(10000, 128, 128,
                      n_layers=2, dropout=0.5)
    decoder = Decoder(128, 128, 10000,
                      n_layers=1, dropout=0.0)
    seq2seq = Seq2Seq(encoder, decoder).cuda()
    output = seq2seq(src, trg)
    # print(output)
    # print(output.shape)
    
    # decoded_batch = seq2seq.decode(src, trg, method='beam-search')
    decoded_batch = seq2seq.decode(src, trg, method='greedy-search')
    # print(decoded_batch)
    # print(decoded_batch.shape)
    
    assert False
    
    cfg_file = Path(data_path, "config", "dummy_cfg.py")
    cfg = Config.fromfile(cfg_file)

    cfg.embedding_length = 10000
    cfg.pad_token_id = 0

    cfg.model.encoder.vocabulary_size = cfg.embedding_length
    cfg.model.encoder.pad_token_id = cfg.pad_token_id
    cfg.model.decoder.vocabulary_size = cfg.embedding_length
    cfg.model.decoder.pad_token_id = cfg.pad_token_id
    
    model = build_from_cfg(cfg.model, GENERATORS).cuda()
    output2, _ = model(src, trg)
    print(output.shape)
    print(output2.shape)
    
    model.generate(src, trg)
    
    print(decoded_batch.shape)
    
    assert False