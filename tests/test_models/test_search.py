from pathlib import Path

import torch
from catbird.core import Config
from catbird.core.utils.registry import build_from_cfg
from catbird.models.registry import GENERATORS


data_path = Path(__file__).parent.parent / "data"

def test_beam_search():
    cfg_file = Path(data_path, "config", "dummy_cfg.py")
    cfg = Config.fromfile(cfg_file)

    cfg.embedding_length = 10000
    cfg.pad_token_id = 0

    cfg.model.encoder.vocabulary_size = cfg.embedding_length
    cfg.model.encoder.pad_token_id = cfg.pad_token_id
    cfg.model.decoder.vocabulary_size = cfg.embedding_length
    cfg.model.decoder.pad_token_id = cfg.pad_token_id

    model = build_from_cfg(cfg.model, GENERATORS).cuda()

    batch_size = 32
    beam_size = 2
    max_output_length=50
    length_normalization_factor = 0

    src = torch.randint(1, 1000, (batch_size, 80)).cuda()
    
    # EOS_TOKEN = 2
    # bos = [[EOS_TOKEN]] * batch_size

    with torch.no_grad():
        seqs = model.generate(src,
                        beam_size=beam_size,
                        max_sequence_length=max_output_length,
                        length_normalization_factor=length_normalization_factor,)
    
    preds = [[el.item() for el in s.output[1:]] for s in seqs]
