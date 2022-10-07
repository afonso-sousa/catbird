from pathlib import Path

from catbird.utils import Config
from catbird.utils.registry import Registry, build_from_cfg
from catbird.models.registry import DECODERS, ENCODERS, GENERATORS

data_path = Path(__file__).parent.parent / "data"


def test_registry():
    MODELS = Registry("models")

    @MODELS.register_module
    class LSTM:
        pass

    _ = build_from_cfg(dict(type="LSTM"), MODELS)

    cfg_file = Path(data_path, "config", "dummy_cfg.py")
    cfg = Config.fromfile(cfg_file)

    assert "model" in cfg
    assert "encoder" in cfg.model and "decoder" in cfg.model

    cfg.model.encoder.vocab_size = 10000
    cfg.model.encoder.pad_token_id = 0
    # cfg.model.encoder.eos_token_id = 1
    # cfg.model.encoder.decoder_start_token_id = 2

    assert all(
        k in cfg.model.encoder
        for k in [
            "vocab_size",
            "pad_token_id",
            # "eos_token_id",
            # "decoder_start_token_id",
        ]
    )

    _ = build_from_cfg(cfg.model.encoder, ENCODERS)

    cfg.model.decoder.vocab_size = 10000
    cfg.model.decoder.pad_token_id = 0
    # cfg.model.decoder.eos_token_id = 1
    # cfg.model.decoder.decoder_start_token_id = 2

    assert all(
        k in cfg.model.decoder
        for k in [
            "vocab_size",
            "pad_token_id",
            # "eos_token_id",
            # "decoder_start_token_id",
        ]
    )

    _ = build_from_cfg(cfg.model.decoder, DECODERS)

    cfg.model.pad_token_id = 0
    cfg.model.eos_token_id = 1
    cfg.model.decoder_start_token_id = 2

    _ = build_from_cfg(cfg.model, GENERATORS)
