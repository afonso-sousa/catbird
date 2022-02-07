from pathlib import Path

from catbird.core import Config
from catbird.core.utils.registry import Registry, build_from_cfg
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

    cfg.model.encoder.vocabulary_size = 10000
    cfg.model.encoder.pad_token_id = 0
    cfg.model.decoder.vocabulary_size = 10000
    cfg.model.decoder.pad_token_id = 0

    _ = build_from_cfg(cfg.model.encoder, ENCODERS)
    _ = build_from_cfg(cfg.model.decoder, DECODERS)

    _ = build_from_cfg(cfg.model, GENERATORS)
