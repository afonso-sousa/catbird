import pytest
from catbird.core import Config
from catbird.models import build_generator


def test_builder():
    cfg = Config.fromfile("configs/edd_quora.yaml")
    cfg.embedding_length = 10000
    cfg.pad_token_id = 0
    _, _ = build_generator(cfg)

    with pytest.raises(ModuleNotFoundError):
        cfg = Config(dict(model=dict(name="testing")))
        _, _ = build_generator(cfg)

