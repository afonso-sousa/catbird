from pathlib import Path

import pytest
from catbird.core import Config
from catbird.models import build_generator
from torch import nn

data_path = Path(__file__).parent.parent / "data"


def test_builder():
    cfg_file = Path(data_path, "config", "dummy_model_cfg.py")
    cfg = Config.fromfile(cfg_file)
    cfg.embedding_length = 10000
    cfg.pad_token_id = 0
    model = build_generator(cfg)
    assert isinstance(model, nn.Module)

    with pytest.raises(AssertionError):
        cfg = Config(dict(model=dict(name="testing")))
        build_generator(cfg)

