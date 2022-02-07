import json
import os.path as osp
import tempfile
from pathlib import Path

import pytest
import yaml
from catbird.core import Config

data_path = Path(__file__).parent.parent / "data"


def test_instancing():
    cfg = Config()
    assert cfg.filename is None
    assert cfg.text == ""
    assert len(cfg) == 0
    assert cfg._cfg_dict == {}

    with pytest.raises(TypeError):
        Config([0, 1])


cfg_dict = dict(item1=[1, 2], item2=dict(a=0), item3=True, item4="test")


def test_construction_python():
    cfg_file = osp.join(data_path, 'config/a.py')
    cfg = Config(cfg_dict, filename=cfg_file)
    assert isinstance(cfg, Config)
    assert cfg.filename == cfg_file
    assert cfg.text == open(cfg_file, 'r').read()


def test_construction_json():
    cfg_file = osp.join(data_path, "config/b.json")
    cfg = Config(cfg_dict, filename=cfg_file)
    assert isinstance(cfg, Config)
    assert cfg.filename == cfg_file
    assert cfg.text == open(cfg_file, "r").read()
    assert cfg.dump() == json.dumps(cfg_dict)
    with tempfile.TemporaryDirectory() as temp_config_dir:
        dump_file = osp.join(temp_config_dir, "b.json")
        cfg.dump(dump_file)
        assert cfg.dump() == open(dump_file, "r").read()
        assert Config.fromfile(dump_file)


def test_construction_yaml():
    cfg_file = osp.join(data_path, "config/c.yaml")
    cfg = Config(cfg_dict, filename=cfg_file)
    assert isinstance(cfg, Config)
    assert cfg.filename == cfg_file
    assert cfg.text == open(cfg_file, "r").read()
    assert cfg.dump() == yaml.dump(cfg_dict)
    with tempfile.TemporaryDirectory() as temp_config_dir:
        dump_file = osp.join(temp_config_dir, "c.yaml")
        cfg.dump(dump_file)
        assert cfg.dump() == open(dump_file, "r").read()
        assert Config.fromfile(dump_file)