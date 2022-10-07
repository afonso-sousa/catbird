import os
import os.path as osp
import tempfile

import pytest
from catbird.utils import dump, load


def _test_handler(file_format, test_obj, str_checker, mode="r+"):
    # dump to a string
    dump_str = dump(test_obj, file_format=file_format)
    str_checker(dump_str)

    # load/dump with filenames from disk
    tmp_filename = osp.join(tempfile.gettempdir(), "test_dump")
    dump(test_obj, tmp_filename, file_format=file_format)
    assert osp.isfile(tmp_filename)
    load_obj = load(tmp_filename, file_format=file_format)
    assert load_obj == test_obj
    os.remove(tmp_filename)

    # json load/dump with a file-like object
    with tempfile.NamedTemporaryFile(mode, delete=False) as f:
        tmp_filename = f.name
        dump(test_obj, f, file_format=file_format)
    assert osp.isfile(tmp_filename)
    with open(tmp_filename, mode) as f:
        load_obj = load(f, file_format=file_format)
    assert load_obj == test_obj
    os.remove(tmp_filename)

    # automatically inference the file format from the given filename
    tmp_filename = osp.join(tempfile.gettempdir(), "test_dump." + file_format)
    dump(test_obj, tmp_filename)
    assert osp.isfile(tmp_filename)
    load_obj = load(tmp_filename)
    assert load_obj == test_obj
    os.remove(tmp_filename)


obj_for_test = [{"a": "abc", "b": 1}, 2, "c"]


def test_json():
    def json_checker(dump_str):
        assert dump_str in [
            '[{"a": "abc", "b": 1}, 2, "c"]',
            '[{"b": 1, "a": "abc"}, 2, "c"]',
        ]

    _test_handler("json", obj_for_test, json_checker)


def test_yaml():
    def yaml_checker(dump_str):
        assert dump_str in [
            "- {a: abc, b: 1}\n- 2\n- c\n",
            "- {b: 1, a: abc}\n- 2\n- c\n",
            "- a: abc\n  b: 1\n- 2\n- c\n",
            "- b: 1\n  a: abc\n- 2\n- c\n",
        ]

    _test_handler("yaml", obj_for_test, yaml_checker)


def test_pickle():
    def pickle_checker(dump_str):
        import pickle

        assert pickle.loads(dump_str) == obj_for_test

    _test_handler("pickle", obj_for_test, pickle_checker, mode="rb+")


def test_exception():
    test_obj = [{"a": "abc", "b": 1}, 2, "c"]

    with pytest.raises(ValueError):
        dump(test_obj)

    with pytest.raises(TypeError):
        dump(test_obj, "tmp.txt")
