import io
from pathlib import Path
from unittest.mock import patch
import os

import pytest
from catbird.core import check_file_exist, fopen, mkdir_or_exist


def test_fopen():
    assert hasattr(fopen(__file__), "read")
    assert hasattr(fopen(Path(__file__)), "read")

    assert isinstance(fopen(__file__), io.TextIOWrapper)
    assert isinstance(fopen(Path(__file__)), io.TextIOWrapper)

    assert fopen(__file__).__dict__ == fopen(Path(__file__)).__dict__


def test_check_file_exist():
    check_file_exist(__file__)
    with pytest.raises(FileNotFoundError):
        check_file_exist("no_such_file.txt")


@patch("catbird.core.utils.path.os.makedirs")
def test_mkdir_or_exists(mock_makedirs):
    assert mkdir_or_exist("") == None
    assert mkdir_or_exist("no_such_file.txt") == None
    mkdir_or_exist("such_file.txt")
    mock_makedirs.assert_called_with("such_file.txt", mode=0o777, exist_ok=True)
