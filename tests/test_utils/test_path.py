import io
from pathlib import Path
from unittest.mock import patch

import pytest
from catbird.utils import check_file_exist, fopen, mkdir_or_exist


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


@patch("catbird.utils.io.path.Path.mkdir")
def test_mkdir_or_exists(mock_makedir):
    assert mkdir_or_exist("") == None
    assert mkdir_or_exist("no_such_file.txt") == None
    mkdir_or_exist("such_file.txt")
    mock_makedir.assert_called_with(mode=0o777, parents=True, exist_ok=True)
