"""File for general path handling."""

import os
from pathlib import Path
from typing import Any, TextIO, Union


def fopen(filepath: Union[str, Path], *args: Any, **kwargs: Any) -> TextIO:
    """Open file.

    Args:
        filepath (Union[str, Path]): filepath

    Returns:
        TextIOWrapper: buffer text stream.
    """
    if isinstance(filepath, str):
        return open(filepath, *args, **kwargs)
    elif isinstance(filepath, Path):
        return filepath.open(*args, **kwargs)


def check_file_exist(filename: Union[str, bytes, os.PathLike]) -> None:
    """Check whether a file exists.

    Args:
        filename (TextIO): filename

    Raises:
        FileNotFoundError: raises exception if file does not exist
    """
    if not Path(filename).is_file():
        raise FileNotFoundError(f'file "{str(filename)}" does not exist')


def mkdir_or_exist(dir_name: str, mode: int = 0o777) -> None:
    """Create directory if it does not exists.

    Args:
        dir_name (str): directiory or file name.
        mode (int, optional): define file assess permissions. Defaults to 0o777.
    """
    if dir_name == "":
        return
    Path(dir_name).mkdir(mode=mode, exist_ok=True)
