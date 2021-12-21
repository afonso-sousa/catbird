"""File for general input-output handling."""

from pathlib import Path
from typing import Any, Optional, Union
import os

from .handlers import JsonHandler, YamlHandler


file_handlers = {
    "json": JsonHandler(),
    "yaml": YamlHandler(),
    "yml": YamlHandler(),
}


def load(
    file: Union[str, bytes, os.PathLike],
    file_format: Optional[str] = None,
    **kwargs: Any
) -> Any:
    """Load data from json/yaml/pickle files. This method provides a unified api for loading data from serialized files.

    Args:
        file (str or :obj:`Path` or file-like object): Filename or a file-like
            object.
        file_format (str, optional): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include "json", "yaml/yml" and
            "pickle/pkl".

    Returns:
        The content from the file.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None and isinstance(file, str):
        file_format = file.split(".")[-1]
    if file_format not in file_handlers:
        raise TypeError("Unsupported format: {}".format(file_format))

    handler = file_handlers[file_format]
    if isinstance(file, str):
        obj = handler.load_from_path(file, **kwargs)
    elif hasattr(file, "read"):
        obj = handler.load_from_fileobj(file, **kwargs)
    else:
        raise TypeError('"file" must be a filepath str or a file-object')
    return obj


def dump(
    obj: Any,
    file: Optional[Union[str, bytes, os.PathLike]] = None,
    file_format: Optional[str] = None,
    **kwargs: Any
) -> Optional[str]:
    """Dump data to json/yaml/pickle strings or files.
    
    This method provides a unified api for dumping data as strings or to files, 
    and also supports custom arguments for each file format.

    Args:
        obj ([Any]): The python object to be dumped.
        file (Optional[Union[str, bytes, os.PathLike]], optional): If not specified, then the object\
        is dump to a str, otherwise to a file specified by the filename or 
        file-like object. Defaults to None.
        file_format (Optional[str], optional): A string specifying the file format. Defaults to None.

    Raises:
        ValueError: File_format must be specified when file is None.
        TypeError: Unsupported format.
        TypeError: "file" must be a filename str or a file-object.

    Returns:
        Optional[str]: String representation of object.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None:
        if isinstance(file, str):
            file_format = file.split(".")[-1]
        elif file is None:
            raise ValueError("file_format must be specified since file is None")
    if file_format not in file_handlers:
        raise TypeError("Unsupported format: {}".format(file_format))

    handler = file_handlers[file_format]
    if file is None:
        return handler.dump_to_str(obj, **kwargs)
    elif isinstance(file, str):
        handler.dump_to_path(obj, file, **kwargs)
    elif hasattr(file, "write"):
        handler.dump_to_fileobj(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')
