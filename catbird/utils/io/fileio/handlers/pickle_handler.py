import pickle
from typing import Any, IO, Union
from os import PathLike

from .base import BaseFileHandler


class PickleHandler(BaseFileHandler):
    """Pickle file handler."""

    def load_from_fileobj(self, file: IO[Any], **kwargs: Any) -> Any:
        return pickle.load(file, **kwargs)

    def load_from_path(self, filepath: Union[str, PathLike], **kwargs: Any) -> Any:
        with open(filepath, mode="rb") as f:
            return self.load_from_fileobj(f, **kwargs)

    def dump_to_path(
        self, obj: Any, filepath: Union[str, PathLike], **kwargs: Any
    ) -> None:
        with open(filepath, mode="wb") as f:
            self.dump_to_fileobj(obj, f, **kwargs)

    def dump_to_fileobj(self, obj: Any, file: IO[Any], **kwargs: Any) -> None:
        pickle.dump(obj, file, **kwargs)

    def dump_to_str(self, obj: Any, **kwargs: Any) -> Union[str, bytes]:
        return pickle.dumps(obj, **kwargs)
