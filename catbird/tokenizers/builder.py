"""Factory to build tokenizers."""
# from urllib.error import HTTPError
from typing import Any

import ignite.distributed as idist
from ignite.utils import setup_logger
from requests import HTTPError
from transformers import AutoTokenizer

from catbird.utils import Config

logger = setup_logger(name="Tokenizer", distributed_rank=idist.get_rank())


def build_tokenizer(cfg: Config) -> Any:
    """Abstraction to build tokenizers based on the given configurations.

    Args:
        cfg (Config): configuration file

    Raises:
        NameError: Exception raised if tokenizer name in config does not match any of the available options

    Returns:
        nn.Module: selected model based on configurations
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
        if cfg.get("tokenizer", None):
            tokenizer.add_special_tokens(cfg.tokenizer.special_tokens)
        return tokenizer
    except HTTPError as err:
        if err.response.status_code == 404:
            raise HTTPError(f"Server error from HuggingFace.\nError message:\n{err}")
        elif err.response.status_code == 401:
            logger.info(
                "The model name does not match an existing tokenizer. Returning BPE pretrained tokenizer from Roberta."
            )
            tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            return tokenizer
        else:
            raise HTTPError(
                "Something went wrong for unknown reasons. Please try again later."
            )
