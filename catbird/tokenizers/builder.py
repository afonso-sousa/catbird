"""Factory to build tokenizers."""
from catbird.core import Config  # type: ignore
from transformers import AutoTokenizer


def build_tokenizer(cfg: Config) -> AutoTokenizer:
    """Abstraction to build tokenizers based on the given configurations.

    Args:
        cfg (Config): configuration file

    Raises:
        NameError: Exception raised if tokenizer name in config does not match any of the available options

    Returns:
        nn.Module: selected model based on configurations
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name.lower())
        if cfg.get("tokenizer", None):
            tokenizer.add_special_tokens(cfg.tokenizer.special_tokens)
    except:
        print(
            "The model name does not match an existing tokenizer. Returning bert-base-uncased pretrained tokenizer."
        )
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    finally:
        return tokenizer
