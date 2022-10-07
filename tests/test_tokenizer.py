from pathlib import Path
from catbird.tokenizers import build_tokenizer
import unittest
from catbird.utils import Config

test_file = Path(__file__).parent.parent / "README.md"
text = "paraphrase generation toolkit"


# def test_word_tokenizer():
#     tokenizer = Tokenizer("test")
#     tokenizer.get_vocab([test_file], from_filenames=True)
#     tokenized = tokenizer.tokenize(text)
#     assert text == tokenizer.detokenize(tokenized)

#     unknown_words = "unknown world"
#     tokenized_unknown_words = tokenizer.tokenize(unknown_words)
#     assert f"{UNK_TOKEN} {UNK_TOKEN}" == tokenizer.detokenize(tokenized_unknown_words)


class TestTokenizers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg_dict = dict(
            tokenizer=dict(
                algorithm="BPE",
                unk_token="[UNK]",
                special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
            )
        )
        cls.cfg = Config(cfg_dict)

    def test_bpe_tokenizer(self):
        tokenizer = build_tokenizer(self.cfg)

        print(tokenizer.pad_token)

        assert False

    def test_roberta_tokenizer(self):
        cfg_dict = dict(
            tokenizer=dict(
                name="roberta-base",
            )
        )
        cfg = Config(cfg_dict)
        tokenizer = build_tokenizer(cfg)

        encoded = tokenizer("Hello world")["input_ids"]
        print(encoded)

        decoded = tokenizer.decode(encoded, skip_special_tokens=False)

        print(decoded)

        assert False
