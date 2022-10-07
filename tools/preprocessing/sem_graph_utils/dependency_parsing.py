from tqdm import tqdm
from allennlp.predictors.predictor import Predictor
from typing import Final


_dependency_parser_path: Final[
    str
] = "https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz"
# "https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz"


def _get_dependency_structure(sentence, parser_path):
    dep_parser = Predictor.from_path(parser_path)
    sentence = dep_parser.predict(sentence=sentence)

    words, pos, heads, dependencies = (
        sentence["words"],
        sentence["pos"],
        sentence["predicted_heads"],
        sentence["predicted_dependencies"],
    )
    result = [
        {"word": w, "pos": p, "head": h - 1, "dep": d}
        for w, p, h, d in zip(words, pos, heads, dependencies)
    ]
    return result


def get_data_with_dependency_tree(data):
    dependency_sentences = [
        {
            f"{k}_dp": _get_dependency_structure(sentence, _dependency_parser_path)
            for k, sentence in sample.items()
        }
        for sample in tqdm(data, desc="Dependency Parsing: ")
    ]

    for idx, sample in enumerate(data):
        sample.update(dependency_sentences[idx])

    return data
