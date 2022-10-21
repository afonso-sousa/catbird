import spacy
from tqdm import tqdm


def _get_dependency_structure(pipeline, sentence):
    sent = pipeline(sentence)
    src_dp = []
    for token in sent:
        dep_parse = {}
        dep_parse["word"] = token.text
        dep_parse["pos"] = token.pos_
        dep_parse["head"] = token.head.i if token.head.i != token.i else -1
        dep_parse["head_token"] = token.head.text
        dep_parse["dep"] = token.dep_
        src_dp.append(dep_parse)
    return src_dp


def get_data_with_dependency_tree(data):
    pipeline = spacy.load("en_core_web_sm")

    dependency_sentences = [
        {
            f"{k}_dp": _get_dependency_structure(pipeline, sentence)
            for k, sentence in sample.items()
        }
        for sample in tqdm(data, desc="Dependency Parsing: ")
    ]

    for idx, sample in enumerate(data):
        sample.update(dependency_sentences[idx])

    return data
