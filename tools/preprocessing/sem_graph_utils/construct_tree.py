from typing import Final, List
import nltk

nltk.download("punkt")

# Penn TreeBank POS
verb_pos: Final[List[str]] = [
    "VBZ",  # Verb, 3rd person singular present
    "VBN",  # Verb, past participle
    "VBD",  # Verb, past tense
    "VBP",  # Verb, non-3rd person singular present
    "VB",  # Verb, base form
    "VBG",  # Verb, gerund or present participle
]
prep_pos: Final[List[str]] = [
    "PP",  # Prepositional Phrase
    "IN",  # Preposition or subordinating conjunction
    "TO",  # to
]
modifier_pos: Final[List[str]] = [
    "JJ",  # Adjective
    "FW",  # Foreign word
    "JJR",  # Adjective, comparative
    "JJS",  # Adjective, superlative
    "RB",  # Adverb
    "RBR",  # Adverb, comparative
    "RBS",  # Adverb, superlative
]

# Stanford typed dependencies
subj_and_obj: Final[List[str]] = [
    "nsubj",  # nominal subject
    "nsubjpass",  # passive nominal subject
    "csubj",  # clausal subject
    "csubjpass",  # clausal passive subject
    "dobj",  # direct object
    "pobj",  # object of a preposition
    "iobj",  # indirect object
]
conj: Final[List[str]] = ["conj", "parataxis"]
modifiers: Final[List[str]] = [
    "amod",  # adjectival modifier
    "nn",  # noun compound modifier
    "mwe",  # multi-word expression
    "advmod",  # adverbial modifier
    "quantmod",  # quantifier phrase modifier
    "npadvmod",  # noun phrase as adverbial modifier
    "advcl",  # adverbial clause modifier
    "poss",  # possession modifier
    "possessive",  # possessive modifier
    "neg",  # negation modifier
    "auxpass",  # passive auxiliary
    "aux",  # auxiliary
    "det",  # determiner
    "dep",  # unspecified dependency
    "predet",  # predeterminer
    "num",  # numeric modifier
]


def _merge_node(raw, sequence):
    node = {k: v for k, v in raw.items()}
    attribute = raw["attribute"]

    attr1, attr2 = [], []  # attr1: ok to merge
    indexes = [idx for idx in node["index"]]
    for a in attribute:
        if "attribute" in a or "noun" in a or "verb" in a:
            attr2.append(a)
        elif (a["dep"] in modifiers or a["pos"] in modifier_pos) and a[
            "pos"
        ] not in prep_pos:
            attr1.append(a)
            indexes += [idx for idx in a["index"]]
        else:
            attr2.append(a)

    if len(attr1) > 0:
        indexes.sort(key=lambda x: x)
        flags = [index not in indexes[:idx] for idx, index in enumerate(indexes)]
        if len(indexes) == indexes[-1] - indexes[0] + 1 and all(
            flags
        ):  # need to be consecutive modifiers
            node["word"] = [sequence[i] for i in indexes]
            node["index"] = indexes
            if len(attr2) > 0:
                node["attribute"] = [a for a in attr2]
            else:
                del node["attribute"]

    return node


def _build_detailed_tree(sequence, all_dep, root, word_type):
    def is_noun(node):
        return node["dep"] in subj_and_obj or (
            all_dep[root]["dep"] in subj_and_obj and node["dep"] == "conj"
        )

    def is_verb(node):
        return (node["dep"] == "cop" and word_type == "A") or (
            word_type == "V" and node["dep"] == "conj"
        )

    ##=== initialize tree-node ===##
    element = all_dep[root]
    word_type = "V" if element["pos"] in verb_pos else "A"
    node = {
        "word": [sequence[root]],
        "index": [root],
        "type": word_type,
        "dep": element["dep"],
        "pos": element["pos"],
    }
    ##=== classify child node sets ===##
    children = [(i, elem) for i, elem in enumerate(all_dep) if elem["head"] == root]
    nouns = [child for child in children if is_noun(child[1])]
    if len(nouns) > 0:
        node["noun"] = [
            _build_detailed_tree(sequence, all_dep, child[0], "A") for child in nouns
        ]
    verbs = [child for child in children if is_verb(child[1])]
    if len(verbs) > 0:
        node["verb"] = [
            _build_detailed_tree(sequence, all_dep, child[0], "V") for child in verbs
        ]
    attributes = [child for child in children if child not in nouns + verbs]
    if len(attributes) > 0:
        node["attribute"] = [
            _build_detailed_tree(sequence, all_dep, child[0], "A")
            for child in attributes
        ]
    ##=== do node-merging ===##
    if "attribute" in node:
        node = _merge_node(node, sequence)

    return node


def build_tree(dep_parse, sequence):
    root = [i for i in range(len(dep_parse)) if dep_parse[i]["head"] == -1]
    heads_dep = [w["dep"] for w in dep_parse if w["head"] == root[0]]

    word_type = (
        "V" if dep_parse[root[0]]["pos"] in verb_pos or "cop" not in heads_dep else "A"
    )
    tree = _build_detailed_tree(sequence, dep_parse, root[0], word_type)

    return tree
