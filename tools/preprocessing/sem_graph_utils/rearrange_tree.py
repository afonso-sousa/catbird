from typing import Final, List


_subj_and_obj: Final[List[str]] = ["nsubj", "nsubjpass", "csubj", "csubjpass"] + ["dobj", "pobj", "iobj"]
_months: Final[List[str]] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def _merge_month(node, sequence):
    indexes = [idx for idx in node["index"]]

    if "attribute" in node:
        for a in node["attribute"]:
            index, _ = _merge_month(a, sequence)
            indexes += index

    indexes.sort(key=lambda x: x)
    words = [sequence[i] for i in indexes]

    return indexes, words


def rearrange(node, sequence):
    ## collect child node sets and do tree-rearranging on child nodes
    noun, verb = None, None
    slf = {k: v for k, v in node.items() if k not in ["noun", "verb"]}
    noun = [rearrange(n, sequence) for n in node["noun"]] if "noun" in node else []
    verb = [rearrange(v, sequence) for v in node["verb"]] if "verb" in node else []
    if "attribute" in node:
        slf["attribute"] = [rearrange(a, sequence) for a in node["attribute"]]
    ## redirect grandchild nodes to the current node
    ## rule: redirect the parallel words to their real parents
    if noun and node["type"] != "V":  # those nodes of type 'V' will be rearranged later
        for id_n in range(len(noun)):
            if (
                noun[id_n]["dep"] in _subj_and_obj
                and "verb" not in noun[id_n]
                and "noun" in noun[id_n]
            ):
                new_nouns, rearrg_nouns = [], []
                dep_list = [
                    n["dep"] == "conj" for n in noun[id_n]["noun"]
                ]  # whether a parallel word
                for i, grandchild in enumerate(noun[id_n]["noun"]):
                    if dep_list[i]:
                        grandchild["dep"] = noun[id_n]["dep"]
                        rearrg_nouns.append(grandchild)
                    else:
                        new_nouns.append(grandchild)
                if len(new_nouns) > 0:
                    noun[id_n]["noun"] = new_nouns
                else:
                    del noun[id_n]["noun"]
                noun += rearrg_nouns
    ## merge preposition and its only child node (pobject) as one node [node type = 'M' (modifier)]
    if noun and (not verb) and ("attribute" not in node):
        if len(noun) == 1 and node["dep"] == "prep" and noun[0]["dep"] in _subj_and_obj:
            if ("noun" not in noun[0]) and ("verb" not in noun[0]):
                indexes = node["index"] + noun[0]["index"]
                indexes.sort(key=lambda x: x)
                wrap = {
                    "dep": noun[0]["dep"],
                    "word": [sequence[i] for i in indexes],
                    "index": indexes,
                    "pos": noun[0]["pos"],
                    "type": "M",
                }
                if "attribute" in noun[0]:
                    wrap["attribute"] = noun[0]["attribute"]
                return wrap
            ## if has more than one nodes, do redirecting
            elif "verb" not in noun[0] and "noun" in noun[0]:
                new_nouns, gg_nouns = [], []
                dep_list = [n["dep"] == "conj" for n in noun[0]["noun"]]
                for i, grandchild in enumerate(noun[0]["noun"]):
                    if dep_list[i]:
                        grandchild["dep"] = noun[0]["dep"]
                        gg_nouns.append(grandchild)
                    else:
                        new_nouns.append(grandchild)
                if len(new_nouns) > 0:
                    noun[0]["noun"] = new_nouns
                else:
                    del noun[0]["noun"]
                noun += gg_nouns
    ## for node which represents time/date (i.e., contain month word),
    #  merge it with all its attribute child nodes
    if "attribute" in slf and any([w in _months for w in slf["word"]]):
        slf["index"], slf["word"] = _merge_month(slf, sequence)
        del slf["attribute"]
    ## get final node
    wrap = {
        "dep": slf["dep"],
        "word": slf["word"],
        "index": slf["index"],
        "pos": slf["pos"],
        "type": slf["type"],
        "noun": noun,
        "verb": verb,
    }
    if "attribute" in slf:
        wrap["attribute"] = slf["attribute"]
    if not noun:
        del wrap["noun"]
    if not verb:
        del wrap["verb"]
    return wrap
