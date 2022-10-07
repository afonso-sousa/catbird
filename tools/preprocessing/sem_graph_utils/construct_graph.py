from typing import Final, List

subj_and_obj: Final[List[str]] = ["nsubj", "nsubjpass", "csubj", "csubjpass"] + [
    "dobj",
    "pobj",
    "iobj",
]
others_dep: Final[List[str]] = ["poss", "npadvmod", "appos", "nn"]
conj: Final[List[str]] = ["conj", "cc", "preconj", "parataxis"]
verb_pos: Final[List[str]] = ["VBZ", "VBN", "VBD", "VBP", "VB", "VBG", "IN", "TO", "PP"]
noun_pos: Final[List[str]] = ["NN", "NNP", "NNS", "NNPS"]
subj: Final[List[str]] = ["nsubj", "nsubjpass", "csubj", "csubjpass"]


def _count_nodes(nodes, tree):
    ## initialize node
    str_words = " ".join(tree["word"])
    node = {
        "type": tree["type"],
        "dep": tree["dep"],
        "pos": tree["pos"],
        "word": str_words,
        "index": tree["index"],
    }
    ## merge almost the same nodes as one node if meet requirements
    for idx, exist in enumerate(nodes):
        # requirement one: has common words and has the same type
        if all([w in exist["word"].split(" ") for w in node["word"].split(" ")]):
            if (
                all([w in node["word"].split(" ") for w in exist["word"].split(" ")])
                and node["type"] == exist["type"]
            ):
                # requirement two: noun-like nodes
                if node["pos"] in noun_pos or node["dep"] in subj_and_obj + others_dep:
                    # requirement three: has upper case and not quite short
                    if (
                        any([w.isupper() for w in node["word"]])
                        and len(node["word"].split(" ")) > 1
                    ):
                        # requirement four: enough high level of overlapping
                        if (
                            len(node["word"].split(" ")) / len(exist["word"].split(" "))
                            > 0.9
                        ):
                            tree["node_num"] = idx
                            break
    ## added as new node if not meet above requirements
    if "node_num" not in tree:
        tree["node_num"] = len(nodes)
        nodes.append(node)
    ## collect child nodes
    if "noun" in tree:
        for child in tree["noun"]:
            _count_nodes(nodes, child)
    if "verb" in tree:
        for child in tree["verb"]:
            _count_nodes(nodes, child)
    if "attribute" in tree:
        for child in tree["attribute"]:
            _count_nodes(nodes, child)


def _draw_graph(graph, tree):
    index = tree["node_num"]
    ## copy existed edges in tree
    children = tree["noun"] if "noun" in tree else []
    children += tree["verb"] if "verb" in tree else []
    children += tree["attribute"] if "attribute" in tree else []
    for child in children:
        if (
            child["dep"] != "punct"
            or "noun" in child
            or "verb" in child
            or "attribute" in child
        ):
            idx = child["node_num"]
            graph[idx][index] = child["dep"]
            _draw_graph(graph, child)
    ## redirect to make it sure that:
    #  1. entity --> predicate <-- entity in each [entity, predicate, entity] triple
    #  2. those parallel words connect to their real parents
    if "noun" in tree and "verb" in tree:
        ## for 'V' type node
        if tree["type"] == "V" or tree["pos"] in verb_pos:
            for verb in tree["verb"]:
                v_idx = verb["node_num"]
                is_replace = False  # whether to redirect
                for noun in tree["noun"]:
                    if noun["dep"] in ["nsubj", "nsubjpass", "csubj", "csubjpass"]:
                        is_replace = True
                        n_idx = noun["node_num"]
                        graph[n_idx][v_idx] = noun["dep"]
                if is_replace and graph[v_idx][index] in conj:
                    graph[v_idx][index] = ""
        ## for 'A'/'M' type node
        else:
            verbs = []
            for v in tree["verb"]:
                # collect verbs to do redirecting with
                if (
                    v["word"] not in [vrb["word"] for vrb in verbs]
                    or "noun" in v
                    or "attribute" in v
                    or "verb" in v
                ):
                    verbs.append(v)
            tree["verb"] = verbs
            for verb in tree["verb"]:
                v_idx = verb["node_num"]
                for noun in tree["noun"]:
                    n_idx = noun["node_num"]
                    # for parallel words
                    if graph[n_idx][index] in conj:
                        if "verb" not in noun:
                            graph[v_idx][n_idx] = verb["dep"]
                            graph[n_idx][index] = ""
                        else:
                            nv_idx = noun["verb"][0]["node_num"]
                            for nn in [
                                tmp for tmp in tree["noun"] if tmp["dep"] in subj
                            ]:
                                graph[nn["node_num"]][nv_idx] = nn["dep"]
                                graph[n_idx][index] = ""
                    # for [entity, predicate, entity] triple where predicate is corpula
                    if noun["dep"] in subj_and_obj and noun["dep"] != "pobj":
                        graph[n_idx][v_idx] = noun["dep"]
                        graph[n_idx][index] = ""


def get_graph(tree):
    ## collect nodes
    nodes = []
    _count_nodes(nodes, tree)
    ## draw edges
    edges = [["" for _ in nodes] for _ in nodes]
    _draw_graph(edges, tree)

    return {"nodes": nodes, "edges": edges}
