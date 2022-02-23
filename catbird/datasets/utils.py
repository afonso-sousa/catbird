import torch
from torch_geometric.data import Data


class TeacherForcing:
    def __init__(self, batch_first=True, token_id=None):
        self.batch_first = batch_first
        self.token_id = token_id

    def __call__(self, src, targets):
        inputs = targets
        if self.batch_first:
            if self.token_id is None:
                # if no token_id is specified, then use the last token in src
                eos_tensor = src[:, 0].reshape(
                    -1, 1
                )  # transpose shape [2] into shape [2, 1]
            else:
                eos_tensor = torch.full((inputs.size(0), 1), self.token_id)
            input_shifted = inputs[:, :-1]
            inputs = torch.cat((eos_tensor, input_shifted), dim=1)
        else:
            if self.token_id is None:
                eos_tensor = src[0].unsqueeze(0)
            else:
                eos_tensor = torch.full((1, inputs.size(1)), self.token_id)
            input_shifted = inputs[:-1]
            inputs = torch.cat((eos_tensor, input_shifted), dim=0)
        return src, inputs, targets


def build_levi_graph(triples, tokenizer):
    # triples is list of dictionaries
    nodes = set()
    src_edges = []
    trg_edges = []
    for triple in triples:
        nodes |= set(triple.values())
        src_edges += [triple["subject"], triple["relation"]]
        trg_edges += [triple["relation"], triple["object"]]

    nodes = list(nodes)
    src_edges = [nodes.index(edge) for edge in src_edges]
    trg_edges = [nodes.index(edge) for edge in trg_edges]

    if nodes:
        nodes = tokenizer(nodes).input_ids

    x = torch.tensor(nodes, dtype=torch.float)
    edge_index = torch.tensor([src_edges, trg_edges], dtype=torch.long)

    graph = Data(x=x, edge_index=edge_index)

    return graph


def generate_ie_graph(text, tokenizer):
    annotators = [
        "tokenize",
        "ssplit",
        "pos",
        "lemma",
        "depparse",
        "ner",
        "coref",
        "natlog",
        "openie",
    ]
    properties = {"openie.resolve_coref": True}
    # triples = extract_triples(text, annotators, properties)
    graph = build_graph_from_triples(triples, tokenizer)
    
    return graph