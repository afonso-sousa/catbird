import torch


def build_levi_graph(triples, tokenizer):
    from torch_geometric.data import Data
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
