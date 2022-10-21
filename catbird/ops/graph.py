import torch
from sentence_transformers import SentenceTransformer
from stanza.server import CoreNLPClient
from torch import nn


def extract_triples(text, annotators=["openie"], properties={}):
    with CoreNLPClient(
        annotators=annotators,
        properties=properties,
        be_quiet=True,
    ) as client:
        ann = client.annotate(text)
        triples = []
        for sentence in ann.sentence:
            for triple in sentence.openieTriple:
                triples.append(
                    {
                        "subject": triple.subject,
                        "relation": triple.relation,
                        "object": triple.object,
                    }
                )

    return triples


def build_graph_from_triples(triples):
    from torch_geometric.data import Data

    # TODO remove sentence encoding from this method
    model = SentenceTransformer("all-MiniLM-L6-v2")
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

    x_embeds = model.encode(nodes)

    x = torch.tensor(x_embeds, dtype=torch.float)
    edge_index = torch.tensor([src_edges, trg_edges], dtype=torch.long)

    graph = Data(x=x, edge_index=edge_index)

    return graph


class IETripleGraph(nn.Module):
    def __init__(self, coref=True):
        super(IETripleGraph, self).__init__()
        if coref:
            self.annotators = [
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
            self.properties = {"openie.resolve_coref": True}
        else:
            self.annotators = ["openie"]
            self.properties = {}


# %%
import os

os.chdir(os.path.join(os.getcwd(), "../.."))


# %%
from torch_geometric.data import Data
import torch
from catbird.utils import load
from catbird.tokenizers import build_tokenizer
from catbird.utils import Config

data = load("data/quora/quora_with_dp.pkl")

# %%
all_dependencies = list(set(token["dep"] for sample in data for token in sample["src_dp"]))
all_pos = list(set(token["pos"] for sample in data for token in sample["src_dp"]))


# %%
sample = data[3]

# nodes = set(token["word"] for token in sample["src_dp"])
# nodes = list(nodes)
nodes = [[token["word"], token["pos"]] for token in sample["src_dp"]]

# %%
edges = []
edge_attr = []
for idx, token in enumerate(sample["src_dp"]):
    if token["head"] != -1:
        edges.append(
            [
                token["head"],
                idx,
            ]
        )
        edge_attr.append([token["dep"]])

# %%
cfg = Config.fromfile("configs/stacked_residual_lstm_quora.py")
tokenizer = build_tokenizer(cfg)

# %%
special_tokens_dict = {"additional_special_tokens": all_pos}
tokenizer.add_special_tokens(special_tokens_dict)

# %%
x = torch.tensor(
    [tokenizer.convert_tokens_to_ids(node) for node in nodes], dtype=torch.float
)

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# %%
graph = Data(x=x, edge_index=edge_index)

# %%
sample = data[3]

# nodes = set(token["word"] for token in sample["src_dp"])
# nodes = list(nodes)
nodes = [token["word"] for token in sample["src_dp"]]

# %%
edges = []
for idx, token in enumerate(sample["src_dp"]):
    if token["head"] != -1:
        nodes.append(token["dep"])
        edges.append(
            [
                token["head"],
                nodes.index(token["dep"]),
            ]
        )
        edges.append(
            [
                nodes.index(token["dep"]),
                idx,
            ]
        )


# %%
cfg = Config.fromfile("configs/stacked_residual_lstm_quora.py")
tokenizer = build_tokenizer(cfg)

# %%
special_tokens_dict = {"additional_special_tokens":  all_pos + all_dependencies}
tokenizer.add_special_tokens(special_tokens_dict)

# %%
x = torch.tensor(
    [tokenizer.convert_tokens_to_ids(node) for node in nodes], dtype=torch.float
)

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
