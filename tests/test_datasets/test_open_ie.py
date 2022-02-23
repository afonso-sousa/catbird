import os
import tempfile
from pathlib import Path
from subprocess import Popen
from sys import stderr

import stanza
from stanza.server import CoreNLPClient

stanza.install_corenlp()


def generate_graphviz_graph(triples: dict, png_filename: str = "./out/graph.png"):
    """
    This method will try to generate the graph using `graphviz`.
    :param (str | unicode) text: raw text for the CoreNLPServer to parse
    :param (list | string) png_filename: list of annotators to use
    """
    entity_relations = triples
    """digraph G {
        # a -> b [ label="a to b" ];
        # b -> c [ label="another label"];
        }"""
    graph = list()
    graph.append("digraph {")
    for er in entity_relations:
        graph.append(
            '"{}" -> "{}" [ label="{}" ];'.format(
                er["subject"], er["object"], er["relation"]
            )
        )
    graph.append("}")

    output_dir = os.path.join(".", os.path.dirname(png_filename))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    out_dot = os.path.join(tempfile.gettempdir(), "graph.dot")
    with open(out_dot, "w") as output_file:
        output_file.writelines(graph)

    command = "dot -Tpng {} -o {}".format(out_dot, png_filename)
    dot_process = Popen(command, stdout=stderr, shell=True)
    dot_process.wait()
    assert (
        not dot_process.returncode
    ), "ERROR: Call to dot exited with a non-zero code status."


def test_triple_extraction():
    # text = "Chris Manning is a nice person. Chris wrote a simple sentence. He also gives oranges to people."
    # text = "Do facts truly exist?"
    text = "A luxury ship passes by the shore of a small village"

    with CoreNLPClient(
        annotators=[
            "tokenize",
            "ssplit",
            "pos",
            "lemma",
            "depparse",
            "ner",
            "coref",
            "natlog",
            "openie",
        ],
        properties={"openie.resolve_coref": True},
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

        print(triples)
        graph_img_path = "graph.png"
        generate_graphviz_graph(triples, graph_img_path)
        print("Graph generated: %s." % graph_img_path)

    assert False

    """
    If true, run coreference (and consequently NER as a dependency of coreference) and replace pronominal mentions with their canonical mention in the text.
    
    coref
    [{'subject': 'Chris Manning', 'relation': 'is', 'object': 'nice person'},
    {'subject': 'Chris Manning', 'relation': 'is', 'object': 'person'},
    {'subject': 'Manning', 'relation': 'is', 'object': 'nice'},
    {'subject': 'Chris', 'relation': 'wrote', 'object': 'sentence'},
    {'subject': 'Chris', 'relation': 'wrote', 'object': 'simple sentence'},
    {'subject': 'Chris Manning', 'relation': 'also gives', 'object': 'oranges'},
    {'subject': 'Chris Manning', 'relation': 'gives', 'object': 'oranges'},
    {'subject': 'Chris Manning', 'relation': 'gives oranges to', 'object': 'people'},
    {'subject': 'Chris Manning', 'relation': 'also gives oranges to', 'object': 'people'}]

    nocoref
    [{'subject': 'Chris Manning', 'relation': 'is', 'object': 'nice person'},
    {'subject': 'Chris Manning', 'relation': 'is', 'object': 'person'},
    {'subject': 'Manning', 'relation': 'is', 'object': 'nice'},
    {'subject': 'Chris', 'relation': 'wrote', 'object': 'sentence'},
    {'subject': 'Chris', 'relation': 'wrote', 'object': 'simple sentence'},
    {'subject': 'He', 'relation': 'also gives', 'object': 'oranges'},
    {'subject': 'He', 'relation': 'also gives oranges to', 'object': 'people'},
    {'subject': 'He', 'relation': 'gives', 'object': 'oranges'},
    {'subject': 'He', 'relation': 'gives oranges to', 'object': 'people'}]
    """


def test_triples2graph():
    import torch
    import pandas as pd
    from collections import defaultdict
    
    triples = [{'subject': 'Chris Manning', 'relation': 'is', 'object': 'nice person'},
               {'subject': 'Chris Manning', 'relation': 'is', 'object': 'person'},
               {'subject': 'Chris Manning', 'relation': 'also gives', 'object': 'oranges'}]
    
    triples_df = pd.DataFrame(triples)
    
    # print(triples_df['relation'].to_numpy())
    
    temp = defaultdict(lambda: len(temp))
    res = [temp[ele] for ele in triples_df['relation'].to_numpy()]
    print(temp)
    print(res)

    edge_features = torch.from_numpy(triples_df['relation'].to_numpy())
    edges_src = torch.from_numpy(triples_df['subject'].to_numpy())
    edges_dst = torch.from_numpy(triples_df['object'].to_numpy())
    num_nodes = len(set(triples_df['subject'].to_list()).intersection(set(triples_df['subject'].to_list())))
    print(num_nodes)
    assert False
    
    graph = dgl.graph((edges_src, edges_dst), num_nodes=num_nodes)


# %%
from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = ['This framework generates embeddings for each input sentence',
'Sentences are passed as a list of string.',
'The quick brown fox jumps over the lazy dog.']

sentence_embeddings = model.encode(sentences)

for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding shape:", embedding.shape)
    print("")

# build graph with triples
triples = [{'subject': 'Chris Manning', 'relation': 'is', 'object': 'nice person'},
            {'subject': 'Chris Manning', 'relation': 'is', 'object': 'person'},
            {'subject': 'Chris Manning', 'relation': 'also gives', 'object': 'oranges'}]

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
edge_index = torch.tensor([src_edges,
                           trg_edges], dtype=torch.long)

from torch_geometric.data import Data

data = Data(x=x, edge_index=edge_index)

# feed graph to network
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(384, 16)
        self.conv2 = GCNConv(16, 8)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# %%
device = torch.device('cuda')
model = GCN().to(device)
output = model(data.to(device))

assert False

