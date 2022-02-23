from stanza.server import CoreNLPClient
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
import torch
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
    # TODO remove sentence encoding from this method
    model = SentenceTransformer('all-MiniLM-L6-v2')
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

    def __init__(self,
                 coref=True):
        super(IETripleGraph, self).__init__()
        if coref:
            self.annotators=[
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
            self.properties={"openie.resolve_coref": True}
        else:
            self.annotators=["openie"]
            self.properties={}


    # def forward(self, input):

    #     return voxelization(input, self.voxel_size, self.point_cloud_range,
    #                         self.max_num_points, max_voxels,
    #                         self.deterministic)

    # def __repr__(self):
    #     tmpstr = self.__class__.__name__ + '('
    #     tmpstr += 'voxel_size=' + str(self.voxel_size)
    #     tmpstr += ', point_cloud_range=' + str(self.point_cloud_range)
    #     tmpstr += ', max_num_points=' + str(self.max_num_points)
    #     tmpstr += ', max_voxels=' + str(self.max_voxels)
    #     tmpstr += ', deterministic=' + str(self.deterministic)
    #     tmpstr += ')'
    #     return tmpstr