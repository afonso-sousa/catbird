from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch

from ..registry import GRAPH_ENCODERS

@GRAPH_ENCODERS.register_module
class GCNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(16, 64)
        self.conv2 = GCNConv(64, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((32, 128))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.unsqueeze(0)
        x = self.avgpool(x)

        return x