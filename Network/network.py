#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, OGB_MAG
from torch_geometric.nn import \
    GCNConv, HeteroConv, GCNConv, SAGEConv, GATConv, HGTConv, \
    Linear, to_hetero

# HomoGraph - GCNConv
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)
        return

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)

# HeteroGraph - SAGEConv
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)
        return

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# HeteroGraph - GATConv
class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)
        return

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x

# HeteroGraph - Mixed
class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('paper', 'cites', 'paper'): GCNConv(-1, hidden_channels),
                ('author', 'writes', 'paper'): SAGEConv((-1, -1), hidden_channels),
                ('paper', 'rev_writes', 'author'): GATConv((-1, -1), hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)
        return

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.lin(x_dict['author'])

# HeteroGraph - HGTConv
class HGT(torch.nn.Module):
    def __init__(self,
                 hidden_channels,
                 out_channels,
                 num_heads,
                 num_layers,
                 data):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)
        return

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict['author'])

def build_homo_network():
    dataset_root = "/home/chli/chLi/Download/DeepLearning/Dataset/PytorchGeometric/"

    dataset = Planetoid(root=dataset_root + 'Cora', name='Cora')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # GCN
    model = GCN(dataset.num_node_features, dataset.num_classes)
    print("build GCN success!")

    model = model.to(device)
    return True

def build_hetero_network():
    dataset_root = "/home/chli/chLi/Download/DeepLearning/Dataset/PytorchGeometric/"

    dataset = OGB_MAG(root=dataset_root + 'OGB_MAG',
                      preprocess='metapath2vec',
                      transform=T.ToUndirected())
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # GNN
    model = GNN(hidden_channels=64, out_channels=dataset.num_classes)
    model = to_hetero(model, data.metadata(), aggr='sum')
    print("build GNN success!")

    # GAT
    model = GAT(hidden_channels=64, out_channels=dataset.num_classes)
    model = to_hetero(model, data.metadata(), aggr='sum')
    print("build GAT success!")

    # HeteroGNN
    model = HeteroGNN(hidden_channels=64,
                      out_channels=dataset.num_classes,
                      num_layers=2)
    print("build HeteroGNN success!")

    # HGT
    model = HGT(hidden_channels=64,
                out_channels=dataset.num_classes,
                num_heads=2,
                num_layers=2,
                data=data)
    print("build HGT success!")

    model = model.to(device)
    return True

def demo():
    build_homo_network()
    build_hetero_network()
    return True

if __name__ == "__main__":
    demo()

