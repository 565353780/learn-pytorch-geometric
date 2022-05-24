#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing, knn_graph
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader

from tqdm import tqdm

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)
        return

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = self.lin(x)

        print("[INFO][GCNConv::forward]")
        print("edge_index =", edge_index)
        row, col = edge_index
        print("row =", row)
        print("col =", col)
        deg = degree(col, x.size(0), dtype=x.dtype)
        print("deg =", deg)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        print("norm =", norm)

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        print("[INFO][GCNConv::message]")
        print("norm.view(-1, 1) =", norm.view(-1, 1))

        return norm.view(-1, 1) * x_j

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max')
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        # tmp has shape [E, 2 * in_channels]
        tmp = torch.cat([x_i, x_j - x_i], dim=1)
        return self.mlp(tmp)

class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, k=6):
        super().__init__(in_channels, out_channels)
        self.k = k

    def forward(self, x, batch=None):
        edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        return super().forward(x, edge_index)

def demo():
    dataset_root = "/home/chli/chLi/Download/DeepLearning/Dataset/PytorchGeometric/"

    dataset = Planetoid(root=dataset_root + 'Cora', name='Cora')

    data = dataset[0]
    print("dataset[0] =", data)

    x, edge_index = data.x, data.edge_index
    conv1 = GCNConv(dataset.num_node_features, 32)
    conv2 = GCNConv(32, dataset.num_classes)
    x = conv1(x, edge_index)
    x = conv2(x, edge_index)
    print("dataset.num_node_features =", dataset.num_node_features)
    print("dataset.num_classes =", dataset.num_classes)
    print("GCNConv(x) =", x.size())

    edge_conv_1 = DynamicEdgeConv(dataset.num_node_features, 128, k=6)
    edge_conv_2 = DynamicEdgeConv(128, dataset.num_classes, k=6)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for data in loader:
        print("data in dataloader =", data)
        x, batch = data.x, data.batch
        x = edge_conv_1(x, batch)
        x = edge_conv_2(x, batch)
        print("DynamicEdgeConv(x) =", x.size())
        break
    return True

if __name__ == "__main__":
    demo()

