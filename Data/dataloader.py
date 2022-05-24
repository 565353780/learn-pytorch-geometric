#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch_scatter import scatter_mean
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

def demo():
    dataset_root = "/home/chli/chLi/Download/DeepLearning/Dataset/PytorchGeometric/"

    dataset = TUDataset(root=dataset_root + 'ENZYMES', name='ENZYMES', use_node_attr=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for data in loader:
        print("data =", data)
        print("data.num_graphs =", data.num_graphs)

        x = scatter_mean(data.x, data.batch, dim=0)
        print("data.x.scatter_mean =", x.size())
        break

    print("====")

    data_list = [Data(), Data()]
    loader = DataLoader(data_list, batch_size=32)

    for data in loader:
        print("data =", data)
    return True

if __name__ == "__main__":
    demo()

