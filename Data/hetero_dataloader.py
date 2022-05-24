#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader

def demo():
    dataset_root = "/home/chli/chLi/Download/DeepLearning/Dataset/PytorchGeometric/"

    dataset = OGB_MAG(root=dataset_root + 'OGB_MAG',
                      preprocess='metapath2vec',
                      transform=T.ToUndirected())
    data = dataset[0]
    print("data =", data)

    train_loader = NeighborLoader(
        data,
        #  num_neighbors=[15, 15],
        num_neighbors={key: [15, 15] for key in data.edge_types},
        batch_size=128,
        input_nodes=('paper', data['paper'].train_mask))

    for batch in train_loader:
        print("batch =", batch)
        break
    return True

if __name__ == "__main__":
    demo()

