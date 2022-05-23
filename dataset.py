#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset, Planetoid, ShapeNet

print("====ENZYMES====")
dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

print("dataset :", dataset)
print("len =", len(dataset))
print("num_classes =", dataset.num_classes)
print("num_node_features =", dataset.num_node_features)

data = dataset[0]

print("data = dataset[0] =", data)
print("data.is_undirected =", data.is_undirected())

dataset = dataset.shuffle()

print("dataset.shuffle =", dataset)

train_dataset = dataset[:540]
test_dataset = dataset[540:]

print("train_dataset =", train_dataset)
print("test_dataset =", test_dataset)

print("====Cora====")
dataset = Planetoid(root='/tmp/Cora', name='Cora')

print("dataset :", dataset)
print("len =", len(dataset))
print("num_classes =", dataset.num_classes)
print("num_node_features =", dataset.num_node_features)

data = dataset[0]

print("data = dataset[0] =", data)
print("data.is_undirected =", data.is_undirected())
print("data.train_mask.sum().item() =", data.train_mask.sum().item())
print("data.val_mask.sum().item() =", data.val_mask.sum().item())
print("data.test_mask.sum().item() =", data.test_mask.sum().item())

print("====ShapeNet====")
dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'])
print("dataset[0] =", dataset[0])

dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'],
                   pre_transform=T.KNNGraph(k=6))
print("create edges, dataset[0] :", dataset[0])

dataset = ShapeNet(root='/tmp/ShapeNet', categories=['Airplane'],
                   pre_transform=T.KNNGraph(k=6),
                   transform=T.RandomTranslate(0.01))
print("augment, dataset[0] =", dataset[0])

