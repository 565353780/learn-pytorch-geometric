#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.datasets import OGB_MAG

dataset_root = "/home/chli/chLi/Download/DeepLearning/Dataset/PytorchGeometric/"

print("====HeteroData====")
data = HeteroData()

data['paper'].x = ... # [num_papers, num_features_paper]
data['author'].x = ... # [num_authors, num_features_author]
data['institution'].x = ... # [num_institutions, num_features_institution]
data['field_of_study'].x = ... # [num_field, num_features_field]

data['paper', 'cites', 'paper'].edge_index = ... # [2, num_edges_cites]
data['author', 'writes', 'paper'].edge_index = ... # [2, num_edges_writes]
data['author', 'affiliated_with', 'institution'].edge_index = ... # [2, num_edges_affiliated]
data['paper', 'has_topic', 'field_of_study'].edge_index = ... # [2, num_edges_topic]

data['paper', 'cites', 'paper'].edge_attr = ... # [num_edges_cites, num_features_cites]
data['author', 'writes', 'paper'].edge_attr = ... # [num_edges_writes, num_features_writes]
data['author', 'affiliated_with', 'institution'].edge_attr = ... # [num_edges_affiliated, num_features_affiliated]
data['paper', 'has_topic', 'field_of_study'].edge_attr = ... # [num_edges_topic, num_features_topic]

print("data =", data)

print("====OGB_MAG====")
dataset = OGB_MAG(root=dataset_root + 'OGB_MAG', preprocess='metapath2vec')
data = dataset[0]
print("dataset[0] =", data)

paper_node_data = data['paper']
cites_edge_data = data['paper', 'cites', 'paper']
# if uniquely identified
cites_edge_data = data['paper', 'paper']
cites_edge_data = data['cites']

data['paper'].year = 6666
del data['field_of_study']
print("after edit, dataset[0] =", data)

data = dataset[0]

node_types, edge_types = data.metadata()
print("node_types =", node_types)
print("edge_types =", edge_types)

data = data.to('cuda:0')
data = data.cpu()
print("data can transferred to gpu")

print("data.has_isolated_nodes() =", data.has_isolated_nodes())
print("data.has_self_loops() =", data.has_self_loops())
print("data.is_undirected() =", data.is_undirected())

homogeneous_data = data.to_homogeneous()
print("homogeneous_data =", homogeneous_data)

data = T.ToUndirected()(data)
data = T.AddSelfLoops()(data)
data = T.NormalizeFeatures()(data)
print("data transforms test succees")

