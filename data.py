#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
edge_index_2 = torch.tensor([[0, 1],
                             [1, 0],
                             [1, 2],
                             [2, 1]], dtype=torch.long)

x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
data_2 = Data(x=x, edge_index=edge_index_2.t().contiguous())

print("[INFO]data:")
print(data)
print()

print("[INFO]keys:")
print(data.keys)
print()

for key, item in data:
    print("[INFO]data[" + key + "]:")
    print(item)
    print(data[key])
    print()

print('edge_attr' in data)
print("num_nodes =", data.num_nodes)
print("num_edges =", data.num_edges)
print("num_node_features =", data.num_node_features)
print("has_isolated_nodes =", data.has_isolated_nodes())
print("has_self_loops =", data.has_self_loops())
print("is_directed =", data.is_directed())

device = torch.device('cuda')
data = data.to(device)

