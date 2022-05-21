#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch_geometric.data import Data

edge_index_format_1 = torch.tensor([[0, 1, 1, 2],
                                    [1, 0, 2, 1]], dtype=torch.long)
edge_index_format_2 = torch.tensor([[0, 1],
                                    [1, 0],
                                    [1, 2],
                                    [2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data_1 = Data(x=x, edge_index=edge_index_format_1)
data_2 = Data(x=x, edge_index=edge_index_format_2.t().contiguous())

print("[INFO][data::]")
print("\t data created success! two edge_index_format datas are:")
print(data_1)
print(data_2)
print()

print("[INFO][data::]")
print("\t keys:")
print(data_1.keys)
print()

print("[INFO][data::]")
print("\t values:")
print(data_1['x'])
print()

print("[INFO][data::]")
for key, item in data_1:
    print("data_1[" + key + "]:")
    print(item)
print()

