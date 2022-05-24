#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import SAGEConv, GATConv, Linear, to_hetero

from tqdm import tqdm

dataset_root = "/home/chli/chLi/Download/DeepLearning/Dataset/PytorchGeometric/"

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x

dataset = OGB_MAG(root=dataset_root + 'OGB_MAG',
                  preprocess='metapath2vec',
                  transform=T.ToUndirected())
data = dataset[0]

model = GNN(hidden_channels=64, out_channels=dataset.num_classes)
#  model = GAT(hidden_channels=64, out_channels=dataset.num_classes)
model = to_hetero(model, data.metadata(), aggr='sum')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print("start training sage...")
model.train()
for i in tqdm(range(1)):
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data['paper'].train_mask
    loss = F.cross_entropy(out['paper'][mask], data['paper'].y[mask])
    loss.backward()
    optimizer.step()

print("start test sage...")
model.eval()
test_mask = data['paper'].test_mask
pred = model(data.x_dict, data.edge_index_dict)
pred['paper'] = pred['paper'].argmax(dim=1)
correct = (pred['paper'][test_mask] == data['paper'].y[test_mask]).sum()
acc = int(correct) / int(data['paper'].y[test_mask].sum())
print("Accuracy =", acc)

