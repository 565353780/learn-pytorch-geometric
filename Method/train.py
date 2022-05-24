#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, OGB_MAG
from torch_geometric.nn import to_hetero
from torch_geometric.loader import DataLoader, NeighborLoader

from tqdm import tqdm

from Network.network import GCN, GNN, GAT, HeteroGNN, HGT

def train_homo():
    dataset_root = "/home/chli/chLi/Download/DeepLearning/Dataset/PytorchGeometric/"

    dataset = Planetoid(root=dataset_root + 'Cora', name='Cora')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # GCN
    model = GCN(dataset.num_node_features, dataset.num_classes)

    model = model.to(device)

    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    print("start training model...")
    model.train()
    for _ in tqdm(range(200)):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    print("start test model")
    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print("Accuracy =", acc)
    return True

def train_hetero():
    dataset_root = "/home/chli/chLi/Download/DeepLearning/Dataset/PytorchGeometric/"

    dataset = OGB_MAG(root=dataset_root + 'OGB_MAG',
                      preprocess='metapath2vec',
                      transform=T.ToUndirected())
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # GNN
    model = GNN(hidden_channels=64, out_channels=dataset.num_classes)
    model = to_hetero(model, data.metadata(), aggr='sum')

    # GAT
    #  model = GAT(hidden_channels=64, out_channels=dataset.num_classes)
    #  model = to_hetero(model, data.metadata(), aggr='sum')

    # HeteroGNN
    #  model = HeteroGNN(hidden_channels=64,
                      #  out_channels=dataset.num_classes,
                      #  num_layers=2)

    # HGT
    #  model = HGT(hidden_channels=64,
                #  out_channels=dataset.num_classes,
                #  num_heads=2,
                #  num_layers=2,
                #  data=data)

    model = model.to(device)

    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    print("start training model...")
    model.train()
    for _ in tqdm(range(1)):
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        mask = data['paper'].train_mask
        loss = F.cross_entropy(out['paper'][mask], data['paper'].y[mask])
        loss.backward()
        optimizer.step()

    print("start test model")
    model.eval()
    test_mask = data['paper'].test_mask
    pred = model(data.x_dict, data.edge_index_dict)
    pred['paper'] = pred['paper'].argmax(dim=1)
    correct = (pred['paper'][test_mask] == data['paper'].y[test_mask]).sum()
    acc = int(correct) / int(data['paper'].y[test_mask].sum())
    print("Accuracy =", acc)
    return True

def train_homo_batch():
    dataset_root = "/home/chli/chLi/Download/DeepLearning/Dataset/PytorchGeometric/"

    dataset = Planetoid(root=dataset_root + 'Cora', name='Cora')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # GCN
    model = GCN(dataset.num_node_features, dataset.num_classes)

    model = model.to(device)

    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    print("start training model...")
    model.train()

    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        batch_size = batch.batch_size
        out = model(batch)
        loss = F.nll_loss(out[:batch_size],
                          batch.y[:batch_size])
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size

    avg_loss = total_loss / total_examples
    print("avg_loss =", avg_loss)

    print("start test model")
    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print("Accuracy =", acc)
    return True

def train_hetero_batch():
    dataset_root = "/home/chli/chLi/Download/DeepLearning/Dataset/PytorchGeometric/"

    dataset = OGB_MAG(root=dataset_root + 'OGB_MAG',
                      preprocess='metapath2vec',
                      transform=T.ToUndirected())
    data = dataset[0]

    train_loader = NeighborLoader(
        data,
        #  num_neighbors=[15, 15],
        num_neighbors={key: [15, 15] for key in data.edge_types},
        batch_size=128,
        input_nodes=('paper', data['paper'].train_mask),
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # GNN
    model = GNN(hidden_channels=64, out_channels=dataset.num_classes)
    model = to_hetero(model, data.metadata(), aggr='sum')

    # GAT
    #  model = GAT(hidden_channels=64, out_channels=dataset.num_classes)
    #  model = to_hetero(model, data.metadata(), aggr='sum')

    # HeteroGNN
    #  model = HeteroGNN(hidden_channels=64,
                      #  out_channels=dataset.num_classes,
                      #  num_layers=2)

    # HGT
    #  model = HGT(hidden_channels=64,
                #  out_channels=dataset.num_classes,
                #  num_heads=2,
                #  num_layers=2,
                #  data=data)

    model = model.to(device)

    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()

    total_examples = total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        batch_size = batch['paper'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)
        loss = F.cross_entropy(out['paper'][:batch_size],
                               batch['paper'].y[:batch_size])
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size

    avg_loss = total_loss / total_examples
    print("avg_loss =", avg_loss)
    return True

def demo():
    #  train_homo()
    #  train_hetero()
    #  train_homo_batch()
    train_hetero_batch()
    return True

if __name__ == "__main__":
    demo()

