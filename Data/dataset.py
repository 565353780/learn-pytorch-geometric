#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path as osp

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, InMemoryDataset, Dataset, download_url
from torch_geometric.datasets import TUDataset, Planetoid, ShapeNet

dataset_root = "/home/chli/chLi/Download/DeepLearning/Dataset/PytorchGeometric/"

print("====ENZYMES====")
dataset = TUDataset(root=dataset_root + 'ENZYMES', name='ENZYMES')

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
dataset = Planetoid(root=dataset_root + 'Cora', name='Cora')

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
dataset = ShapeNet(root=dataset_root + 'ShapeNet', categories=['Airplane'])
print("dataset[0] =", dataset[0])

dataset = ShapeNet(root=dataset_root + 'ShapeNet', categories=['Airplane'],
                   pre_transform=T.KNNGraph(k=6))
print("create edges, dataset[0] :", dataset[0])

dataset = ShapeNet(root=dataset_root + 'ShapeNet', categories=['Airplane'],
                   pre_transform=T.KNNGraph(k=6),
                   transform=T.RandomTranslate(0.01))
print("augment, dataset[0] =", dataset[0])

print("====Custom Dataset====")

class CustomInMemoryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['file_name_1', 'file_name_2']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        url = None
        if url is None:
            return
        download_url(url, self.raw_dir)
        return

    def process(self):
        data_list = [Data(), Data()]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class CustomDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['file_name_1', 'file_name_2']

    @property
    def processed_file_names(self):
        return ['data_1.pt', 'data_2.pt']

    def download(self):
        url = None
        if url is None:
            return
        path = download_url(url, self.raw_dir)
        print("[INFO][CustomDataset::download]")
        print("path =", path)
        return

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            print("[INFO][CustomDataset::process]")
            print("raw_path =", raw_path)

            data = Data()

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir,
                                      "data_" + str(idx) + ".pt"))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir,
                                   "data_" + str(idx) + ".pt"))
        return data

exit()
custom_inmemory_dataset = CustomInMemoryDataset(
    root=dataset_root + "custom_inmemory_example",
    transform=T.RandomTranslate(0.01),
    pre_transform=T.KNNGraph(k=6))

custom_dataset = CustomDataset(
    root=dataset_root + "custom_example",
    transform=T.RandomTranslate(0.01),
    pre_transform=T.KNNGraph(k=6))

