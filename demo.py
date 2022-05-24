#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Data.data import demo as data_demo
from Data.dataset import demo as dataset_demo
from Data.dataloader import demo as dataloader_demo
from Data.hetero_data import demo as hetero_data_demo

from Network.gcn_layer import demo as layer_demo
from Network.network import demo as network_demo
from Network.hetero_network import demo as hetero_network_demo

if __name__ == "__main__":
    demo = data_demo
    demo()

