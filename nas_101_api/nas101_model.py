#Adapted from https://github.com/BayesWatch/nas-without-training/

# This source code is licensed under the license found in the
# LICENSE file in the nas_101_api directory
# LICENSE filename : Mellor_MIT_LICENSE
#

from nas_101_api.model import *
import numpy as np

from nas_101_api.ntk import *
from easydict import EasyDict as edict

import os, json
from os import path as osp
from pathlib import Path
from collections import namedtuple

import sys, torch

import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from copy import deepcopy
from PIL import Image

from nas_101_api.SearchDatasetWrap import SearchDataset
from nas_101_api.configure_utils import *

def get_ntk(spec):
    #model part
    args = edict({'in_channels': 3, "num_stacks": 3, 'num_modules_per_stack':3,
            'stem_out_channels': 128, 'num_labels': 10})

    classifier = Network(spec, args, searchspace=[]).to(torch.device('cuda:6'))

    #data parameters
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std  = [x / 255 for x in [63.0, 62.1, 66.7]]
    root = '../data/cifardata'
    batch_size = 32
    pin_memory=True

    lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)]
    train_transform = transforms.Compose(lists)
    test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    xshape = (1, 3, 32, 32)
    train_data = dset.CIFAR10 (root, train=True , transform=train_transform, download=True)
    test_data  = dset.CIFAR10 (root, train=False, transform=test_transform , download=True)
    assert len(train_data) == 50000 and len(test_data) == 10000

    cifar_split = load_config('nas_101_api/cifar-split.txt', None, None)
    train_split, valid_split = cifar_split.train, cifar_split.valid # search over the proposed training and validation set
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               num_workers=0, pin_memory=pin_memory, sampler= torch.utils.data.sampler.SubsetRandomSampler(train_split))

    ntk_value = get_ntk_n(train_loader, [classifier], recalbn=0, train_mode=True, num_batch=1)
    print("The condition number of the network is ", ntk_value)

    return ntk_value
