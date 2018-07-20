#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import json
import math
from PIL import Image
from collections import OrderedDict

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

from utils import query_yes_no

parser = argparse.ArgumentParser(
    description='This is trainer program.',
)

parser.add_argument('data_directory', default='flowers')
parser.add_argument('--save_dir', action='store', default='.')
parser.add_argument('--arch', action='store', default='densenet121')
parser.add_argument('--learning_rate', action='store', default=.01, type=float)
parser.add_argument('--epochs', action='store', default=40, type=int)
parser.add_argument('--gpu', action='store_true', default=False)
parser.add_argument('--hidden_units', action='store', default=512)

args = parser.parse_args()


try:
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
except OSError:
    print('Error: Creating directory. ' + args.save_dir)
    exit(1)

model_file = args.save_dir + '/' + args.arch + '.mdl'
print(model_file)

if os.path.isfile(model_file):
    answer = query_yes_no('model already exists on save dir, do you want to overwrite?')
    if not answer:
        exit(0)

train_dir = args.data_directory + '/train'
valid_dir = args.data_directory + '/valid'
test_dir = args.data_directory + '/test'

invalid_data_dir = not os.path.exists(train_dir) \
                   or not os.path.exists(valid_dir) \
                   or not os.path.exists(test_dir)

try:
    if invalid_data_dir:
        print('Error: Not a valid data directory. ' + args.data_directory)
        print('Data directory should have train, valid and test sub directories')
        exit(1)
except OSError:
    print('Error: Checking data directory.')
    exit(1)

data_transforms = {
    "training": transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),
    "validation": transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])]),
    "test": transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
}

image_datasets = {
    "training": datasets.ImageFolder(train_dir, transform=data_transforms["training"]),
    "validation": datasets.ImageFolder(valid_dir, transform=data_transforms["validation"]),
    "test": datasets.ImageFolder(test_dir, transform=data_transforms["test"])
}

dataloaders = {
    "training": torch.utils.data.DataLoader(image_datasets["training"], batch_size=64, shuffle=True),
    "validation": torch.utils.data.DataLoader(image_datasets["validation"], batch_size=32),
    "test": torch.utils.data.DataLoader(image_datasets["test"], batch_size=32)
}

