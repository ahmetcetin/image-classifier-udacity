#!/usr/bin/env python3
import argparse
import os
import sys
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
from pick import pick

parser = argparse.ArgumentParser(
    description='This is trainer program.',
)

parser.add_argument('image_file')
parser.add_argument('--top_k', action='store', default=3, type=int)
parser.add_argument('--models_dir', action='store', default='.')
parser.add_argument('--category_names', action='store', default='cat_to_name.json')
parser.add_argument('--gpu', action='store_true', default=False)

args = parser.parse_args()

files = os.listdir(args.models_dir)
trained_models = [i for i in files if i.endswith('.mdl')]

if len(trained_models) == 0:
    print('Error: There is no pretrained model in models directory - ', args.models_dir)
    print('Please change the models directory using --models_dir option')
    exit(1)

title = 'Please choose pretrained model to use: '
model_file, _ = pick(trained_models, title)

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

check_gpu = 'cuda:0' if args.gpu and torch.cuda.is_available() else 'cpu'

if args.gpu and check_gpu == 'cpu':
    confirm_cpu = query_yes_no('There is no cuda compatible gpu exists, do you want to continue using cpu?')
    if not confirm_cpu:
        exit(0)

print('Running prediction, please wait...')

device = torch.device(check_gpu)

state = torch.load(model_file, map_location=check_gpu)


model = getattr(models, state['arch'])(pretrained=True)
for param in model.parameters():
    param.requires_grad = False


if 'vgg' in state['arch']:
    first_layer_input = 25088
elif 'densenet' in state['arch']:
    first_layer_input = 1024

layers = []
prev_layer = first_layer_input

for i, layer in enumerate(state['hidden_units']):
    layers.append(('class_layer_' + str(i + 1), nn.Linear(prev_layer, layer)))
    layers.append(('relu_' + str(i + 1), nn.ReLU()))
    prev_layer = layer

layers.append(('fc_last', nn.Linear(prev_layer, 102)))
layers.append(('output', nn.LogSoftmax(dim=1)))

classifier = nn.Sequential(OrderedDict(layers))
model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=state['learning_rate'])

model.load_state_dict(state['state_dict'])
optimizer.load_state_dict(state['optimizer'])

class_to_idx = state['class_to_idx']
class_to_idx = {str(x): str(y) for x, y in class_to_idx.items()}
idx_to_class = {y: x for x, y in class_to_idx.items()}


def process_image(image_file):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image_file)
    width, height = img.size
    ratio = height / width
    shortsize = 256

    if height > width:
        longsize = int(math.floor(shortsize * ratio))
        newsize = (shortsize, longsize)
    else:
        longsize = int(math.floor(shortsize / ratio))
        newsize = (longsize, shortsize)

    img = img.resize(newsize)

    w, h = newsize

    left = (w - 224)/2
    top = (h - 224)/2
    right = (w + 224)/2
    bottom = (h + 224)/2

    cropped = img.crop((left, top, right, bottom))

    imgplot = plt.imshow(cropped)
    imgplot.axes.get_xaxis().set_visible(False)
    imgplot.axes.get_yaxis().set_visible(False)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image_file = np.array(cropped) / 255
    image_file = (image_file - mean) / std
    image_file = image_file.transpose((2, 0, 1))

    return torch.from_numpy(image_file)


# image = process_image('flowers/test/28/image_05230.jpg')
def predict(image_path, model, topk=5):
    """ Predict the class (or classes) of an image using a trained deep learning model.
    """
    model.eval()
    model = model.to(device)
    image = process_image(image_path)
    image = torch.unsqueeze(image, 0)
    image = image.to(device).float()
    result = model(Variable(image))
    result = F.softmax(result, dim=1)
    result = result.cpu()
    p, c = result.topk(topk)
    p, c = p.detach().numpy()[0], c.detach().numpy()[0]
    return p, c


def get_class_name(idx):
    return cat_to_name[idx_to_class[str(idx)]]


probs, classes = predict(args.image_file, model, args.top_k)
class_names = [get_class_name(x) for x in classes]

print('Most probably: ', get_class_name(classes[0]))
print('\nProbabilities:')
for a, b in zip(class_names, probs):
    print("{}: {:.4%}".format(a, b))
