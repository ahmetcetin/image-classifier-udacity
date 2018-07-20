#!/usr/bin/env python3
import argparse
import os
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

from utils import query_yes_no

parser = argparse.ArgumentParser(
    description='This is trainer program.',
)

parser.add_argument('data_directory')
parser.add_argument('--save_dir', action='store', default='.')
parser.add_argument('--arch', action='store', default='densenet121')
parser.add_argument('--learning_rate', action='store', default=.01, type=float)
parser.add_argument('--epochs', action='store', default=40, type=int)
parser.add_argument('--gpu', action='store_true', default=False)
parser.add_argument('--hidden_units', action='append', default=[], type=int)

args = parser.parse_args()

if len(args.hidden_units) == 0:
    args.hidden_units = [512]

try:
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
except OSError:
    print('Error: Creating directory. ' + args.save_dir)
    exit(1)

model_file = args.save_dir + '/' + args.arch + '.mdl'

if os.path.isfile(model_file):
    overwrite = query_yes_no('model already exists on save dir, do you want to overwrite?')
    if not overwrite:
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


def get_pretrained_model(name):
    conv_models = {'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'densenet121', 'densenet169', 'densenet161', 'densenet201'}
    if name not in conv_models:
        print('Error: Sorry, there is no such pretrained model. ' + name)
        print('You can use one of the following predefined models:')
        print(str(conv_models))
        exit(1)

    return getattr(models, name)


init_lr = args.learning_rate
epochs = args.epochs

model = get_pretrained_model(args.arch)(pretrained=True)
for param in model.parameters():
    param.requires_grad = False


if 'vgg' in args.arch:
    first_layer_input = 25088
elif 'densenet' in args.arch:
    first_layer_input = 1024

layers = []
prev_layer = first_layer_input

for i, layer in enumerate(args.hidden_units):
    layers.append(('class_layer_' + str(i + 1), nn.Linear(prev_layer, layer)))
    layers.append(('relu_' + str(i + 1), nn.ReLU()))
    prev_layer = layer

layers.append(('fc_last', nn.Linear(prev_layer, 102)))
layers.append(('output', nn.LogSoftmax(dim=1)))

classifier = nn.Sequential(OrderedDict(layers))
model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=init_lr)


check_gpu = 'cuda:0' if args.gpu and torch.cuda.is_available() else 'cpu'

if args.gpu and check_gpu == 'cpu':
    confirm_cpu = query_yes_no('There is no cuda compatible gpu exists, do you want to continue using cpu?')
    if not confirm_cpu:
        exit(0)

device = torch.device(check_gpu)
model.to(device)

print_every = 50
steps = 0


def adjust_lr(optimizer, epoch):
    lr = init_lr * (0.1 ** (epoch // 20))
    print('Starting epoch {}, learning rate {}'.format(epoch+1, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def validation(model, validationloader, criterion):
    test_loss = 0
    accuracy = 0
    for inputs, labels in validationloader:
        inputs, labels = inputs.to(device), labels.to(device)

        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy

print('*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')
print('Epochs: ', args.epochs)
print('Starting learning rate: ', args.learning_rate)
print('Pre-trained architecture: ', args.arch)
print('Classification layers: ', args.hidden_units)
print('*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')

confirm_start = query_yes_no('All checks good, do you want to start training with above parameters?')

if not confirm_start:
    exit(0)

for e in range(epochs):
    running_loss = 0
    adjust_lr(optimizer, e)
    for inputs, labels in dataloaders['training']:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            model.eval()

            with torch.no_grad():
                test_loss, accuracy = validation(model, dataloaders['validation'], criterion)

            print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Training Loss: {:.3f} ".format(running_loss/print_every),
                  "Validation Loss: {:.3f} ".format(test_loss/len(dataloaders['validation'])),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(dataloaders['validation'])))

            running_loss = 0

            model.train()

correct = 0
total = 0
model = model.to(device)
model.eval()

with torch.no_grad():
    for data in dataloaders['test']:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

confirm_save = query_yes_no('Are you happy with the test results? Do you want to save the model?')

if not confirm_save:
    exit(0)

model.class_to_idx = image_datasets['training'].class_to_idx


def save_checkpoint():
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'hidden_units': args.hidden_units,
        'arch': args.arch
    }

    torch.save(state, model_file)


save_checkpoint()
