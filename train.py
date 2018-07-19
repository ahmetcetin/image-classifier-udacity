#!/usr/bin/env python3
import argparse
parser = argparse.ArgumentParser(
    description='This is trainer program.',
)

parser.add_argument('data_directory')
parser.add_argument('--save_dir', action='store', default='.')
parser.add_argument('--arch', action='store', default='densenet121')
parser.add_argument('--learning_rate', action='store', default=.01, type=float)
parser.add_argument('--epochs', action='store', default=40, type=int)
parser.add_argument('--gpu', action='store_true', default=False)

args = parser.parse_args()

print(args)