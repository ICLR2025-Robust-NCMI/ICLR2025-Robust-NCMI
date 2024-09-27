"""
Evaluation with AutoAttack.
"""

import json
import time
import argparse
import shutil

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from autoattack import AutoAttack
    
from core.data import get_data_info
from core.data import load_data
from core.models import create_model
from core.utils import Logger
from core.utils import parser_eval
from core.utils import seed
from core.utils.losses import CE_KL_PWCE_loss
from core.utils.centroid import *


# Setup

parse = parser_eval()
args = parse.parse_args()

LOG_DIR = args.log_dir + '/' + args.desc
with open(LOG_DIR+'/args.txt', 'r') as f:
    old = json.load(f)
    args.__dict__ = dict(vars(args), **old)

n_classes = 10

if args.data in ['cifar10', 'cifar10s']:
    da = '/cifar10/'
elif args.data in ['cifar100', 'cifar100s']:
    da = '/cifar100/'
    n_classes = 100
elif args.data in ['svhn', 'svhns']:
    da = '/svhn/'
elif args.data in ['tiny-imagenet', 'tiny-imagenets']:
    da = '/tiny-imagenet/'
    n_classes = 200


DATA_DIR = args.data_dir + da
WEIGHTS = LOG_DIR + '/weights-best.pt'

log_path = LOG_DIR + '/log-aa.log'
logger = Logger(log_path)
info = get_data_info(DATA_DIR)

BATCH_SIZE = 128
BATCH_SIZE_VALIDATION = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.log('Using device: {}'.format(device))


# Load data

seed(args.seed)
_, _, train_dataloader, test_dataloader = load_data(DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=False, 
                                                    shuffle_train=False)

if args.train:
    logger.log('Evaluating on training set.')
    l = [x for (x, y) in train_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in train_dataloader]
    y_test = torch.cat(l, 0)
else:
    l = [x for (x, y) in test_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_dataloader]
    y_test = torch.cat(l, 0)


# Model
print(args.model)
model = create_model(args.model, args.normalize, info, device)
checkpoint = torch.load(WEIGHTS)
if 'tau' in args and args.tau:
    print ('Using WA model.')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
del checkpoint


# AA Evaluation

seed(args.seed)
norm = 'Linf' if args.attack in ['fgsm', 'linf-pgd', 'linf-df'] else 'L2'
adversary = AutoAttack(model, norm=norm, eps=args.attack_eps, log_path=log_path, version=args.version, seed=args.seed)
centroids = Centroid(n_classes, None, None, Ctemp=1)
centroids.centroids = torch.zeros_like(centroids.centroids)

if args.version == 'custom':
    adversary.attacks_to_run = ['apgd-ce', 'apgd-t']
    adversary.apgd.n_restarts = 1
    adversary.apgd_targeted.n_restarts = 1

with torch.no_grad():
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=BATCH_SIZE_VALIDATION)
    num_batches = len(y_test) // BATCH_SIZE_VALIDATION + 1
    output_total = []
    for i in range(num_batches):
        start_idx = i * BATCH_SIZE_VALIDATION
        end_idx = len(y_test) if i >= num_batches - 1 else (i + 1) * BATCH_SIZE_VALIDATION
        logit = model(x_adv[start_idx:end_idx])
        output = F.softmax(logit, 1)
        
        Classes = y_test[start_idx:end_idx].cpu().unique()
        for c in Classes:
            try:
                centroids.centroids[c] += torch.sum(output[y_test[start_idx:end_idx].cpu() == c], axis = 0).detach().cpu()
            except Exception:
                breakpoint()

        output_total.append(output)
    
    centroids.centroids = centroids.centroids / centroids.centroids.sum(1)[:, None]
    output_total = torch.cat(output_total, dim=0).to(device)
    Cs = centroids.get_centroids(y_test).to(device)
    # AllCs = centroids.get_centroids(torch.arange(10)).to(device)
    y_test = y_test.to(device)
    loss_func = CE_KL_PWCE_loss(n_classes)
    loss, con, sep = loss_func(output_total, y_test, Cs)
    logger.log('Concentration: {:.4f}, Separation: {:.4f}, NCMI: {:.4f}'.format(con.item(), -sep.item(), (-con / sep).item()))


print ('Script Completed.')