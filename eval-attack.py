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
from core.metrics import accuracy
from core.attacks import create_attack
from core.attacks.utils import CWLoss


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

log_path = LOG_DIR + '/log-eval.log'
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
adversary = {
    'FGSM': create_attack(model, nn.CrossEntropyLoss(), 'fgsm', 8/255, 40, 2/255),
    'PGD': create_attack(model, nn.CrossEntropyLoss(), 'linf-pgd', 8/255, 40, 2/255),
    'CW': create_attack(model, CWLoss, 'linf-pgd', 8/255, 40, 2/255)
}

for name, attack in adversary.items():
    logger.log(name + ':\n')
    centroids = Centroid(n_classes, None, None, Ctemp=1)
    centroids.centroids = torch.zeros_like(centroids.centroids)
    
    output_total, y_total = [], []
    acc = 0.
    
    for x, y in tqdm(test_dataloader):
        x, y = x.to(device), y.to(device)
        
        with ctx_noparamgrad_and_eval(model):
            x_adv, _ = attack.perturb(x, y)
        
        with torch.no_grad():
            logit = model(x_adv)
            output = F.softmax(logit, 1)
            acc += accuracy(y, logit)
            
            Classes = y.cpu().unique()
            for c in Classes:
                try:
                    centroids.centroids[c] += torch.sum(output[y.cpu() == c], axis = 0).detach().cpu()
                except Exception:
                    breakpoint()

            output_total.append(output)
            y_total.append(y)
            
    acc /= len(test_dataloader)
    logger.log('Accuracy: {:.2f}%'.format(acc * 100))
    centroids.centroids = centroids.centroids / centroids.centroids.sum(1)[:, None]
    output_total = torch.cat(output_total, dim=0).to(device)
    y_total = torch.cat(y_total, dim=0).to(device)
    Cs = centroids.get_centroids(y_total).to(device)
    # AllCs = centroids.get_centroids(torch.arange(10)).to(device)
    y_total = y_total.to(device)
    
    with torch.no_grad():
        loss_func = CE_KL_PWCE_loss(n_classes)
        loss, con, sep = loss_func(output_total, y_total, Cs)
    
    logger.log('Concentration: {:.4f}, Separation: {:.4f}, NCMI: {:.4f}'.format(con.item(), -sep.item(), (-con / sep).item()))
    logger.log('\n')


print ('Script Completed.')