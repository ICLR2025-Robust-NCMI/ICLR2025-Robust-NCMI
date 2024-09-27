import json
import time
import argparse
import shutil

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from core.data import get_data_info
from core.data import load_data
from core.data import SEMISUP_DATASETS

from core.utils import format_time
from core.utils import Logger
from core.utils import parser_train
from core.utils import Trainer
from core.utils import seed
from core.utils.centroid import *
from core.utils.sampler import *
from core.utils.utils_awp import *
from core.models import create_model
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from gowal21uncovering.utils import WATrainer


# Setup

parse = parser_train()
parse.add_argument('--tau', type=float, default=0.995, help='Weight averaging decay.')
args = parse.parse_args()

if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
if not os.path.exists(args.data_dir):
    os.mkdir(args.data_dir)

DATA_DIR = os.path.join(args.data_dir, args.data)
LOG_DIR = os.path.join(args.log_dir, args.desc)
WEIGHTS = os.path.join(LOG_DIR, 'weights-best.pt')
if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)
os.makedirs(LOG_DIR)
logger = Logger(os.path.join(LOG_DIR, 'log-train.log'))

with open(os.path.join(LOG_DIR, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=4)


info = get_data_info(DATA_DIR)
BATCH_SIZE = args.batch_size
BATCH_SIZE_VALIDATION = args.batch_size_validation
NUM_ADV_EPOCHS = args.num_adv_epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.log('Using device: {}'.format(device))
if args.debug:
    NUM_ADV_EPOCHS = 1

# To speed up training
torch.backends.cudnn.benchmark = True


# Load data

# seed(args.seed)
if args.data in SEMISUP_DATASETS:
    train_dataset, test_dataset, eval_dataset, train_dataloader, test_dataloader, eval_dataloader = load_data(
        DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=args.augment, use_consistency=args.consistency, shuffle_train=True, 
        aux_data_filename=args.aux_data_filename, unsup_fraction=args.unsup_fraction, validation=args.data in SEMISUP_DATASETS
    )
    # breakpoint()
    train_dataset_orig, _, train_dataloader_orig, _ = load_data(
        DATA_DIR[:-1], BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=args.augment, use_consistency=args.consistency, shuffle_train=True, 
        aux_data_filename=args.aux_data_filename, unsup_fraction=args.unsup_fraction, validation=False
    )
    del train_dataset, test_dataset, eval_dataset
    del train_dataset_orig
else:
    train_dataset, test_dataset, train_dataloader, test_dataloader = load_data(
        DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=args.augment, use_consistency=args.consistency, shuffle_train=True, 
        aux_data_filename=args.aux_data_filename, unsup_fraction=args.unsup_fraction, validation=args.data in SEMISUP_DATASETS
    )
    eval_dataloader = test_dataloader
    del train_dataset, test_dataset


# Adversarial Training

if args.tau:
    print ('Using WA.')
    trainer = WATrainer(info, args)
else:
    trainer = Trainer(info, args)
last_lr = args.lr


if NUM_ADV_EPOCHS > 0:
    logger.log('\n\n')
    metrics = pd.DataFrame()
    logger.log('Standard Accuracy-\tTest: {:2f}%.'.format(trainer.eval(test_dataloader)*100))
    
    old_score = [0.0, 0.0]
    logger.log('RST Adversarial training for {} epochs'.format(NUM_ADV_EPOCHS))
    trainer.init_optimizer(args.num_adv_epochs)
    test_adv_acc = 0.0    

if args.resume_path:
    start_epoch = trainer.load_model_resume(os.path.join(args.resume_path, 'state-last.pt')) + 1
    logger.log(f'Resuming at epoch {start_epoch}')
else:
    start_epoch = 1

if args.NCMI:
    n_classes = 10
    if 'cifar100' in args.data:
        n_classes = 100
    elif 'tiny-imagenet' in args.data:
        n_classes = 200

    train_label_list = torch.tensor(train_dataloader_orig.dataset.targets) if args.data.endswith('s') else torch.tensor(train_dataloader.dataset.targets)

    samples_data, samples_tar = [], []
    sample_loader = train_dataloader_orig if args.data.endswith('s') else train_dataloader
    
    idx_total = [[] for _ in range(n_classes)]
    for Class in range(n_classes):
        idx = (train_label_list == Class).nonzero().squeeze().numpy()
        sampler = SubsetRandomSampler(idx)
        batch_size = 500 if '100' in args.data else 5000
        class_loader = DataLoader(sample_loader.dataset, batch_size = batch_size, sampler = sampler, pin_memory = True)
        img, tar = next(iter(class_loader))
        samples_data.append(img.numpy())
        samples_tar.append(tar)
    
    samples_data = np.array(samples_data)
    samples_tar = torch.cat(samples_tar, 0)
    
    centroids = Centroid(n_classes, samples_data, samples_tar, Ctemp=1, CdecayFactor=args.cdf)
    centroids.update_epoch(trainer.model, train_dataloader, device)
    centroid_dir = os.path.join(LOG_DIR, 'centroids')
    logits_dir = os.path.join(LOG_DIR, 'logits')
    os.makedirs(centroid_dir, exist_ok=True)
    os.makedirs(logits_dir, exist_ok=True)

for epoch in range(start_epoch, NUM_ADV_EPOCHS+1):
    start = time.time()
    logger.log('======= Epoch {} ======='.format(epoch))
    
    if args.scheduler:
        last_lr = trainer.scheduler.get_last_lr()[0]
    
    if args.NCMI:
        os.makedirs(os.path.join(logits_dir, 'epoch_{}'.format(epoch)), exist_ok=True)
        awp = None
        if args.awp:
            proxy = create_model(args.model, args.normalize, info, device)
            proxy_optim = torch.optim.SGD(proxy.parameters(), lr = args.lr)
            awp = TradesAWP(trainer.model, proxy, proxy_optim, args.awp_gamma)
        res, centroids = trainer.train(train_dataloader, epoch=epoch, adversarial=True, centroids=centroids, awp=awp)
    else:
        res = trainer.train(train_dataloader, epoch=epoch, adversarial=True)
    test_acc = trainer.eval(test_dataloader)
    
    logger.log('Loss: {:.4f}.\tLR: {:.4f}'.format(res['loss'], last_lr))
    if 'clean_acc' in res:
        logger.log('Standard Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(res['clean_acc']*100, test_acc*100))
    else:
        logger.log('Standard Accuracy-\tTest: {:.2f}%.'.format(test_acc*100))
    epoch_metrics = {'train_'+k: v for k, v in res.items()}
    epoch_metrics.update({'epoch': epoch, 'lr': last_lr, 'test_clean_acc': test_acc, 'test_adversarial_acc': ''})
    
    if epoch % args.adv_eval_freq == 0 or epoch == NUM_ADV_EPOCHS:        
        test_adv_acc, test_adv_loss, test_adv_con, test_adv_sep = trainer.eval(test_dataloader, adversarial=True)
        logger.log('Adversarial Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(res['adversarial_acc']*100, 
                                                                                   test_adv_acc*100))
        epoch_metrics.update({'test_adversarial_acc': test_adv_acc,
                              'test_adversarial_loss': test_adv_loss,
                              'test_adversarial_con': test_adv_con,
                              'test_adversarial_sep': test_adv_sep,
                              'test_adversarial_NCMI': test_adv_con / test_adv_sep})
    else:
        logger.log('Adversarial Accuracy-\tTrain: {:.2f}%.'.format(res['adversarial_acc']*100))
    eval_adv_acc, eval_adv_loss, eval_adv_con, eval_adv_sep = trainer.eval(eval_dataloader, adversarial=True, epoch=epoch)
    logger.log('Adversarial Accuracy-\tEval: {:.2f}%.'.format(eval_adv_acc*100))
    epoch_metrics['eval_adversarial_acc'] = eval_adv_acc
    epoch_metrics['eval_adversarial_loss'] = eval_adv_loss
    epoch_metrics['eval_adversarial_con'] = eval_adv_con
    epoch_metrics['eval_adversarial_sep'] = eval_adv_sep
    epoch_metrics['eval_adversarial_NCMI'] = eval_adv_con / eval_adv_sep
    
    if eval_adv_acc > old_score[1]:
        old_score[0], old_score[1] = test_acc, eval_adv_acc
        trainer.save_model(WEIGHTS)
    if epoch % 10 == 0:
        trainer.save_model_resume(os.path.join(LOG_DIR, 'state-last.pt'), epoch) 
    if epoch % 200 == 0:
        shutil.copyfile(WEIGHTS, os.path.join(LOG_DIR, f'weights-best-epoch{str(epoch)}.pt'))

    logger.log('Time taken: {}'.format(format_time(time.time()-start)))
    metrics = metrics._append(pd.DataFrame(epoch_metrics, index=[0]), ignore_index=True)
    metrics.to_csv(os.path.join(LOG_DIR, 'stats_adv.csv'), index=False)

    
    
# Record metrics

train_acc = res['clean_acc'] if 'clean_acc' in res else trainer.eval(train_dataloader)
logger.log('\nTraining completed.')
logger.log('Standard Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(train_acc*100, old_score[0]*100))
if NUM_ADV_EPOCHS > 0:
    logger.log('Adversarial Accuracy-\tTrain: {:.2f}%.\tEval: {:.2f}%.'.format(res['adversarial_acc']*100, old_score[1]*100)) 

logger.log('Script Completed.')
