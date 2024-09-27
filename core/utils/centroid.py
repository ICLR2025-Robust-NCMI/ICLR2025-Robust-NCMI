import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
from .context import ctx_noparamgrad_and_eval
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, SubsetRandomSampler

class Centroid():
    def __init__(self, n_classes, samples_data, samples_tar, Ctemp, CdecayFactor=0.9999):
        self.n_classes = n_classes
        self.centroids = torch.ones((n_classes, n_classes)) / n_classes
        self.CdecayFactor = CdecayFactor
        self.Ctemp = Ctemp
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.batch_counter = 0
        try:
            self.samples_data = torch.tensor(samples_data).to(self.device)
            self.samples_tar = samples_tar.to(self.device)
        except Exception:
            pass

    def update_batch(self, model, targets, sample_size=64, attack=None):
        idx = torch.randperm(5000)[:sample_size].to(self.device)
        img = torch.index_select(self.samples_data, 1, idx).view(-1,3,32,32)
        tar_idx = torch.cat([5000 * i + idx for i in range(10)], 0)
        tar = torch.index_select(self.samples_tar, 0, tar_idx)
        if attack is not None:
            with ctx_noparamgrad_and_eval(model):
                img, tar = attack.perturb(img, tar)
        with torch.no_grad():
            logits = model(img).detach()
            output = F.softmax(logits.float(), 1)
            for Class in range(self.n_classes):
                try:
                    self.centroids[Class] = self.CdecayFactor * self.centroids[Class] + \
                        (1- self.CdecayFactor) * torch.mean(output[Class * sample_size : (Class + 1) * sample_size], axis = 0).detach().cpu()
                except Exception:
                    breakpoint()

        self.centroids =  self.centroids/(self.centroids.sum(1)[:,None])

    def update_batch_with_logits(self, logits, targets, device=None, softmax=True):
            
        with torch.no_grad():

            Classes =  targets.cpu().unique()
            output = F.softmax(logits.float(), 1).cpu() if softmax else logits.cpu()
        
        for Class in Classes:
            self.centroids[Class] = self.CdecayFactor * self.centroids[Class] + \
                (1 - self.CdecayFactor) * torch.mean(output[targets.cpu() == Class], axis = 0).detach().cpu()

        self.centroids =  self.centroids/(self.centroids.sum(1)[:,None])
    
    def update_epoch(self, model, data_loader, device, attack=None):
        self.centroids = torch.zeros_like(self.centroids)
        model.eval()
        
        for image,target in tqdm(data_loader):
            image,target = image.to(device), target.to(device)
            if attack is not None:
                image = attack(image, target)
            
            with torch.no_grad():
                logit = model(image).detach()

                Classes =  target.cpu().unique()
                logit = logit.cpu()
                output = F.softmax(logit.float(), 1)
            
            for Class in Classes:
                self.centroids[Class] += torch.sum(output[target.cpu() == Class], axis = 0).detach().cpu()

        self.centroids =  self.centroids/(self.centroids.sum(1)[:,None])
    
    def get_centroids(self, target):
        return torch.index_select(self.centroids, 0, target.cpu()).to(target.device)
    
    def record_logits(self, logits, targets, reset=False, update=False):
        if reset:
            self.backup_centroids = torch.zeros_like(self.centroids)
        
        with torch.no_grad():
            output = F.softmax(logits.detach().float(), 1).cpu()
            Classes =  targets.cpu().unique()
        
        for Class in Classes:
            self.backup_centroids[Class] += torch.sum(output[targets.cpu() == Class], axis = 0).detach().cpu()
            
        if update:
            self.backup_centroids = self.backup_centroids / (self.backup_centroids.sum(1)[:, None])
            self.centroids = self.CdecayFactor * self.centroids + (1 - self.CdecayFactor) * self.backup_centroids

    def sample_data(self, sample_size=64):
        idx = torch.randperm(5000)[:sample_size].to(self.device)
        img = torch.index_select(self.samples_data, 1, idx).view(-1,3,32,32)
        tar_idx = torch.cat([5000 * i + idx for i in range(10)], 0)
        tar = torch.index_select(self.samples_tar, 0, tar_idx)
        return img, tar

    def update_sample_batch(self, model, img, targets, sample_size=64):
        with torch.no_grad():
            logits = model(img).detach()
            output = F.softmax(logits.float(), 1)
            for Class in targets.cpu().unique():
                try:
                    self.centroids[Class] = self.CdecayFactor * self.centroids[Class] + \
                        (1- self.CdecayFactor) * torch.mean(output[targets.cpu()==Class], axis = 0).detach().cpu()
                except Exception:
                    breakpoint()
