import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.attacks import create_attack
from core.metrics import accuracy
from core.models import create_model
from core.utils import SmoothCrossEntropyLoss

from .context import ctx_noparamgrad_and_eval
from .utils import seed, track_bn_stats

from .mart import mart_loss
from .rst import CosineLR, WarmUpLR
from .trades import trades_loss

from .centroid import *
from .losses import CE_KL_PWCE_loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SCHEDULERS = ['cyclic', 'step', 'cosine', 'cosinew']


class Trainer(object):
    """
    Helper class for training a deep neural network.
    Arguments:
        info (dict): dataset information.
        args (dict): input arguments.
    """
    def __init__(self, info, args):
        super(Trainer, self).__init__()
        
        self.model = create_model(args.model, args.normalize, info, device)

        self.params = args
        self.n_classes = 10
        if 'cifar100' in args.data:
            self.n_classes = 100
        elif 'tiny-imagenet' in args.data:
            self.n_classes = 200
        self.criterion = nn.CrossEntropyLoss()
        self.init_optimizer(self.params.num_adv_epochs)
        
        if self.params.pretrained_file is not None:
            self.load_model(os.path.join(self.params.log_dir, self.params.pretrained_file, 'weights-best.pt'))
        
        self.attack, self.eval_attack = self.init_attack(self.model, self.criterion, self.params.attack, self.params.attack_eps, 
                                                         self.params.attack_iter, self.params.attack_step)
        
    
    @staticmethod
    def init_attack(model, criterion, attack_type, attack_eps, attack_iter, attack_step):
        """
        Initialize adversary.
        """
        attack = create_attack(model, criterion, attack_type, attack_eps, attack_iter, attack_step, rand_init_type='uniform')
        if attack_type in ['linf-pgd', 'l2-pgd']:
            eval_attack = create_attack(model, criterion, attack_type, attack_eps, 40, attack_step)
        elif attack_type in ['fgsm', 'linf-df']:
            eval_attack = create_attack(model, criterion, 'linf-pgd', 8/255, 20, 2/255)
        elif attack_type in ['fgm', 'l2-df']:
            eval_attack = create_attack(model, criterion, 'l2-pgd', 128/255, 20, 15/255)
        return attack,  eval_attack
    
    
    def init_optimizer(self, num_epochs):
        """
        Initialize optimizer and scheduler.
        """
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay, 
                                         momentum=0.9, nesterov=self.params.nesterov)
        if num_epochs <= 0:
            return
        self.init_scheduler(num_epochs)
    
        
    def init_scheduler(self, num_epochs):
        """
        Initialize scheduler.
        """
        if self.params.scheduler == 'cyclic':
            num_samples = 50000 if 'cifar10' in self.params.data else 73257
            num_samples = 100000 if 'tiny-imagenet' in self.params.data else num_samples
            update_steps = int(np.floor(num_samples/self.params.batch_size) + 1)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.params.lr, pct_start=0.25,
                                                                 steps_per_epoch=update_steps, epochs=int(num_epochs))
        elif self.params.scheduler == 'step':
            warmup_steps = int(.025 * num_epochs)
            warmup = WarmUpLR(self.optimizer, warmup_steps)
            multistep = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=0.1, milestones=[100 - warmup_steps, 150 - warmup_steps])
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[warmup, multistep], milestones=[warmup_steps])    
        elif self.params.scheduler == 'cosine':
            self.scheduler = CosineLR(self.optimizer, max_lr=self.params.lr, epochs=int(num_epochs))
        elif self.params.scheduler == 'cosinew':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.params.lr, pct_start=0.025, 
                                                                 total_steps=int(num_epochs))
        else:
            self.scheduler = None
    
    
    def train(self, dataloader, epoch=0, adversarial=False, centroids=None, verbose=False):
        """
        Run one epoch of training.
        """
        metrics = pd.DataFrame()
        self.model.train()
        
        for data in tqdm(dataloader, desc='Epoch {}: '.format(epoch)):
            x, y = data
            x, y = x.to(device), y.to(device)
            
            if adversarial:
                if self.params.beta is not None and self.params.mart:
                    loss, batch_metrics = self.mart_loss(x, y, beta=self.params.beta)
                elif self.params.beta is not None and (not self.params.NCMI):
                    loss, batch_metrics = self.trades_loss(x, y, beta=self.params.beta)
                elif self.params.NCMI:
                    if self.params.beta is None:
                        if self.params.param[1] != 0 or self.params.param[2] != 0:
                            loss, batch_metrics = self.NCMI_attack_loss(x, y, centroids)
                        else:
                            loss, batch_metrics = self.NCMI_loss(x, y, centroids)
                    else:
                        loss, batch_metrics = self.NCMI_trades_loss(x, y, centroids, beta=self.params.beta)
                else:
                    loss, batch_metrics = self.adversarial_loss(x, y)
            else:
                loss, batch_metrics = self.standard_loss(x, y)
                
            if self.params.NCMI:
                centroids.update_batch(self.model, y, self.params.SampleSize, self.attack)
            
            loss.backward()
            if self.params.clip_grad:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_grad)
            self.optimizer.step()
            if self.params.scheduler in ['cyclic']:
                self.scheduler.step()
            
            metrics = metrics.append(pd.DataFrame(batch_metrics, index=[0]), ignore_index=True)
        
        if self.params.scheduler in ['step', 'converge', 'cosine', 'cosinew']:
            self.scheduler.step()
        if self.params.NCMI:
            return dict(metrics.mean()), centroids
        return dict(metrics.mean())
    
    
    def standard_loss(self, x, y):
        """
        Standard training.
        """
        self.optimizer.zero_grad()
        out = self.model(x)
        loss = self.criterion(out, y)
        
        preds = out.detach()
        batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, preds)}
        return loss, batch_metrics
    
    
    def adversarial_loss(self, x, y):
        """
        Adversarial training (Madry et al, 2017).
        """
        with ctx_noparamgrad_and_eval(self.model):
            x_adv, _ = self.attack.perturb(x, y)
        
        self.optimizer.zero_grad()
        if self.params.keep_clean:
            x_adv = torch.cat((x, x_adv), dim=0)
            y_adv = torch.cat((y, y), dim=0)
        else:
            y_adv = y
        out = self.model(x_adv)
        loss = self.criterion(out, y_adv)
        
        preds = out.detach()
        batch_metrics = {'loss': loss.item()}
        if self.params.keep_clean:
            preds_clean, preds_adv = preds[:len(x)], preds[len(x):]
            batch_metrics.update({'clean_acc': accuracy(y, preds_clean), 'adversarial_acc': accuracy(y, preds_adv)})
        else:
            batch_metrics.update({'adversarial_acc': accuracy(y, preds)})    
        return loss, batch_metrics
    
    def NCMI_loss(self, x, y, centroids):

        with ctx_noparamgrad_and_eval(self.model):
            x_adv, _ = self.attack.perturb(x, y)
        
        self.optimizer.zero_grad()
        y_adv = y
        out = self.model(x_adv)
        try:
            Cs = centroids.get_centroids(y).to(device)
        except Exception:
            breakpoint()
        loss_func = CE_KL_PWCE_loss(self.n_classes)
        ce, con, sep = loss_func(out, y_adv, Cs)
        loss = self.params.param[0] * ce + self.params.param[1] * con + self.params.param[2] * sep
        
        preds = out.detach()
        batch_metrics = {'loss': loss.item(), 'con': con.item(), 'sep': -sep.item(), 'NCMI': (-con / sep).item()}

        batch_metrics.update({'adversarial_acc': accuracy(y, preds)})    
        return loss, batch_metrics
    
    def NCMI_attack_loss(self, x_natural, y, centroids, step_size=2/255, epsilon=8/255, perturb_steps=10, attack='linf-pgd', epoch=0, reset=False, update=False):
    
        with ctx_noparamgrad_and_eval(self.model):
            x_adv, _ = self.attack.perturb(x_natural, y)

        self.optimizer.zero_grad()
        logits_pgd_ce = self.model(x_adv)
        ce = F.cross_entropy(logits_pgd_ce, y) if self.params.ls == 0 else F.cross_entropy(logits_pgd_ce, y, label_smoothing=self.params.ls)
        adv_acc = accuracy(y, logits_pgd_ce.detach())
        
        criterion = CE_KL_PWCE_loss(self.n_classes)

        self.model.train()
        track_bn_stats(self.model, False)
        batch_size = len(x_natural)
        
        x_adv = x_natural.detach() +  torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        Cs = centroids.get_centroids(y).to(device)
        
        if attack == 'linf-pgd':
            for i in range(perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    logits_adv = self.model(x_adv)
                    _, con, sep = criterion(logits_adv, y, Cs)

                    loss_NCMI = con + (self.params.param[2] / max(1e-6, self.params.param[1])) * sep
                grad = torch.autograd.grad(loss_NCMI, [x_adv])[0]
                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        else:
            raise ValueError(f'Attack={attack} not supported for TRADES training!')
        self.model.train()
        track_bn_stats(self.model, True)
    
        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        
        self.optimizer.zero_grad()
        # calculate robust loss
        logits_natural = self.model(x_natural)
        logits_adv = self.model(x_adv)

        _, con, sep = criterion(logits_adv, y, Cs)
        loss = self.params.param[0] * ce + self.params.param[1] * con + self.params.param[2] * sep
        
        batch_metrics = {'loss': loss.item(), 'con': con.item(), 'sep': -sep.item(), 'NCMI': (-con / sep).item()}
        batch_metrics.update({'clean_acc': accuracy(y, logits_natural.detach()), 'adversarial_acc': adv_acc})
            
        return loss, batch_metrics, centroids, x_adv, y
    
    def NCMI_attack_loss_clean(self, x_natural, y, centroids, step_size=2/255, epsilon=8/255, perturb_steps=10, attack='linf-pgd'):

        criterion = CE_KL_PWCE_loss(self.n_classes)
        try:
            Cs = centroids.get_centroids(y).to(device)
        except Exception:
            breakpoint()
        self.model.train()
        track_bn_stats(self.model, False)
        batch_size = len(x_natural)
        
        x_adv = x_natural.detach() +  torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        p_natural = F.softmax(self.model(x_natural), dim=1)
        p_natural = p_natural.detach()
        
        if attack == 'linf-pgd':
            for _ in range(perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    _, con, sep = criterion(self.model(x_adv), y, Cs)
                    loss_NCMI = con + (self.params.param[2] / self.params.param[1]) * sep
                grad = torch.autograd.grad(loss_NCMI, [x_adv])[0]
                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        else:
            raise ValueError(f'Attack={attack} not supported for TRADES training!')
        self.model.train()
        track_bn_stats(self.model, True)
    
        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        
        self.optimizer.zero_grad()
        # calculate robust loss
        logits_natural = self.model(x_natural)
        logits_adv = self.model(x_adv)

        _, con, sep = criterion(logits_adv, y, Cs)
        ce = F.cross_entropy(logits_natural, y)
        adv_acc = accuracy(y, logits_adv.detach())
        loss = self.params.param[0] * ce + self.params.param[1] * con + self.params.param[2] * sep
        
        batch_metrics = {'loss': loss.item(), 'con': con.item(), 'sep': -sep.item(), 'NCMI': (-con / sep).item()}
        batch_metrics.update({'clean_acc': accuracy(y, logits_natural.detach()), 'adversarial_acc': adv_acc})
            
        return loss, batch_metrics
    
    def NCMI_attack_trades_loss(self, x_natural, y, centroids, step_size=2/255, epsilon=8/255, perturb_steps=10, beta=5.0, 
                                attack='linf-pgd', label_smoothing=0., epoch=0, reset=False, update=False, awp=None):
        """
        TRADES training (Zhang et al, 2019).
        """
    
        criterion_ce = SmoothCrossEntropyLoss(reduction='mean', smoothing=label_smoothing)
        criterion_kl = nn.KLDivLoss(reduction='batchmean')
        criterion_NCMI = CE_KL_PWCE_loss(self.n_classes)

        self.model.train()
        track_bn_stats(self.model, False)
        batch_size = len(x_natural)
        
        x_adv = x_natural.detach() +  torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()
        x_adv_kl = torch.clamp(x_adv, 0.0, 1.0)
        x_adv_NCMI = torch.clamp(x_adv, 0.0, 1.0)

        p_natural = F.softmax(self.model(x_natural), dim=1)
        p_natural = p_natural.detach()
        logits_adv_NCMI = self.model(x_adv_NCMI)
        Cs = centroids.get_centroids(y).to(device)
        
        if attack == 'linf-pgd':
            for _ in range(perturb_steps):
                x_adv_kl.requires_grad_()
                with torch.enable_grad():
                    loss_kl = criterion_kl(F.log_softmax(self.model(x_adv_kl), dim=1), p_natural)
                grad_kl = torch.autograd.grad(loss_kl, [x_adv_kl])[0]
                x_adv_kl = x_adv_kl.detach() + step_size * torch.sign(grad_kl.detach())
                x_adv_kl = torch.min(torch.max(x_adv_kl, x_natural - epsilon), x_natural + epsilon)
                x_adv_kl = torch.clamp(x_adv_kl, 0.0, 1.0)

                if self.params.param[1] == 0 and self.params.param[2] == 0:
                    continue

                x_adv_NCMI.requires_grad_()
                with torch.enable_grad():
                    logits_adv_NCMI = self.model(x_adv_NCMI)
                    _, con, sep = criterion_NCMI(logits_adv_NCMI, y, Cs)
                    loss_NCMI = con + (self.params.param[2] / self.params.param[1]) * sep
                grad_NCMI = torch.autograd.grad(loss_NCMI, [x_adv_NCMI])[0]
                x_adv_NCMI = x_adv_NCMI.detach() + step_size * torch.sign(grad_NCMI.detach())
                x_adv_NCMI = torch.min(torch.max(x_adv_NCMI, x_natural - epsilon), x_natural + epsilon)
                x_adv_NCMI = torch.clamp(x_adv_NCMI, 0.0, 1.0)
        
        else:
            raise ValueError(f'Attack={attack} not supported for TRADES training!')
        self.model.train()
        track_bn_stats(self.model, True)
    
        x_adv_kl = Variable(torch.clamp(x_adv_kl, 0.0, 1.0), requires_grad=False)
        if self.params.param[1] != 0 or self.params.param[2] != 0:
            x_adv_NCMI = Variable(torch.clamp(x_adv_NCMI, 0.0, 1.0), requires_grad=False)
        
        w_adv = None
        if self.params.awp:
            warmup = 2 * self.params.awp_warmup if self.params.data.endswith('s') else self.params.awp_warmup
            if epoch >= warmup:
                w_adv = awp.calc_awp(inputs_adv=x_adv,
                                     inputs_clean=x_natural,
                                     targets=y,
                                     beta=self.params.beta)
                awp.perturb(w_adv)
        
        self.optimizer.zero_grad()
        # calculate robust loss
        logits_natural = self.model(x_natural)
        logits_adv_kl = self.model(x_adv_kl)
        if self.params.param[1] != 0 or self.params.param[2] != 0:
            logits_adv_NCMI = self.model(x_adv_NCMI)
        
        loss_natural = criterion_ce(logits_natural, y)

        loss_robust = criterion_kl(F.log_softmax(logits_adv_kl, dim=1), F.softmax(logits_natural, dim=1))
        _, con, sep = criterion_NCMI(logits_adv_NCMI, y, Cs) if (self.params.param[1] != 0 or self.params.param[2] != 0) else criterion_NCMI(logits_adv_kl, y, Cs)
        loss = self.params.param[0] * (loss_natural + beta * loss_robust) + self.params.param[1] * con + self.params.param[2] * sep
        
        batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, logits_natural.detach()), 
                    'adversarial_acc': accuracy(y, logits_adv_kl.detach()), 
                    'con': con.item(), 'sep': -sep.item(), 'NCMI': (-con / sep).item()}
            
        return loss, batch_metrics, centroids, w_adv, x_adv_NCMI, y
    
    def NCMI_trades_loss(self, x, y, centroids, beta):

        with ctx_noparamgrad_and_eval(self.model):
            x_adv, _ = self.attack.perturb(x, y)
        
        loss, batch_metrics = self.trades_loss(x, y, beta)
        y_adv = y
        out = self.model(x_adv)
        try:
            Cs = centroids.get_centroids(y).to(device)
        except Exception:
            breakpoint()
        loss_func = CE_KL_PWCE_loss(self.n_classes)
        _, con, sep = loss_func(out, y_adv, Cs)
        loss_NCMI = self.params.param[1] * con + self.params.param[2] * sep
        
        preds = out.detach()
        batch_metrics.update({'loss': loss.item() + loss_NCMI.item(), 'con': con.item(), 'sep': -sep.item(), 'NCMI': (-con / sep).item()})
        batch_metrics.update({'adversarial_acc': accuracy(y, preds)})    
        return loss, batch_metrics
    
    
    def NCMI_attack_mart_loss(self, x_natural, y, centroids, step_size=2/255, epsilon=8/255, perturb_steps=10, beta=6.0, 
                              attack='linf-pgd', epoch=0, reset=False, update=False):
        """
        MART training (Wang et al, 2020).
        """
    
        kl = nn.KLDivLoss(reduction='none')
        criterion_NCMI = CE_KL_PWCE_loss(self.n_classes)
        self.model.eval()
        track_bn_stats(self.model, False)
        batch_size = len(x_natural)
        
        # generate adversarial example
        x_adv_CE = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
        x_adv_NCMI = x_natural.detach() +  torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()
        x_adv_CE = torch.clamp(x_adv_CE, 0.0, 1.0)
        x_adv_NCMI = torch.clamp(x_adv_NCMI, 0.0, 1.0)
        Cs = centroids.get_centroids(y).to(device)
        
        if attack == 'linf-pgd':
            for _ in range(perturb_steps):
                x_adv_CE.requires_grad_()
                with torch.enable_grad():
                    loss_ce = F.cross_entropy(self.model(x_adv_CE), y)
                grad = torch.autograd.grad(loss_ce, [x_adv_CE])[0]
                x_adv_CE = x_adv_CE.detach() + step_size * torch.sign(grad.detach())
                x_adv_CE = torch.min(torch.max(x_adv_CE, x_natural - epsilon), x_natural + epsilon)
                x_adv_CE = torch.clamp(x_adv_CE, 0.0, 1.0)
                
                if (not self.params.NCMI) or (self.params.param[1] == 0 and self.params.param[2] == 0):
                    continue
                
                x_adv_NCMI.requires_grad_()
                with torch.enable_grad():
                    logits_adv_NCMI = self.model(x_adv_NCMI)
                    _, con, sep = criterion_NCMI(logits_adv_NCMI, y, Cs)
                    loss_NCMI = con + (self.params.param[2] / self.params.param[1]) * sep
                grad_NCMI = torch.autograd.grad(loss_NCMI, [x_adv_NCMI])[0]
                x_adv_NCMI = x_adv_NCMI.detach() + step_size * torch.sign(grad_NCMI.detach())
                x_adv_NCMI = torch.min(torch.max(x_adv_NCMI, x_natural - epsilon), x_natural + epsilon)
                x_adv_NCMI = torch.clamp(x_adv_NCMI, 0.0, 1.0)
                    
        else:
            raise ValueError(f'Attack={attack} not supported for MART training!')
        self.model.train()
        track_bn_stats(self.model, True)
        
        x_adv_CE = Variable(torch.clamp(x_adv_CE, 0.0, 1.0), requires_grad=False)
        if self.params.NCMI and (self.params.param[1] != 0 or self.params.param[2] != 0):
            x_adv_NCMI = Variable(torch.clamp(x_adv_NCMI, 0.0, 1.0), requires_grad=False)
        # zero gradient
        self.optimizer.zero_grad()

        logits = self.model(x_natural)
        logits_adv_CE = self.model(x_adv_CE)
        if self.params.NCMI and (self.params.param[1] != 0 or self.params.param[2] != 0):
            logits_adv_NCMI = self.model(x_adv_NCMI)
        
        adv_probs = F.softmax(logits_adv_CE, dim=1)
        tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
        new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
        loss_adv = F.cross_entropy(logits_adv_CE, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

        nat_probs = F.softmax(logits, dim=1)
        true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

        loss_robust = (1.0 / batch_size) * torch.sum(
            torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
        _, con, sep = criterion_NCMI(logits_adv_NCMI, y, Cs) if (self.params.param[1] != 0 or self.params.param[2] != 0) else criterion_NCMI(logits_adv_CE, y, Cs)
        loss = self.params.param[0] * (loss_adv + float(beta) * loss_robust) + self.params.param[1] * con + self.params.param[2] * sep

        batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, logits.detach()), 
                        'adversarial_acc': accuracy(y, logits_adv_CE.detach()),
                        'con': con.item(), 'sep': -sep.item(), 'NCMI': (-con / sep).item()}
            
        return loss, batch_metrics, centroids, x_adv_NCMI, y
    
    
    def trades_loss(self, x, y, beta):
        """
        TRADES training.
        """
        loss, batch_metrics = trades_loss(self.model, x, y, self.optimizer, step_size=self.params.attack_step, 
                                          epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                          beta=beta, attack=self.params.attack)
        return loss, batch_metrics  

    
    def mart_loss(self, x, y, beta):
        """
        MART training.
        """
        loss, batch_metrics = mart_loss(self.model, x, y, self.optimizer, step_size=self.params.attack_step, 
                                        epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                        beta=beta, attack=self.params.attack)
        return loss, batch_metrics  
    
    
    def eval(self, dataloader, adversarial=False):
        """
        Evaluate performance of the model.
        """
        acc = 0.0
        self.model.eval()
        
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if adversarial:
                with ctx_noparamgrad_and_eval(self.model):
                    x_adv, _ = self.eval_attack.perturb(x, y)            
                out = self.model(x_adv)
            else:
                out = self.model(x)
            acc += accuracy(y, out)
        acc /= len(dataloader)
        return acc

    
    def save_model(self, path):
        """
        Save model weights.
        """
        torch.save({'model_state_dict': self.model.state_dict()}, path)

    
    def load_model(self, path, load_opt=True):
        """
        Load model weights.
        """
        checkpoint = torch.load(path)
        if 'model_state_dict' not in checkpoint:
            raise RuntimeError('Model weights not found at {}.'.format(path))
        self.model.load_state_dict(checkpoint['model_state_dict'])
