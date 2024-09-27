import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.attacks import create_attack
from core.attacks import CWLoss
from core.metrics import accuracy
from core.models import create_model

from core.utils import ctx_noparamgrad_and_eval
from core.utils import Trainer
from core.utils import set_bn_momentum
from core.utils import seed
from core.utils.centroid import *
from core.utils.losses import CE_KL_PWCE_loss

from .trades import trades_loss, trades_loss_LSE
from .cutmix import cutmix


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WATrainer(Trainer):
    """
    Helper class for training a deep neural network with model weight averaging (identical to Gowal et al, 2020).
    Arguments:
        info (dict): dataset information.
        args (dict): input arguments.
    """
    def __init__(self, info, args):
        super(WATrainer, self).__init__(info, args)
        
        self.wa_model = copy.deepcopy(self.model)
        self.eval_attack = create_attack(self.wa_model, CWLoss, args.attack, args.attack_eps, 4*args.attack_iter, 
                                         args.attack_step)
        num_samples = 50000 if 'cifar' in self.params.data else 73257
        num_samples = 100000 if 'tiny-imagenet' in self.params.data else num_samples
        if self.params.data in ['cifar10', 'cifar10s', 'svhn', 'svhns']:
            self.num_classes = 10
        elif self.params.data in ['cifar100', 'cifar100s']:
            self.num_classes = 100
        elif self.params.data == 'tiny-imagenet':
            self.num_classes = 200
        self.params.batch_size = self.params.batch_size if self.params.imbalance else 640
        self.update_steps = int(np.floor(num_samples/self.params.batch_size) + 1)
        if self.params.scheduler == ['cosinew', 'step']:
            self.warmup_steps = 0.025 * self.params.num_adv_epochs * self.update_steps
        elif self.params.scheduler == 'cyclic':
            self.warmup_steps = 0.25 * self.params.num_adv_epochs * self.update_steps
        else:
            self.warmup_steps = 0
    
    
    def init_optimizer(self, num_epochs):
        """
        Initialize optimizer and schedulers.
        """
        def group_weight(model):
            group_decay = []
            group_no_decay = []
            for n, p in model.named_parameters():
                if 'batchnorm' in n:
                    group_no_decay.append(p)
                else:
                    group_decay.append(p)
            assert len(list(model.parameters())) == len(group_decay) + len(group_no_decay)
            groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
            return groups
        
        self.optimizer = torch.optim.SGD(group_weight(self.model), lr=self.params.lr, weight_decay=self.params.weight_decay, 
                                         momentum=0.9, nesterov=self.params.nesterov)
        if num_epochs <= 0:
            return
        self.init_scheduler(num_epochs)
    
    
    def train(self, dataloader, epoch=0, adversarial=False, verbose=False, centroids=None, awp=None):
        """
        Run one epoch of training.
        """
        metrics = pd.DataFrame()
        self.model.train()
        
        update_iter = 0
        for data in tqdm(dataloader, desc='Epoch {}: '.format(epoch)):
            global_step = (epoch - 1) * self.update_steps + update_iter
            if global_step == 0:
                # make BN running mean and variance init same as Haiku
                set_bn_momentum(self.model, momentum=1.0)
            elif global_step == 1:
                set_bn_momentum(self.model, momentum=0.01)
            update_iter += 1
            
            x, y = data
            if self.params.consistency:
                x_aug1, x_aug2, y = x[0].to(device), x[1].to(device), y.to(device)
                if self.params.beta is not None:
                    loss, batch_metrics = self.trades_loss_consistency(x_aug1, x_aug2, y, beta=self.params.beta)

            else:
                if self.params.CutMix:
                    x_all, y_all = torch.tensor([]), torch.tensor([])
                    for i in range(4): # 128 x 4 = 512 or 256 x 4 = 1024
                        x_tmp, y_tmp = x.detach(), y.detach()
                        x_tmp, y_tmp = cutmix(x_tmp, y_tmp, alpha=1.0, beta=1.0, num_classes=self.num_classes)
                        x_all = torch.cat((x_all, x_tmp), dim=0)
                        y_all = torch.cat((y_all, y_tmp), dim=0)
                    x, y = x_all.to(device), y_all.to(device)
                else:
                    x, y = x.to(device), y.to(device)
                
                if adversarial:
                    if self.params.beta is not None and self.params.mart:
                        if self.params.NCMI:
                            loss, batch_metrics, centroids, x_adv_sample, y_sample = self.NCMI_attack_mart_loss(x, y, centroids, self.params.attack_step,
                                                                                                                self.params.attack_eps, self.params.attack_iter,
                                                                                                                beta=self.params.beta, epoch=epoch, reset=(update_iter==1), 
                                                                                                                update=(update_iter==self.update_steps))
                        else:
                            loss, batch_metrics = self.mart_loss(x, y, beta=self.params.beta)
                    elif self.params.beta is not None and self.params.LSE:
                        loss, batch_metrics = self.trades_loss_LSE(x, y, beta=self.params.beta)
                    elif self.params.beta is not None and (not self.params.NCMI):
                        loss, batch_metrics = self.trades_loss(x, y, beta=self.params.beta)
                    elif self.params.NCMI:
                        if self.params.beta is None:
                            if self.params.param[1] != 0 or self.params.param[2] != 0:
                                if self.params.clean_center:
                                    loss, batch_metrics = self.NCMI_attack_loss_clean(x, y, centroids, self.params.attack_step, 
                                                                                      self.params.attack_eps, self.params.attack_iter, self.params.attack)
                                else:
                                    loss, batch_metrics, centroids, x_adv_sample, y_sample = self.NCMI_attack_loss(x, y, centroids, self.params.attack_step, 
                                                                                           self.params.attack_eps, self.params.attack_iter, self.params.attack, epoch=epoch,
                                                                                           reset=(update_iter==1), update=(update_iter==self.update_steps))
                            else:
                                loss, batch_metrics, centroids, x_adv_sample, y_sample = self.NCMI_attack_loss(x, y, centroids, self.params.attack_step, 
                                                                                       self.params.attack_eps, self.params.attack_iter, self.params.attack, epoch=epoch,
                                                                                       reset=(update_iter==1), update=(update_iter==self.update_steps))
                        else:
                            loss, batch_metrics, centroids, w_adv, x_adv_sample, y_sample = self.NCMI_attack_trades_loss(x, y, centroids, self.params.attack_step, 
                                                                                                                         self.params.attack_eps, self.params.attack_iter, 
                                                                                                                         beta=self.params.beta, label_smoothing=self.params.ls,
                                                                                                                         epoch=epoch, reset=(update_iter==1), update=(update_iter==self.update_steps), awp=awp)
                    else:
                        loss, batch_metrics = self.adversarial_loss(x, y)
                else:
                    loss, batch_metrics = self.standard_loss(x, y)
                    
            loss.backward()
            if self.params.clip_grad:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_grad)
            self.optimizer.step()
            if self.params.awp:
                warmup = self.params.awp_warmup * 2 if self.params.data.endswith('s') else self.params.awp_warmup
                if epoch >= warmup:
                    awp.restore(w_adv)
            if self.params.NCMI:
                centroids.update_sample_batch(self.model, x_adv_sample, y_sample)
            if self.params.scheduler in ['cyclic']:
                self.scheduler.step()
            
            global_step = (epoch - 1) * self.update_steps + update_iter
            ema_update(self.wa_model, self.model, global_step, decay_rate=self.params.tau, 
                       warmup_steps=self.warmup_steps, dynamic_decay=True)
            metrics = metrics._append(pd.DataFrame(batch_metrics, index=[0]), ignore_index=True)
        
        if self.params.scheduler in ['step', 'converge', 'cosine', 'cosinew']:
            self.scheduler.step()
        
        update_bn(self.wa_model, self.model)
        if self.params.NCMI:
            return dict(metrics.mean()), centroids
        return dict(metrics.mean())
    
    
    def trades_loss(self, x, y, beta):
        """
        TRADES training.
        """
        loss, batch_metrics = trades_loss(self.model, x, y, self.optimizer, step_size=self.params.attack_step, 
                                          epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                          beta=beta, attack=self.params.attack, label_smoothing=self.params.ls,
                                          use_cutmix=self.params.CutMix)
        return loss, batch_metrics

    def trades_loss_consistency(self, x_aug1, x_aug2, y, beta):
        """
        TRADES training with Consistency.
        """
        x = torch.cat([x_aug1, x_aug2], dim=0)
        loss, batch_metrics = trades_loss(self.model, x, y.repeat(2), self.optimizer, step_size=self.params.attack_step, 
                                          epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                          beta=beta, attack=self.params.attack, label_smoothing=self.params.ls,
                                          use_cutmix=self.params.CutMix, use_consistency=True, cons_lambda=self.params.cons_lambda, cons_tem=self.params.cons_tem)
        return loss, batch_metrics

    def trades_loss_LSE(self, x, y, beta):
        """
        TRADES training with LSE loss.
        """
        loss, batch_metrics = trades_loss_LSE(self.model, x, y, self.optimizer, step_size=self.params.attack_step, 
                                          epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                          beta=beta, attack=self.params.attack, label_smoothing=self.params.ls,
                                          clip_value=self.params.clip_value,
                                          use_cutmix=self.params.CutMix,
                                          num_classes=self.num_classes)
        return loss, batch_metrics  

    
    def eval(self, dataloader, adversarial=False, test_NCMI=False, perturb_steps=10, epsilon=8/255, step_size=2/255, epoch=None):
        """
        Evaluate performance of the model.
        """
        acc = 0.0
        self.wa_model.eval()
        loss_func = CE_KL_PWCE_loss(self.num_classes)
        centroids = Centroid(self.num_classes, None, None, Ctemp=1)
        centroids.centroids = torch.zeros_like(centroids.centroids)
        out_total, y_total = [], []
        loss_total, con_total, sep_total = [], [], []
        dummy_centroids = Centroid(self.n_classes, None, None, Ctemp=1, CdecayFactor=0)
        loss_total = 0
        if epoch is not None:
            logits_dir = os.path.join(self.params.log_dir, self.params.desc, 'logits', 'epoch_{}'.format(epoch))
            centroids_dir = os.path.join(self.params.log_dir, self.params.desc, 'centroids')
        
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            if adversarial:
                with ctx_noparamgrad_and_eval(self.wa_model):
                    x_adv, _ = self.eval_attack.perturb(x, y)            
                out = self.wa_model(x_adv)
            else:
                out = self.wa_model(x)
            if epoch is not None:
                torch.save(out.cpu(), os.path.join(logits_dir, 'logits_CE_batch_{}.pth'.format(batch)))
            acc += accuracy(y, out)
            loss_total += F.cross_entropy(out, y).cpu().item()

            if adversarial:
                if test_NCMI:
                    x_adv = x.detach() +  torch.FloatTensor(x.shape).uniform_(-epsilon, epsilon).cuda().detach()
                    x_adv = torch.clamp(x_adv, 0.0, 1.0)
                    dummy_centroids.update_batch_with_logits(self.wa_model(x_adv).detach(), y, device)
                    try:
                        Cs = dummy_centroids.get_centroids(y).to(device)
                    except Exception:
                        breakpoint()
                    for i in range(perturb_steps):
                        x_adv.requires_grad_()
                        with torch.enable_grad():
                            logits_adv = self.model(x_adv)
                            _, con, sep = loss_func(logits_adv, y, Cs)
                            loss_NCMI = con + (self.params.param[2] / max(1e-6, self.params.param[1])) * sep
                        grad = torch.autograd.grad(loss_NCMI, [x_adv])[0]
                        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
                        x_adv = torch.clamp(x_adv, 0.0, 1.0)

                        if (i + 1) < perturb_steps and (i + 1) % 2 == 0:
                            dummy_centroids.update_batch_with_logits(logits_adv.detach(), y, device)
                            Cs = dummy_centroids.get_centroids(y).to(device)

                    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
                    out = self.wa_model(x_adv)
                    if epoch is not None:
                        torch.save(out.cpu(), os.path.join(logits_dir, 'logits_NCMI_batch_{}.pth'.format(batch)))
                
                with torch.no_grad():
                    out = F.softmax(out, 1)
                    out_total.append(out)
                    y_total.append(y)
                    Classes = y.cpu().unique()
                    for c in Classes:
                        try:
                            centroids.centroids[c] += torch.sum(out[y.cpu() == c], axis = 0).detach().cpu()
                        except Exception:
                            breakpoint()

        
        acc /= len(dataloader)
        if adversarial:
            con_total, sep_total = 0, 0
            with torch.no_grad():
                centroids.centroids = centroids.centroids / centroids.centroids.sum(1)[:, None]
                if epoch is not None:
                    torch.save(centroids.centroids.cpu(), os.path.join(centroids_dir, 'epoch_{}.pth'.format(epoch)))
                for i in range(len(out_total)):
                    out, y = out_total[i].to(device), y_total[i].to(device)
                    Cs = centroids.get_centroids(y).to(device)
                    _, con, sep = loss_func(out, y, Cs)
                    con_total += con.item()
                    sep_total += -sep.item()

            return acc, loss_total / len(dataloader), con_total / len(dataloader), sep_total / len(dataloader)
        
        return acc


    def save_model(self, path):
        """
        Save model weights.
        """
        torch.save({
            'model_state_dict': self.wa_model.state_dict(), 
            'unaveraged_model_state_dict': self.model.state_dict()
        }, path)


    def save_model_resume(self, path, epoch):
        """
        Save model weights and optimizer.
        """
        torch.save({
            'model_state_dict': self.wa_model.state_dict(), 
            'unaveraged_model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(), 
            'scheduler_state_dict': self.scheduler.state_dict(), 
            'epoch': epoch
        }, path)

    
    def load_model(self, path):
        """
        Load model weights.
        """
        checkpoint = torch.load(path)
        if 'model_state_dict' not in checkpoint:
            raise RuntimeError('Model weights not found at {}.'.format(path))
        self.wa_model.load_state_dict(checkpoint['model_state_dict'])
    

    def load_model_resume(self, path):
        """
        load model weights and optimizer.
        """
        checkpoint = torch.load(path)
        if 'model_state_dict' not in checkpoint:
            raise RuntimeError('Model weights not found at {}.'.format(path))
        self.wa_model.load_state_dict(checkpoint['model_state_dict'])
        self.model.load_state_dict(checkpoint['unaveraged_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']


def ema_update(wa_model, model, global_step, decay_rate=0.995, warmup_steps=0, dynamic_decay=True):
    """
    Exponential model weight averaging update.
    """
    factor = int(global_step >= warmup_steps)
    if dynamic_decay:
        delta = global_step - warmup_steps
        decay = min(decay_rate, (1. + delta) / (10. + delta)) if 10. + delta != 0 else decay_rate
    else:
        decay = decay_rate
    decay *= factor
    
    for p_swa, p_model in zip(wa_model.parameters(), model.parameters()):
        p_swa.data *= decay
        p_swa.data += p_model.data * (1 - decay)


@torch.no_grad()
def update_bn(avg_model, model):
    """
    Update batch normalization layers.
    """
    avg_model.eval()
    model.eval()
    for module1, module2 in zip(avg_model.modules(), model.modules()):
        if isinstance(module1, torch.nn.modules.batchnorm._BatchNorm):
            module1.running_mean = module2.running_mean
            module1.running_var = module2.running_var
            module1.num_batches_tracked = module2.num_batches_tracked
