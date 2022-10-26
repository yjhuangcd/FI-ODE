import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

import ExpConfig
from ExpConfig import ExpCfg, RobustExpCfg
from sl_pipeline_test import SLExperiment
from pl_modules import AdversarialLearning
from pathlib import Path
import torch as th
import os
import itertools
# import torchattacks
from autoattack import AutoAttack
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd.functional import jvp
import torch.nn as nn
from torch.nn import functional as F
import math
import wandb
import itertools
from tqdm import tqdm


def sample_decision_boundary(n=10, T=10):
    """
    n: number of classes
    T: density of sampling
    label: ground truth label
    """
    sol = [[np.zeros((0,j)) for j in range(n+1)] for k in range(T+1)]

    for j in tqdm(range(T+1)):
        for k in range(n+1):
            if(j==0):
                sol[j][k] = np.array([[0]*k])
            elif(k<2 or j==1):
                pass
            elif(k==2 and np.mod(j,2)==0):
                sol[j][k] = np.array([[j/2,j/2]])
            elif(k==2 and np.mod(j,2)==1):
                pass
            else:
                for l in range(k-1):
                    if(j-k+l>=0 and k-l>=0):
                        tmp = sol[j-k+l][k-l] + 1
                        combinations = list(itertools.combinations(range(1, k), k-l-1))
                        for c in combinations:
                            new_sol = np.zeros((tmp.shape[0], k))
                            if(tmp.shape[0]>0):
                                new_sol[:, [0]+list(c)] = tmp
                                sol[j][k] = np.concatenate((sol[j][k], new_sol))
    grid = sol[T][n] / T

    return grid


def get_grid_for_label(grid, label):
    if label != 0:
        grid[:, [label, 0]] = grid[:, [0, label]]

    grid_tensor = th.from_numpy(grid).float()
    return grid_tensor


def hydra_conf_load_from_checkpoint_nonstrict(chkpt_file, cfg):
    instance_args = dict()
    cfg_mask = list()
    for k in cfg.keys():
        if OmegaConf.is_dict(cfg[k]) and '_target_' in cfg[k]:
            instance_args[k] = hydra.utils.instantiate(cfg[k])
        else:
            cfg_mask += [k]
    ModuleType = type(hydra.utils.instantiate(cfg))
    return ModuleType.load_from_checkpoint(
        chkpt_file,
        map_location=lambda storage, loc: storage,
        strict=False,
        **OmegaConf.masked_copy(cfg, cfg_mask),
        **instance_args
    )


class AutoLirpaModelRawDynamics(nn.Module):
    def __init__(self, param_map, dynamics):
        super().__init__()
        self.param_map = param_map
        self.dynamics = dynamics

    def forward(self, eta, x):
        return self.dynamics._h_dot_raw(eta, self.param_map(x))


class AutoAttackModelRawDynamics(nn.Module):
    def __init__(self, param_map, dynamics):
        super().__init__()
        self.param_map = param_map
        self.dynamics = dynamics

    def forward(self, x):
        eta = th.ones(1, 10).to(x.device) / 10
        return self.dynamics._h_dot_raw(eta, self.param_map(x))


class VdotDecisionBoundary(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_Lips = True
        self.LVeta = math.sqrt(2)

    def forward(self, eta, f, label):
        max_wrong = th.max(eta, dim=-1, keepdim=True).values
        ind_wrong = eta == max_wrong
        ind_wrong[:, label] = False
        y = th.ones(eta.shape[0], dtype=th.long).cuda() * label
        # perturbed Vdot
        f_y = f[list(range(f.shape[0])), y]
        f_wrong = th.max(f.masked_fill(~ind_wrong, -float('inf')), dim=-1).values
        vdot = -f_y + f_wrong
        return vdot

    def perturb(self, eta, eps, label, f_lb, f_ub):
        # find the label and runner-up index for each sampled eta
        max_wrong = th.max(eta, dim=-1, keepdim=True).values
        # when perturb eta, runner up eta may change
        ind_wrong = eta >= (max_wrong - 2*eps)
        ind_wrong[:, label] = False
        y = th.ones(eta.shape[0], dtype=th.long).cuda() * label
        # perturbed Vdot
        f_y = f_lb[list(range(f_lb.shape[0])), y]
        f_wrong = th.max(f_ub.masked_fill(~ind_wrong, -float('inf')), dim=-1).values
        worst_vdot = -f_y + f_wrong
        return worst_vdot

