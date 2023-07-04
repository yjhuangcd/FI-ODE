import math
from math import sqrt, log, floor
import numpy as np
import torch
import torch as th
import torch.nn.functional as F
from torch import nn as nn

def random_uniform(sizes, batch_size):
    # sample uniform random vectors in a cube around origin
    # sizes: magnitude of each dimension
    dist = th.distributions.Uniform(
        th.tensor(-1., device=sizes.device),
        th.tensor(1., device=sizes.device))
    etas = dist.sample((batch_size, len(sizes))) * sizes
    return etas

def reject_sampling(x, lya, level_lb, level_ub, return_mask=False):
    # given x, reject x outside the level sets
    val = lya(x)
    mask = (val >= level_lb) & (val <= level_ub)
    if return_mask:
        return x[mask.squeeze()], mask.squeeze()
    else:
        return x[mask.squeeze()]

def grid_uniform_2d(sizes, r):
    # sample grids in 3d around origin
    d0 = np.arange(-sizes[0] + r[0].numpy()/2, sizes[0], r[0])
    d1 = np.arange(-sizes[1] + r[1].numpy()/2, sizes[1], r[1])
    phi, v = np.meshgrid(d0, d1)
    grid = np.concatenate((phi.reshape(-1, 1), v.reshape(-1, 1)), axis=1)
    grid = torch.from_numpy(grid).float()
    return grid, phi, v

def grid_uniform_3d(sizes, r):
    # sample grids in 3d around origin
    d0 = np.arange(-sizes[0] + r[0].numpy()/2, sizes[0], r[0])
    d1 = np.arange(-sizes[1] + r[1].numpy()/2, sizes[1], r[1])
    d2 = np.arange(-sizes[2] + r[1].numpy()/2, sizes[2], r[2])
    phi, v, phi_dot = np.meshgrid(d0, d1, d2)
    grid = np.concatenate((phi.reshape(-1, 1), v.reshape(-1, 1), phi_dot.reshape(-1, 1)), axis=1)
    grid = torch.from_numpy(grid).float()
    return grid, phi, v, phi_dot

def grid_uniform_4d(sizes, r):
    if sizes.device.type != 'cpu':
        sizes = sizes.cpu()
    if r.device.type != 'cpu':
        r = r.cpu()
    # sample grids in 4d around origin
    d0 = np.arange(-sizes[0] + r[0].numpy()/2, sizes[0], r[0])
    d1 = np.arange(-sizes[1] + r[1].numpy()/2, sizes[1], r[1])
    d2 = np.arange(-sizes[2] + r[2].numpy()/2, sizes[2], r[2])
    d3 = np.arange(-sizes[3] + r[3].numpy()/2, sizes[3], r[3])
    v, theta_dot, phi, phi_dot = np.meshgrid(d0, d1, d2, d3)
    grid = np.concatenate((v.reshape(-1, 1), theta_dot.reshape(-1, 1), phi.reshape(-1, 1), phi_dot.reshape(-1, 1)), axis=1)
    grid = torch.from_numpy(grid).float()
    return grid, v, theta_dot, phi, phi_dot
