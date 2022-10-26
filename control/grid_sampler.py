import math
from math import sqrt, log, floor
import numpy as np
import torch
import torch as th
import torch.nn.functional as F
from torch import nn as nn


# def grid_uniform_2d(origin, width, height, r):
#     # sample grids uniformly in a square, with edge r between points
#     if th.is_tensor(origin):
#         origin = origin.numpy()
#     w_lower, w_upper = origin[0, 0] - width/2, origin[0, 0] + width/2
#     h_lower, h_upper = origin[0, 1] - height/2, origin[0, 1] + height/2
#     # grid_tmp = np.mgrid[w_lower:w_upper:r, h_lower:h_upper:r]
#     # grid_tmp = torch.from_numpy(grid_tmp)
#     # grid = grid_tmp.reshape(grid_tmp.shape[0], -1).t()
#     # X, Y coordinates for plotting
#     X = np.arange(w_lower, w_upper, r)
#     Y = np.arange(h_lower, h_upper, r)
#     xx, yy = np.meshgrid(X, Y)
#     grid = np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1)
#     grid = torch.from_numpy(grid)
#     return grid, xx, yy

def random_uniform(sizes, batch_size):
    # sample uniform random vectors in a cube around origin
    # sizes: magnitude of each dimension
    dist = th.distributions.Uniform(
        th.tensor(-1., device=sizes.device),
        th.tensor(1., device=sizes.device))
    etas = dist.sample((batch_size, len(sizes))) * sizes
    return etas

def random_uniform_extend(sizes, batch_size, alpha_1=1., margin=0.):
    # sample uniform random vectors in a cube around origin
    # phi and phi_dot in a band
    # sizes: magnitude of each dimension: th.tensor([5., 2*np.pi, np.pi/12])
    dist = th.distributions.Uniform(
        th.tensor(-1., device=sizes.device),
        th.tensor(1., device=sizes.device))
    tmp = dist.sample((batch_size, len(sizes))) * sizes
    phi = tmp[:, 2:3]
    ub = alpha_1 * (np.pi/12 - phi) + margin
    lb = -alpha_1 * (np.pi/12 + phi) - margin
    phi_dot = (ub - lb) * th.rand(batch_size, 1) + lb
    etas = th.concat((tmp, phi_dot), dim=1)
    return etas

def random_polytope(sizes, batch_size, alphas=[10., 0.1, 2.], margin=0.):
    # phi, v, phi_dot
    # sample phi uniformly
    # find phi_dot and v according to barrier functions
    # sizes: magnitude of each dimension: th.tensor([np.pi/12])
    dist = th.distributions.Uniform(
        th.tensor(-1., device=sizes.device),
        th.tensor(1., device=sizes.device))
    phi = dist.sample((batch_size, len(sizes))) * sizes
    ub = alphas[0] * (np.pi/12 - phi) + margin
    lb = -alphas[0] * (np.pi/12 + phi) - margin
    phi_dot = (ub - lb) * th.rand(batch_size, 1) + lb
    # lb_v = 1/alphas[1] * phi - 3.
    # ub_v = 1/alphas[1] * phi + 3.
    lb_v = th.max(1/alphas[1] * phi - 3., -1/alphas[2] * phi_dot - 2.25) + margin
    ub_v = th.min(1/alphas[1] * phi + 3., -1/alphas[2] * phi_dot + 2.25) + margin
    v = th.rand_like(phi) * (ub_v - lb_v) + lb_v
    etas = th.concat((phi, v, phi_dot), dim=1)
    return etas

def random_polytope_clipv(sizes, batch_size, alphas=[10., 0.1, 2.], margin=0.):
    # phi, v, phi_dot
    # sample phi uniformly
    # find phi_dot and v according to barrier functions
    # sizes: magnitude of each dimension: th.tensor([np.pi/12])
    dist = th.distributions.Uniform(
        th.tensor(-1., device=sizes.device),
        th.tensor(1., device=sizes.device))
    phi = dist.sample((batch_size, len(sizes))) * sizes
    ub = alphas[0] * (np.pi/12 - phi) + margin
    lb = -alphas[0] * (np.pi/12 + phi) - margin
    phi_dot = (ub - lb) * th.rand(batch_size, 1) + lb
    # lb_v = 1/alphas[1] * phi - 3.
    # ub_v = 1/alphas[1] * phi + 3.
    lb_v = th.clamp(th.max(1/alphas[1] * phi - 3., -1/alphas[2] * phi_dot - 2.25), min=-2.5-margin)
    ub_v = th.clamp(th.min(1/alphas[1] * phi + 3., -1/alphas[2] * phi_dot + 2.25), max=2.5+margin)
    v = th.rand_like(phi) * (ub_v - lb_v) + lb_v
    etas = th.concat((phi, v, phi_dot), dim=1)
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
    d0 = np.arange(-sizes[0], sizes[0], r[0])
    d1 = np.arange(-sizes[1], sizes[1], r[1])
    phi, v = np.meshgrid(d0, d1)
    grid = np.concatenate((phi.reshape(-1, 1), v.reshape(-1, 1)), axis=1)
    grid = torch.from_numpy(grid).float()
    return grid, phi, v

def grid_uniform_3d(sizes, r):
    # sample grids in 3d around origin
    d0 = np.arange(-sizes[0], sizes[0], r[0])
    d1 = np.arange(-sizes[1], sizes[1], r[1])
    d2 = np.arange(-sizes[2], sizes[2], r[2])
    phi, v, phi_dot = np.meshgrid(d0, d1, d2)
    grid = np.concatenate((phi.reshape(-1, 1), v.reshape(-1, 1), phi_dot.reshape(-1, 1)), axis=1)
    grid = torch.from_numpy(grid).float()
    return grid, phi, v, phi_dot

def grid_uniform_4d(sizes, r):
    # sample grids in 4d around origin
    d0 = np.arange(-sizes[0], sizes[0], r[0])
    d1 = np.arange(-sizes[1], sizes[1], r[1])
    d2 = np.arange(-sizes[2], sizes[2], r[2])
    d3 = np.arange(-sizes[3], sizes[3], r[3])
    v, theta_dot, phi, phi_dot = np.meshgrid(d0, d1, d2, d3)
    grid = np.concatenate((v.reshape(-1, 1), theta_dot.reshape(-1, 1), phi.reshape(-1, 1), phi_dot.reshape(-1, 1)), axis=1)
    grid = torch.from_numpy(grid)
    return grid, v, theta_dot, phi, phi_dot

def check_valid_range(lb, ub):
    if (lb.min()).item() <= (ub.max()).item():
        flag = False
    else:
        flag = True
    return flag

class SamplingPhiPhiDot(nn.Module):
    def __init__(self, alphas, rs, side):
        super().__init__()
        # alphas and rs are lists
        # return true_rs for certify interval
        # lb: A1
        self.alphas = alphas
        self.rs = rs
        if side == 'lb':
            self.sign = 1.
        else:
            self.sign = -1.

    def forward(self):
        phi = th.arange(-np.pi/12, np.pi/12, self.rs[0]).unsqueeze(dim=1)
        phi_dot = -self.alphas[0] * (phi + self.sign * np.pi/12)
        lb_v = th.clamp(th.max(1/self.alphas[1] * phi - 3., -1/self.alphas[2] * phi_dot - 2.25), min=-2.5)
        ub_v = th.clamp(th.min(1/self.alphas[1] * phi + 3., -1/self.alphas[2] * phi_dot + 2.25), max=2.5)
        v = th.arange(lb_v.min(), ub_v.max(), self.rs[1]).unsqueeze(dim=1)
        num_h = phi.shape[0]
        num_v = v.shape[0]
        phi_re = phi.repeat_interleave(num_v, dim=0)
        phi_dot_re = phi_dot.repeat_interleave(num_v, dim=0)
        v_re = v.repeat(num_h, 1)
        grid = th.concat((phi_re, v_re, phi_dot_re), dim=1)
        mask = (grid[:, 0:1] >= self.alphas[1]*(grid[:, 1:2]-3.)) & (grid[:, 0:1] <= self.alphas[1]*(grid[:, 1:2]+3.)) & \
               (grid[:, 2:3] >= -self.alphas[2]*(grid[:, 1:2]+2.25)) & (grid[:, 2:3] <= -self.alphas[2]*(grid[:, 1:2]-2.25))
        true_grid = grid[mask.squeeze()]
        r_phi_dot = self.alphas[0] * self.rs[0]
        true_rs = [self.rs[0], self.rs[1], r_phi_dot]
        return true_grid, true_rs

class SamplingPhiV(nn.Module):
    def __init__(self, alphas, rs, side):
        super().__init__()
        # alphas and rs are lists
        # return true_rs for certify interval
        # lb: B1
        self.alphas = alphas
        self.rs = rs
        if side == 'lb':
            self.sign = 1.
        else:
            self.sign = -1.

    def forward(self):
        phi = th.arange(-np.pi/12, np.pi/12, self.rs[0]).unsqueeze(dim=1)
        v = 1/self.alphas[1] * phi + self.sign * 3.
        lb_phi_dot = th.max(-self.alphas[0]*(phi+np.pi/12), -self.alphas[2]*(v+2.25))
        ub_phi_dot = th.min(-self.alphas[0]*(phi-np.pi/12), -self.alphas[2]*(v-2.25))
        flag = check_valid_range(lb_phi_dot, ub_phi_dot)
        if flag:
            return None, None
        else:
            phi_dot = th.arange(lb_phi_dot.min(), ub_phi_dot.max(), self.rs[2]).unsqueeze(dim=1)
            num_h = phi.shape[0]
            num_v = phi_dot.shape[0]
            phi_re = phi.repeat_interleave(num_v, dim=0)
            v_re = v.repeat_interleave(num_v, dim=0)
            phi_dot_re = phi_dot.repeat(num_h, 1)
            grid = th.concat((phi_re, v_re, phi_dot_re), dim=1)
            mask = (grid[:, 2:3] >= -self.alphas[0]*(grid[:, 0:1]+np.pi/12)) & \
                   (grid[:, 2:3] <= -self.alphas[0]*(grid[:, 0:1]-np.pi/12)) & \
                   (grid[:, 2:3] >= -self.alphas[2]*(grid[:, 1:2]+2.25)) & (grid[:, 2:3] <= -self.alphas[2]*(grid[:, 1:2]-2.25)) & \
                   (grid[:, 1:2] >= -th.ones_like(grid[:, 1:2])*2.5) & (grid[:, 1:2] <= th.ones_like(grid[:, 1:2])*2.5)
            true_grid = grid[mask.squeeze()]
            r_v = 1/self.alphas[1] * self.rs[0]
            true_rs = [self.rs[0], r_v, self.rs[2]]
            return true_grid, true_rs

class SamplingPhiDotV(nn.Module):
    def __init__(self, alphas, rs, side):
        super().__init__()
        # alphas and rs are lists
        # return true_rs for certify interval
        # lb: C1
        self.alphas = alphas
        self.rs = rs
        if side == 'lb':
            self.sign = -1.
        else:
            self.sign = 1.

    def forward(self):
        phi_dot = th.arange(-self.alphas[0]*np.pi/12*2, self.alphas[0]*np.pi/12*2, self.rs[2]).unsqueeze(dim=1)
        v = -1/self.alphas[2] * phi_dot + self.sign * 2.25
        lb_phi = th.clamp(th.max(-1/self.alphas[0]*phi_dot-np.pi/12, self.alphas[1]*(v-3)), min=-np.pi/12)
        ub_phi = th.clamp(th.min(-1/self.alphas[0]*phi_dot+np.pi/12, self.alphas[1]*(v+3)), max=np.pi/12)
        phi = th.arange(lb_phi.min(), ub_phi.max(), self.rs[0]).unsqueeze(dim=1)
        num_h = phi_dot.shape[0]
        num_v = phi.shape[0]
        phi_dot_re = phi_dot.repeat_interleave(num_v, dim=0)
        v_re = v.repeat_interleave(num_v, dim=0)
        phi_re = phi.repeat(num_h, 1)
        grid = th.concat((phi_re, v_re, phi_dot_re), dim=1)
        mask = (grid[:, 2:3] >= -self.alphas[0]*(grid[:, 0:1]+np.pi/12)) & \
               (grid[:, 2:3] <= -self.alphas[0]*(grid[:, 0:1]-np.pi/12)) & \
               (grid[:, 0:1] >= self.alphas[1]*(grid[:, 1:2]-3.)) & (grid[:, 0:1] <= self.alphas[1]*(grid[:, 1:2]+3.)) & \
               (grid[:, 1:2] >= -th.ones_like(grid[:, 1:2])*2.5) & (grid[:, 1:2] <= th.ones_like(grid[:, 1:2])*2.5)
        true_grid = grid[mask.squeeze()]
        r_v = 1/self.alphas[1] * self.rs[0]
        true_rs = [self.rs[0], r_v, self.rs[2]]
        return true_grid, true_rs

class SamplingV(nn.Module):
    def __init__(self, alphas, rs, side):
        super().__init__()
        # alphas and rs are lists
        # return true_rs for certify interval
        # lb: D1
        self.alphas = alphas
        self.rs = rs
        if side == 'lb':
            self.sign = -1.
        else:
            self.sign = 1.

    def forward(self):
        phi = th.arange(-np.pi/12, np.pi/12, self.rs[0]).unsqueeze(dim=1)
        phi_dot = th.arange(-self.alphas[0]*np.pi/12*2, self.alphas[0]*np.pi/12*2, self.rs[2]).unsqueeze(dim=1)
        v = th.ones_like(phi) * self.sign * 2.5
        num_h = v.shape[0]
        num_v = phi_dot.shape[0]
        phi_re = phi.repeat_interleave(num_v, dim=0)
        v_re = v.repeat_interleave(num_v, dim=0)
        phi_dot_re = phi_dot.repeat(num_h, 1)
        grid = th.concat((phi_re, v_re, phi_dot_re), dim=1)
        mask = (grid[:, 2:3] >= -self.alphas[0]*(grid[:, 0:1]+np.pi/12)) & \
               (grid[:, 2:3] <= -self.alphas[0]*(grid[:, 0:1]-np.pi/12)) & \
               (grid[:, 0:1] >= self.alphas[1]*(grid[:, 1:2]-3.)) & (grid[:, 0:1] <= self.alphas[1]*(grid[:, 1:2]+3.)) & \
               (grid[:, 2:3] >= -self.alphas[2]*(grid[:, 1:2]+2.25)) & (grid[:, 2:3] <= -self.alphas[2]*(grid[:, 1:2]-2.25))
        true_grid = grid[mask.squeeze()]
        r_v = 0.
        true_rs = [self.rs[0], r_v, self.rs[2]]
        return true_grid, true_rs
