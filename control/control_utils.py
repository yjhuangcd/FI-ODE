import math
from math import sqrt, log, floor
import numpy as np
import torch
import torch as th
import torch.nn.functional as F
from torch import nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm


def Plot3D(X, Y, V):
    # Plot Lyapunov functions
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X,Y,V, rstride=5, cstride=5, alpha=0.5, cmap=cm.coolwarm)
    ax.contour(X,Y,V,10, zdir='z', offset=0, cmap=cm.coolwarm)
    return ax

class AutoLirpaModel(nn.Module):
    def __init__(self, dyn, ctrl):
        super().__init__()
        self.dyn = dyn
        self.ctrl = ctrl

    def forward(self, x):
        return self.dyn(x, self.ctrl(x, 0.0), 0.0)


class AutoLirpaModel_params(nn.Module):
    def __init__(self, dyn, ctrl, num_states):
        super().__init__()
        self.dyn = dyn
        self.ctrl = ctrl
        self.num_states = num_states

    def forward(self, x):
        return self.dyn(x, self.ctrl(x[:, :self.num_states], 0.0), 0.0)


class AutoLirpaModelTimeVarying(nn.Module):
    def __init__(self, dyn, ctrl, num_states):
        super().__init__()
        self.dyn = dyn
        self.ctrl = ctrl
        self.num_states = num_states

    def forward(self, x):
        eta = x[:, :self.num_states]
        t = x[:, self.num_states:self.num_states+1]
        return self.dyn(eta, self.ctrl(eta, t), t)


class QuadraticVdotModel(nn.Module):
    def __init__(self, close_dyn, P):
        super().__init__()
        self.close_dyn = close_dyn
        self.P = nn.Linear(P.shape[1], P.shape[0], bias=False)
        self.P.weight.data = P

    def forward(self, x):
        f = self.close_dyn(x)
        P_x = self.P(x)
        out = (f * P_x).sum(dim=1, keepdim=True)
        return out


class SegwaySingleBarrierModel(nn.Module):
    def __init__(self, close_dyn, barrier):
        super().__init__()
        self.close_dyn = close_dyn
        self.barrier = barrier

    def forward(self, x):
        f = self.close_dyn(x)
        h = self.barrier.h_dot(f, x)
        return h

    def forward_adv(self, x, yvar=None, batch=None):
        f = self.close_dyn(x)
        h = self.barrier.h_dot(f, x)
        return h


class SegwaySingleBarrierModel_params(nn.Module):
    def __init__(self, close_dyn, barrier, num_states):
        super().__init__()
        self.close_dyn = close_dyn
        self.barrier = barrier
        self.num_states = num_states

    def forward(self, x):
        f = self.close_dyn(x)
        h = self.barrier.h_dot(f, x[:, :self.num_states])
        return h

    def forward_adv(self, x, yvar=None, batch=None):
        f = self.close_dyn(x)
        h = self.barrier.h_dot(f, x[:, :self.num_states])
        return h


class LyaQuadratic(nn.Module):
    # Learnable
    def __init__(self, P, goal):
        super().__init__()
        dim = P.shape[1]
        self.P = nn.Linear(dim, dim, bias=False)
        self.P.weight.data = P
        self.goal = goal
        self.P_t = nn.Linear(dim, dim, bias=False)
        self.P_t.weight.data = self.P.weight.data.t()

    def forward(self, eta):
        v = th.sum(self.P(eta - self.goal) * self.P(eta - self.goal), dim=1, keepdim=True)
        return v

    def h_dot(self, f, eta):
        self.P_t.weight.data = self.P.weight.data.t()
        # vdot = th.sum(self.P_t(self.P(eta - self.goal)) * f, dim=1, keepdim=True)
        # goal is set to zeros by default, delete for now because of an autolirpa broadcast bug
        vdot = 2 * th.sum(self.P_t(self.P(f)) * eta, dim=1, keepdim=True)
        return vdot

