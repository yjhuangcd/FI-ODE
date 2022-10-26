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


class SegwayBarrierModel(nn.Module):
    # Deprecated: used in segway3d
    def __init__(self, close_dyn, alpha_1, alpha_2):
        super().__init__()
        self.close_dyn = close_dyn
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2

    def forward(self, x):
        f = self.close_dyn(x)
        barrier_2 = f[:, 3:4] + (self.alpha_1 + self.alpha_2) * x[:, 3:4] - self.alpha_1 * self.alpha_2 * (np.pi/12 - x[:, 2:3])
        barrier_1 = -f[:, 3:4] - (self.alpha_1 + self.alpha_2) * x[:, 3:4] - self.alpha_1 * self.alpha_2 * (np.pi/12 + x[:, 2:3])
        # max(barrier_1, barrier_2)
        barrier = F.relu(barrier_1 - barrier_2) + barrier_2
        return barrier

    def forward_adv(self, x, yvar=None, batch=None):
        f = self.close_dyn(x)
        barrier_2 = f[:, 3:4] + (self.alpha_1 + self.alpha_2) * x[:, 3:4] - self.alpha_1 * self.alpha_2 * (np.pi/12 - x[:, 2:3])
        barrier_1 = -f[:, 3:4] - (self.alpha_1 + self.alpha_2) * x[:, 3:4] - self.alpha_1 * self.alpha_2 * (np.pi/12 + x[:, 2:3])
        # max(barrier_1, barrier_2)
        barrier = F.relu(barrier_1 - barrier_2) + barrier_2
        return barrier


class SegwayCompositeBarrierModel(nn.Module):
    def __init__(self, close_dyn, barriers):
        super().__init__()
        self.close_dyn = close_dyn
        self.barriers = barriers

    def forward(self, x):
        f = self.close_dyn(x)
        for i in range(len(self.barriers)):
            hi = self.barriers[i](f, x)
            if i == 0:
                comp_barrier = hi
            else:
                comp_barrier = th.min(comp_barrier, hi)
        return comp_barrier

    def forward_adv(self, x, yvar=None, batch=None):
        f = self.close_dyn(x)
        for i in range(len(self.barriers)):
            hi = self.barriers[i](f, x)
            if i == 0:
                comp_barrier = hi
            else:
                comp_barrier = th.min(comp_barrier, hi)
        return F.relu(-comp_barrier)


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


class BarrierExt(nn.Module):
    def __init__(self, alpha, alpha_ext, side='lb'):
        super().__init__()
        self.alpha = alpha
        self.alpha_ext = alpha_ext
        if side == 'lb':
            sign = -1.
        else:
            sign = 1.
        self.sign = sign

    def forward(self, f, x):
        term = self.alpha * self.alpha_ext * np.pi/12
        signed_term = - f[:, 2:3] - (self.alpha + self.alpha_ext) * x[:, 2:3] - self.alpha * self.alpha_ext * x[:, 0:1]
        barrier = self.sign * signed_term + term
        return barrier

    def h_dot(self, f, x):
        signed_term = - f[:, 2:3] - self.alpha * x[:, 2:3]
        return self.sign * signed_term


class BarrierPhiV(nn.Module):
    def __init__(self, alpha, alpha_ext, side='lb'):
        super().__init__()
        self.alpha = alpha
        self.alpha_ext = alpha_ext
        if side == 'lb':
            sign = -1.
        else:
            sign = 1.
        self.sign = sign

    def forward(self, f, x):
        term = self.alpha * self.alpha_ext * 3.
        signed_term = - x[:, 2:3] + self.alpha * f[:, 1:2] + self.alpha_ext * (- x[:, 0:1] + self.alpha * x[:, 1:2])
        barrier = self.sign * signed_term + term
        return barrier

    def h_dot(self, f, x):
        signed_term = - x[:, 2:3] + self.alpha * f[:, 1:2]
        return self.sign * signed_term


class BarrierPhiDotV(nn.Module):
    def __init__(self, alpha, alpha_ext, side='lb'):
        super().__init__()
        self.alpha = alpha
        self.alpha_ext = alpha_ext
        if side == 'lb':
            sign = -1.
        else:
            sign = 1.
        self.sign = sign

    def forward(self, f, x):
        term = self.alpha * self.alpha_ext * 2.25
        signed_term = - (f[:, 2:3] + self.alpha * f[:, 1:2] + self.alpha_ext * (x[:, 2:3] + self.alpha * x[:, 1:2]))
        barrier = self.sign * signed_term + term
        return barrier

    def h_dot(self, f, x):
        signed_term = - (f[:, 2:3] + self.alpha * f[:, 1:2])
        return self.sign * signed_term


class BarrierV(nn.Module):
    def __init__(self, alpha, alpha_ext, side='lb'):
        super().__init__()
        self.alpha = alpha
        self.alpha_ext = alpha_ext
        if side == 'lb':
            sign = -1.
        else:
            sign = 1.
        self.sign = sign

    def forward(self, f, x):
        term = self.alpha_ext * 2.5
        signed_term = - (f[:, 1:2] + self.alpha_ext * x[:, 1:2])
        barrier = self.sign * signed_term + term
        return barrier

    def h_dot(self, f, x):
        signed_term = - f[:, 1:2]
        return self.sign * signed_term


class BarrierMaskPair(nn.Module):
    def __init__(self, barrier, mask):
        super().__init__()
        self.barrier = barrier
        self.mask = mask

    def forward(self, x):
        return self.barrier(x)

    def mask(self, x):
        return self.mask(x)


class LyaQuadratic(nn.Module):
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

    def h_dot(self, eta, f):
        self.P_t.weight.data = self.P.weight.data.t()
        # vdot = th.sum(self.P_t(self.P(eta - self.goal)) * f, dim=1, keepdim=True)
        # goal is set to zeros by default, delete for now because of an autolirpa broadcast bug
        vdot = th.sum(self.P_t(self.P(eta)) * f, dim=1, keepdim=True)
        return vdot

