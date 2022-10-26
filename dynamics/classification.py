import math
from math import sqrt, log
import numpy as np
import torch
import torch as th
import torch.nn.functional as F
from torch import nn as nn
from libs.ortho_conv.layers import *
# import cuosqp as osqp
# import cvxpy as cp
# from cvxpylayers.torch import CvxpyLayer
# from qpth.qp import QPFunction
from torch.autograd.functional import jvp, jacobian
# from functorch import jacrev, vmap
from barrier_projection.barrier_projection import FastBarrierProjection, FastBarrierProjectionNoUpper


# Layers with singular vectors to compute Lips
class LipsConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('singular_u', None)


class LipsLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('singular_u', None)


class OrthoClassDynProjectSimplexLips(nn.Module):
    def __init__(self, n_hidden=10,
                 activation='ReLU',
                 dropout=0.5,
                 mlp_size=128,
                 kappa=5.,
                 kappa_length=3e+4,
                 alpha_1=100.,
                 alpha_2=5.,
                 sigma_1=0.02, scale_nominal=False, x_dim=10,
                 cayley=True):
        super().__init__()

        if activation != 'GroupSort':
            act_maker = getattr(nn, activation)
            self.activation = act_maker()
        elif activation == 'GroupSort':
            self.activation = GroupSort()
        self.dropout = nn.Dropout(dropout)
        self.mlp_size = mlp_size
        self.n_hidden = n_hidden
        self.kappa = kappa
        self.kappa_length = kappa_length
        # For crown evaluation
        self.register_buffer("static_state", None)
        self.apply(self._init_parameters)

        # barrier conditions for safety inside simplex
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.sigma_1 = sigma_1

        self.fast_qp_solver = FastBarrierProjectionNoUpper(max_iter=30, tol=1e-4,
                                                           verbose=False)
        self.scale_nominal = scale_nominal
        self.cayley = cayley

        if self.cayley:
            self.hidden_to_mlp = CayleyLinear(in_features=n_hidden,
                                              out_features=mlp_size, bias=True)
            self.mlp_to_mlp = CayleyLinear(in_features=mlp_size,
                                           out_features=mlp_size)
            self.mlp_to_hidden = CayleyLinear(in_features=mlp_size,
                                              out_features=n_hidden)
            self.U_x = CayleyLinear(in_features=x_dim, out_features=mlp_size)
        else:
            self.hidden_to_mlp = LipsLinear(in_features=n_hidden,
                                            out_features=mlp_size, bias=True)
            self.mlp_to_mlp = LipsLinear(in_features=mlp_size,
                                         out_features=mlp_size)
            self.mlp_to_hidden = LipsLinear(in_features=mlp_size,
                                            out_features=n_hidden)
            self.U_x = LipsLinear(in_features=x_dim, out_features=mlp_size)

    def _init_parameters(self, m):
        with th.no_grad():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _h_dot_raw(self, h, x):
        h_dot = self.hidden_to_mlp(h) + self.U_x(x)
        h_dot = self.activation(self.dropout(h_dot))
        h_dot = self.mlp_to_mlp(h_dot)
        h_dot = self.activation(self.dropout(h_dot))
        h_dot = self.mlp_to_hidden(h_dot)
        return h_dot

    def eval_dot(self, t, h_tuple, x):
        h = h_tuple[0]
        f_tilde = self._h_dot_raw(h, x)
        # lower = -self.alpha_1 * h
        lower = -self.alpha_1*(torch.exp(self.sigma_1*h)-1)
        upper = self.alpha_2 * (1 - h)
        if self.scale_nominal:
            # f_tilde = (upper - lower) * th.sigmoid(f_tilde)
            f_tilde = (upper - lower) * th.sigmoid(f_tilde) + lower
        # f = self.fast_qp_solver(lower, upper, f_tilde)
        f = self.fast_qp_solver(lower, f_tilde)
        return f

    def eval_dot_light(self, h, x):
        f_tilde = self._h_dot_raw(h, x)
        # lower = -self.alpha_1 * h
        lower = -self.alpha_1*(torch.exp(self.sigma_1*h)-1)
        upper = self.alpha_2 * (1 - h)
        if self.scale_nominal:
            f_tilde = (upper - lower) * th.sigmoid(f_tilde) + lower
        # f = self.fast_qp_solver(lower, upper, f_tilde)
        f = self.fast_qp_solver(lower, f_tilde)
        return f

    def ode_forward(self, t, h_tuple):
        assert self.static_state is not None, "[ERROR] You forgot to set " \
                                              "static state before calling " \
                                              "forward."
        return self.eval_dot(t, h_tuple, self.static_state)


class CrownOrthoClassDynProjectSimplexLips(OrthoClassDynProjectSimplexLips):
    def __init__(self, n_hidden=10,
                 activation='ReLU',
                 dropout=0.5,
                 mlp_size=128,
                 kappa=5.,
                 kappa_length=3e+4,
                 alpha_1=5.,
                 alpha_2=5.,
                 sigma_1=0.02, scale_nominal=False, x_dim=10, cayley=True):
        super().__init__(n_hidden, activation, dropout, mlp_size, kappa, kappa_length, alpha_1, alpha_2, sigma_1,
                         scale_nominal, x_dim, cayley)

        self.hidden_to_mlp = nn.Linear(in_features=n_hidden,
                                       out_features=mlp_size)
        self.mlp_to_mlp = nn.Linear(in_features=mlp_size,
                                    out_features=mlp_size)
        self.mlp_to_hidden = nn.Linear(in_features=mlp_size,
                                       out_features=n_hidden)
        self.U_x = nn.Linear(in_features=x_dim, out_features=mlp_size)

    def _h_dot_raw(self, h, x):
        h_dot = self.hidden_to_mlp(h) + self.U_x(x)
        h_dot = self.activation(self.dropout(h_dot))
        h_dot = self.mlp_to_mlp(h_dot)
        h_dot = self.activation(self.dropout(h_dot))
        h_dot = self.mlp_to_hidden(h_dot)
        return h_dot

    def eval_dot_light(self, h, x):
        f_tilde = self._h_dot_raw(h, x)
        # lower = -self.alpha_1 * h
        lower = -self.alpha_1*(torch.exp(self.sigma_1*h)-1)
        upper = self.alpha_2 * (1 - h)
        if self.scale_nominal:
            f_tilde = (upper - lower) * th.sigmoid(f_tilde) + lower
        # f = self.fast_qp_solver(lower, upper, f_tilde)
        f = self.fast_qp_solver(lower, f_tilde)
        return f

    def ibp_sigmoid(self, f_lb, f_ub, h_lb, h_ub):
        # upper - lower is monotonically decreasing
        lower_lb = -self.alpha_1*(torch.exp(self.sigma_1*h_ub)-1)
        lower_ub = -self.alpha_1*(torch.exp(self.sigma_1*h_lb)-1)
        out_f_lb = (self.alpha_2 * (1 - h_ub) - lower_lb) * th.sigmoid(f_lb) + lower_lb
        out_f_ub = (self.alpha_2 * (1 - h_lb) - lower_ub) * th.sigmoid(f_ub) + lower_ub
        return out_f_lb, out_f_ub

    def ibp_cbf_qp_individual(self, h, eps, lb, ub):
        # eps: size of the super box on sampled grids
        h_lower = h.repeat(self.n_hidden, 1) - eps
        h_upper = h.repeat(self.n_hidden, 1) + eps
        h_lower_diag = th.diag(h.squeeze() - eps)
        h_upper_diag = th.diag(h.squeeze() + eps)
        lower_lb = -self.alpha_1 * (h_lower - h_lower_diag + h_upper_diag)
        lower_ub = -self.alpha_1 * (h_upper - h_upper_diag + h_lower_diag)
        upper_lb = self.alpha_2 * (1 - (h_lower - h_lower_diag + h_upper_diag))
        upper_ub = self.alpha_2 * (1 - (h_upper - h_upper_diag + h_lower_diag))
        f_lb = th.zeros_like(h)
        f_ub = th.zeros_like(h)
        f_lb_rep = lb.repeat(self.n_hidden, 1)
        f_ub_rep = ub.repeat(self.n_hidden, 1)
        lb_diag = th.diag(lb.squeeze())
        ub_diag = th.diag(ub.squeeze())
        f_tilde_lb = f_ub_rep - ub_diag + lb_diag
        f_tilde_ub = f_lb_rep - lb_diag + ub_diag
        for i in range(self.n_hidden):
            f_lb_tmp = self.fast_qp_solver(lower_lb[i][None], upper_lb[i][None], f_tilde_lb[i][None])
            f_ub_tmp = self.fast_qp_solver(lower_ub[i][None], upper_ub[i][None], f_tilde_ub[i][None])
            f_lb[0][i] = f_lb_tmp[0][i]
            f_ub[0][i] = f_ub_tmp[0][i]
        return f_lb, f_ub

    def ibp_cbf_qp(self, h, eps, lb, ub, upper=False):
        # eps: size of the super box on sampled grids
        h_lower = h.repeat_interleave(self.n_hidden, dim=0) - eps
        h_upper = h.repeat_interleave(self.n_hidden, dim=0) + eps
        row_ind = list(range(h_lower.shape[0]))
        diag_ind = list(range(self.n_hidden))*h.shape[0]
        upper_diag = h_upper[row_ind, diag_ind]
        lower_diag = h_lower[row_ind, diag_ind]
        h_lower[row_ind, diag_ind] = upper_diag
        h_upper[row_ind, diag_ind] = lower_diag
        if upper:
            lower_lb = -self.alpha_1 * h_lower
            lower_ub = -self.alpha_1 * h_upper
            upper_lb = self.alpha_2 * (1 - h_lower)
            upper_ub = self.alpha_2 * (1 - h_upper)
        else:
            lower_lb = -self.alpha_1*(torch.exp(self.sigma_1 * h_lower)-1)
            lower_ub = -self.alpha_1*(torch.exp(self.sigma_1 * h_upper)-1)
        f_tilde_lb = ub.repeat_interleave(self.n_hidden, dim=0)
        f_tilde_ub = lb.repeat_interleave(self.n_hidden, dim=0)
        ub_diag = f_tilde_lb[row_ind, diag_ind]
        lb_diag = f_tilde_ub[row_ind, diag_ind]
        f_tilde_lb[row_ind, diag_ind] = lb_diag
        f_tilde_ub[row_ind, diag_ind] = ub_diag
        if upper:
            f_lb_tmp = self.fast_qp_solver(lower_lb, upper_lb, f_tilde_lb)
            f_ub_tmp = self.fast_qp_solver(lower_ub, upper_ub, f_tilde_ub)
        else:
            f_lb_tmp = self.fast_qp_solver(lower_lb, f_tilde_lb)
            f_ub_tmp = self.fast_qp_solver(lower_ub, f_tilde_ub)
        f_lb_flatten = f_lb_tmp[row_ind, diag_ind]
        f_ub_flatten = f_ub_tmp[row_ind, diag_ind]
        f_lb = f_lb_flatten.reshape(h.shape[0], -1)
        f_ub = f_ub_flatten.reshape(h.shape[0], -1)
        return f_lb, f_ub

    def ibp_cbf_qp_band(self, h_lb, h_ub, lb, ub, upper=False):
        # eps: size of the super box on sampled grids
        h_lower = h_lb.repeat_interleave(self.n_hidden, dim=0)
        h_upper = h_ub.repeat_interleave(self.n_hidden, dim=0)
        row_ind = list(range(h_lower.shape[0]))
        diag_ind = list(range(self.n_hidden))*h_lb.shape[0]
        upper_diag = h_upper[row_ind, diag_ind]
        lower_diag = h_lower[row_ind, diag_ind]
        h_lower[row_ind, diag_ind] = upper_diag
        h_upper[row_ind, diag_ind] = lower_diag
        if upper:
            lower_lb = -self.alpha_1 * h_lower
            lower_ub = -self.alpha_1 * h_upper
            upper_lb = self.alpha_2 * (1 - h_lower)
            upper_ub = self.alpha_2 * (1 - h_upper)
        else:
            lower_lb = -self.alpha_1*(torch.exp(self.sigma_1 * h_lower)-1)
            lower_ub = -self.alpha_1*(torch.exp(self.sigma_1 * h_upper)-1)
        f_tilde_lb = ub.repeat_interleave(self.n_hidden, dim=0)
        f_tilde_ub = lb.repeat_interleave(self.n_hidden, dim=0)
        ub_diag = f_tilde_lb[row_ind, diag_ind]
        lb_diag = f_tilde_ub[row_ind, diag_ind]
        f_tilde_lb[row_ind, diag_ind] = lb_diag
        f_tilde_ub[row_ind, diag_ind] = ub_diag
        if upper:
            f_lb_tmp = self.fast_qp_solver(lower_lb, upper_lb, f_tilde_lb)
            f_ub_tmp = self.fast_qp_solver(lower_ub, upper_ub, f_tilde_ub)
        else:
            f_lb_tmp = self.fast_qp_solver(lower_lb, f_tilde_lb)
            f_ub_tmp = self.fast_qp_solver(lower_ub, f_tilde_ub)
        f_lb_flatten = f_lb_tmp[row_ind, diag_ind]
        f_ub_flatten = f_ub_tmp[row_ind, diag_ind]
        f_lb = f_lb_flatten.reshape(h_lb.shape[0], -1)
        f_ub = f_ub_flatten.reshape(h_lb.shape[0], -1)
        return f_lb, f_ub

    # convert cayley linear to standard linear layer for Crown
    def convert_cayley(self, cayleymodel):
        self.hidden_to_mlp.weight.data = cayley(
            cayleymodel.hidden_to_mlp.alpha * cayleymodel.hidden_to_mlp.weight / cayleymodel.hidden_to_mlp.weight.norm()).detach()
        self.hidden_to_mlp.bias = cayleymodel.hidden_to_mlp.bias
        self.mlp_to_mlp.weight.data = cayley(
            cayleymodel.mlp_to_mlp.alpha * cayleymodel.mlp_to_mlp.weight / cayleymodel.mlp_to_mlp.weight.norm()).detach()
        self.mlp_to_mlp.bias = cayleymodel.mlp_to_mlp.bias
        self.mlp_to_hidden.weight.data = cayley(
            cayleymodel.mlp_to_hidden.alpha * cayleymodel.mlp_to_hidden.weight / cayleymodel.mlp_to_hidden.weight.norm()).detach()
        self.mlp_to_hidden.bias = cayleymodel.mlp_to_hidden.bias
        self.U_x.weight.data = cayley(
            cayleymodel.U_x.alpha * cayleymodel.U_x.weight / cayleymodel.U_x.weight.norm()).detach()
        self.U_x.bias = cayleymodel.U_x.bias
        return

    # # For Crown evaluation
    # def forward(self, h):
    #     h_dot = self.hidden_to_mlp(h) + self.U_x(self.static_state)
    #     h_dot = self.activation(h_dot)
    #     h_dot = self.mlp_to_mlp(h_dot)
    #     h_dot = self.activation(h_dot)
    #     h_dot = self.mlp_to_hidden(h_dot)
    #     return h_dot

    # Forward with two inputs, latest
    def forward(self, h, x):
        h_dot = self.hidden_to_mlp(h) + self.U_x(x)
        h_dot = self.activation(h_dot)
        h_dot = self.mlp_to_mlp(h_dot)
        h_dot = self.activation(h_dot)
        h_dot = self.mlp_to_hidden(h_dot)
        return h_dot
