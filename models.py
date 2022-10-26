import torch
import torch as th
import torch.nn.functional as F
import torchvision.models
from torch import nn as nn
from torchdiffeq import odeint, odeint_adjoint

from dynamics.output_coordinates import DefaultOutputFun
from dynamics.classification import LipsConv, LipsLinear
from math import log, sqrt

import libs.ortho_conv.models_ortho as ortho_models
import libs.ortho_conv.models_ortho_test as ortho_models_test
import libs.ortho_conv.layers as layers


class Normalize(nn.Module):
    def __init__(self, mu, std):
        super(Normalize, self).__init__()
        self.register_buffer("mu", th.Tensor(mu).view(-1,1,1))
        self.register_buffer("std", th.Tensor(std).view(-1,1,1))

    def forward(self, x):
        if self.std is not None:
            return (x - self.mu) / self.std
        return (x - self.mu)


def make_ortho_KWLarge_Concat(n_in_channels, n_outputs, mu=None, std=None, out_dim=128, act='GroupSort'):
    norm_layer = Normalize(mu, std)
    model = nn.Sequential(
        norm_layer,
        ortho_models.KWLarge_Concat(out_dim=out_dim, act=act)
    )
    return model


def make_ortho_KWLargeMNIST_Concat(n_in_channels, n_outputs, mu=None, std=None, out_dim=128, act='GroupSort'):
    norm_layer = Normalize(mu, std)
    model = nn.Sequential(
        norm_layer,
        ortho_models.KWLargeMNIST_Concat(out_dim=out_dim, act=act)
    )
    return model


def make_4C3F(mu=None, std=None, out_dim=10, act='GroupSort'):
    norm_layer = Normalize(mu, std)
    if act == 'GroupSort':
        act_fun = layers.GroupSort()
    elif act == 'ReLU':
        act_fun = nn.ReLU()
    elif act == 'GroupSortTest':
        act_fun = layers.GroupSortTest()
    actual_model = nn.Sequential(
        LipsConv(3, 32, 3, stride=1, padding=1),
        act_fun,
        LipsConv(32, 32, 4, stride=2, padding=1),
        act_fun,
        LipsConv(32, 64, 3, stride=1, padding=1),
        act_fun,
        LipsConv(64, 64, 4, stride=2, padding=1),
        act_fun,
        nn.Flatten(),
        LipsLinear(64*8*8, 512),
        act_fun,
        LipsLinear(512, 512),
        act_fun,
        LipsLinear(512, 10)
    )
    for m in actual_model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, sqrt(2. / n))
            m.bias.data.zero_()

    model = nn.Sequential(
        norm_layer,
        actual_model
    )
    return model


def make_4C3F_nolips(mu=None, std=None, out_dim=10, act='GroupSort'):
    norm_layer = Normalize(mu, std)
    if act == 'GroupSort':
        act_fun = layers.GroupSort()
    elif act == 'ReLU':
        act_fun = nn.ReLU()
    elif act == 'GroupSortTest':
        act_fun = layers.GroupSortTest()
    actual_model = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        act_fun,
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        act_fun,
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        act_fun,
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        act_fun,
        nn.Flatten(),
        nn.Linear(64*8*8, 512),
        act_fun,
        nn.Linear(512, 512),
        act_fun,
        nn.Linear(512, 10)
    )
    for m in actual_model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, sqrt(2. / n))
            m.bias.data.zero_()

    model = nn.Sequential(
        norm_layer,
        actual_model
    )
    return model


def make_6C2F(mu=None, std=None):
    norm_layer = Normalize(mu, std)
    actual_model = nn.Sequential(
        LipsConv(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        LipsConv(32, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        LipsConv(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        LipsConv(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        LipsConv(64, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        LipsConv(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        LipsLinear(4096, 512),
        nn.ReLU(),
        LipsLinear(512, 10)
    )
    for m in actual_model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, sqrt(2. / n))
            m.bias.data.zero_()

    model = nn.Sequential(
        norm_layer,
        actual_model
    )
    return model


def make_ortho_KWLarge_Concat_test(n_in_channels, n_outputs, mu=None, std=None, out_dim=128, act='GroupSort'):
    norm_layer = Normalize(mu, std)
    model = nn.Sequential(
        norm_layer,
        ortho_models_test.KWLarge_Concat(out_dim=out_dim, act=act)
    )
    return model


def make_ortho_KWLargeMNIST_Concat_test(n_in_channels, n_outputs, mu=None, std=None, out_dim=128, act='GroupSort'):
    norm_layer = Normalize(mu, std)
    model = nn.Sequential(
        norm_layer,
        ortho_models_test.KWLargeMNIST_Concat(out_dim=out_dim, act=act)
    )
    return model


def make_ortho_KWLarge_inter(n_in_channels, n_outputs, mu=None, std=None):
    norm_layer = Normalize(mu, std)
    model = nn.Sequential(
        norm_layer,
        ortho_models.KWLarge_inter()
    )
    return model


class IVP(nn.Module):
    def __init__(self,
                 n_input,
                 n_output,
                 dyn_fun,
                 init_coordinates,
                 output_fun=DefaultOutputFun(),
                 ode_tol=1e-2,
                 ts=th.linspace(0, 1, 200)):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.dyn_fun = dyn_fun
        self.ode_tol = ode_tol
        self.register_buffer('ts', ts)
        self.output_fun = output_fun
        self.init_coordinates = init_coordinates
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def h_dot(self, t, h):
        return self.dyn_fun.ode_forward(t, h)

    def forward(self, x, ts=None, int_params=None, use_adjoint=False, return_traj=False):
        solution = self.integrate(x, ts=ts, int_params=int_params,
                                  use_adjoint=use_adjoint)
        if return_traj:
            return self.output_fun(solution)
        else:
            return self.output_fun(solution)[-1]

    def integrate(self, x, ts=None, int_params=None, use_adjoint=False):
        if ts is None:
            ts = self.ts
        if int_params is None:
            int_params = dict(
                rtol=self.ode_tol,
                atol=self.ode_tol
            )
        static_state, state = self.init_coordinates(x, self.dyn_fun)
        self.dyn_fun.static_state = static_state
        if use_adjoint:
            ode_call = odeint_adjoint
            # if we are differentiating the model but not training,
            # and the inputs require gradient,
            # we are probably computing adversarial robustness compute gradients
            # w.r.t to the inputs rather than the parameters.
            if not self.training and torch.is_grad_enabled() and x.requires_grad:
                int_params['adjoint_params'] = (x,)
            else:
                int_params['adjoint_params'] = tuple(self.parameters())
            int_params["adjoint_options"] = dict(norm="seminorm")
            int_params["adjoint_atol"] = int_params["atol"]
            int_params["adjoint_rtol"] = int_params["rtol"]
        else:
            ode_call = odeint
        solution = ode_call(self.h_dot, state, ts,
                            **int_params,
                            # method='dopri8',
                            # method='rk4',
                            # options=dict(step_size=self.ode_tol, perturb=True)
                            )
        return solution

