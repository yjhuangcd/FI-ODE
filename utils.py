from omegaconf import OmegaConf
import hydra
import hydra.utils

import plotly.express as px
import plotly.io as pio
import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F

pio.renderers.default = 'browser'

def hydra_conf_load_from_checkpoint(chkpt_file, cfg):
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
        **OmegaConf.masked_copy(cfg, cfg_mask),
        **instance_args
    )

simplex_transform = [[0, 1/2, 1], [0, np.sqrt(3)/2, 0]]

def plot_samples_on_3_simplex(h_sample, fig=None):
    transform = th.tensor([[0, 1 / 2, 1], [0, np.sqrt(3) / 2, 0]],
                          device=h_sample.device, dtype=h_sample.dtype)
    edge_points = (th.eye(3, device=h_sample.device,
                          dtype=h_sample.dtype) @ transform.T).detach().cpu().numpy()
    projected_samples = (h_sample @ transform.T).detach().cpu().numpy()
    if fig is None:
        fig = px.scatter(x=projected_samples[:, 0], y=projected_samples[:, 1])
    else:
        fig.add_scatter(x=projected_samples[:, 0], y=projected_samples[:, 1])
    fig.add_shape(dict(type='line',
                       x0=edge_points[0, 0], y0=edge_points[0, 1],
                       x1=edge_points[1, 0], y1=edge_points[1, 1],
                       line=dict(color='red', width=1))
                  )
    fig.add_shape(dict(type='line',
                       x0=edge_points[2, 0], y0=edge_points[2, 1],
                       x1=edge_points[1, 0], y1=edge_points[1, 1],
                       line=dict(color='red', width=1))
                  )
    fig.add_shape(dict(type='line',
                       x0=edge_points[0, 0], y0=edge_points[0, 1],
                       x1=edge_points[2, 0], y1=edge_points[2, 1],
                       line=dict(color='red', width=1))
                  )
    return fig

def plot_labeled_samples_on_simplex(y, h_sample, fig=None):
    if h_sample.ndim == 3:
        y = y[:, None].expand(-1, h_sample.shape[1])
        h_sample = h_sample.flatten(0,1)
        y = y.flatten(0,1)
    transform = th.tensor([[0, 1 / 2, 1], [0, np.sqrt(3) / 2, 0]],
                          device=h_sample.device, dtype=h_sample.dtype)
    edge_points = (th.eye(3, device=h_sample.device,
                          dtype=h_sample.dtype) @ transform.T)
    projected_samples = (h_sample @ transform.T)
    projected_samples_y = th.cat(
        [projected_samples,
        y[:, None]], dim=-1)
    coord = ['x', 'y', 'label']
    batch_idx = np.arange(0, y.shape[0])
    index = pd.MultiIndex.from_product([batch_idx, coord], names=['batch', 'coord'])
    df = pd.Series(data=projected_samples_y.flatten().cpu().detach().numpy(),
                   index=index).unstack(level=[1])
    if fig is None:
        fig = px.scatter(df, x='x', y='y', color='label')
    else:
        fig.add_scatter(df, x='x', y='y', color='label')
    fig.add_shape(dict(type='line',
                       x0=edge_points[0, 0], y0=edge_points[0, 1],
                       x1=edge_points[1, 0], y1=edge_points[1, 1],
                       line=dict(color='red', width=1))
                  )
    fig.add_shape(dict(type='line',
                       x0=edge_points[2, 0], y0=edge_points[2, 1],
                       x1=edge_points[1, 0], y1=edge_points[1, 1],
                       line=dict(color='red', width=1))
                  )
    fig.add_shape(dict(type='line',
                       x0=edge_points[0, 0], y0=edge_points[0, 1],
                       x1=edge_points[2, 0], y1=edge_points[2, 1],
                       line=dict(color='red', width=1))
                  )
    return fig

import pandas as pd
def plot_traj_on_3_simplex(traj,y, fig=None):
    transform = th.tensor([[0, 1 / 2, 1], [0, np.sqrt(3) / 2, 0]],
                          device=traj.device, dtype=traj.dtype)
    edge_points = (th.eye(3, device=traj.device,
                          dtype=traj.dtype) @ transform.T).detach().cpu().numpy()

    projected_samples = traj.permute(1, 0, 2) @ transform.T
    projected_samples_y = th.cat([projected_samples,
                                  y[:, None, None].expand(-1, projected_samples.shape[1], -1)], dim=-1)

    ts_idx = np.arange(projected_samples_y.shape[1])
    coord = ['x', 'y', 'label']
    batch_idx = np.arange(0, projected_samples_y.shape[0])
    index = pd.MultiIndex.from_product([batch_idx, ts_idx, coord], names=[
        'batch', 't', 'coord'])
    df = pd.Series(data=projected_samples_y.flatten().cpu().detach().numpy(),
                   index=index).unstack(level=2)
    df = df.reset_index(level=0)


    if fig is None:
        fig = px.line(data_frame=df, x='x', y='y', color='label',
                      line_group='batch')
    else:
        fig.add_line(data_frame=df, x='x', y='y', color='label',
                     line_group='batch')
    fig.add_shape(dict(type='line',
                       x0=edge_points[0, 0], y0=edge_points[0, 1],
                       x1=edge_points[1, 0], y1=edge_points[1, 1],
                       line=dict(color='red', width=1))
                  )
    fig.add_shape(dict(type='line',
                       x0=edge_points[2, 0], y0=edge_points[2, 1],
                       x1=edge_points[1, 0], y1=edge_points[1, 1],
                       line=dict(color='red', width=1))
                  )
    fig.add_shape(dict(type='line',
                       x0=edge_points[0, 0], y0=edge_points[0, 1],
                       x1=edge_points[2, 0], y1=edge_points[2, 1],
                       line=dict(color='red', width=1))
                  )
    return fig


def _conv2d(x, w, b, stride=1, padding=0):
    return F.conv2d(x, w, bias=b, stride=stride, padding=padding)


def _conv_trans2d(x, w, stride=1, padding=0, output_padding=0):
    return F.conv_transpose2d(x, w,stride=stride, padding=padding, output_padding=output_padding)


def power_iteration_evl(A, u=None, num_iter=1):
    if u is None:
        u = th.randn((A.size()[0],1), device=A.device)

    B = A.t()
    for i in range(num_iter):
        u1 = B.mm(u)
        u1_norm = u1.norm(2)
        v = u1 / u1_norm
        u_tmp = u

        v1 = A.mm(v)
        v1_norm = v1.norm(2)
        u = v1 / v1_norm

        if (u-u_tmp).norm(2)<1e-4 or (i+1)==num_iter:
            break

    out = u.t().mm(A).mm(v)[0][0]

    return out, u


def power_iteration_conv_evl(mu, layer, u=None, num_iter=1):
    EPS = 1e-12
    output_padding = 0
    if u is None:
        u = th.randn((1,*mu.size()[1:]), device=mu.device)

    W = layer.weight
    if layer.bias is not None:
        b = th.zeros_like(layer.bias)
    else:
        b = None

    for i in range(num_iter):
        u1 = _conv2d(u, W, b, stride=layer.stride, padding=layer.padding)
        u1_norm = u1.norm(2)
        v = u1 / (u1_norm+EPS)
        u_tmp = u

        v1 = _conv_trans2d(v, W, stride=layer.stride, padding=layer.padding, output_padding=output_padding)
        #  When the output size of conv_trans differs from the expected one.
        if v1.shape != u.shape:
            output_padding = 1
            v1 = _conv_trans2d(v, W, stride=layer.stride, padding=layer.padding, output_padding=output_padding)
        v1_norm = v1.norm(2)
        u = v1 / (v1_norm+EPS)

        if (u-u_tmp).norm(2)<1e-4 or (i+1)==num_iter:
            break

    out = (v*(_conv2d(u, W, b, stride=layer.stride, padding=layer.padding))).view(v.size()[0],-1).sum(1)[0]
    return out, u


def compute_Lfx(module, x):
    param_map = module.model.init_coordinates.param_map
    # param_map = {Normalize, nn.Sequential}, use param_map[1] to take nn.Sequential
    Lfx = 1.
    for i, layer in enumerate(param_map[1]):
        if isinstance(layer, nn.Conv2d):
            lips, u = power_iteration_conv_evl(x, layer, layer.singular_u)
            Lfx *= lips
            setattr(layer, "singular_u", u.clone().detach())
        elif isinstance(layer, nn.Linear):
            lips, u = power_iteration_evl(layer.weight, layer.singular_u)
            Lfx *= lips
            setattr(layer, "singular_u", u.clone().detach())
        x = layer(x)

    if module.model.dyn_fun.cayley:
        return Lfx
    else:
        dyn_fun = module.model.dyn_fun
        # dyn_fun doesn't have sequential, need to manually compute matrix norm for each layer
        lips_1, u = power_iteration_evl(dyn_fun.U_x.weight, dyn_fun.U_x.singular_u)
        setattr(dyn_fun.U_x, "singular_u", u.clone().detach())
        lips_2, u = power_iteration_evl(dyn_fun.mlp_to_mlp.weight, dyn_fun.mlp_to_mlp.singular_u)
        setattr(dyn_fun.mlp_to_mlp, "singular_u", u.clone().detach())
        lips_3, u = power_iteration_evl(dyn_fun.mlp_to_hidden.weight, dyn_fun.mlp_to_hidden.singular_u)
        setattr(dyn_fun.mlp_to_hidden, "singular_u", u.clone().detach())
        Lfx *= (lips_1*lips_2*lips_3)
        return Lfx
