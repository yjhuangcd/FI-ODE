import math
from math import sqrt, log, floor
import numpy as np
import torch
import torch as th
import torch.nn.functional as F
from torch import nn as nn
import pytorch_lightning as pl
from torch.autograd.functional import jvp, jacobian
from barrier_projection.barrier_projection import FastBarrierProjection

class AbstractSampler(nn.Module):
    def __init__(self, h_dims):
        super().__init__()
        self.h_dims = h_dims

    def device_initialize(self, device):
        self.device = device

    def forward(self, x, y, model, h_dim, sample_size):
        raise NotImplementedError()


class UniformSimplexSampling(AbstractSampler):

    def __init__(self, h_dims):
        super().__init__(h_dims)
        self.h_dist_warmup = None

    def device_initialize(self, device):
        super().device_initialize(device)
        self.h_dist_warmup = th.distributions.exponential.Exponential(th.tensor(1., device=device))

    def forward(self, x, y, model, h_dim, sample_size):
        with torch.no_grad():
            h_vec = self.h_dist_warmup.sample((sample_size, h_dim))
            h_sample_uniform = F.normalize(h_vec, p=1.0, dim=1)
            return h_sample_uniform


class BandSimplexSampling(AbstractSampler):

    def __init__(self, h_dims):
        super().__init__(h_dims)
        self.h_dist_warmup = None

    def device_initialize(self, device):
        super().device_initialize(device)
        self.h_dist_warmup = th.distributions.exponential.Exponential(th.tensor(1., device=device))
        self.h_dist_uniform = th.distributions.Uniform(
            th.tensor(0.1, device=device),
            th.tensor(1., device=device))

    def forward(self, x, y, model, h_dim, sample_size):
        with torch.no_grad():
            h_vec = self.h_dist_warmup.sample((sample_size, h_dim))
            h_sample_uniform = F.normalize(h_vec, p=1.0, dim=1)
            h_gt = self.h_dist_uniform.sample((sample_size, 1))
            h_sample_uniform[:, y] = h_gt
            return h_sample_uniform


class ProjectedBiasedHyperSphereSampling(AbstractSampler):

    def __init__(self, h_dims, n_output, h_dist_lim):
        super().__init__(h_dims)
        self.n_output = n_output
        self.h_dist_lim = h_dist_lim

    def device_initialize(self, device):
        super().device_initialize(device)
        self.h_dist = th.distributions.Uniform(
            th.tensor(0., device=device),
            th.tensor(sqrt(self.n_output * self.h_dist_lim ** 2),
            device=device))

    def forward(self, x, y, model, h_dim, sample_size):
        with torch.no_grad():
            h_radius = self.h_dist.sample((sample_size, 1))
            h_vec = th.randn(sample_size, h_dim, device=self.device)
            F.normalize(h_vec, p=2.0, dim=1, out=h_vec)
            h_sample_softmax = F.softmax(h_vec * h_radius, dim=1)
            return h_sample_softmax

class ProjectedHyperCubeSampling(AbstractSampler):

    def __init__(self, h_dims, h_dist_lim):
        super().__init__(h_dims)
        self.h_dist_lim = h_dist_lim

    def device_initialize(self, device):
        super().device_initialize(device)
        self.h_dist = th.distributions.Uniform(
            th.tensor(-self.h_dist_lim, device=device),
            th.tensor(self.h_dist_lim, device=device))

    def forward(self, x, y, model, h_dim, sample_size):
        with torch.no_grad():
            h_logits = self.h_dist.sample((sample_size, h_dim))
            F.normalize(h_logits, p=2.0, dim=1, out=h_logits)
            h_logits = F.softmax(h_logits, dim=1)
            return h_logits

class CorrectConeSampling(AbstractSampler):

    def __init__(self, h_dims):
        super().__init__(h_dims)

    def device_initialize(self, device):
        super().device_initialize(device)
        self.exp_dist = th.distributions.exponential.Exponential(th.tensor(1., device=device))

    def forward(self, x, y, model, h_dim, sample_size):
        n_batch = x.shape[0]
        with torch.no_grad():
            h_sample = F.normalize(self.exp_dist.sample((n_batch, sample_size, h_dim)),
                                   dim=-1, p=1)
            h_max_obj = h_sample.max(dim=-1)
            h_max, h_max_idx = h_max_obj.values, h_max_obj.indices
            y_one_hot = F.one_hot(y, num_classes=h_dim).bool()
            h_label = h_sample[y_one_hot[:, None, :].expand(-1, sample_size, -1)]
            h_sample[y_one_hot[:, None, :].expand(-1, sample_size, -1)] = h_max.flatten()
            h_sample.scatter_(2, h_max_idx[:, :, None],
                                 h_label.unflatten(0,h_max_idx.shape)[:, :, None])
            # For Debugging only
            # from utils import plot_labeled_samples_on_simplex
            # plot_labeled_samples_on_simplex(y, h_sample)
            return h_sample

class DecisionBoundarySampling(AbstractSampler):

    def __init__(self, h_dims):
        super().__init__(h_dims)

    def device_initialize(self, device):
        super().device_initialize(device)
        self.exp_dist = th.distributions.exponential.Exponential(th.tensor(1., device=device))

    def forward(self, x, y, model, h_dim, sample_size):
        n_batch = x.shape[0]
        with torch.no_grad():
            zs = self.exp_dist.sample((n_batch, sample_size, h_dim - 1))
            z1 = zs.max(dim=-1).values[:, :, None]
            raw_h_sample = F.normalize(th.cat([z1, zs], dim=-1), p=1, dim=-1)
            h_sample = th.zeros((n_batch, sample_size, h_dim), device=x.device, dtype=x.dtype)
            y_one_hot = F.one_hot(y).bool()
            not_y_one_hot = ~y_one_hot.bool()
            h_sample.masked_scatter_(y_one_hot[:, None, :], raw_h_sample[:, :, 0, None])
            h_sample.masked_scatter_(not_y_one_hot[:, None, :], raw_h_sample[:, :, 1:, None])
            # For Debugging only
            # from utils import plot_labeled_samples_on_simplex
            # plot_labeled_samples_on_simplex(y, h_sample)
            return h_sample


class TrajectorySampler(AbstractSampler):

    def __init__(self, h_dims):
        super().__init__(h_dims)

    def forward(self, x, y, model, h_dim, sample_size):
        with torch.no_grad():
            net_out = model.model(x, ts=th.linspace(0., model.t_max, sample_size, device=self.device),
                       int_params=model.train_solver_params,
                       use_adjoint=model.use_adjoint, return_traj=True)
            return net_out.transpose(0,1)


class CompositeSampler(nn.Module):

    def __init__(self, h_dims, samplers):
        super().__init__()
        self.samplers = samplers
        self.h_dims = h_dims


    def device_initialize(self, device):
        for sampler in self.samplers:
            sampler.device_initialize(device)

    def _coefficient_to_num_samples(self, sample_size, mixer_coefficients):
        mixed_samples = list()
        samples_added = 0
        for coeff in mixer_coefficients:
            if len(mixed_samples) == (len(mixer_coefficients) - 1):
                mixed_samples += [sample_size - samples_added]
                break
            sample_slice = floor(sample_size*coeff)
            samples_added += sample_slice
            mixed_samples += [sample_slice]
        assert sum(mixed_samples) == sample_size
        return mixed_samples


    def forward(self, x, y, model, sample_size, batch_size, mixer_coefficients):
        all_samples = []
        assert len(mixer_coefficients) == len(self.samplers), \
            "[ERROR] Each sampler must have a mixer coefficient"
        assert abs(sum(mixer_coefficients) - 1.0) < 1e-8, "[ERROR] mixer " \
                                                          "coefficeints need " \
                                                          "to sum to one."
        mixed_samples = self._coefficient_to_num_samples(sample_size, mixer_coefficients)
        for h_dim in self.h_dims:
            h_samples = []
            for sampler, n_samples in zip(self.samplers, mixed_samples):
                if n_samples == 0:
                    continue
                added_samples = sampler(x, y, model, h_dim, n_samples)
                if added_samples.ndim == 2:
                    added_samples = added_samples[None].repeat(batch_size, 1, 1)
                else:
                    added_samples = added_samples
                h_samples += [added_samples]
            h_samples = th.cat(h_samples, dim=1).flatten(0, 1)
            all_samples += [h_samples]
        return th.Tensor(), all_samples
