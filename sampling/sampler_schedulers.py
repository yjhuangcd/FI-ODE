import torch
import torch as th
import torch.nn.functional as F
from torch import nn as nn
import numpy as np

class AbstractScheduler:
    def __init__(self):
        pass
    def sampler_weight(self, epoch_num):
        raise NotImplementedError("[ERROR] Not Implemented")


class LinearScheduler(AbstractScheduler):

    def __init__(self, rate, bias=0., clamp='min', clamp_val=0., start=0):
        super().__init__()
        assert clamp_val >= 0, "Schedulers must return positive number"
        self.rate = rate
        self.bias = bias
        self.clamp = clamp
        self.clamp_val = clamp_val
        self.start = start

    def sampler_weight(self, epoch_num):
        if epoch_num < self.start:
            if self.rate > 0:
                return 0.
            else:
                return 1.
        else:
            weight = (epoch_num - self.start) * self.rate + self.bias
            if self.clamp != 'min' and self.clamp != 'max':
                return weight
            elif self.clamp == 'max':
                return min(weight, self.clamp_val)
            else:
                return max(weight, self.clamp_val)


class ConstantScheduler(AbstractScheduler):
    def __init__(self, constant):
        super().__init__()
        assert constant >= 0, "Schedulers must return positive number"
        self.constant = constant

    def sampler_weight(self, epoch_num):
        return self.constant

class SwitchScheduler(AbstractScheduler):
    def __init__(self, start, end, trigger):
        super().__init__()
        assert start >= 0 , "Schedulers must return positive number"
        assert end >= 0 , "Schedulers must return positive number"
        self.start = start
        self.end = end
        self.trigger = trigger

    def sampler_weight(self, epoch_num):
        if epoch_num < self.trigger:
            return self.start
        else:
            return self.end

class CompositeSamplerScheduler:

    def __init__(self, schedulers, scheduler_weights):
        assert len(schedulers) == len(scheduler_weights), "each scheduler needs a weight"
        self.schedulers = schedulers
        self.scheduler_weights = np.array(scheduler_weights)

    def get_mixer_coefficients(self, epoch_num):
        unnormalized_coefficients = \
            np.array([sch.sampler_weight(epoch_num) for sch in self.schedulers])
        weighted_coefficients = unnormalized_coefficients * self.scheduler_weights
        norm = np.linalg.norm(weighted_coefficients, ord=1) + 1e-12
        return weighted_coefficients / norm