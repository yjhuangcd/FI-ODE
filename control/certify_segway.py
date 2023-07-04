import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from core.systems import Segway3DReduced, Segway, Segway_params
from core.controllers import LQRController
from core.controllers.linear_controller import LinearController
from core.controllers.nn_controller import NNController
from core.controllers.constant_controller import ConstantController
import torch as th
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy.linalg._solvers import solve_continuous_are
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from grid_sampler import *
from control_utils import *
from advertorch.attacks import LinfPGDAttack, L2PGDAttack
from advertorch.context import ctx_noparamgrad_and_eval

plt.rcParams.update({'font.size': 18})

def main():
    th.set_default_dtype(th.float32)
    model_name = 'trained_models/robust_segway.pt'
    model_name_lya = 'trained_models/lya_' + model_name

    goal = th.tensor([[0.0, 0.0, 0.0]])
    lya = LyaQuadratic(th.eye(3), goal)
    lya.load_state_dict(th.load(model_name_lya))
    lya.eval()
    U, S, Vh = th.linalg.svd(lya.P.weight.data)
    max_sigma = (S.max()).item()
    region = 1.5
    phi_region = np.pi / 6
    level = 0.15
    r = 0.01
    level_ub = (math.sqrt(level) + math.sqrt(3)/2*r*max_sigma)**2
    level_lb = (math.sqrt(level) - math.sqrt(3)/2*r*max_sigma)**2

    system = Segway()
    nn_controller = NNController(system, 3, 1, 32)
    nn_controller.load_state_dict(th.load(model_name))
    nn_controller.eval()
    dynamics = AutoLirpaModel(system, nn_controller)
    dummy_input = th.zeros(1, 3)

    model = SegwaySingleBarrierModel(dynamics, lya)
    bounded_model = BoundedModule(model, dummy_input, bound_opts={"conv_mode": "matrix"})
    bounded_model.eval()

    sizes = th.tensor([phi_region, region, region])
    rs = th.ones(3)*r
    grid, phi, v, phi_dot = grid_uniform_3d(sizes, rs)
    eta = reject_sampling(grid, lya, level_lb, level_ub)
    val = model(eta)
    ptb = PerturbationLpNorm(norm=2, eps=math.sqrt(3)/2*r)
    bounded_eta = BoundedTensor(eta, ptb)
    lb, ub = bounded_model.compute_bounds(x=(bounded_eta,), method='CROWN')
    print("###### val max: {}".format(val.max()))
    print("###### ub max: {}".format(ub.max()))

    print("segway")


if __name__ == '__main__':
    main()

# def input_to_lirpa(x):
#     return system(x, linear_lqr_controller(x, 0.0), 0.0)
