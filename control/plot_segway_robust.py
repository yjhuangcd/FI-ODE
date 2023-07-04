import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from core.systems import Segway, Segway_params, Segway_params_sim
from core.controllers import LQRController
from core.controllers.linear_controller import LinearController
from core.controllers.nn_controller import NNController, NNController_sim
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
from tqdm import tqdm
import random

plt.rcParams.update({'font.size': 18})

def main():
    th.set_default_dtype(th.float32)
    perturb_init = True
    perturb_params = True
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
    r = 0.02
    level_ub = (math.sqrt(level) + math.sqrt(3)/2*r*max_sigma)**2
    level_lb = (math.sqrt(level) - math.sqrt(3)/2*r*max_sigma)**2

    # test perturbed system parameters
    system = Segway_params()
    nn_controller = NNController_sim(system, 3, 1, 32)
    nn_controller.load_state_dict(th.load(model_name))
    nn_controller.eval()
    dynamics = AutoLirpaModel_params(system, nn_controller, num_states=3)
    dummy_input = th.zeros(1, 14)
    model = SegwaySingleBarrierModel_params(dynamics, lya, num_states=3)
    bounded_model = BoundedModule(model, dummy_input, bound_opts={"conv_mode": "matrix"})
    bounded_model.eval()

    sizes = th.tensor([phi_region, region, region])
    rs = th.ones(3)*r
    grid, phi, v, phi_dot = grid_uniform_3d(sizes, rs)
    eta = reject_sampling(grid, lya, level_lb, level)

    # find adversarial system params
    params_mean = th.ones(1, 11)
    params_eps_percentage = 0.02
    # prepare eta for adv
    eta_small = th.concat((eta, params_mean.repeat(eta.shape[0], 1)), dim=1)
    if perturb_init and perturb_params:
        state_eps = r
        params_eps = params_mean * params_eps_percentage  # 2% error
        adv_steps = 100
    elif perturb_init and not perturb_params:
        state_eps = r
        params_eps = params_mean * 0.
        adv_steps = 100
    elif not perturb_init and perturb_params:
        state_eps = 0.
        params_eps = params_mean * params_eps_percentage  # 2% error
        adv_steps = 100
    else:
        state_eps = 0.
        params_eps = params_mean * 0.0
        adv_steps = 1
    eps_adv = th.concat((th.ones(1, 3) * state_eps, params_eps), dim=1).squeeze()
    clip_min = th.concat((-th.ones((1, 3)) * 2*np.pi, params_mean - params_eps), dim=1).squeeze()
    clip_max = th.concat((th.ones((1, 3)) * 2*np.pi, params_mean + params_eps), dim=1).squeeze()
    adversary = LinfPGDAttack(
        model.forward_adv, loss_fn=lambda x: th.mean(x), eps=eps_adv,
        nb_iter=adv_steps, eps_iter=2.5 * eps_adv / adv_steps, rand_init=True, clip_min=clip_min,
        clip_max=clip_max, targeted=False)
    with ctx_noparamgrad_and_eval(model):
        eta_adv = adversary.perturb(eta_small)
    mask = (lya(eta_adv[:, :3]) <= level).squeeze()
    eta_adv = eta_adv[mask]
    val_adv = model(eta_adv)
    print("###### val_adv max: {}".format(val_adv.max()))

    batch_size = 1000
    indice = random.sample(range(eta_adv.shape[0]), batch_size)
    indice = torch.tensor(indice)
    eta_adv_sub = eta_adv[indice]
    ts, hk = np.linspace(0, 10, 10000, retstep=True)
    system = Segway_params_sim()
    xs, us = system.simulate(eta_adv_sub, nn_controller, ts)
    xs = xs[:, :, :3]
    xs_np = xs.detach().numpy()
    not_FI = 0
    lya_val_max = []
    for i in range(xs.shape[0]):
        xs1 = xs[i]
        lya_val = lya(xs1)
        lya_val_max.append(lya_val.max().item())
        if (lya_val > level).sum() > 0:
            not_FI += 1
    print(f"###### FI percentage: {(1 - not_FI / batch_size) * 100} %")

    lya_val_max = th.tensor(lya_val_max)
    lya_val_topk, lya_val_inds = th.topk(lya_val_max, 5)
    for i in lya_val_inds:
        xs1 = xs[i]
        lya_val = lya(xs1).detach().numpy()
        plt.plot(ts, lya_val)
    plt.xlabel("time (s)")
    plt.ylabel("V")
    plt.tight_layout()
    plt.show()

    # plot 3d trajectory
    sizes = th.tensor([phi_region, region, region])
    rs = th.ones(3) * 0.005
    grid, phi, v, phi_dot = grid_uniform_3d(sizes, rs)
    eta = reject_sampling(grid, lya, level_lb, level)
    eta = eta.numpy()
    fig = plt.figure(figsize=(6.4, 6.4))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(eta[:, 0], eta[:, 2], eta[:, 1], c="grey", s=0.1, alpha=0.002)
    for i in lya_val_inds:
        xs1 = xs_np[i]
        ax.plot3D(xs1[:, 0], xs1[:, 2], xs1[:, 1])
        ax.scatter(xs1[0, 0], xs1[0, 2], xs1[0, 1], marker='o', color='c', s=5)
        ax.scatter(xs1[-1, 0], xs1[-1, 2], xs1[-1, 1], marker='s', color="y", s=2.5)
    ax.set_xlabel('\n $\phi$', linespacing=2.)
    ax.set_zlabel('\n v', linespacing=3.2)
    ax.set_ylabel('\n $\dot{\phi}$', linespacing=2.6)
    ax.dist = 10
    plt.show()

    print("segway")


if __name__ == '__main__':
    main()

