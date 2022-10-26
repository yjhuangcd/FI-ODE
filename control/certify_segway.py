import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from core.systems import Segway3DReduced, Segway
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
from tqdm import tqdm
import plotly.express as px
# import plotly.io as pio
# pio.renderers.default = "browser"

plt.rcParams.update({'font.size': 18})

def main():
    th.set_default_dtype(th.float32)
    model_name = 'nn_inv_rej_sg2d_adv_train_best.pt'
    model_name_lya = 'lya_' + model_name

    goal = th.tensor([[0.0, 0.0, 0.0]])
    lya = LyaQuadratic(th.eye(3), goal)
    lya.load_state_dict(th.load(model_name_lya))
    lya.eval()
    # lya = th.load(model_name_lya)
    U, S, Vh = th.linalg.svd(lya.P.weight.data)
    max_sigma = (S.max()).item()
    region = 1.5
    phi_region = np.pi/12
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
    perturb_r = rs
    # ptb = PerturbationLpNorm(norm=np.inf, eps=None,
    #                          x_L=eta-perturb_r/2,
    #                          x_U=eta+perturb_r/2)
    ptb = PerturbationLpNorm(norm=2, eps=math.sqrt(3)/2*r)
    bounded_eta = BoundedTensor(eta, ptb)
    lb, ub = bounded_model.compute_bounds(x=(bounded_eta,), method='CROWN')
    print("###### ub max: {}".format(ub.max()))

    # plot contour in 2d for phi and phi_dot
    sizes = th.tensor([phi_region, region])
    eta, phi, phi_dot = grid_uniform_2d(sizes, rs)
    zeros = th.zeros(eta.shape[0], 1) + 0.
    tmp = th.concat((eta, zeros), dim=1)
    eta = tmp.clone()
    eta[:, 1] = tmp[:, 2]
    eta[:, 2] = tmp[:, 1]
    f = dynamics(eta)
    f = f / f.norm(dim=1, keepdim=True)
    f_phi = f[:, 0].reshape(phi.shape[0], phi.shape[1]).detach().numpy()
    f_phi_dot = f[:, 2].reshape(phi.shape[0], phi.shape[1]).detach().numpy()

    bounded_eta = BoundedTensor(eta, ptb)
    lb_2d, ub_2d = bounded_model.compute_bounds(x=(bounded_eta,), method='CROWN')
    val_2d = model(eta)
    plot_ub = ub_2d.reshape(phi.shape[0], phi.shape[1]).detach().numpy()
    plot_val = val_2d.reshape(phi.shape[0], phi.shape[1]).detach().numpy()

    lya_val = lya(eta)
    plot_lya_val_phi_dot = lya_val.reshape(phi.shape[0], phi.shape[1]).detach().numpy()
    ax = plt.gca()
    c1 = ax.contourf(phi, phi_dot, plot_val, levels=8, alpha=0.4, cmap=cm.coolwarm)
    ax.contour(phi, phi_dot, plot_lya_val_phi_dot, [level,])
    ax.contour(phi, phi_dot, plot_lya_val_phi_dot, [level_lb,], linestyles=["dashed",], linewidth=1.5)
    ax.contour(phi, phi_dot, plot_lya_val_phi_dot, [level_ub,], linestyles=["dashed",], linewidth=1.5)
    plt.streamplot(phi, phi_dot, f_phi, f_phi_dot, color=('grey'), linewidth=0.5,
                   density=0.5, arrowstyle='-|>', arrowsize=1.5)
    plt.colorbar(c1)
    ax.set_xlabel('$\phi$')
    ax.set_ylabel('$\dot{\phi}$')
    plt.tight_layout()
    plt.show()

    # plot trajectories
    sizes = th.tensor([phi_region, region, region])
    x_0 = random_uniform(sizes, batch_size=1000)
    x_0_in, _ = reject_sampling(x_0, lya, level-0.02, level, return_mask=True)
    x_0 = x_0_in[:5, :]
    ts, hk = np.linspace(0, 50, 10000, retstep=True)
    xs, us = system.simulate(x_0, nn_controller, ts)
    xs_np = xs.numpy()

    for i in range(x_0.shape[0]):
        xs1 = xs[i]
        lya_val = lya(xs1).numpy()
        plt.plot(ts, lya_val)
    plt.xlabel("time (s)")
    plt.ylabel("V")
    plt.tight_layout()
    plt.show()

    # plot 3d trajectory
    sizes = th.tensor([phi_region, region, region])
    rs = th.ones(3) * 0.005
    grid, phi, v, phi_dot = grid_uniform_3d(sizes, rs)
    eta = reject_sampling(grid, lya, level_lb, level_ub)
    fig = plt.figure(figsize=(6.4, 6.4))
    ax = fig.add_subplot(projection='3d')
    eta = eta.numpy()
    ax.scatter(eta[:, 0], eta[:, 2], eta[:, 1], c="grey", s=1., alpha=0.002)
    for i in range(x_0.shape[0]):
        xs1 = xs_np[i]
        ax.plot3D(xs1[:, 0], xs1[:, 2], xs1[:, 1])
        # ax.scatter(xs1[0, 0], xs1[0, 1], marker="*", color="c", s=5)
        ax.scatter(xs1[-1, 0], xs1[-1, 1], marker="s", color="g", s=2.5)
    ax.set_xlabel('\n $\phi$', linespacing=2.)
    ax.set_zlabel('\n v', linespacing=3.2)
    ax.set_ylabel('\n $\dot{\phi}$', linespacing=2.6)
    ax.dist = 10
    plt.show()


    print("segway")


if __name__ == '__main__':
    main()

# def input_to_lirpa(x):
#     return system(x, linear_lqr_controller(x, 0.0), 0.0)
