import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from core.systems import Segway3DReduced, Segway, Segway_params
from core.controllers import LQRController
from core.controllers.linear_controller import LinearController
from core.controllers.constant_controller import ConstantController
from core.controllers.nn_controller import NNController
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


def main():
    th.set_default_dtype(th.float32)
    adv_train = True
    adap_sample = True
    save_name = 'robust_segway.pt'
    save_name_lya = 'lya_' + save_name
    pretrained_lya_name = 'trained_models/pretrained_lya.pt'

    system = Segway()
    # initialize the neural network by LQR
    Q = 10 * th.eye(3)
    R = th.eye(1)
    goal = th.tensor([[0.0, 0.0, 0.0]])

    FF, GG = system.jacobian(goal, th.zeros(1, 1), th.tensor(0.0))

    P = solve_continuous_are(a=FF[0].detach().numpy(),
                             b=GG[0].detach().numpy(),
                             q=Q,
                             r=R)
    P = th.tensor(P)
    K = th.tensor(np.linalg.inv(R)) @ GG[0].T @ P.float()
    linear_lqr_controller = LinearController(system, K)

    level_ub = 0.2
    level_lb = 0.1
    region = 1.5
    phi_region = np.pi / 6
    nn_controller = NNController(system, 3, 1, 32)
    lya = LyaQuadratic(th.eye(3), goal)
    # train nn controller
    optimizer = optim.Adam(nn_controller.parameters(), lr=0.01)
    max_epochs_fit_lqr = 300
    sizes = th.tensor([phi_region, region, region])
    batch_size = 5120
    losses_fit_lqr = []
    for i in range(max_epochs_fit_lqr):
        eta = random_uniform(sizes, batch_size)
        if adap_sample:
            eta = reject_sampling(eta, lya, level_lb=level_lb, level_ub=level_ub)
        optimizer.zero_grad()
        nn_out = nn_controller(eta, 0.)
        lqr_out = linear_lqr_controller(eta, 0.)
        loss = F.mse_loss(nn_out, lqr_out)
        loss.backward()
        optimizer.step()
        losses_fit_lqr.append(loss.item())

    plt.plot(losses_fit_lqr)
    plt.title("fit_lqr loss")
    plt.xlabel("epochs")
    plt.show()

    system = Segway_params()
    dynamics = AutoLirpaModel_params(system, nn_controller, num_states=3)
    # load in pretrained lya
    lya.load_state_dict(th.load(pretrained_lya_name))
    model = SegwaySingleBarrierModel_params(dynamics, lya, num_states=3)

    max_epochs = 1000
    optimizer = optim.Adam([
        {'params': list(nn_controller.parameters())},
        {'params': list(lya.P.parameters()), 'lr': 0.002}
    ], lr=0.01)
    losses = []
    best_loss = 10000
    sizes = th.tensor([phi_region, region, region])
    rs = th.tensor([0.02, 0.02, 0.02])
    grid, phi, v, phi_dot = grid_uniform_3d(sizes, rs)
    params = th.ones(1, 11)
    params_eps_percentage = 0.02  # 2% error

    for i in tqdm(range(max_epochs)):
        if adap_sample:
            eta = reject_sampling(grid, lya, level_lb, level_ub)
        else:
            eta = random_uniform(sizes, batch_size)
        params_rep = params.repeat((eta.shape[0], 1))
        eta = th.concat((eta, params_rep), dim=1)

        optimizer.zero_grad()
        if adv_train:
            state_eps = 0.02
        else:  # only adv on x, no adv on eta
            state_eps = 0.0
        params_eps = params * params_eps_percentage
        eps_adv = th.concat((th.ones(1, 3) * state_eps, params_eps), dim=1).squeeze()
        clip_min = th.concat((-th.ones((1, 3)) * 2 * np.pi, params - params_eps), dim=1).squeeze()
        clip_max = th.concat((th.ones((1, 3)) * 2 * np.pi, params + params_eps), dim=1).squeeze()
        adversary = LinfPGDAttack(
            model.forward_adv, loss_fn=lambda x: th.sum(x), eps=eps_adv,
            nb_iter=7, eps_iter=2.5 * eps_adv / 7, rand_init=True, clip_min=clip_min,
            clip_max=clip_max, targeted=False)
        with ctx_noparamgrad_and_eval(model):
            eta_adv = adversary.perturb(eta)
        loss = th.mean(F.relu(model(eta_adv) + 0.01))
        loss.backward()
        optimizer.step()
        # scheduler.step()
        losses.append(loss.item())
        if loss.item() < best_loss:
            th.save(nn_controller.state_dict(), save_name)
            th.save(lya.state_dict(), save_name_lya)
            best_loss = loss.item()
            print("epoch: {}, v_dot_adv: {}, v_dot: {}".format(i, model(eta_adv).max().item(), model(eta).max().item()))
    plt.plot(losses)
    plt.title("barrier loss")
    plt.xlabel("epochs")
    plt.show()
    print("####### best loss: {}".format(min(losses)))

    # when plotting, using system that only takes in states
    system = Segway()
    # c0_controller = ConstantController(system, 0.)
    dynamics = AutoLirpaModel(system, nn_controller)
    model = SegwaySingleBarrierModel(dynamics, lya)

    # check lyapunov condition
    eta_all = random_uniform(sizes, 5120)
    eta_all[:, 1] = 0.
    eta = reject_sampling(eta_all, lya, level_lb=level_lb, level_ub=level_ub)
    val = model(eta)
    val = val.detach().cpu().numpy()
    plt.scatter(eta[:, 0], eta[:, 2], c=val)
    plt.xlim(-sizes[0], sizes[0])
    plt.ylim(-sizes[2], sizes[2])
    plt.xlabel("$\phi$")
    plt.ylabel("$\dot{\phi}$")
    plt.colorbar()
    plt.show()

    eta_all = random_uniform(sizes, 5120)
    eta_all[:, 0] = 0.
    eta = reject_sampling(eta_all, lya, level_lb=level_lb, level_ub=level_ub)
    val = model(eta)
    val = val.detach().cpu().numpy()
    plt.scatter(eta[:, 1], eta[:, 2], c=val)
    plt.xlim(-sizes[1], sizes[1])
    plt.ylim(-sizes[2], sizes[2])
    plt.xlabel("v")
    plt.ylabel("$\dot{\phi}$")
    plt.colorbar()
    plt.show()

    # check lyapunov value
    eta_all = random_uniform(sizes, 5120)
    eta_all[:, 1] = 0.
    eta = reject_sampling(eta_all, lya, level_lb=level_lb, level_ub=level_ub)
    lya_val = lya(eta)
    lya_val = lya_val.detach().cpu().numpy()
    plt.scatter(eta[:, 0], eta[:, 2], c=lya_val)
    plt.xlabel("$\phi$")
    plt.ylabel("$\dot{\phi}$")
    plt.colorbar()
    plt.show()


    print("Segway")


if __name__ == '__main__':
    main()

