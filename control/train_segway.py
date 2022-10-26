import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from core.systems import Segway3DReduced, Segway
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
    save_name = 'nn_inv_rej_sg2d_adv_train_best.pt'
    save_name_lya = 'lya_' + save_name

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
    # level_lb = -0.01
    region = 1.5
    nn_controller = NNController(system, 3, 1, 32)
    lya = LyaQuadratic(th.eye(3), goal)
    # train nn controller
    # optimizer = optim.SGD(nn_controller.parameters(), lr=0.01)
    optimizer = optim.Adam(nn_controller.parameters(), lr=0.01)
    max_epochs_fit_lqr = 300
    sizes = th.tensor([np.pi/12, region, region])
    batch_size = 512
    losses_fit_lqr = []
    for i in range(max_epochs_fit_lqr):
        eta_all = random_uniform(sizes, batch_size)
        eta = reject_sampling(eta_all, lya, level_lb=level_lb, level_ub=level_ub)
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

    # test sampling shape
    # eta_all = random_uniform(sizes, 512)
    # eta = reject_sampling(eta_all, lya, level_lb=level_lb, level_ub=0.05)
    # plt.scatter(eta[:,1], eta[:,0])
    # plt.xlabel("v")
    # plt.ylabel("phi")
    # plt.show()
    #
    # plt.scatter(eta[:,1], eta[:,2])
    # plt.xlabel("v")
    # plt.ylabel("phi_dot")
    # plt.show()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # grid = eta.numpy()
    # ax.scatter(grid[:, 0], grid[:, 2], grid[:, 1], c="grey", s=5.)
    # ax.set_xlabel('$\phi$')
    # ax.set_zlabel('v')
    # ax.set_ylabel('$\dot{\phi}$')
    # plt.show()

    # train with barriers
    # c0_controller = ConstantController(system, 0.)
    dynamics = AutoLirpaModel(system, nn_controller)
    model = SegwaySingleBarrierModel(dynamics, lya)

    # # check lyapunov condition
    # eta_all = random_uniform(sizes, 5120)
    # eta = reject_sampling(eta_all, lya, level_lb=level_lb, level_ub=level_ub)
    # val = model(eta)
    # val = val.detach().cpu().numpy()
    # plt.scatter(eta[:, 0], eta[:, 2], c=val)
    # plt.xlabel("$\phi$")
    # plt.ylabel("$\dot{\phi}$")
    # plt.colorbar()
    # plt.show()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(eta[:, 0], eta[:, 2], eta[:, 1], c=val, s=5.)
    # ax.set_xlabel('$\phi$')
    # ax.set_zlabel('v')
    # ax.set_ylabel('$\dot{\phi}$')
    # plt.show()

    if adv_train:
        eps = 0.02
        adversary = LinfPGDAttack(
            model.forward_adv, loss_fn=lambda x: th.mean(x), eps=eps,
            nb_iter=7, eps_iter=2.5*eps/7, rand_init=True, clip_min=-2*np.pi, clip_max=2*np.pi, targeted=False)

    max_epochs = 300
    optimizer = optim.Adam([
        {'params': list(nn_controller.parameters())},
        {'params': list(lya.P.parameters()), 'lr': 0.02}
    ], lr=0.01)
    # optimizer = optim.Adam(list(nn_controller.parameters()) + list(lya.P.parameters()), lr=0.001)
    losses = []
    best_loss = 10000
    sizes = th.tensor([np.pi/12, region, region])
    rs = th.tensor([0.02, 0.02, 0.02])
    grid, phi, v, phi_dot = grid_uniform_3d(sizes, rs)

    for i in tqdm(range(max_epochs)):
        # eta_all = random_uniform(sizes, batch_size)
        # eta = reject_sampling(eta_all, lya, level_lb=level_lb, level_ub=level_ub)
        eta = reject_sampling(grid, lya, level_lb, level_ub)
        optimizer.zero_grad()
        if adv_train:
            with ctx_noparamgrad_and_eval(model):
                eta_adv = adversary.perturb(eta)
            loss = th.sum(F.relu(model(eta_adv) + 0.01))
            # loss = th.sum(F.leaky_relu(model(eta_adv), negative_slope=0.01))
        else:
            loss = th.sum(F.relu(model(eta)))
        loss.backward()
        optimizer.step()
        # scheduler.step()
        losses.append(loss.item())
        if loss.item() < best_loss:
            th.save(nn_controller.state_dict(), save_name)
            th.save(lya.state_dict(), save_name_lya)
            best_loss = loss.item()
            print("epoch: {}, v_dot: {}".format(i, model(eta).max().item()))
    plt.plot(losses)
    plt.title("barrier loss")
    plt.xlabel("epochs")
    plt.show()
    print("####### best loss: {}".format(min(losses)))

    # check lyapunov condition
    val = model(eta)
    val = val.detach().cpu().numpy()
    plt.scatter(eta[:, 1], eta[:, 0], c=val)
    plt.colorbar()
    plt.show()

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

