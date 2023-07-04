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
from tqdm import tqdm
import time
import plotly.express as px
# import plotly.io as pio
# pio.renderers.default = "browser"

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
    r = 0.005
    level_ub = (math.sqrt(level) + math.sqrt(3)/2*r*max_sigma)**2
    level_lb = (math.sqrt(level) - math.sqrt(3)/2*r*max_sigma)**2


    # test perturbed system parameters
    system = Segway_params()
    nn_controller = NNController(system, 3, 1, 32)
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
    eta = reject_sampling(grid, lya, level_lb, level_ub)

    # find adversarial system params
    params_mean = th.ones(1, 11)
    # prepare eta for adv
    eta_small = th.concat((eta, params_mean.repeat(eta.shape[0], 1)), dim=1)
    # prepare eta for certification
    state_eps = 0.02
    params_eps = params_mean * 0.02  # 1% error
    eps_adv = th.concat((th.ones(1, 3) * state_eps, params_eps), dim=1).squeeze()
    clip_min = th.concat((-th.ones((1, 3)) * 2*np.pi, params_mean - params_eps), dim=1).squeeze()
    clip_max = th.concat((th.ones((1, 3)) * 2*np.pi, params_mean + params_eps), dim=1).squeeze()
    adversary = LinfPGDAttack(
        model.forward_adv, loss_fn=lambda x: th.mean(x), eps=eps_adv,
        nb_iter=100, eps_iter=2.5 * eps_adv / 100, rand_init=True, clip_min=clip_min,
        clip_max=clip_max, targeted=False)
    with ctx_noparamgrad_and_eval(model):
        eta_adv = adversary.perturb(eta_small)
    val_adv = model(eta_adv)
    print("###### val_adv max: {}".format(val_adv.max()))

    # certification
    sizes_params = th.tensor([params_eps[0,3], params_eps[0,6], params_eps[0,7], params_eps[0,-3]])
    rs_params = th.tensor([params_eps[0,3]/2, params_eps[0,6], params_eps[0,7]/2, params_eps[0,-3]/2])
    grid_params, _, _, _, _ = grid_uniform_4d(sizes_params, rs_params)
    grid_params += params_mean[:, [3, 6, 7, -3]]
    params = params_mean.repeat((grid_params.shape[0], 1))
    params[:, [3,6,7,-3]] = grid_params
    eta_unique_num = eta.shape[0]
    eta_rep = eta.repeat_interleave(params.shape[0], dim=0)
    params = params.repeat((eta_unique_num, 1))
    eta_rep = th.concat((eta_rep, params), dim=1)
    params_eps[0,3] /= 4
    params_eps[0,6] /= 2
    params_eps[0,7] /= 4
    params_eps[0,-3] /= 4
    perturb_r = th.concat((rs.unsqueeze(dim=0), params_eps * 2), dim=1)
    batch_size = eta_small.shape[0]
    val_maxs = []
    ub_maxs = []
    violate_ind = -1
    start_time = time.time()
    with th.no_grad():
        for i in tqdm(range(0, eta_rep.shape[0], batch_size)):
            if i+batch_size <= eta_rep.shape[0]:
                eta_batch = eta_rep[i: i+batch_size, :]
            else:
                eta_batch = eta_rep[i:, :]
            val = model(eta_batch)
            ptb = PerturbationLpNorm(norm=np.inf, eps=None,
                                     x_L=eta_batch-perturb_r/2,
                                     x_U=eta_batch+perturb_r/2)
            # ptb = PerturbationLpNorm(norm=2, eps=math.sqrt(3)/2*r)
            bounded_eta = BoundedTensor(eta_batch, ptb)
            lb, ub = bounded_model.compute_bounds(x=(bounded_eta,), method='CROWN')
            print(f"###### batch: {i}, val max: {val.max().item()}")
            print(f"###### batch: {i}, ub max: {ub.max().item()}")
            val_maxs.append(val.max().item())
            ub_maxs.append(ub.max().item())
            if ub.max().item() > 0:
                violate_ind = i
                break
        print(f"###### Phase 1: val max: {max(val_maxs)}")
        print(f"###### Phase 1: ub max: {max(ub_maxs)}")

        # sampling denser for vulnerable region
        eta_ind = violate_ind // grid_params.shape[0]
        eta_vulnerable = eta[eta_ind:, :]
        grid_single_eta, _, _, _ = grid_uniform_3d(rs / 2, rs / 2)
        for k in tqdm(range(eta_vulnerable.shape[0])):
            cur_eta = eta_vulnerable[k].unsqueeze(dim=0).repeat((grid_single_eta.shape[0], 1))
            cur_eta += grid_single_eta
            if k == 0:
                eta_vulnerable_denser = cur_eta
            else:
                eta_vulnerable_denser = th.concat((eta_vulnerable_denser, cur_eta), dim=0)
        eta_vulnerable = eta_vulnerable_denser
        rs /= 2
        params_eps[0, 6] /= 2
        params_eps[0, -3] /= 4
        perturb_r = th.concat((rs.unsqueeze(dim=0), params_eps * 2), dim=1)
        rs_params[1] /= 2
        rs_params[-1] /= 4
        grid_params, _, _, _, _ = grid_uniform_4d(sizes_params, rs_params)
        grid_params += params_mean[:, [3, 6, 7, -3]]
        params = params_mean.repeat((grid_params.shape[0], 1))
        params[:, [3,6,7,-3]] = grid_params
        eta_rep = eta_vulnerable.repeat_interleave(params.shape[0], dim=0)
        params = params.repeat((eta_vulnerable.shape[0], 1))
        eta_rep = th.concat((eta_rep, params), dim=1)
        val_maxs = []
        ub_maxs = []
        for i in tqdm(range(0, eta_rep.shape[0], batch_size)):
            if i+batch_size <= eta_rep.shape[0]:
                eta_batch = eta_rep[i: i+batch_size, :]
            else:
                eta_batch = eta_rep[i:, :]
            val = model(eta_batch)
            ptb = PerturbationLpNorm(norm=np.inf, eps=None,
                                     x_L=eta_batch-perturb_r/2,
                                     x_U=eta_batch+perturb_r/2)
            # ptb = PerturbationLpNorm(norm=2, eps=math.sqrt(3)/2*r)
            bounded_eta = BoundedTensor(eta_batch, ptb)
            lb, ub = bounded_model.compute_bounds(x=(bounded_eta,), method='CROWN')
            print(f"###### batch: {i}, val max: {val.max().item()}")
            print(f"###### batch: {i}, ub max: {ub.max().item()}")
            val_maxs.append(val.max().item())
            ub_maxs.append(ub.max().item())
    end_time = time.time()
    print(ub_maxs)
    print(f"###### Phase 2: val max: {max(val_maxs)}")
    print(f"###### Phase 2: ub max: {max(ub_maxs)}")
    print(f"time: {end_time - start_time} s")

    print("segway")


if __name__ == '__main__':
    main()

