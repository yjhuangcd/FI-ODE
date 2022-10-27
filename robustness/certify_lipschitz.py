"""
Ceritfy with Lipschitz
Using exp alpha function
"""

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from ExpConfig import ExpCfg, RobustExpCfg, CertifyExpCfg
from sl_pipeline_test import SLExperiment
from pl_modules import AdversarialLearning
from pathlib import Path
import torch as th
import os
import itertools
# import torchattacks
# from autoattack import AutoAttack
# from auto_LiRPA import BoundedModule, BoundedTensor
# from auto_LiRPA.perturbations import *
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd.functional import jvp
from torch.nn import functional as F
import math
from eval_utils import *
import wandb
from tqdm import tqdm

file_path = Path(__file__).parent
config_path = Path(os.path.relpath(file_path.parent / 'configs' /
                                   'certify', start=file_path))
save_path = 'certify_results/'

def h_vdot(dyn_fun, eta, x_in, label, runner_up):
    f = dyn_fun.eval_dot_light(eta, x_in)
    f_y = f[list(range(f.shape[0])), label]
    f_wrong = th.max(f.masked_fill(~runner_up, -float('inf')), dim=-1).values
    vdot = -f_y + f_wrong
    return vdot

@hydra.main(config_path=config_path, config_name='classical')
def main(cfg: CertifyExpCfg) -> None:
    exp = SLExperiment(cfg, project='robustness')
    exp_name = exp.create_log_name()
    # Load from Wandb logger
    module_file_dir = Path('../run_data/robustness/' + cfg.model_file)
    module_file_path = Path('../run_data/robustness/' + cfg.model_file + '/model.ckpt')

    # load checkpoint
    module = hydra_conf_load_from_checkpoint_nonstrict(module_file_path, cfg.module)

    N_class = cfg.dataset.N_CLASSES
    T = cfg.T
    if cfg.load_grid:
        grid = torch.load(cfg.grid_name)
    else:
        grid_label_0 = sample_decision_boundary(n=N_class, T=T)
        grid = []
        for i in range(N_class):
            grid.append(get_grid_for_label(grid_label_0.copy(), i))
    # stength of barrier condition
    alpha = module.model.dyn_fun.alpha_1
    # Lipschitz of the dynamics with respect to eta
    if cfg.module.dynamics.scale_nominal:
        Lfx = alpha / min(module.model.init_coordinates.param_map[0].std).item()
    else:
        Lfx = 1. / min(module.model.init_coordinates.param_map[0].std).item()
    # kappa to certify
    kappa = math.sqrt(2) * Lfx * cfg.eps
    # number of batches for eta_grid
    batches = cfg.batches
    # params for exp alpha function
    alpha_1 = cfg.module.dynamics.alpha_1
    sigma_1 = cfg.module.dynamics.sigma_1
    eps = 1/T
    if eps == 1/T:
        # distance between any input and a grid point
        dist = math.sqrt(N_class) / T
    elif eps == 2/T:
        dist = (math.sqrt(2) + math.sqrt(N_class)/2) / T

    # load datasets
    test_data = datasets.CIFAR10(
        "./data", train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor()]))

    count_correct = 0
    count_certify = 0
    count_certify_larger_T = 0

    certified_idx = []
    violations_store = []

    module.cuda()
    module.eval()

    eta_batch_size = grid[0].shape[0] // batches
    if grid[0].shape[0] % batches != 0:
        batches += 1

    for idx in tqdm(range(len(test_data))):
        image = test_data[idx][0].view(1,3,32,32).cuda()
        label = test_data[idx][1]
        violations = []
        violation_larger_Ts = []

        with th.no_grad():
            net_out = module(image)
            y_hat = th.argmax(net_out, dim=-1)
            static_state, _ = module.model.init_coordinates(image, module.model.dyn_fun)

            for batch_idx in range(batches):
                if (batch_idx+1) * eta_batch_size < grid[label].shape[0]:
                    eta = grid[label][batch_idx*eta_batch_size:(batch_idx+1)*eta_batch_size, :].float().cuda()
                else:
                    eta = grid[label][batch_idx*eta_batch_size:, :].float().cuda()
                # Compute Lfeta and perturb
                # Take the worst local lipschitz within the neighborhood
                eta_ub = eta.amax(dim=1) + eps
                Lf_eta = math.sqrt(N_class) * (sigma_1*alpha_1*torch.exp(sigma_1*eta_ub)) + 1
                perturb = (math.sqrt(2) * Lf_eta) * dist
                # find the label and runner-up index for each sampled eta
                max_wrong = th.max(eta, dim=-1, keepdim=True).values
                ind_wrong = eta==max_wrong
                ind_wrong[:,label] = False
                y = th.ones(eta.shape[0], dtype=th.long).cuda() * label
                h_x0 = h_vdot(module.model.dyn_fun, eta, static_state, y, ind_wrong)
                violation_larger_T = h_x0 + kappa
                violation = h_x0 + perturb + kappa
                violation_larger_Ts.append(violation_larger_T.max().item())
                violations.append(violation.max().item())

        violations_store.append(max(violations))
        if y_hat == label:
            count_correct += 1
        if max(violations) < 0:
            count_certify += 1
            certified_idx.append(idx)
        if max(violation_larger_Ts) < 0:
            count_certify_larger_T += 1

        if (idx+1) % 100 == 0:
            clean_acc = count_correct / (idx+1)
            certify_acc = count_certify / (idx+1)
            larger_T_certify_acc = count_certify_larger_T / (idx+1)
            print("# Images: {}, clean acc: {}, certify acc: {}, "
                  "larger T certify acc: {}".format(idx, clean_acc, certify_acc, larger_T_certify_acc))
        if idx == 200:
            print(certified_idx)
            torch.save(certified_idx, str(module_file_dir)+'/margin_certified_idx_200.pt')
            torch.save(violations_store, str(module_file_dir)+'/margin_violation_store_200.pt')

    clean_acc = count_correct / len(test_data)
    certify_acc = count_certify / len(test_data)
    larger_T_certify_acc = count_certify_larger_T / len(test_data)
    print("clean acc: {}, certify acc: {}, "
          "larger T certify acc: {}".format(clean_acc, certify_acc, larger_T_certify_acc))

    torch.save(certified_idx, str(module_file_dir)+'/margin_certified_idx.pt')
    torch.save(violations_store, str(module_file_dir)+'/margin_violations_store.pt')


if __name__ == '__main__':
    main()
