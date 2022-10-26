import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from ExpConfig import ExpCfg, RobustExpCfg, CertifyExpCfg
from sl_pipeline_test import SLExperiment
from pl_modules import AdversarialLearning
from pathlib import Path
import torch as th
import os
import itertools
from utils import hydra_conf_load_from_checkpoint
# import torchattacks
from autoattack import AutoAttack
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import wandb
from dataset_loaders import *
from eval_utils import *
from dynamics.classification import *

file_path = Path(__file__).parent
config_path = Path(os.path.relpath(file_path.parent / 'configs' /
                                   'certify', start=file_path))
save_path = 'certify_results/'

def perturbed_vdot(f_lb, f_ub, label, runner_up):
    # perturbed Vdot
    f_y = f_lb[list(range(f_lb.shape[0])), label]
    f_wrong = th.max(f_ub.masked_fill(~runner_up, -float('inf')), dim=-1).values
    vdot = -f_y + f_wrong
    return vdot

@hydra.main(config_path=config_path,
            config_name='classical')
def main(cfg: CertifyExpCfg) -> None:
    exp = SLExperiment(cfg, project='robustness')
    exp_name = exp.create_log_name()
    module_file_dir = Path('../run_data/robustness/' + cfg.model_file)
    module_file_path = Path('../run_data/robustness/' + cfg.model_file + '/model.ckpt')

    # load checkpoint
    module = hydra_conf_load_from_checkpoint_nonstrict(module_file_path, cfg.module)
    module.eval()

    # load in grids
    N_class = cfg.dataset.N_CLASSES
    T = cfg.T
    if cfg.load_grid:
        grid = torch.load(cfg.grid_name)
    else:
        grid_label_0 = sample_decision_boundary(n=N_class, T=T)
        grid = []
        for i in range(N_class):
            grid.append(get_grid_for_label(grid_label_0.copy(), i))
    batches = cfg.batches
    norm = np.inf
    if norm == np.inf:
        eps = 1 / T
    elif norm == 2:
        eps = 1/T * (math.sqrt(N_class)/2 + math.sqrt(2))
    # Lipschitz of the dynamics with respect to x
    Lfx = 1 / min(module.model.init_coordinates.param_map[0].std).item()
    # kappa to certify
    kappa = math.sqrt(2) * Lfx * cfg.eps

    # load datasets
    if cfg.dataset.name == 'CIFAR10':
        image_shape = (1, 3, 32, 32)
        test_data = datasets.CIFAR10(
            "./data", train=False, download=True,
            transform=transforms.Compose([transforms.ToTensor()]))
    elif cfg.dataset.name == 'MNIST':
        image_shape = (1, 1, 28, 28)
        test_data = datasets.MNIST(
            "./data", train=False, download=True,
            transform=transforms.Compose([transforms.ToTensor()]))
    if N_class != 10:
        test_data = reduce_to_n_classes(test_data, N_class)

    count_correct = 0
    count_certify = 0
    certified_idx = []

    # convert cayley layer to standard linear layer
    module.cuda()
    dynamics = CrownOrthoClassDynProjectSimplexLips(n_hidden=cfg.module.dynamics.n_hidden,
                                                    activation=cfg.module.dynamics.activation,
                                                    dropout=cfg.module.dynamics.dropout,
                                                    mlp_size=cfg.module.dynamics.mlp_size,
                                                    kappa=cfg.module.dynamics.kappa,
                                                    kappa_length=cfg.module.dynamics.kappa_length,
                                                    alpha_1=cfg.module.dynamics.alpha_1,
                                                    sigma_1=cfg.module.dynamics.sigma_1,
                                                    alpha_2=cfg.module.dynamics.alpha_2,
                                                    scale_nominal=cfg.module.dynamics.scale_nominal,
                                                    x_dim=cfg.module.dynamics.x_dim)

    if cfg.module.dynamics.cayley:
        dynamics.convert_cayley(module.model.dyn_fun)
    dynamics.eval()
    dummy_image = test_data[0][0].view(image_shape).cuda()
    dummy_eta = grid[0][0].view(1,10).cuda()
    dummy_static_state = module.model.init_coordinates.param_map(dummy_image)
    # Wrap model with BoundedModule
    bounded_model = BoundedModule(dynamics, (dummy_eta, dummy_static_state), bound_opts={"conv_mode": "matrix"})
    bounded_model.eval()
    # Define perturbation. Here we use a Linf perturbation on input image.
    ptb = PerturbationLpNorm(norm=norm, eps=eps)

    eta_batch_size = grid[0].shape[0] // batches
    if grid[0].shape[0] % batches != 0:
        batches += 1
    start_ind = cfg.start_ind
    end_ind = cfg.end_ind
    if end_ind == -1:
        end_ind = len(test_data)

    with th.no_grad():
        for idx in tqdm(range(start_ind, end_ind)):
            image = test_data[idx][0].view(image_shape).cuda()
            label = test_data[idx][1]
            violate = False
            net_out = module(image)
            y_hat = th.argmax(net_out, dim=-1)

            if y_hat == label:
                static_state = module.model.init_coordinates.param_map(image)
                for batch_idx in tqdm(range(0, batches)):
                    if (batch_idx+1) * eta_batch_size < grid[label].shape[0]:
                        eta = grid[label][batch_idx*eta_batch_size:(batch_idx+1)*eta_batch_size, :].float().cuda()
                    else:
                        eta = grid[label][batch_idx*eta_batch_size:, :].float().cuda()
                    # Input tensor is wrapped in a BoundedTensor object.
                    bounded_image = BoundedTensor(eta, ptb)
                    # Step 3: compute bounds for NN
                    lb, ub = bounded_model.compute_bounds(x=(bounded_image, static_state), method='CROWN')
                    if cfg.module.dynamics.scale_nominal:
                        lb, ub = dynamics.ibp_sigmoid(lb, ub, eta-eps, eta+eps)
                    f_lb, f_ub = dynamics.ibp_cbf_qp(eta, eps, lb, ub)
                    # find the label and runner-up index for each sampled eta
                    max_wrong = th.max(eta, dim=-1, keepdim=True).values
                    # when perturb eta, runner up eta may change
                    ind_wrong = eta >= (max_wrong - 2*eps)
                    ind_wrong[:,label] = False
                    y = th.ones(eta.shape[0], dtype=th.long).cuda() * label
                    vdot_crown = perturbed_vdot(f_lb, f_ub, y, ind_wrong)
                    violation = vdot_crown + kappa
                    if violation.max() > 0:
                        violate = True
                        break
                del lb, ub

            if y_hat == label:
                count_correct += 1
            if not violate and y_hat == label:
                count_certify += 1
                certified_idx.append(idx)
            if (idx-start_ind+1) % 10 == 0:
                clean_acc = count_correct / (idx-start_ind+1)
                certify_acc = count_certify / (idx-start_ind+1)
                print("# Images: {}, clean acc: {}, certify acc: {}".format(idx, clean_acc, certify_acc))
            if (idx-start_ind+1) % 100 == 0:
                print(certified_idx)

    clean_acc = count_correct / (end_ind - start_ind)
    certify_acc = count_certify / (end_ind - start_ind)
    print("range: {} to {}, clean acc: {}, certify acc: {}".format(start_ind, end_ind, clean_acc, certify_acc))

    torch.save(certified_idx, str(module_file_dir)+'/crown_margin_certified_idx' + '_st' + str(start_ind) + 'end' + str(end_ind) + '_eps_' + str(cfg.eps) + '.pt')

if __name__ == '__main__':
    main()
