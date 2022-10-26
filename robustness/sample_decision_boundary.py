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
from autoattack import AutoAttack
# from auto_LiRPA import BoundedModule, BoundedTensor
# from auto_LiRPA.perturbations import *
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


@hydra.main(config_path=config_path, config_name='classical')
def main(cfg: CertifyExpCfg) -> None:
    N_class = cfg.dataset.N_CLASSES
    T = cfg.T
    grid_label_0 = sample_decision_boundary(n=N_class, T=T)
    grid = []
    for i in range(N_class):
        grid.append(get_grid_for_label(grid_label_0.copy(), i))

    torch.save(grid, 'grid_' + str(T) + '.pt')

if __name__ == '__main__':
    main()
