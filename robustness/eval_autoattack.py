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
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd.functional import jvp
from torch.nn import functional as F
from eval_utils import *
import math
import wandb

file_path = Path(__file__).parent
config_path = Path(os.path.relpath(file_path.parent / 'configs' /
                                   'certify', start=file_path))


@hydra.main(config_path=config_path, config_name='classical')
def main(cfg: CertifyExpCfg) -> None:
    exp = SLExperiment(cfg, project='robustness')
    exp_name = exp.create_log_name()

    # Load from Wandb logger
    # run = wandb.init(project=f"RobustLyaNet")
    module_file_dir = Path('../run_data/robustness/' + cfg.model_file)
    module_file_path = Path('../run_data/robustness/' + cfg.model_file + '/model.ckpt')
    if cfg.download:
        run = wandb.init(project=f"RobustLyaNet", mode=f"online")
        artifact = run.use_artifact(cfg.model_file, type='model')
        artifact_dir = artifact.download(root=module_file_dir)

    # load checkpoint
    module = hydra_conf_load_from_checkpoint_nonstrict(module_file_path, cfg.module)
    # verify model works
    # exp.run(checkpoint_module=module, test_only=True)

    print("[INFO] Using L2 Attack with eps {}".format(cfg.eps))
    atk = AutoAttack(module, norm='L2', eps=36 / 255, version='standard')
    # atk.attacks_to_run = ['apgd-ce', 'apgd-t']

    # load datasets
    test_data = datasets.CIFAR10(
        "./data", train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor()]))

    test_data_loader = th.utils.data.DataLoader(dataset=test_data,
                                                batch_size=cfg.val_batch_size,
                                                shuffle=False,
                                                num_workers=4, pin_memory=True)

    count_correct = 0
    module.cuda()
    module.eval()

    for idx, (image, label) in enumerate(test_data_loader):
        image = image.cuda()
        label = label.cuda()

        with torch.enable_grad():
            im_adv = atk.run_standard_evaluation(image, label, bs=image.shape[0])
        with torch.no_grad():
            net_out = module(im_adv)
        y_hat = net_out.argmax(dim=-1)
        count_correct += (y_hat == label).sum()
        if idx == 0:
            robust_tensor_idx = (y_hat == label).nonzero()
            print(robust_tensor_idx.view(-1))
            torch.save(robust_tensor_idx, str(module_file_dir) + '/aa_robust_idx_first_batch.pt')
        else:
            robust_tensor_idx = th.cat((robust_tensor_idx, (y_hat == label).nonzero() + (idx * cfg.val_batch_size)),
                                       dim=0)

    autoattack_acc = count_correct / len(test_data)
    print("autoattack acc: {}".format(autoattack_acc))

    torch.save(robust_tensor_idx, str(module_file_dir) + '/aa_robust_idx.pt')


if __name__ == '__main__':
    main()
