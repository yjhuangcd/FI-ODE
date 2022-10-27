# FI-ODE: Certified and Robust Forward Invariance in Neural ODEs

This repository contains the implementation of the paper: FI-ODE: Certified and Robust Forward Invariance in Neural ODEs.
We develop a general approach to certifiably enforce forward invariance properties in neural ODEs using tools from non-linear control theory and sampling-based verification.

## Getting Started
This repository depends on several submodules. To clone this repository, you can use:
```bash
git clone git@github.com:yjhuangcd/FI-ODE.git --recurse-submodules
```

The environment requirements are in `env.yml`. You can create a conda environment using:
```bash
conda env create --name $envname --file=env.yml
```
And you need install the AutoAttack package manually via:
```bash
pip install git+https://github.com/fra31/auto-attack
```


## Train a Certifiably Robust Neural ODE
You can run the training code under the current directory.
To train a certifiably robust neural ODE on CIFAR-10, you can run:

```bash
python3 sl_pipeline.py --config-name cifar_train +module/lya_cand=DecisionBoundary +dataset=CIFAR10 ++gpus=1 ++batch_size=128 ++val_batch_size=256 ++data_loader_workers=4 ++module.h_dist_lim=15. ++module.opt_name=Adam ++module.lr=5e-3 ++module.t_max=1 ++module.weight_decay=0. ++module.warmup=-1 ++module.dynamics.kappa=2.0 ++module.max_epochs=300 ++module.h_sample_size=256 ++module.dynamics.alpha_1=100. ++module.dynamics.sigma_1=0.02 ++module.dynamics.alpha_2=20. ++module.val_ode_tol=1e-3 ++module.val_ode_solver=dopri5 ++module.dynamics.scale_nominal=True ++module.adv_train=False ++module.dynamics.cayley=True ++module.dynamics.kappa_length=0
```

## Certify a Neural ODE for Adversarial Robustness
To run certified robustness tests, first go to the `robustness` directory. 
Then you need to sample deterministically using:
```bash
python3 sample_decision_boundary.py --config-name cifar_certify +dataset=CIFAR10 ++T=40 hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled
```
You can also directly download the sampled points from: , and put it in the `robustness` folder.

To certify the robustness of a trained neural ODE using CROWN, you can run:

```bash
python3 certify_crown.py --config-name cifar_certify +dataset=CIFAR10 +model_file='cifar' +module/lya_cand=DecisionBoundary ++start_ind=0 ++end_ind=10000 ++T=40 ++batches=400 ++load_grid=True ++grid_name="grid_40.pt" ++norm="2" ++gpus=1 ++data_loader_workers=4 ++module.h_dist_lim=15. ++module.dynamics.alpha_1=100. ++module.dynamics.sigma_1=0.02 ++module.dynamics.alpha_2=20. ++module.val_ode_tol=1e-3 ++module.val_ode_solver=dopri5 ++module.dynamics.scale_nominal=False ++module.dynamics.cayley=True ++module.dynamics.activation=ReLU ++module.lya_cand.log_mode=False hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled
```

To certify using Lipschitz bound, you can run:
```bash
python3 certify_lipschitz.py --config-name cifar_certify +dataset=CIFAR10 +model_file='cifar' +module/lya_cand=DecisionBoundary ++T=40 ++batches=10 ++load_grid=True ++grid_name="grid_40.pt" ++norm="2" ++gpus=1 ++data_loader_workers=4 ++module.h_dist_lim=15. ++module.dynamics.alpha_1=100. ++module.dynamics.sigma_1=0.02 ++module.dynamics.alpha_2=20. ++module.val_ode_tol=1e-3 ++module.val_ode_solver=dopri5 ++module.dynamics.scale_nominal=False ++module.dynamics.cayley=True ++module.dynamics.activation=ReLU ++module.lya_cand.log_mode=False hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled
```

To evaluate the adversarial robustness of the trained models using AutoAttack, you can run:
```bash
python3 eval_autoattack.py --config-name cifar_certify +dataset=CIFAR10 +model_file='cifar' +module/lya_cand=DecisionBoundary ++module.dynamics.activation=ReLU ++norm="2" ++gpus=1 ++batch_size=128 ++val_batch_size=512 ++module.dynamics.alpha_1=100. ++module.dynamics.sigma_1=0.02 ++module.dynamics.alpha_2=20. ++module.dynamics.scale_nominal=False ++module.dynamics.cayley=True ++module.t_max=0.1 hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled
```

## Train and Certify Safe Neural Network Controller
To run certified nn controller, first go to the `control` directory.

To train a neural network controller that keeps the states of a segway system stay within a safe set, run:
```bash
python3 train_segway.py
```

To certify that the controller satisfies the safe condition and plot the trajectories, run:
```bash
python3 certify_segway.py
```

## Refererences
This repository is based upon the [LyaNet](https://github.com/ivandariojr/LyapunovLearning) repository.
The submodules use the code from the following repositories: [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA), [orthogonal-convolutions](https://github.com/locuslab/orthogonal-convolutions), [learning-and-control](https://github.com/learning-and-control/core), and [advertorch](https://github.com/BorealisAI/advertorch). 