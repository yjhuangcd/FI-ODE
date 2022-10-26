# FI-ODE Certified and Robust Forward Invariance in Neural ODEs

This repository depends on a few submodules: advertorch, auto_LiRPA, core, and ortho_conv.
The environment requirements are in the file "env.yml".

## Train a Certifiably Robust Neural ODE
For the code assumes the project root is the current directory.

Example commands:

```bash
python3 sl_pipeline.py --config-name lyapunov_robust_orthodyn_project_simplex_lips_cone +module/lya_cand=DecisionBoundary +dataset=CIFAR10 ++gpus=1 ++batch_size=128 ++val_batch_size=256 ++data_loader_workers=4 ++module.h_dist_lim=15. ++module.opt_name=Adam ++module.lr=5e-3 ++module.t_max=1 ++module.weight_decay=0. ++module.warmup=-1 ++module.dynamics.kappa=2.0 ++module.max_epochs=300 ++module.h_sample_size=256 ++module.dynamics.alpha_1=100. ++module.dynamics.sigma_1=0.02 ++module.dynamics.alpha_2=20. ++module.val_ode_tol=1e-3 ++module.val_ode_solver=dopri5 ++module.dynamics.scale_nominal=True ++module.adv_train=False ++module.dynamics.cayley=True ++module.dynamics.kappa_length=0
```

## Certify a Neural ODE for Adversarial Robustness

Example commands:

```bash
python3 certify_bias_decision_boundary.py --config-name lyapunov_robust_orthodyn_project_simplex_lips +dataset=CIFAR10 +model_file=<MODEL_PATH> +module/lya_cand=DecisionBoundary ++start_ind=0 ++end_ind=10000 ++T=40 ++batches=400 ++load_grid=True ++grid_name="grid_40.pt" ++norm="2" ++gpus=1 ++data_loader_workers=4 ++module.h_dist_lim=15. ++module.dynamics.alpha_1=100. ++module.dynamics.sigma_1=0.02 ++module.dynamics.alpha_2=20. ++module.val_ode_tol=1e-3 ++module.val_ode_solver=dopri5 ++module.dynamics.scale_nominal=False ++module.dynamics.cayley=True ++module.dynamics.activation=ReLU ++module.lya_cand.log_mode=False ++download=False hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled
```

## Train a Neural Network Controller that Keeps States Stay in a Safe Region

Example commands:

```bash
python3 train_segway.py
```

## Train a Neural Network Controller that Keeps States Stay in a Safe Region

Example commands:

```bash
python3 certified_segway_plot.py
```