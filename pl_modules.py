from typing import Any
from math import sqrt
import numpy as np
import pytorch_lightning as pl
import torch
import torch as th
from utils import plot_samples_on_3_simplex, plot_traj_on_3_simplex, compute_Lfx
from torch import nn
from torch.autograd.functional import jvp
from torch.nn import functional as F
from advertorch.attacks import LinfPGDAttack, L2PGDAttack
from advertorch.context import ctx_noparamgrad_and_eval
import matplotlib.pyplot as plt
from autoattack import AutoAttack
import torchattacks
from models import IVP

ADAPTIVE_SOLVERS = ['dopri8', 'dopri5', 'bosh3', 'fehlberg2', 'adaptive_heun',
                    'scipy_solver']
FIXED_SOVLERS = ['euler', 'midpoint', 'rk4', 'explicit_adams',
                 'implicit_adams', 'fixed_adams']


def make_solver_params(solver_name, ode_tol):
    if solver_name in ADAPTIVE_SOLVERS:
        return dict(method=solver_name, rtol=ode_tol, atol=ode_tol)
    elif solver_name in FIXED_SOVLERS:
        return dict(
            method=solver_name,
            options=dict(
                step_size=ode_tol
            )
        )
    else:
        raise RuntimeError('[ERROR] Invalid Solver Name')


class AdversarialLearning(pl.LightningModule):

    def __init__(self, attacker, model) -> None:
        super().__init__()
        self.attacker = attacker
        self.model = model
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.run_adv = False
        self.check = False

    def test_step(self, batch, batch_idx):
        im, label = batch
        if self.run_adv:
            self.attacker.device = im.device
            with torch.enable_grad():
                im_adv = self.attacker.run_standard_evaluation(im, label, bs=im.shape[0])
            with torch.no_grad():
                net_out = self.model(im_adv)
            y_hat = net_out.argmax(dim=-1)
            error = (y_hat != label).float().mean()
            self.log('adv_test_error', error, on_epoch=True, on_step=False, logger=True)
            return error
        else:
            net_out = self.model(im)
            _, y_hat = th.max(net_out, dim=-1)
            error = (y_hat != label).float().mean()
            self.log('nominal_test_error', error, on_epoch=True, on_step=False, logger=True)
            return error


class GeneralLearning(pl.LightningModule):
    def __init__(self, opt_name="SGD",
                 lr=1e-3, momentum=0.9, weight_decay=1e-4,
                 decay_epochs=[30, 60, 90], beta1=0.9, beta2=0.999,
                 scheduler_name="cos_anneal", max_epochs=200, warmup=20,
                 adv_train=False, eps=36/255, norm='L2', simplex=False, act='relu',
                 fix_backbone=False, val_adv=True):
        super().__init__()
        self.opt_name = opt_name
        self.lr = lr
        self.betas = (beta1, beta2)
        self.decay_epochs = decay_epochs
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_per_el = nn.CrossEntropyLoss(reduction='none')
        self.scheduler_name = scheduler_name
        self.max_epochs = max_epochs
        self.warmup = warmup
        self.adv_train = adv_train
        self.eps = eps
        self.norm = norm
        self.simplex = simplex
        self.adversary = None
        self.attacker = None
        self.act = act
        self.fix_backbone = fix_backbone
        self.val_adv = val_adv

    def configure_optimizers(self):
        if self.opt_name == 'Adam':
            optimizer = th.optim.Adam(self.parameters(),
                                      lr=self.lr,
                                      weight_decay=self.weight_decay,
                                      amsgrad=False,
                                      betas=self.betas)
        elif self.opt_name == "AdamW":
            optimizer = th.optim.AdamW(self.parameters(), lr=self.lr,
                                       weight_decay=self.weight_decay,
                                       amsgrad=False,
                                       betas=self.betas)
        elif self.opt_name == 'SGD':
            if self.fix_backbone:
                optimizer = torch.optim.SGD(self.model.dyn_fun.parameters(),
                                            lr=self.lr,
                                            momentum=self.momentum,
                                            weight_decay=self.weight_decay)
            else:
                optimizer = torch.optim.SGD(self.parameters(),
                                            lr=self.lr,
                                            momentum=self.momentum,
                                            weight_decay=self.weight_decay)
        else:
            raise RuntimeError(
                f"[ERROR] Invalid Optimizer Param: {self.opt_name}")

        if self.scheduler_name == 'cos_anneal':
            scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.max_epochs)
        elif self.scheduler_name == 'step':
            scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.decay_epochs, gamma=0.1)
        else:
            # Can't have none scheduler
            scheduler = None
        lr_scheduler = {
            'scheduler': scheduler,
            # 'name': 'learning_rate',
            'monitor': 'training_loss',
            'interval': 'epoch',
            'frequency': 1
        }

        if self.current_epoch < self.warmup:
            optimizer = th.optim.Adam(self.parameters(),
                                      lr=1e-3,
                                      weight_decay=5e-4,
                                      amsgrad=False,
                                      betas=self.betas)
            return [optimizer]
        else:
            return [optimizer], [lr_scheduler]

    def on_train_start(self) -> None:
        if self.adv_train:
            if self.norm == 'L2':
                self.adversary = L2PGDAttack(
                    self.compute_loss, loss_fn=lambda x: x, eps=self.eps,
                    nb_iter=7, eps_iter=2.5*self.eps/7, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
            elif self.norm == 'Linf':
                self.adversary = LinfPGDAttack(
                    self.compute_loss, loss_fn=lambda x: x, eps=self.eps,
                    nb_iter=7, eps_iter=2.5*self.eps/7, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)

    def on_train_epoch_start(self):
        if self.current_epoch == self.warmup:
            new_opt, new_sched = self.configure_optimizers()
            self.trainer.optimizers = new_opt
            self.trainer.lr_schedulers = self.trainer._configure_schedulers(new_sched, monitor=False, is_manual_optimization=False)

    def training_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.shape[0]

        if self.adv_train:
            with ctx_noparamgrad_and_eval(self):
                x_adv = self.adversary.perturb(x, y)

            loss = self.compute_loss(x_adv, y, batch_size, self.act)
        else:
            loss = self.compute_loss(x, y, batch_size, self.act)

        self.log('training_loss', loss, on_step=True, on_epoch=True,
                 logger=True)
        return loss

    def compute_loss(self, x, y, batch_size, act):
        raise NotImplementedError('[ERROR] Abstract Method.')

    def validation_step(self, batch, batch_idxs):
        x, y = batch
        # run attack
        # if self.trainer.current_epoch >= self.max_epochs // 2 and self.val_adv:
        if self.val_adv:
            if self.norm == 'L2':
                atk = torchattacks.PGDL2(self, eps=self.eps, alpha=self.eps*2.5/10, steps=5)
            elif self.norm == 'Linf':
                atk = torchattacks.PGD(self, eps=self.eps, alpha=self.eps*2.5/10, steps=5)
            with torch.enable_grad():
                adv_images = atk(x, y)
            with torch.no_grad():
                net_out = self(x)
                net_out_adv = self(adv_images)
            y_hat_adv = net_out_adv.argmax(dim=-1)
            error_adv = (y_hat_adv != y).float().mean()
            y_hat = net_out.argmax(dim=-1)
            error = (y_hat != y).float().mean()
        else:
            with torch.no_grad():
                net_out = self(x)
            y_hat = net_out.argmax(dim=-1)
            error = (y_hat != y).float().mean()
            # if not passing half of the training, do not run adv attacks
            error_adv = error
        if not self.simplex:
            loss = self.criterion(net_out, y)
        else:
            print(f"Check Simplex! min: {net_out.min().detach().cpu().item()}, max: {net_out.max().detach().cpu().item()} ")
            # assert net_out.min().detach().cpu().item() >= -1e-1 and net_out.max().detach().cpu().item() <= 1+1e-1, \
            #     f"Violating Simplex! min: {net_out.min().detach().cpu().item()}, max: {net_out.max().detach().cpu().item()} "
            loss = F.nll_loss(torch.log(torch.clamp(net_out, min=1e-12)), y)
        self.log('validation_loss', loss, on_epoch=True, on_step=False, logger=True, sync_dist=True)
        self.log('validation_error', error, on_epoch=True, on_step=False, logger=True, sync_dist=True)
        self.log('validation_adv_error', error_adv, on_epoch=True, on_step=False, logger=True, sync_dist=True)
        return loss

    def on_test_start(self) -> None:
        if self.norm == 'L2':
            self.attacker = AutoAttack(self, norm='L2', eps=self.eps, version='standard')
        elif self.norm == 'Linf':
            self.attacker = AutoAttack(self, norm='Linf', eps=self.eps, version='standard')

    def test_step(self, batch, batch_idx):
        x, y = batch
        self.attacker.device = x.device
        self.attacker.attacks_to_run = ['apgd-ce', 'apgd-t']
        im_adv = self.attacker.run_standard_evaluation(x, y, bs=x.shape[0])
        with torch.no_grad():
            net_out_clean = self(x)
            net_out_adv = self(im_adv)
        loss = self.criterion(net_out_clean, y)
        y_hat_clean = net_out_clean.argmax(dim=-1)
        error_clean = (y_hat_clean != y).float().mean()
        y_hat_adv = net_out_adv.argmax(dim=-1)
        error_adv = (y_hat_adv != y).float().mean()
        self.log('test_loss_clean', loss, on_epoch=True, on_step=False, logger=True)
        self.log('test_error_clean', error_clean, on_epoch=True, on_step=False, logger=True)
        self.log('test_error_adv', error_adv, on_epoch=True, on_step=False, logger=True)
        return loss


class ClassicalLearning(GeneralLearning):

    def __init__(self, model: nn.Module, opt_name="SGD",
                 lr=1e-3,
                 momentum=0.9,
                 weight_decay=1e-4,
                 decay_epochs=[30, 60, 90],
                 beta1=0.9, beta2=0.999, eps=1e-8,
                 scheduler_name="cos_anneal", max_epochs=200, warmup=20):
        super().__init__(
            opt_name=opt_name, 
            lr=lr, 
            momentum=momentum,
            weight_decay=weight_decay, 
            decay_epochs=decay_epochs,
            beta1=beta1, beta2=beta2, eps=eps,
            scheduler_name=scheduler_name, max_epochs=max_epochs, warmup=warmup)
        self.model = model

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, x, y, batch_size):
        return self.criterion(self.model(x), y)


class ODELearning(GeneralLearning):
    def __init__(self, dynamics: nn.Module,
                 output,
                 n_input,
                 n_output,
                 init_fun,
                 t_max=1.0,
                 train_ode_solver='dopri5',
                 train_ode_tol=1e-6,
                 val_ode_solver='dopri5',
                 val_ode_tol=1e-6,
                 opt_name="SGD",
                 lr=1e-3,
                 momentum=0.9,
                 weight_decay=1e-4,
                 decay_epochs=[30, 60, 90],
                 beta1=0.9, beta2=0.999,
                 scheduler_name="cos_anneal", max_epochs=200, warmup=20,
                 adv_train=False, eps=127/255, norm='L2',
                 simplex=False, act='relu', fix_backbone=False, val_adv=True):
        super().__init__(
            opt_name=opt_name, lr=lr, momentum=momentum, weight_decay=weight_decay,
            decay_epochs=decay_epochs, beta1=beta1, beta2=beta2, scheduler_name=scheduler_name,
            max_epochs=max_epochs, warmup=warmup, adv_train=adv_train, eps=eps, norm=norm, simplex=simplex, act=act,
            fix_backbone=fix_backbone, val_adv=val_adv)
        self.t_max = t_max
        self.train_ode_solver = train_ode_solver
        self.train_ode_tol = train_ode_tol
        self.val_ode_solver = val_ode_solver
        self.val_ode_tol = val_ode_tol
        self.use_adjoint = False
        self.model = IVP(n_input=n_input,
                         n_output=n_output,
                         init_coordinates=init_fun,
                         # n_hidden=tuple(h_dims),
                         ts=th.linspace(0, t_max, 2),
                         ode_tol=train_ode_tol,
                         dyn_fun=dynamics,
                         output_fun=output)
    @property
    def train_solver_params(self):
        return make_solver_params(self.train_ode_solver,
                                  self.train_ode_tol)

    @property
    def val_solver_params(self):
        return make_solver_params(self.val_ode_solver,
                                  self.val_ode_tol)

    def forward(self, x, t_steps=2, return_traj=False):
        return self.model(x, ts=th.linspace(0., self.t_max, t_steps, device=x.device),
                          int_params=self.val_solver_params,
                          use_adjoint=self.use_adjoint, return_traj=return_traj)


    def compute_loss(self, x, y, batch_size, act='identity'):
        y_hat = self.model(x, ts=th.linspace(0., self.t_max, 2, device=x.device),
                            int_params=self.train_solver_params,
                            use_adjoint=self.use_adjoint)
        if not self.simplex:
            return self.criterion(y_hat, y)
        else:
            return F.nll_loss(torch.log(y_hat), y)


class LyapunovLearning(ODELearning):

    def __init__(self, order, h_sample_size, h_dist_lim, sampler,
                 sampler_scheduler,
                 dynamics: nn.Module, output, n_input, n_output, init_fun,
                 lya_cand,
                 t_max=1.0, train_ode_solver='dopri5', train_ode_tol=1e-6,
                 val_ode_solver='dopri5', val_ode_tol=1e-6, opt_name="SGD",
                 lr=1e-3, momentum=0.9, weight_decay=1e-4,
                 decay_epochs=[30, 60, 90], beta1=0.9, beta2=0.999,
                 scheduler_name="cos_anneal", max_epochs=200, warmup=20,
                 adv_train=False, eps=36/255, norm='L2',
                 simplex=False, act='relu', fix_backbone=False, val_adv=True,
                 barrier_loss=False, lips_train=False, relax_exp_stable=False, scaleLeps=3.,
                 train_ode=False, train_ode_epoch=100, epoch_off_scale=10, lips_warmup=0):
        super().__init__(dynamics, output, n_input, n_output, init_fun, t_max,
                         train_ode_solver, train_ode_tol, val_ode_solver,
                         val_ode_tol, opt_name, lr, momentum, weight_decay,
                         decay_epochs, beta1, beta2, scheduler_name, max_epochs, warmup,
                         adv_train, eps, norm,
                         simplex, act, fix_backbone, val_adv)
        self.barrier_loss = barrier_loss
        self.lips_train = lips_train
        self.relax_exp_stable = relax_exp_stable
        self.scaleLeps = scaleLeps
        self.train_ode = train_ode
        self.train_ode_epoch = train_ode_epoch
        self.epoch_off_scale = epoch_off_scale
        self.lips_warmup = lips_warmup
        self.sampler = sampler
        self.sampler_scheduler = sampler_scheduler
        self.order = order
        self.h_sample_size = h_sample_size
        self.h_dims = self.model.init_coordinates.h_dims
        self.h_dist_lim = h_dist_lim
        self.softmax = nn.Softmax(dim=1)
        self.lya_cand = lya_cand
        self.last_h_sample = None
        self.last_val_batch = None
        self.plot_saved_epoch = -1

    def on_train_start(self) -> None:
        GeneralLearning.on_train_start(self)
        self.sampler.device_initialize(self.device)

    def make_trajectories(self, x, t_steps=100):
        net_out = self(x, t_steps=t_steps, return_traj=True)
        h_sample_in = []
        h_sample_in += [net_out.transpose(0,1).reshape(-1, net_out.shape[-1])]
        t_sample = 0.
        return t_sample, h_sample_in

    def compute_loss(self, x, y, batch_size, act='identity'):
        # Turn off scale nominal after 10 epochs
        if self.current_epoch == self.epoch_off_scale:
            self.model.dyn_fun.scale_nominal = False
        static_state, _ = self.model.init_coordinates(x, self.model.dyn_fun)
        if isinstance(static_state, list):
            x_in = []
            for weight_dyn in static_state:
                x_in.append(weight_dyn[:, None].expand(-1, self.h_sample_size, *((-1,)*(weight_dyn.ndim-1))).flatten(0, 1))
        else:
            x_in = static_state[:, None].expand(-1, self.h_sample_size, *((-1,)*(static_state.ndim-1))).flatten(0, 1)
        y_in = y[:, None].expand(-1, self.h_sample_size).flatten(0, 1)

        def v_ndot(order: int, t_sample, *oc_in):
            assert isinstance(order, int) and order >= 0, \
                f"[ERROR] Order({order}) must be non-negative integer."
            if order == 0:
                return self.lya_cand(self.model.output_fun(oc_in), y_in)
            elif order == 1:
                return jvp(func=lambda *x: v_ndot(0, t_sample, *x),
                           inputs=tuple(oc_in),
                           v=self.model.dyn_fun.eval_dot(t_sample, tuple(oc_in), x_in),
                           create_graph=True)
            else:
                returns = tuple()
                for i in range(1, order):
                    returns += v_ndot(i, t_sample, *oc_in)
                returns += (jvp(func=lambda *x: v_ndot(order-1,t_sample, *x)[-1],
                               inputs=tuple(oc_in),
                               v=self.model.dyn_fun.eval_dot(t_sample, tuple(oc_in), x_in),
                               create_graph=True)[-1],)
                return returns

        mixing_weights = self.sampler_scheduler.get_mixer_coefficients(self.current_epoch)
        for i, mi_w in enumerate(mixing_weights):
            self.log(f'mixing_weight_{i}', mi_w)

        t_samples, h_sample_in = self.sampler(x, y, self, self.h_sample_size, batch_size, mixing_weights)

        if self.plot_saved_epoch < self.current_epoch:
            self.plot_saved_epoch = self.current_epoch
            self.last_h_sample = h_sample_in[0]
        if self.order == 0:
            raise NotImplementedError('[TODO] Implement this.')
        elif self.order == 1:
            v, vdot = v_ndot(1, t_samples, *h_sample_in)
            if self.lips_train:
                Lfx = compute_Lfx(self, x)
                if self.trainer.global_step < self.lips_warmup:  # around 10 warmup epochs
                    current_eps = 0.
                elif self.trainer.global_step < (self.model.dyn_fun.kappa_length + self.lips_warmup):
                    current_eps = (self.trainer.global_step - self.lips_warmup) / self.model.dyn_fun.kappa_length * self.eps
                else:
                    current_eps = self.eps
                current_kappa = max(current_eps * sqrt(2) * Lfx, self.model.dyn_fun.kappa) + 1.
                self.log('Lips', Lfx, on_step=True, logger=True, sync_dist=True)
            else:
                if self.trainer.global_step < self.model.dyn_fun.kappa_length:
                    current_kappa = self.trainer.global_step / self.model.dyn_fun.kappa_length * self.model.dyn_fun.kappa
                else:
                    current_kappa = self.model.dyn_fun.kappa
            self.log('kappa', current_kappa, on_step=True, logger=True, sync_dist=True)
            if self.relax_exp_stable:
                margin = torch.clamp(current_kappa * v.detach(), max=self.scaleLeps * self.model.dyn_fun.alpha_1 * self.eps)
            else:
                margin = current_kappa * v.detach()
            if act == 'relu':
                violations = th.relu(vdot + margin)
            elif act == 'elu':
                violations = F.elu(vdot + margin)
            else:
                violations = vdot + margin

        violation_mask = violations > 0
        effective_batch_size = (violation_mask).sum()

        loss = violations.mean()
        if self.barrier_loss:
            f_tilde = self.model.dyn_fun._h_dot_raw(h_sample_in[0], x_in)
            lower = -self.model.dyn_fun.alpha_1 * h_sample_in[0]
            upper = self.model.dyn_fun.alpha_2 * (1 - h_sample_in[0])
            self.log('train_monte_carlo_loss', loss, on_step=True, logger=True, sync_dist=True)
            loss_barrier = 100*th.relu(f_tilde - upper).mean() + th.relu(lower - f_tilde).mean()
            self.log('train_barrier_loss', loss_barrier, on_step=True, logger=True, sync_dist=True)
        if hasattr(self.model.dyn_fun, 'scale_nominal'):
            # if not self.model.dyn_fun.scale_nominal:
            with torch.no_grad():
                f = self.model.dyn_fun.eval_dot(0, h_sample_in, x_in)
                lower = -self.model.dyn_fun.alpha_1 * h_sample_in[0]
                upper = self.model.dyn_fun.alpha_2 * (1-h_sample_in[0])
                active_constraint_mask = ((f - lower).abs() <= 1e-6) | (
                        (f - upper).abs() <= 1e-6)
                mean_active = active_constraint_mask.float().mean()
                self.log('mean_active_constraints', mean_active, on_step=True, logger=True, sync_dist=True)
        self.log('effective_batch_size', effective_batch_size, on_step=True, logger=True, sync_dist=True)

        h_sample_in = None
        x_in = None
        y_in = None

        if self.train_ode and self.trainer.current_epoch > self.train_ode_epoch:
            y_hat = self.model(x, ts=th.linspace(0., self.t_max, 2, device=self.device),
                               int_params=self.train_solver_params,
                               use_adjoint=self.use_adjoint, return_traj=False)
            if not self.simplex:
                loss_ode = self.criterion(y_hat, y)
            else:
                loss_ode = F.nll_loss(torch.log(y_hat), y)
            loss_ode_portion = min(0.98, (self.trainer.current_epoch - self.train_ode_epoch) / 50.)
            loss_portion = 1 - loss_ode_portion
            return loss * loss_portion + loss_ode * loss_ode_portion
        else:
            return loss

    def validation_step(self, batch, batch_idxs):
        super(ODELearning, self).validation_step(batch, batch_idxs)
        self.last_val_batch = batch

    def on_validation_end(self):
        if self.h_dims[0] == 3:
            if self.last_h_sample is not None:
                self.logger.experiment.log({
                    "H samples": plot_samples_on_3_simplex(self.last_h_sample),
                    "epoch": self.current_epoch})
            if self.last_val_batch is not None:
                x, y = self.last_val_batch
                trajectory = self.model(x, ts=th.linspace(0., self.t_max, 100, device=x.device),
                               int_params=self.val_solver_params,
                               use_adjoint=self.use_adjoint, return_traj=True)
                self.logger.experiment.log({
                    "sample_trajectories": plot_traj_on_3_simplex(trajectory, y),
                    "epoch": self.current_epoch
                })
        super(ODELearning, self).on_validation_end()
