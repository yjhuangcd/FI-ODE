import torch as th
from torch.autograd import Function

"""
The Barrier Projection QP solves the following optimization problem:

v^*(v_, v^_, hat{v}) &= argmin_v 0.5 ||v - hat{v}||_2^2
                                          & text{s.t. } sum(v) = 0
                                          & v_ <= v <= v^_
                                
Here We assume v in R^{nv}
"""

def BarrierProjection(max_iter=10, tol=1e-4, step_size=1e-1, verbose=False):
    class BarrierProjectionFn(Function):
        @staticmethod
        def forward(ctx, lower, upper, nominal):
            """

            :param ctx:
            :param lower: A (n_batch, nv) Tensor
            :param upper: A (n_batch, nv) Tensor
            :param nominal: A (n_batch, nv) Tensor
            :return: v^*: A (n_batch, nv) Tensor
            """
            with th.no_grad():
                device = nominal.device
                n_batch = nominal.shape[0]
                mu = th.zeros((n_batch, 1), device=device)
                v = th.zeros_like(nominal)
                epsilon = th.zeros((n_batch, 1), device=device)

                converged = False
                for i in range(max_iter):
                    v.data.copy_(nominal.data)
                    v.data -= mu
                    v.data.clamp_(lower, upper)
                    epsilon.data = v.sum(dim=-1)[:, None]
                    mu += step_size * epsilon
                    if epsilon.abs().max() < tol:
                        converged = True
                        break
                    if verbose:
                        print(f"Iteration: {i:5d} "
                              f"Primal Residual: "
                              f"{epsilon.abs().max().item(): 2.3f} ")
                        # print(epsilon)
                        # print(mu)
                if not converged and verbose:
                    print(f'[WARNING] Possibly innacurate solution residual:'
                          f' {epsilon.abs().max().item()}')

                ctx.save_for_backward(v, mu, lower, upper, nominal)
                return v
        @staticmethod
        def backward(ctx, dl_dvhat):
            v, mu, lower, upper, nominal = ctx.saved_tensors
            n_batch = v.shape[0]
            n_dim = v.shape[1]
            device = dl_dvhat.device
            lambda_upper = th.zeros_like(nominal)
            lambda_lower = th.zeros_like(nominal)


            lambda_vals = v - nominal + mu
            lambda_upper += (lambda_vals < 0) * lambda_vals.abs()
            lambda_lower += (lambda_vals > 0) * lambda_vals

            Q = th.eye(n_dim)[None].expand(n_batch, -1, -1).to(device)
            G = th.cat([-th.eye(n_dim), th.eye(n_dim)], dim=0)[None].expand(n_batch, -1, -1).to(device)
            A = th.ones((1, n_dim,))[None].expand(n_batch, -1, -1).to(device)

            Gzmh = th.diag_embed(th.cat([-v + lower, v - upper], dim=-1))
            Dlambda = th.diag_embed(th.cat([lambda_lower, lambda_upper], dim=-1))
            Gtlambda = G.transpose(2,1) @ Dlambda

            KKT_r1 = th.cat([Q, Gtlambda, A.transpose(2,1)], dim=-1)
            cols = KKT_r1.shape[-1]
            r2_zeros = cols - G.shape[-1] - Gzmh.shape[-1]
            KKT_r2 = th.cat([G, Gzmh, th.zeros((n_batch, G.shape[1],
                                                r2_zeros), device=device)], dim=-1)
            r3_zeros = cols - A.shape[-1]
            KKT_r3 = th.cat([A, th.zeros((n_batch, A.shape[1], r3_zeros), device=device)],
                            dim=-1)
            KKT_mat = th.cat([KKT_r1, KKT_r2, KKT_r3], dim=1)

            rhs = th.cat([dl_dvhat.unsqueeze(-1),
                    th.zeros((n_batch, KKT_mat.shape[-1]-dl_dvhat.shape[1],
                              1),  device=device)], dim=1)

            ds = th.linalg.lstsq(KKT_mat, -rhs)
            # ds_pinv = th.linalg.pinv(KKT_mat) @ rhs
            # dz = ds_pinv[:, :n_dim,0]
            # dlambda = ds_pinv[:, n_dim: n_dim + 2*n_dim,0]

            dz = ds.solution[:, :n_dim,0]
            dlambda = ds.solution[:, n_dim: n_dim + 2*n_dim,0]

            # dnu = ds.solution[:, -1:, 0]
            grad_box = -Dlambda @ dlambda.unsqueeze(-1)

            # we return -dz because our nominal parameter v_hat is -q in optnet paper
            # we return -grad_box[:, :n_dim, 0] because h = [-lower, upper]›
            return -grad_box[:, :n_dim, 0], grad_box[:, n_dim:, 0], -dz
    return BarrierProjectionFn.apply


def FastBarrierProjection(max_iter=10, tol=1e-4,  verbose=False):
    class BarrierProjectionFn(Function):
        @staticmethod
        def forward(ctx, lower, upper, nominal):
            """

            :param ctx:
            :param lower: A (n_batch, nv) Tensor
            :param upper: A (n_batch, nv) Tensor
            :param nominal: A (n_batch, nv) Tensor
            :return: v^*: A (n_batch, nv) Tensor
            """
            with th.no_grad():
                device = nominal.device
                n_batch = nominal.shape[0]

                mu = th.zeros((n_batch, 1), device=device)
                mu_ceil = th.ones_like(mu) * (nominal - lower).max(dim=-1).values[:, None]
                mu_floor = th.ones_like(mu) * (nominal - upper).min(dim=-1).values[:, None]

                v = th.zeros_like(nominal)
                epsilon = th.zeros((n_batch, 1), device=device)
                epsilon_pos = th.zeros_like(epsilon, dtype=th.bool)
                epsilon_neg = th.zeros_like(epsilon, dtype=th.bool)
                converged = False
                for i in range(max_iter):
                    mu = (mu_ceil - mu_floor) / 2 + mu_floor
                    v.data.copy_(nominal.data)
                    v.data -= mu
                    v.data.clamp_(lower, upper)
                    epsilon.data = v.sum(dim=-1)[:, None]
                    if epsilon.abs().max() < tol:
                        converged = True
                        break

                    epsilon_pos = epsilon > 0
                    epsilon_neg = epsilon < 0

                    mu_floor.masked_scatter_(epsilon_pos, mu[epsilon_pos])
                    mu_ceil.masked_scatter_(epsilon_neg, mu[epsilon_neg])


                    # if verbose:
                    #     print(f"Iteration: {i:5d} "
                    #           f"Primal Residual: "
                    #           f"{epsilon.abs().max().item(): 1.4e} ")
                        # print(epsilon)
                        # print(mu)
                if not converged and verbose:
                    print(f'[WARNING] Possibly innacurate solution residual:'
                          f' {epsilon.abs().max().item()}')

                ctx.save_for_backward(v, mu, lower, upper, nominal)
                return v

        @staticmethod
        def backward(ctx, dl_dvhat):
            v, mu, lower, upper, nominal = ctx.saved_tensors

            n_batch = v.shape[0]
            n_dim = v.shape[1]
            device = dl_dvhat.device
            # lambda_upper = th.zeros_like(nominal)
            # lambda_lower = th.zeros_like(nominal)

            def diag_mask(constraint_tensor):
                return th.diag_embed(
                    constraint_tensor[:, :,None].expand(-1, -1, n_dim).diagonal(dim1=-2, dim2=-1)
                )
            def recp_card(active_mask):
                return 1/active_mask.sum(dim=-1)[:,None, None].expand(-1, n_dim, n_dim)

            lambda_vals = v - nominal + mu
            upper_active = lambda_vals < 0
            lower_active = lambda_vals > 0
            not_lower_active = ~lower_active
            not_upper_active = ~upper_active
            any_active = upper_active | lower_active
            not_any_active = ~any_active
            # sum_active = any_active.sum(dim=-1)[:, None]

            jac_dnominal = th.zeros((n_batch, n_dim, n_dim), device=device)
            jac_dlower = th.zeros((n_batch, n_dim, lower.shape[1]), device=device)
            jac_dupper = th.zeros((n_batch, n_dim, upper.shape[1]), device=device)

            jac_dnominal_mask = not_any_active[:, None, :] * not_any_active[:, :, None]
            jac_dnominal.masked_scatter_(
                jac_dnominal_mask,
                -recp_card(not_any_active)[jac_dnominal_mask])
            jac_dnominal[diag_mask(not_any_active)]+= 1

            jac_lower_mask = lower_active[:, None, :] * not_lower_active[:, :, None]
            jac_dlower.masked_scatter_(
                jac_lower_mask,
                -recp_card(not_lower_active)[jac_lower_mask])
            jac_dlower[diag_mask(lower_active)]+=1

            jac_upper_mask = upper_active[:, None, :] * not_upper_active[:, :, None]
            jac_dupper.masked_scatter_(
                jac_upper_mask,
                -recp_card(not_upper_active)[jac_upper_mask])
            jac_dupper[diag_mask(upper_active)]+=1
            return (dl_dvhat[:, None, :] @ jac_dlower)[:, 0, :], \
                   (dl_dvhat[:, None, :] @ jac_dupper)[:, 0, :], \
                   (dl_dvhat[:, None, :] @ jac_dnominal)[:, 0, :]

    return BarrierProjectionFn.apply


def FastBarrierProjectionNoUpper(max_iter=10, tol=1e-4,  verbose=False):
    class BarrierProjectionFn(Function):
        @staticmethod
        def forward(ctx, lower, nominal):
            """

            :param ctx:
            :param lower: A (n_batch, nv) Tensor
            :param nominal: A (n_batch, nv) Tensor
            :return: v^*: A (n_batch, nv) Tensor
            """
            with th.no_grad():
                device = nominal.device
                n_batch = nominal.shape[0]

                mu = th.zeros((n_batch, 1), device=device)
                mu_ceil = th.ones_like(mu) * (nominal - lower).max(dim=-1).values[:, None]
                mu_floor = th.ones_like(mu) * nominal.min(dim=-1).values[:, None]

                v = th.zeros_like(nominal)
                epsilon = th.zeros((n_batch, 1), device=device)
                epsilon_pos = th.zeros_like(epsilon, dtype=th.bool)
                epsilon_neg = th.zeros_like(epsilon, dtype=th.bool)
                converged = False
                for i in range(max_iter):
                    mu = (mu_ceil - mu_floor) / 2 + mu_floor
                    v.data.copy_(nominal.data)
                    v.data -= mu
                    v.data.clamp_(lower)
                    epsilon.data = v.sum(dim=-1)[:, None]
                    if epsilon.abs().max() < tol:
                        converged = True
                        break

                    epsilon_pos = epsilon > 0
                    epsilon_neg = epsilon < 0

                    mu_floor.masked_scatter_(epsilon_pos, mu[epsilon_pos])
                    mu_ceil.masked_scatter_(epsilon_neg, mu[epsilon_neg])


                    # if verbose:
                    #     print(f"Iteration: {i:5d} "
                    #           f"Primal Residual: "
                    #           f"{epsilon.abs().max().item(): 1.4e} ")
                    # print(epsilon)
                    # print(mu)
                if not converged and verbose:
                    print(f'[WARNING] Possibly innacurate solution residual:'
                          f' {epsilon.abs().max().item()}')

                ctx.save_for_backward(v, mu, lower, nominal)
                return v

        @staticmethod
        def backward(ctx, dl_dvhat):
            v, mu, lower, nominal = ctx.saved_tensors

            n_batch = v.shape[0]
            n_dim = v.shape[1]
            device = dl_dvhat.device
            # lambda_upper = th.zeros_like(nominal)
            # lambda_lower = th.zeros_like(nominal)

            def diag_mask(constraint_tensor):
                return th.diag_embed(
                    constraint_tensor[:, :,None].expand(-1, -1, n_dim).diagonal(dim1=-2, dim2=-1)
                )
            def recp_card(active_mask):
                return 1/active_mask.sum(dim=-1)[:,None, None].expand(-1, n_dim, n_dim)

            lambda_vals = v - nominal + mu
            lower_active = lambda_vals > 0
            not_lower_active = ~lower_active
            any_active = lower_active
            not_any_active = ~any_active
            # sum_active = any_active.sum(dim=-1)[:, None]

            jac_dnominal = th.zeros((n_batch, n_dim, n_dim), device=device)
            jac_dlower = th.zeros((n_batch, n_dim, lower.shape[1]), device=device)

            jac_dnominal_mask = not_any_active[:, None, :] * not_any_active[:, :, None]
            jac_dnominal.masked_scatter_(
                jac_dnominal_mask,
                -recp_card(not_any_active)[jac_dnominal_mask])
            jac_dnominal[diag_mask(not_any_active)]+= 1

            jac_lower_mask = lower_active[:, None, :] * not_lower_active[:, :, None]
            jac_dlower.masked_scatter_(
                jac_lower_mask,
                -recp_card(not_lower_active)[jac_lower_mask])
            jac_dlower[diag_mask(lower_active)]+=1

            return (dl_dvhat[:, None, :] @ jac_dlower)[:, 0, :], \
                   (dl_dvhat[:, None, :] @ jac_dnominal)[:, 0, :]

    return BarrierProjectionFn.apply