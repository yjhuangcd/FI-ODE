import torch as th
import torch.nn as nn
import torch.nn.functional as F

class DynCrossEntropy(nn.Module):
    def __init__(self, on_simplex=False):
        super().__init__()
        self.on_simplex = on_simplex

    def forward(self, state_output, y):
        if not self.on_simplex:
            return F.cross_entropy(state_output, y, reduction='none')
        else:
            return F.nll_loss(
                th.log(th.clamp(state_output, min=1e-12)),
                y,
                reduction='none'
            )

class MSELoss(nn.Module):
    def __init__(self, on_simplex=False, num_class=10):
        super().__init__()
        self.on_simplex = on_simplex
        self.num_class = num_class

    def forward(self, state_output, y):
        assert self.on_simplex, f"States need to be on simplex"
        return F.mse_loss(state_output, F.one_hot(y, self.num_class).float(), reduction='none')


class OnemEtay(nn.Module):
    def __init__(self, on_simplex=False):
        super().__init__()
        self.on_simplex = on_simplex

    def forward(self, state_output, y):
        if not self.on_simplex:
            return F.cross_entropy(state_output, y, reduction='none')
        else:
            return F.nll_loss(
                state_output,
                y,
                reduction='none'
            )

class CompositeDynCrossEntropy(nn.Module):
    def __init__(self, on_simplex=False, norm_type='L1'):
        super().__init__()
        self.on_simplex = on_simplex
        assert norm_type in ['L1', 'L2'], "Invalid Norm Type"
        self.norm_type = norm_type

    def forward(self, state_output, y):
        prob = state_output
        if not self.on_simplex:
            prob = F.softmax(state_output, dim=1)
        prob_y = prob[list(range(y.shape[0])), y]
        loss_terms = -th.log(th.clamp(1 - prob, min=1e-12))
        if self.norm_type == 'L2':
            mod_prob_y = -th.pow(
                th.log(th.clamp(1 - prob_y, min=1e-12)), 2) + th.pow(
                th.log(th.clamp(prob_y, min=1e-12)), 2)
            loss_terms_sq = th.pow(loss_terms, 2)
            return (loss_terms_sq.sum(dim=-1) + mod_prob_y) / prob.shape[1]
        elif self.norm_type == 'L1':
            mod_prob_y = th.log(
                th.clamp(1 - prob_y, min=1e-12)) - th.log(
                th.clamp(prob_y, min=1e-12))
            loss_tmp = -th.log(th.clamp(1 - prob, min=1e-12)).sum(dim=1)
            return (loss_tmp + mod_prob_y) / prob.shape[1]

class DecisionBoundary(nn.Module):
    def __init__(self, on_simplex=False, log_mode=False, num_class=10):
        super().__init__()
        self.on_simplex = on_simplex
        self.log_mode = log_mode
        self.num_class = num_class

    def forward(self, state_output, y):
        prob = state_output
        if not self.on_simplex:
            prob = F.softmax(state_output, dim=1)

        prob_y = th.gather(prob, dim=1, index=y[:, None])[:, 0]
        flattened_wrong_probs = th.masked_select(prob, ~F.one_hot(y, self.num_class).bool())
        wrongs_shape = (prob.shape[0], prob.shape[1] - 1)
        wrong_probs = flattened_wrong_probs.unflatten(0, wrongs_shape)
        max_wrong = th.max(wrong_probs, dim=-1).values
        if self.log_mode:
            #There doesn't seem to be a good analogue for log probability loss
            #if we use the margin.
            return th.log(1+max_wrong - prob_y)
        else:
            return 1+max_wrong - prob_y