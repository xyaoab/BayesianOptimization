from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import gpytorch
from .acquisition_function import DiscreteAcquisitionFunction
from torch.distributions.normal import Normal


class DiscreteMES(DiscreteAcquisitionFunction):
    def __init__(self, GPModel, nK=100):
        self.nK = nK
        super(DiscreteMES, self).__init__(GPModel)

    def forward(self, candidate_set):
        if not torch.is_tensor(candidate_set):
            raise RuntimeError("Candidate set must be a tensor")

        with gpytorch.beta_features.fast_pred_var(), gpytorch.beta_features.fast_pred_samples():
            observed_pred = self.model(candidate_set)
            f_samples = observed_pred.sample(self.nK)
            y_star_sample = f_samples.max(dim=0)
            y_max = y_star_sample[0].repeat(candidate_set.size()[0], 1).t()

            mean = observed_pred.mean()
            std = torch.sqrt(observed_pred.var())
            # TODO: keep gamma in gpu
            gamma = ((y_max - mean) / std).cpu()
            acq_func = gamma * Normal(0, 1).log_prob(gamma).exp()\
                / (2 * Normal(0, 1).cdf(gamma)) - torch.log(Normal(0, 1).cdf(gamma))

            acq_func = acq_func.sum(dim=0) / self.nK
            next_point = candidate_set[torch.argmax(acq_func)].view(1, -1)
        return acq_func, next_point, observed_pred
