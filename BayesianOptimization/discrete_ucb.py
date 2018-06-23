from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import gpytorch
from .acquisition_function import DiscreteAcquisitionFunction


class DiscreteUCB(DiscreteAcquisitionFunction):
    def __init__(self, GPModel, kappa=5):
        self.kappa = kappa
        super(DiscreteUCB, self).__init__(GPModel)

    def forward(self, candidate_set):
        if not torch.is_tensor(candidate_set):
            raise RuntimeError("Candidate set must be a tensor")

        with gpytorch.beta_features.fast_pred_var(), gpytorch.beta_features.fast_pred_samples():
            observed_pred = self.model(candidate_set)
            mean = observed_pred.mean()
            std = torch.sqrt(observed_pred.var())
            acq_func = mean + self.kappa * std
            next_point = candidate_set[torch.argmax(acq_func)].view(1, -1)
        return acq_func, next_point, observed_pred
