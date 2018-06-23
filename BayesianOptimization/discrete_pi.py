from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import gpytorch
from .acquisition_function import DiscreteAcquisitionFunction
from torch.distributions.normal import Normal


class DiscretePI(DiscreteAcquisitionFunction):

    def __init__(self, GPModel):
        super(DiscretePI, self).__init__(GPModel)
        self.train_ouput = self.model.train_targets

    def forward(self, candidate_set):
        if not torch.is_tensor(candidate_set):
            raise RuntimeError("Candidate set must be a tensor")

        with gpytorch.beta_features.fast_pred_var(), gpytorch.beta_features.fast_pred_samples():
            observed_pred = self.model(candidate_set)
            mean = observed_pred.mean()
            std = torch.sqrt(observed_pred.var())

            y_max = torch.max(self.train_ouput)
            z = ((mean - y_max) / std).cpu()
            acq_func = Normal(0, 1).cdf(z)
            next_point = candidate_set[torch.argmax(acq_func)].view(1, -1)
        return acq_func, next_point, observed_pred
