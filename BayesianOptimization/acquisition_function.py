from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from gpytorch.module import Module
from gpytorch.models.exact_gp import ExactGP


class AcquisitionFunction(Module):
    def __init__(self, GPModel):
        if not isinstance(GPModel, ExactGP):
            raise RuntimeError("AcquisitionFunction can only handle instance of ExactGP")
        super(AcquisitionFunction, self).__init__()
        self.model = GPModel

    def forward(self, x):
        raise NotImplementedError


class DiscreteAcquisitionFunction(AcquisitionFunction):
    def __call__(self, x):
        return super(DiscreteAcquisitionFunction, self).__call__(x)
