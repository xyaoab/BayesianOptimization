from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from gpytorch.module import Module


class BayesianOptimization(Module):

    def step(self):
        raise NotImplementedError
