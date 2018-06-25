from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import unittest

import torch
import gpytorch

from torch.autograd import Variable
from BayesianOptimization.standard_bayesian_optimization import StandardBayesianOptimization
from torch import nn
from gpytorch.kernels import RBFKernel, GridInterpolationKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.random_variables import GaussianRandomVariable
from BayesianOptimization.dimension import Real
torch.cuda.set_device(2)


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(constant_bounds=[-3, 3])
        # Put a grid interpolation kernel over the RBF kernel
        self.base_covar_module = RBFKernel(log_lengthscale_bounds=(-6, 6))
        self.covar_module = GridInterpolationKernel(self.base_covar_module, grid_size=400,
                                                    grid_bounds=[(0, 1.2)])
        # Register kernel lengthscale as parameter
        self.register_parameter('log_outputscale', nn.Parameter(torch.Tensor([0])), bounds=(-6, 6))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        covar_x = covar_x.mul(self.log_outputscale.exp())
        return GaussianRandomVariable(mean_x, covar_x)


def target(x):
    return -(1.4 - 3 * x) * torch.sin(18 * x)


train_x = Variable(torch.linspace(0, 1.2, 4)).cuda()
train_y = Variable(-1 * target(train_x)).cuda()
search_space = [Real(0, 1.2)]

likelihood = GaussianLikelihood(log_noise_bounds=(-8, -7)).cuda()
model = GPRegressionModel(train_x.data, train_y.data, likelihood).cuda()
model.train()
likelihood.train()
optimizer = torch.optim.Adam([{'params': model.parameters()}, ], lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for _ in range(20):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()


class TestBayesOptSin(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_minimum_ucb(self):
        bo = StandardBayesianOptimization(model, likelihood, optimizer, target, search_space, "discrete_ucb", {"kappa": 4})
        bo.optimal(10)
        y_actual = -1.48907
        x_actual = 0.96609
        self.assertLess(torch.norm(bo.x_star - x_actual), 1e-3)
        self.assertLess(torch.norm(bo.y_star - y_actual), 1e-3)

    def test_minimum_mes(self):
        bo = StandardBayesianOptimization(model, likelihood, optimizer, target, search_space, "discrete_mes", {"nK": 1000})
        bo.optimal(10)
        y_actual = -1.48907
        x_actual = 0.96609
        self.assertLess(torch.norm(bo.x_star - x_actual), 1e-3)
        self.assertLess(torch.norm(bo.y_star - y_actual), 1e-3)

    def test_minimum_pi(self):
        bo = StandardBayesianOptimization(model, likelihood, optimizer, target, search_space, "discrete_pi")
        bo.optimal(10)
        y_actual = -1.48907
        x_actual = 0.96609
        self.assertLess(torch.norm(bo.x_star - x_actual), 1e-3)
        self.assertLess(torch.norm(bo.y_star - y_actual), 1e-3)

    def test_minimum_ei(self):
        bo = StandardBayesianOptimization(model, likelihood, optimizer, target, search_space, "discrete_ei")
        bo.optimal(10)
        y_actual = -1.48907
        x_actual = 0.96609
        self.assertLess(torch.norm(bo.x_star - x_actual), 1e-3)
        self.assertLess(torch.norm(bo.y_star - y_actual), 1e-3)


if __name__ == '__main__':
    unittest.main()
