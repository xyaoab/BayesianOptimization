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
        self.mean_module = ConstantMean(constant_bounds=[-6, 6])
        # Put a grid interpolation kernel over the RBF kernel
        self.base_covar_module = RBFKernel(log_lengthscale_bounds=(-5, 5))
        self.covar_module = GridInterpolationKernel(self.base_covar_module, grid_size=400,
                                                    grid_bounds=[(-10, 10)])
        # Register kernel lengthscale as parameter
        self.register_parameter('log_outputscale', nn.Parameter(torch.Tensor([0])), bounds=(-5, 5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        covar_x = covar_x.mul(self.log_outputscale.exp())
        return GaussianRandomVariable(mean_x, covar_x)


def target(x):
    return -(torch.cos(2 * x + 1) + 2 * torch.cos(3 * x + 2) + 3 * torch.cos(4 * x + 3) +
             4 * torch.cos(5 * x + 4) + 5 * torch.cos(6 * x + 5))


train_x = Variable(torch.linspace(-10, 10, 6)).cuda()
train_y = Variable(-1 * target(train_x)).cuda()
search_space = [Real(-10, 10)]

likelihood = GaussianLikelihood(log_noise_bounds=(-8, -6)).cuda()
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


class TestBayesOptCos(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_minimum_ucb(self):
        bo = StandardBayesianOptimization(model, likelihood, optimizer, target, search_space, "discrete_ucb", {"kappa": 5})
        bo.optimal(10)
        y_actual = -14.508
        # x_actual = -7.083506
        # self.assertLess(torch.norm(bo.x_star - x_actual), 1e-2)
        self.assertLess(torch.norm(bo.y_star - y_actual), 1e-2)

    def test_minimum_mes(self):
        bo = StandardBayesianOptimization(model, likelihood, optimizer, target, search_space, "discrete_mes", {"nK": 1000})
        bo.optimal(10)
        y_actual = -14.508
        # x_actual = -7.083506
        # self.assertLess(torch.norm(bo.x_star - x_actual), 1e-2)
        self.assertLess(torch.norm(bo.y_star - y_actual), 1e-2)

    def test_minimum_pi(self):
        bo = StandardBayesianOptimization(model, likelihood, optimizer, target, search_space, "discrete_pi")
        bo.optimal(10)
        y_actual = -14.508
        # x_actual = -7.083506
        # self.assertLess(torch.norm(bo.x_star - x_actual), 1e-2)
        self.assertLess(torch.norm(bo.y_star - y_actual), 1e-2)

    def test_minimum_ei(self):
        bo = StandardBayesianOptimization(model, likelihood, optimizer, target, search_space, "discrete_ei")
        bo.optimal(10)
        y_actual = -14.508
        # x_actual = -7.083506
        # self.assertLess(torch.norm(bo.x_star - x_actual), 1e-2)
        self.assertLess(torch.norm(bo.y_star - y_actual), 1e-2)


if __name__ == '__main__':
    unittest.main()
