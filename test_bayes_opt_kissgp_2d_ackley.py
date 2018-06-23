from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import unittest
import math
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
        self.mean_module = ConstantMean(constant_bounds=[-5,5])
        self.base_covar_module = RBFKernel(log_lengthscale_bounds=(-5, 6))
        self.covar_module = GridInterpolationKernel(self.base_covar_module, grid_size=500,
                                                    grid_bounds=[(-10, 10), (-10, 10)])
        self.register_parameter('log_outputscale', nn.Parameter(torch.Tensor([0])), bounds=(-5,6))
        
    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        covar_x = covar_x.mul(self.log_outputscale.exp())
        return GaussianRandomVariable(mean_x, covar_x)


def target(x):
    return -20 * torch.exp(-0.2 * torch.sqrt(0.5* torch.sum(x**2, dim = 1))) - torch.exp(.5 * torch.sum(torch.cos(2 * math.pi * x), dim = 1)) + 20 + torch.exp(torch.ones(x.size(0)).cuda())

x = torch.linspace(-10,10,5)
y =  torch.linspace(-10,10,5)
train_x = Variable(torch.stack([x.repeat(y.size(0)), y.repeat(x.size(0),1).t().contiguous().view(-1)],1)).cuda()
train_y = Variable(-1*target(train_x)).cuda()

likelihood = GaussianLikelihood(log_noise_bounds=(-8, -7)).cuda()
model = GPRegressionModel(train_x.data, train_y.data, likelihood).cuda()
search_space = [Real(-10,10,steps=300),Real(-10,10,steps=300)]

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


class TestBayesOptAckley(unittest.TestCase):
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
        y_actual = torch.zeros((1,2)).cuda()
        x_actual = 0
        print(torch.norm(bo.x_star - x_actual),torch.norm(bo.y_star - y_actual) )
        self.assertLess(torch.norm(bo.x_star - x_actual), 1e-1)
        self.assertLess(torch.norm(bo.y_star - y_actual), 3e-1)

    def test_minimum_mes(self):
        bo = StandardBayesianOptimization(model, likelihood, optimizer, target, search_space, "discrete_mes", {"nK": 1000})
        bo.optimal(10)
        y_actual = torch.zeros((1,2)).cuda()
        x_actual = 0
        self.assertLess(torch.norm(bo.x_star - x_actual), 1e-1)
        self.assertLess(torch.norm(bo.y_star - y_actual), 3e-1)

    def test_minimum_pi(self):
        bo = StandardBayesianOptimization(model, likelihood, optimizer, target, search_space, "discrete_pi")
        bo.optimal(10)
        y_actual = torch.zeros((1,2)).cuda()
        x_actual = 0
        self.assertLess(torch.norm(bo.x_star - x_actual), 1e-1)
        self.assertLess(torch.norm(bo.y_star - y_actual), 3e-1)

    def test_minimum_ei(self):
        bo = StandardBayesianOptimization(model, likelihood, optimizer, target, search_space, "discrete_ei")
        bo.optimal(10)
        y_actual = torch.zeros((1,2)).cuda()
        x_actual = 0
        self.assertLess(torch.norm(bo.x_star - x_actual), 1e-1)
        self.assertLess(torch.norm(bo.y_star - y_actual), 3e-1)


if __name__ == '__main__':
    unittest.main()
