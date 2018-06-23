from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from gpytorch.module import Module
from gpytorch.likelihoods import GaussianLikelihood
import torch
import gpytorch
from torch.autograd import Variable

from .discrete_mes import DiscreteMES
from .discrete_ei import DiscreteEI
from .discrete_ucb import DiscreteUCB
from .discrete_pi import DiscretePI
from .dimension import Dimension
from .bayesian_optimization import BayesianOptimization

class StandardBayesianOptimization(BayesianOptimization):
    def __init__(self, GPModel,
                 likelihood,
                 optimizer,
                 target,
                 search_space,
                 acq_func="discrete_mes",
                 acq_func_kwargs=None,
                 sub_samples=False):

        if not isinstance(GPModel, Module):
            raise RuntimeError("BayesianOptimization can only handle Module")

        for param in GPModel.parameters():
            if param.grad is not None and param.grad.norm().item() == 0:
                raise RuntimeError("Model is not trained")

        if not isinstance(likelihood, GaussianLikelihood):
            raise RuntimeError("BayesianOptimization can only handle GaussianLikelihood")

        allowed_acq_funcs = ["discrete_mes", "discrete_ei", "discrete_ucb", "discrete_pi"]
        if acq_func not in allowed_acq_funcs:
            raise RuntimeError("expected acq_func to be in %s, got %s" %
                               (",".join(allowed_acq_funcs), acq_func))

        super(BayesianOptimization, self).__init__()
        self.model = GPModel
        self.likelihood = likelihood
        self.optimizer = optimizer
        self.acq_func = acq_func
        if acq_func_kwargs is None:
            acq_func_kwargs = dict()
        self.acq_func_kwargs = acq_func_kwargs
        # negative wrapper to maximize the target function
        self.function = lambda x: -1 * target(x)
        if sub_samples:
            self._x_samples = Dimension(search_space).get_subset_samples().cuda()
        else:
            self._x_samples = Dimension(search_space).get_samples().cuda()
        self._y_samples = Variable(self.function(self._x_samples)).cuda()

    def update_model(self, next_point):
        train_x = Variable(torch.cat((self.model.train_inputs[0], next_point))).cuda()
        train_targets = Variable(torch.cat((self.model.train_targets, self.function(next_point).view(-1)))).cuda()
        train_inputs = tuple(tri.unsqueeze(-1) if tri.ndimension() == 1 else tri for tri in (train_x,))
        self.model.set_train_data(train_inputs, train_targets, strict=False)

        self.model.train()
        self.likelihood.train()

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        training_iter = 10
        for i in range(training_iter):
            # Zero gradients from previous iteration
            self.optimizer.zero_grad()
            # Output from model
            output = self.model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_targets)
            loss.backward()
            self.optimizer.step()

    def step(self):
        if self.acq_func == "discrete_mes":
            nK = self.acq_func_kwargs.get("nK", 500)
            self.acq_func = DiscreteMES(self.model, nK)
        elif self.acq_func == "discrete_ei":
            self.acq_func = DiscreteEI(self.model)
        elif self.acq_func == "discrete_pi":
            self.acq_func = DiscretePI(self.model)
        elif self.acq_func == "discrete_ucb":
            kappa = self.acq_func_kwargs.get("kappa", 5)
            self.acq_func = DiscreteUCB(self.model, kappa)
        # next point to query
        self.model.eval()
        self.likelihood.eval()
        acq, next_point, observed_pred = self.acq_func(self._x_samples)

        return next_point

    def optimal(self, n_calls):
        for _ in range(n_calls):
            next_point = self.step()
            self.update_model(next_point)

    @property
    def x_star(self):
        return self._x_samples[torch.argmax(self._y_samples)].view(-1)

    @property
    def y_star(self):
        return -1 * torch.max(self._y_samples)

    @property
    def x_samples(self):
        return self._x_samples

    @property
    def y_samples(self):
        return -1 * self._y_samples
