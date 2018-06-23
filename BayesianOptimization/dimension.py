from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import itertools
from torch.autograd import Variable
import random


class Space():

    def get_samples(self):
        raise NotImplementedError


# upper_bound inclusive
class Real(Space):
    def __init__(self, lower, upper, sampling="linear", steps=1000):
        if upper <= lower:
            raise RuntimeError("the lower bound {} has to be less than the"
                               "upper bound {}".format(lower, upper))
        super(Real, self).__init__()
        self.lower = lower
        self.upper = upper
        self.sampling = sampling
        self.steps = steps
        
    def get_samples(self, steps=None):
        if steps == None:
            steps = self.steps
        if self.sampling == "linear":
            candidate_set = torch.linspace(self.lower, self.upper, steps)
        elif self.sampling == "log_uniform":
            candidate_set = torch.logspace(self.lower, self.upper, steps)
        else:
            raise ValueError("Sampling can only handle linear or log_uniform")
        return candidate_set


# upper_bound inclusive
class Integer(Space):
    def __init__(self, lower, upper):
        if upper <= lower:
            raise RuntimeError("the lower bound {} has to be less than the"
                               "upper bound {}".format(lower, upper))
        super(Integer, self).__init__()
        self.lower = lower
        self.upper = upper
        self.steps = self.upper - self.lower + 1

    def get_samples(self, steps=None):
        if steps == None:
            steps = self.steps
        candidate_set = torch.range(self.lower, self.upper, (self.upper - self.lower + 1)/ steps)
        return candidate_set


class Categorical(Space):
    def __init__(self, space):
        if len(space) < 1:
            raise RuntimeError("the number of class has to be greater than 0")
        super(Categorical, self).__init__()
        self.space = space
        self.steps = len(space)

    def get_samples(self, steps=None):
        if steps == None:
            steps = self.steps
        candidate_set = torch.Tensor(random.sample(self.space, steps))
        return candidate_set


class Dimension():
    def __init__(self, dimensions):
        if dimensions is None:
            raise RuntimeError("dimensions can't be none")
        self.dimensions = dimensions

    def get_samples(self):      
        candidate_set = []
        D = []
        for dimension in self.dimensions:
            D.append(dimension.get_samples())
        for e in itertools.product(*D):
            e = torch.stack(e)
            candidate_set.append(e)
        candidate_set = Variable(torch.stack(candidate_set)).cuda()
        return candidate_set

    def get_subset_samples(self):
        candidate_set = []
        D = []
        for dimension in self.dimensions:
            if dimension.steps / len(self.dimensions) > 1:
                steps = int(dimension.steps/ len(self.dimensions))
            else:
                steps = dimension.steps
            D.append(dimension.get_samples(steps))

        for e in itertools.product(*D):
            e = torch.stack(e)
            candidate_set.append(e)
        candidate_set = Variable(torch.stack(candidate_set)).cuda()
        return candidate_set
