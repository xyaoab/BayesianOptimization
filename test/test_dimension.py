from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch
import unittest
from torch.autograd import Variable
from BayesianOptimization.dimension import Real, Integer, Dimension, Categorical


class TestDimension(unittest.TestCase):

    def test_real_bound(self):
        self.assertRaises(RuntimeError, Real, 2., -2.)

    def test_real_linear(self):
        a = Real(-2., 2., "linear", 9).get_samples()
        actual = torch.tensor([-2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2.])
        self.assertLess(torch.norm(a - actual), 1e-7)

    def test_real_log_uniform(self):
        a = Real(-2., 2., "log_uniform", 5).get_samples()
        actual = torch.tensor([1e-2, 1e-1, 1, 1e1, 1e2])
        self.assertLess(torch.norm(a - actual), 1e-7)

    def test_integer_bound(self):
        self.assertRaises(RuntimeError, Integer, 2, -2)

    def test_integer_step(self):
        a = Integer(-2, 2).get_samples()
        actual = torch.tensor([-2., -1., 0., 1., 2.])
        self.assertLess(torch.norm(a.data - actual), 1e-7)

    def test_categorical_bound(self):
        self.assertRaises(RuntimeError, Categorical, [])

    def test_categorical_step(self):
        a = Categorical([-1, 3, -5]).get_samples()
        actual = torch.tensor([-5., -1., 3.])
        self.assertLess(torch.norm(torch.sort(a)[0] - actual), 1e-7)

    def test_dimension_get_samples(self):
        a = Dimension([Real(-1., 1., "linear", 3), Integer(4, 5), Categorical([-2])]).get_samples()
        actual = Variable(torch.tensor([[-1., 4., -2.], [-1., 5., -2.],
                                        [0., 4., -2.], [0., 5., -2.],
                                        [1., 4., -2.], [1., 5., -2.]
                                        ])).cuda()
        self.assertLess(torch.norm(a - actual), 1e-7)

    def test_dimension_get_subset_samples_real(self):
        a = Dimension([Real(-1., 1., "linear", 20), Real(-1., 1., "linear", 20)]).get_subset_samples()
        actual = 100
        self.assertEqual(a.size(0) - actual, 0)

    def test_dimension_get_subset_samples_integer(self):
        a = Dimension([Integer(-10, 9), Integer(-10, 9)]).get_subset_samples()
        actual = 100
        self.assertEqual(a.size(0) - actual, 0)
        
    # def test_dimension_get_subset_samples_categorical(self):


if __name__ == '__main__':
    unittest.main()
