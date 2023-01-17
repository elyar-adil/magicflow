import unittest

import numpy as np

import mf

mf.config(framework='torch')  # noqa
from mf import Tensor


class TestTensor(unittest.TestCase):

    def test_sum(self):
        a = Tensor(np.ones((32, 32)))
        b = Tensor(np.ones((32, 1)))

        self.assertEqual([32, 32], (a + b).shape)
        self.assertTrue(np.allclose(2 * np.ones((32, 32)), (a + b)))

    def test_backward(self):
        a = Tensor(np.ones((32, 32)))

        self.assertEqual(a.__raw_tensor__.requires_grad, False)
        self.assertEqual(a.requires_grad, False)

        a.requires_grad = True
        self.assertEqual(a.__raw_tensor__.requires_grad, True)
        self.assertEqual(a.requires_grad, True)

        a = a.sum()
        a.backward()


if __name__ == '__main__':
    unittest.main()
