import unittest

import numpy as np

import mf

mf.config(framework='torch')  # noqa
from mf import Tensor
from mf import Linear


class TestLinear(unittest.TestCase):

    def test_shape(self):
        a = Tensor(np.ones((32, 32)).astype("float32"))
        fc = Linear(16)
        b = fc(a)
        self.assertEqual([32, 16], b.shape)


if __name__ == '__main__':
    unittest.main()
