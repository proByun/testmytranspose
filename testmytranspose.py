import unittest
import numpy as np
import pandas as pd
import torch
from mytranspose import mytranspose

class TestMyTranspose(unittest.TestCase):

    def test_matrix(self):
        mat = np.array([[1, 2], [3, 4], [5, 6]])
        expected = np.array([[1, 3, 5], [2, 4, 6]])
        result = mytranspose(mat)
        np.testing.assert_array_equal(result, expected)

if __name__ == "__main__":
    unittest.main()
