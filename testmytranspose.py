import unittest
import numpy as np
import pandas as pd
from mytranspose import mytranspose

class TestMyTranspose(unittest.TestCase):

    def test_matrix(self):
        mat = np.array([[1, 2], [3, 4], [5, 6]])
        expected = np.array([[1, 3, 5], [2, 4, 6]])
        result = mytranspose(mat)
        np.testing.assert_array_equal(result, expected)

    def test_dataframe(self):
        d = np.array([1, 2, 3, 4])
        e = np.array(["red", "white", "red", np.nan])
        f = np.array([True, True, True, False])
        df = pd.DataFrame({"d": d, "e": e, "f": f})
        expected = df.transpose()
        result = mytranspose(df)
        pd.testing.assert_frame_equal(result, expected)
