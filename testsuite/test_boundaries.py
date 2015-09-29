from grass.gunittest.case import TestCase
from grass.gunittest.main import test
import numpy as np
from boundares import Boundary

class TestBoundary(TestCase):

    @classmethod
    def setUpClass(cls):
        """create test data"""
        cls.shape = (3, 3)
        cls.dtype = np.float32
        arr_z = np.array([[0,1,2], [0,1,2], [0,1,2]], dtype = cls.dtype)
        arr_h = np.array([[2,1,0], [2,1,0], [2,1,0]], dtype = cls.dtype)
        cls.arrp_z = np.pad(arr_z, 1, 'edge')
        cls.arrp_h = np.pad(arr_h, 1, 'edge')
        cls.arr_z = cls.arrp_z[1:-1,1:-1]
        cls.arr_h = cls.arrp_h[1:-1,1:-1]

    @classmethod
    def tearDownClass(cls):
        pass

    def test_type1(self):
        """
        """
        pass

    def test_type2(self):
        """
        """
        pass

    def test_type3(self):
        """
        """
        pass

if __name__ == '__main__':
    test()
