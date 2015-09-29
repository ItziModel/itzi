from grass.gunittest.case import TestCase
from grass.gunittest.main import test
import numpy as np
from domain import SurfaceDomain

class TestHFlow(TestCase):

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

    # test method must start with test_
    def test_hf(self):
        """
        """
        dom = SurfaceDomain()
        dom.arrp_z = self.arrp_z
        dom.arrp_h_new = self.arrp_h
        dom.arr_z = self.arr_z
        dom.arr_h_new = self.arr_h
        dom.arr_hfw = np.zeros(shape=self.shape, dtype=self.dtype)
        dom.arr_hfn = np.zeros(shape=self.shape, dtype=self.dtype)

        dom.solve_hflow()
        # boundary
        self.assertEqual(dom.arr_hfw[1,0], 0)
        self.assertEqual(dom.arr_hfn[0,1], 0)
        # WSE == Z
        self.assertEqual(dom.arr_hfw[1,2], 0)
        self.assertEqual(dom.arr_hfn[1,2], 0)
        # middle of the grid
        self.assertEqual(dom.arr_hfw[1,1], 1)
        self.assertEqual(dom.arr_hfn[1,1], 1)
        # W boundary
        self.assertEqual(dom.arr_hfn[1,0], 2)


if __name__ == '__main__':
    test()
