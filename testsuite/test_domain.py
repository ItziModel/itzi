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
        cls.arr_z = np.array([[0,1,2], [0,1,2], [0,1,2]], dtype = cls.dtype)
        cls.arr_h = np.array([[2,1,0], [2,1,0], [2,1,0]], dtype = cls.dtype)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_hf(self):
        """
        """
        dom = SurfaceDomain()
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


class TestH(TestCase):

    @classmethod
    def setUpClass(cls):
        """create test data"""
        cls.shape = (3, 3)
        cls.dtype = np.float32

        cls.dom = SurfaceDomain()
        cls.dom.arr_h_old = np.ones(shape=cls.shape, dtype = cls.dtype)
        cls.dom.arr_h_new = np.copy(cls.dom.arr_h_old)
        cls.dom.arr_ext = np.zeros(shape=cls.shape, dtype = cls.dtype)
        cls.dom.dt = cls.dom.cell_surf = 1

    @classmethod
    def tearDownClass(cls):
        pass

    def test_h_q_uniform(self):
        """
        """
        self.dom.arr_qn = np.ones(shape=self.shape, dtype = self.dtype)
        self.dom.arr_qw = np.ones(shape=self.shape, dtype = self.dtype)
        self.dom.arrp_qn = np.pad(self.dom.arr_qn, 1, mode='edge')
        self.dom.arrp_qw = np.pad(self.dom.arr_qw, 1, mode='edge')

        self.dom.solve_h()

        # middle of the grid
        self.assertEqual(self.dom.arr_h_old[1,1], self.dom.arr_h_new[1,1])

    def test_h_xq_var(self):
        """
        """
        self.dom.arr_qn = np.ones(shape=self.shape, dtype = self.dtype)
        self.dom.arr_qw = np.ones(shape=self.shape, dtype = self.dtype)
        self.dom.arrp_qw = np.pad(self.dom.arr_qn, 1, mode='edge')
        self.dom.arrp_qn = np.pad(self.dom.arr_qw, 1, mode='edge')
        # set a superior inflow at the grid center
        self.dom.arr_qw[1,1] = 2

        self.dom.solve_h()

        # middle of the grid
        self.assertEqual(self.dom.arr_h_old[1,1] + 1, self.dom.arr_h_new[1,1])

    def test_h_yq_var(self):
        """
        """
        self.dom.arr_qn = np.ones(shape=self.shape, dtype = self.dtype)
        self.dom.arr_qw = np.ones(shape=self.shape, dtype = self.dtype)
        self.dom.arrp_qw = np.pad(self.dom.arr_qn, 1, mode='edge')
        self.dom.arrp_qn = np.pad(self.dom.arr_qw, 1, mode='edge')
        # set a superior inflow at the grid center
        self.dom.arr_qn[1,1] = 2

        self.dom.solve_h()

        # middle of the grid
        self.assertEqual(self.dom.arr_h_old[1,1] + 1, self.dom.arr_h_new[1,1])

if __name__ == '__main__':
    test()
