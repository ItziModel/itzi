# coding=utf8
"""
Copyright (C) 2015  Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""
from __future__ import division
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

    #~ def test_hf(self):
        #~ """
        #~ """
        #~ arr_def = np.zeros(shape=self.shape, dtype=self.dtype)
        #~ arr_h = np.zeros(shape=self.shape, dtype=self.dtype)
        #~ dom = SurfaceDomain(dx=1, dy=1, arr_def=arr_def, arr_h=arr_h)
        #~ dom.arr_z = self.arr_z
        #~ dom.arr_h_new = self.arr_h
        #~ dom.arr_hfw = np.zeros(shape=self.shape, dtype=self.dtype)
        #~ dom.arr_hfn = np.zeros(shape=self.shape, dtype=self.dtype)
#~ 
        #~ dom.solve_hflow()
        #~ # boundary
        #~ self.assertEqual(dom.arr_hfw[1,0], 0)
        #~ self.assertEqual(dom.arr_hfn[0,1], 0)
        #~ # WSE == Z
        #~ self.assertEqual(dom.arr_hfw[1,2], 0)
        #~ self.assertEqual(dom.arr_hfn[1,2], 0)
        #~ # middle of the grid
        #~ self.assertEqual(dom.arr_hfw[1,1], 1)
        #~ self.assertEqual(dom.arr_hfn[1,1], 1)
        #~ # W boundary
        #~ self.assertEqual(dom.arr_hfn[1,0], 2)


class TestH(TestCase):

    @classmethod
    def setUpClass(cls):
        """create test data"""
        cls.shape = (3, 3)
        cls.dtype = np.float32

        arr_def = np.zeros(shape=cls.shape, dtype=cls.dtype)
        arr_h = np.zeros(shape=cls.shape, dtype=cls.dtype)
        cls.dom = SurfaceDomain(dx=1, dy=1, arr_def=arr_def, arr_h=arr_h)

        cls.dom.arr_h_old = np.ones(shape=cls.shape, dtype = cls.dtype)
        cls.dom.arr_h_new = np.copy(cls.dom.arr_h_old)
        cls.dom.arr_ext = np.zeros(shape=cls.shape, dtype = cls.dtype)
        cls.dom.dt = cls.dom.cell_surf = 1
        cls.dom.arr_qn = np.ones(shape=cls.shape, dtype = cls.dtype)
        cls.dom.arr_qw = np.ones(shape=cls.shape, dtype = cls.dtype)
        cls.dom.arrp_qn = np.pad(cls.dom.arr_qn, 1, mode='edge')
        cls.dom.arrp_qw = np.pad(cls.dom.arr_qw, 1, mode='edge')

    @classmethod
    def tearDownClass(cls):
        pass

    def test_h_q_uniform(self):
        """
        """
        self.dom.solve_h()

        # middle of the grid
        self.assertEqual(self.dom.arr_h_old[1,1], self.dom.arr_h_new[1,1])

    def test_h_xq_var(self):
        """
        """
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


class TestFlow(TestCase):

    @classmethod
    def setUpClass(cls):
        """create test data"""
        cls.shape = (3, 3)
        cls.dtype = np.float32
        arr_def = np.zeros(shape=cls.shape, dtype=cls.dtype)
        arr_h = np.zeros(shape=cls.shape, dtype=cls.dtype)
        cls.dom = SurfaceDomain(dx=1, dy=1, arr_def=arr_def, arr_h=arr_h)


    @classmethod
    def tearDownClass(cls):
        pass

    def test_bates2010(self):
        """
        """
        self.dom.dt = 2.3
        slope = .1
        hf = .2
        q0 = 2
        n = 0.03
        length = 1
        width = 1
        wse_up = 0.08
        wse0 = 0.1
        # length, width, wse_0, wse_up, hf, q0, n
        q_result = self.dom.bates2010(length, width, wse0, wse_up, hf, q0, n)

        np.testing.assert_approx_equal(q_result, 0.6981, significant=4)


if __name__ == '__main__':
    test()
