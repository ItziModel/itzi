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
from boundary import Boundary

class TestBoundary(TestCase):

    @classmethod
    def setUpClass(cls):
        """create test data"""
        # instantiate the class to be tested
        cls.bound = Boundary(1, 1, 'E')

        cls.shape = (3)
        cls.celldim = 1
        cls.dtype = np.float32
        cls.arr_z = np.zeros(shape=cls.shape, dtype = cls.dtype)
        cls.n = np.full(cls.shape, 0.03, dtype = cls.dtype)


    @classmethod
    def tearDownClass(cls):
        pass

    def test_init(self):
        """Test if the initialization is done properly
        """
        self.assertEqual(self.bound.pos, 'E')
        self.assertEqual(self.bound.postype, 'downstream')

    def test_closed_boundary(self):
        """
        """
        qin = np.ones(shape=self.shape, dtype = self.dtype)
        hflow = np.ones(shape=self.shape, dtype = self.dtype)
        depth = np.ones(shape=self.shape, dtype = self.dtype)
        qres = np.ones(shape=self.shape, dtype = self.dtype)
        bctype = np.array([0,1,10], dtype = self.dtype)
        bcvalue = np.ones(shape=self.shape, dtype = self.dtype)
        n = self.n

        # copy of input arrays for before/after test
        qin_check = np.copy(qin)
        hflow_check = np.copy(hflow)
        depth_check = np.copy(depth)
        qres_check = np.copy(qres)
        bctype_check = np.copy(bctype)
        bcvalue_check = np.copy(bcvalue)
        n_check = np.copy(n)

        # set boundary conditions
        self.bound.get_boundary_flow(qin, qres, hflow, n, self.arr_z,
                    depth, bctype, bcvalue)

        # check if none of the input arrays changed (appart from qres type 1)
        np.testing.assert_array_equal(qin, qin_check)
        np.testing.assert_array_equal(hflow, hflow_check)
        np.testing.assert_array_equal(depth, depth_check)
        np.testing.assert_array_equal(bctype, bctype_check)
        np.testing.assert_array_equal(bcvalue, bcvalue_check)
        np.testing.assert_array_equal(qres[0], qres_check[0])
        np.testing.assert_array_equal(qres[2], qres_check[2])
        # check if the relevant value is well set to zero
        np.testing.assert_array_equal(qres[1], 0)

    def test_open_boundary(self):
        """
        """
        qin = np.ones(shape=self.shape, dtype = self.dtype)
        hflow = np.ones(shape=self.shape, dtype = self.dtype)
        depth = np.ones(shape=self.shape, dtype = self.dtype)
        qres = np.ones(shape=self.shape, dtype = self.dtype)
        bctype = np.array([2,0,10], dtype = self.dtype)
        bcvalue = np.ones(shape=self.shape, dtype = self.dtype)
        n = self.n
        hflow[0] = 1.5
        depth[0] = 1.2
        # copy of input arrays for before/after test
        qin_check = np.copy(qin)
        hflow_check = np.copy(hflow)
        depth_check = np.copy(depth)
        qres_check = np.copy(qres)
        bctype_check = np.copy(bctype)
        bcvalue_check = np.copy(bcvalue)
        n_check = np.copy(n)

        # set boundary conditions
        self.bound.get_boundary_flow(qin, qres, hflow, n, self.arr_z,
                    depth, bctype, bcvalue)

        # check if none of the input arrays changed (appart from qres type 1)
        np.testing.assert_array_equal(qin, qin_check)
        np.testing.assert_array_equal(hflow, hflow_check)
        np.testing.assert_array_equal(depth, depth_check)
        np.testing.assert_array_equal(bctype, bctype_check)
        np.testing.assert_array_equal(n, n_check)
        np.testing.assert_array_equal(bcvalue, bcvalue_check)
        np.testing.assert_array_equal(qres[1:], qres_check[1:])
        # check if the relevant value is correctly set
        np.testing.assert_approx_equal(qres[0], 0.8)


    def test_user_wse(self):
        """
        """
        qin = np.zeros(shape=self.shape, dtype = self.dtype)
        hflow = np.zeros(shape=self.shape, dtype = self.dtype)
        depth = np.zeros(shape=self.shape, dtype = self.dtype)
        qres = np.ones(shape=self.shape, dtype = self.dtype)
        bctype = np.array([0,10,3], dtype = self.dtype)
        bcvalue = np.full(self.shape, 0.2, dtype = self.dtype)
        n = self.n
        # copy of input arrays for before/after test
        qin_check = np.copy(qin)
        hflow_check = np.copy(hflow)
        depth_check = np.copy(depth)
        qres_check = np.copy(qres)
        bctype_check = np.copy(bctype)
        bcvalue_check = np.copy(bcvalue)
        n_check = np.copy(n)

        # set boundary conditions
        self.bound.get_boundary_flow(qin, qres, hflow, n, self.arr_z,
                    depth, bctype, bcvalue)

        # check if none of the input arrays changed (appart from qres type 1)
        np.testing.assert_array_equal(qin, qin_check)
        np.testing.assert_array_equal(hflow, hflow_check)
        np.testing.assert_array_equal(depth, depth_check)
        np.testing.assert_array_equal(bctype, bctype_check)
        np.testing.assert_array_equal(n, n_check)
        np.testing.assert_array_equal(bcvalue, bcvalue_check)
        np.testing.assert_array_equal(qres[:-1], qres_check[:-1])
        # check if the relevant value is correctly set
        np.testing.assert_approx_equal(qres[2], -1.01963, significant=5)

        # case of W or N boundary
        self.bound.postype = 'upstream'
        self.bound.get_boundary_flow(qin, qres, hflow, n, self.arr_z,
                    depth, bctype, bcvalue)
        np.testing.assert_approx_equal(qres[2], 1.01963, significant=5)

if __name__ == '__main__':
    test()
