# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 14:51:00 2025

@author: gaura
"""

import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.preprocessing import StandardScaler


class TestScaler(unittest.TestCase):

    def setUp(self):
        # Example dataset
        self.X = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0]
        ])
    
    def test_fit_mode0(self):
        from scaler import Scaler
        scaler = Scaler(mode=0)
        sk_scaler = StandardScaler()
        scaler.fit(self.X)
        sk_scaler.fit(self.X)
        expected_mean = sk_scaler.mean_
        expected_var = sk_scaler.var_

        assert_almost_equal(scaler.mean, expected_mean)
        assert_almost_equal(scaler.var, expected_var)

    def test_transform_mode0(self):
        from scaler import Scaler
        scaler = Scaler(mode=0)
        sk_scaler = StandardScaler()
        
        scaler.fit(self.X)
        X_scaled = scaler.transform(self.X)
        
        sk_scaler.fit(self.X)
        sk_X_scaled = sk_scaler.transform(self.X)
        
        # scaled data should have mean 0 and std 1
        assert_almost_equal(X_scaled, sk_X_scaled)
        


    def test_fit_transform_consistency(self):
        from scaler import Scaler
        scaler = Scaler(mode=0)
        sk_scaler = StandardScaler()
        
        
        X_scaled = scaler.fit_transform(self.X)
        sk_X_scaled = sk_scaler.fit_transform(self.X)
        
        assert_almost_equal(X_scaled, sk_X_scaled)

    def test_inverse_transform_mode0(self):
        from scaler import Scaler
        scaler = Scaler(mode=0)
        X_scaled = scaler.fit_transform(self.X)
        X_recovered = scaler.inverse_transform(X_scaled)
        assert_almost_equal(X_recovered, self.X)

   


if __name__ == "__main__":
    unittest.main()