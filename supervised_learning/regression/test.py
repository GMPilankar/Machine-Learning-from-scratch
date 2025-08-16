# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 19:49:03 2025

@author: gaura
"""

import unittest
import numpy as np
from sklearn.linear_model import LinearRegression
from linear_regression import LinearReg
from sklearn.datasets import make_regression



class TestCustomLinearRegression(unittest.TestCase):
    
    def setUp(self):
        # Generate random, reproducible test data
        np.random.seed(42)
        self.X_train, self.y_train, self.coef = make_regression(
            n_samples=100, 
            n_features=5, 
            noise=10, 
            random_state=42, 
            coef=True  # Returns the true coefficients of the underlying linear model
        )

        # Instantiate both models
        self.custom_model = LinearReg()
        self.sklearn_model = LinearRegression()

    def test_fit_method(self):
        # Fit both models
        self.custom_model.fit(self.X_train, self.y_train)
        self.sklearn_model.fit(self.X_train, self.y_train)

        # Assert that the learned weights are nearly equal
        np.testing.assert_array_almost_equal(self.custom_model.W, self.sklearn_model.coef_, decimal=5)

        # Assert that the learned intercept is nearly equal
        self.assertAlmostEqual(self.custom_model.b, self.sklearn_model.intercept_, places=5)

    def test_predict_method(self):
        # First, fit both models to have weights for prediction
        self.custom_model.fit(self.X_train, self.y_train)
        self.sklearn_model.fit(self.X_train, self.y_train)

        # Make predictions on the training data
        custom_predictions = self.custom_model.predict(self.X_train)
        sklearn_predictions = self.sklearn_model.predict(self.X_train)

        # Assert that the predictions are nearly equal
        np.testing.assert_array_almost_equal(custom_predictions, sklearn_predictions, decimal=5)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)