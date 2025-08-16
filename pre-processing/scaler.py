# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 12:29:05 2025

@author: gaura
"""

import numpy as np


class Scaler:
    def __init__(self, mode = 0):
        """
        mode 0 : make mean zero and variance 1
        mode 1 : only make mean 0 dont alter variance
        """
        self.mode = mode
        self.mean = None
        self.std = None
        self.var = None
        
    
    
    def fit(self, X):
        self.X = X
        self.mean = np.mean(self.X, axis = 0)
        self.std = np.std(self.X, axis=0)
        self.var = self.std ** 2
        
    
    def transform(self, X):
        if self.mode:
            return X - self.mean
        else:
            X = X - self.mean
            X = X / self.std
            return X
    
    def fit_transform(self, X):
        self.X = X
        self.mean = np.mean(self.X, axis=0)
        self.std = np.std(self.X, axis = 0)
        self.var = self.std ** 2
        return self.transform(X)
    
    def inverse_transform(self, X):
        if self.mode:
            X = X + self.mean
        else:
            X = X * self.std
            X = X + self.mean
        return X


if __name__ == '__main__':
    sc = Scaler()
    X = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0]
    ])
    
    print("Original Data:\n", X)
    
    
    X_scaled = sc.fit_transform(X)
    
    print("\nScaled Data:\n", X_scaled)
    print("\nFeature-wise Mean:", sc.mean)
    print("Feature-wise Variance:", sc.var)
    
    
    X_original = sc.inverse_transform(X_scaled)
    print("\nRecovered Original Data:\n", X_original)



