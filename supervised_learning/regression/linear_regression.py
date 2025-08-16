# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 19:27:29 2025

@author: gaura
"""

import numpy as np


class LinearReg:
    def __init__(self,):
        self.W = None
        self.b = None
        
    
    def fit(self, X, Y):
        '''
        X (shape: n_samples , n_features)
        Y ( shape : n_samples)
        '''
        X_new = np.hstack( (np.ones( (X.shape[0], 1) ), X ) )
        self.W = np.linalg.pinv(X_new) @ Y
        self.b = self.W[0]
        self.W = self.W[1:]
    
    
    def predict(self, X):
        return (X @ self.W) + self.b
    
    


        