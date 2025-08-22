# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 01:34:02 2025

@author: gaura
"""

import numpy as np




class Logistic_regression:
    def __init__(self, max_iter=10):
        self.max_iter = max_iter
        self.theta = None
        self.bias = None
        self.lr = 0.1
        self.threshold = 0.5
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, Y):
        Y = Y.reshape((-1,1))
        #print(X.shape)
        X = np.hstack((np.ones((X.shape[0],1)), X))
        #print(X.shape)
        self.theta = np.ones((X.shape[1], 1))
        for epoch in range(self.max_iter):
            y_hat = self.sigmoid(X @ self.theta)
            #print(np.transpose(self.lr * np.sum((Y-y_hat) * X , axis=0)).shape)
            self.theta += np.transpose(self.lr * np.sum((Y-y_hat) * X , axis=0)).reshape((-1,1))
        self.bias = self.theta[0,0]
        self.theta = self.theta[1:]
        
    def predict(self, X):
        y_ = self.sigmoid((X @ self.theta) + self.bias)
        return np.where(y_ >= self.threshold, 1 ,0)


