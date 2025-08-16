# -*- coding: utf-8 -*-
"""
Created on Tue May  6 19:06:48 2025

@author: gaurav
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 

class PCA_scratch:
    def __init__(self):
        self.data = None
        self.cov_mat = None
    
    def fit_transform(self, data, n_components):
        self.data = data
        self.sub_mean()
        
        #compute covariance matrix
        self.cov_mat = (1 / len(self.data)) * np.matmul(np.transpose(self.data), self.data)
        
        #compute eigen_values and eigen_vectors of cov matrix
        eigen_values , eigen_vectors = np.linalg.eig(self.cov_mat)
        eigen_values = list(eigen_values)
        
        #sort eigen values 
        for i in range(len(eigen_values)):
            eigen_values[i] = (eigen_values[i], i)
        eigen_values.sort(reverse = True)
        
        #select top eigenvectors corresponding to top eigenvectors
        eigen_vectors_sorted = []
        for i in range(n_components):
            eigen_vectors_sorted.append(list(eigen_vectors[:,eigen_values[i][1]]))
        
        #project data on top (n_components) eigen vectors
        eigen_vectors_sorted = np.transpose(np.array(eigen_vectors_sorted))
        transformed_data = np.matmul(self.data, eigen_vectors_sorted)
        
        return transformed_data
    
    
    def sub_mean(self,):
        self.data = self.data - np.mean(self.data, axis=0)
    



X = np.array([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0],
    [2.3, 2.7],
    [2, 1.6],
    [1, 1.1],
    [1.5, 1.6],
    [1.1, 0.9]
])

# Step 1: Standardize the data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply PCA (reduce to 2 components here, but you can try 1 as well)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


pca_scratch = PCA_scratch()
X_pca_scratch = pca_scratch.fit_transform(X_scaled,2)
#X_pca_scratch = pca_scratch.transform(2)

