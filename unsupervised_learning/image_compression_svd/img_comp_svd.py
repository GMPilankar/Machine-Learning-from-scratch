# -*- coding: utf-8 -*-
"""
Created on Tue May 20 19:42:19 2025

@author: gaura
"""

import cv2
import numpy as np


im = cv2.imread("Lenna.png", 0)
cv2.imshow('original_image', im)
cv2.waitKey(0)
cv2.destroyAllWindows()

U, S, Vt = np.linalg.svd(im)
print(U.shape)
print(S.shape)
print(Vt.shape)

n = 40

print(U[:,:n].shape)
print(np.diag(S[:n]).shape)
print(Vt[:n,:].shape)
compr_im = np.matmul(U[:,:n], np.matmul(np.diag(S[:n]), Vt[:n,:]))
#compr_im = U[:,:n] * (np.diag(S[:n]) * Vt[:n,:])
#compr_im = np.clip(compr_im, 1, 255)
compr_im = compr_im.astype(np.uint8)
#compr_im = U[:,:n] * np.diag(S[:n]) * Vt[:n,:]
print(compr_im.shape)
cv2.imshow('compr_image', compr_im)
cv2.waitKey(0)
cv2.destroyAllWindows()

