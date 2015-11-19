import numpy as np
import operator
import os
import matplotlib.pyplot as plt
import pandas as pd

from numpy import *
from os import listdir
from numpy.linalg import *
from scipy.stats.stats import pearsonr
from scipy.sparse.linalg import eigs
from numpy import abs, array, average, corrcoef, mat, shape, std, sum, transpose, zeros
from numpy.linalg import svd



#Preprocessing
#normalize and remove mean
#data = mat(log10(data[:,:4]))
#N = 2



#PCA: an orthogonal linear transformation that transforms the data to a new coordinate system such that the greatest variance by any projection of the data comes to lie on the first coordinate

#PCA (TL;DR version): "machine to summarize information"
#1) Normalize and remove mean
#2) Calculate covariance matrix
#3) Calculate eigenvalues and sort it
#4) transform data into new dimensions
#5) complete

#PCA for summarizing into lower dimensions
def PCA(data_4d, N):
    """
    Paramters
    --------
    data_4d: 4-dimensional data (array)
    N: how many dimensions we want to reduce to
    
    Returns
    -------
    newdata: new data sorted in the newest N dimension
    """
    
    means = mean(data, axis = 0)
    data_4d = data_4d-means
    
    #Calculating covariance matrix
    covar = cov(data_4d, rowvar = 0)
    
    #eigvalues
    eigvalues, eigvectors = eigs(cov)
    
    #sorting from descending order
    index = argsort(eigvalues)
    index = index[:-(N+1):-1]
    eigvector_sorted = eigvectors[:,index]
    
    #summary of transformation to lower dimensions
    newdata = data_4d * eigvector_sorted
    
    return newdata
    
#SVD: data summary method
#-extracts important features from data
#-reconstructs original dataset into smaller dataset (can be used for image compression)
    
#SVD (TL;DR) version
#1) Calculate SVD
#2) Decide how many Singular Values 'S' you want to keep
#3) Take out columns more than S of U

#normalize and remove mean
#data_4d = mat(data[:,:4])

def svd(data_4d, S = 2):
    """
    Parameters
    ---------
    data_4d: 4-dimensional data (array)
    
    Outputs
    ------
    SVD of the dataset
        
    """
    #calculate SVD
    U, s, V = linalg.svd(data_4d)
    sig = mat(eye(S)*s[:S])
    #removing columns that you don't need
    newdata2 = U[:,:S]
    svd(data_4d, 2)
    return newdata2
  