import numpy as numpy
import matplotlib.pyplot as plt 
import numpy.linalg as npl


def pca_1(data, dims_rescaled_data = 2):
	"""
	Returns a data transformed in 2 dims/columns and our
	original data

	Parameters
	----------
	pass in: data in 2D numpy array

	Returns
	-------

	"""
	m, n = data.shape
	#mean of the center of data
	data -= data.mean(axis = 0) 
	#Covariance
	R = np.cov(data, rowvar = False) 
	#eigenvectors/eigenvalues of covariance matrix
	evals, evecs = npl.eigh(R) 
	index = np.argsort(evals)[::-1]
	evecs = evals[:,index]
	#sorting by similar index
	evals = evals[index]
	#transforming data with eigenvectors, returning
	evecs = evecs[:, :dims_rescaled_data] 
	return np.dot(evecs.T, data.T).T, eigenvalues, eigenvectors

def pca_2((data, pc_count = None):
	"""
	Principal component analysis using eigenvalues
	This method centers the mean and auto-scales the data

	Parameters
	----------
	passes in: data in 2D numpy array
	"""
