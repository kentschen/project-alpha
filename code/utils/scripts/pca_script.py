"""Script for PCA tests function. 
 Run with 
     python PCA_script.py 
     
 in the scripts directory
"""

import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import nibabel as nib
import os
import sys

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

# Relative path to subject 1 data
pathtodata = "../../../data/ds009/sub001/"
condition_location=pathtodata+"model/model001/onsets/task001_run001/"
location_of_images="../../../images/"

#this doesn't work on my laptop 
sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))
#this works though
sys.path.append("../functions/")

# Load the image data for subject 1.
img = nib.load(pathtodata+"BOLD/task001_run001/bold.nii.gz")
data = img.get_data()
data = data[...,6:] # Knock off the first 6 observations.

cond1=np.loadtxt(condition_location+"cond001.txt")
cond2=np.loadtxt(condition_location+"cond002.txt")
cond3=np.loadtxt(condition_location+"cond003.txt")


#######################################
###Let's look at cond001

rx = cond1[:,0] #Taking out the column
ry = cond1[:,1]

#Calculating the average
averx = np.average(rx)
avery = np.average(ry)

#Convert r vector to vector x (to center the origin)
xx = rx - averx
xy = ry - avery

#Calculate the variance-covariance matrix
V = np.zeros((2,2))
#Achieve sigma calculated by the inner product of the transposed vector and vector
#Do I need to divide by some number here???
V[0][0] = np.dot(rx-averx,rx-averx.T) 
V[0][1] = np.dot(rx-averx,ry-avery.T) 
V[1][0] = np.dot(ry-avery,rx-averx.T) 
V[1][1] = np.dot(ry-avery,ry-avery.T) 

#Eigenvalues, eigenvectors of the variance-covariance matrix
la, u = np.linalg.eig(V)
print(averx)
print(avery)
print(V)
print(la)
print(u)


#################################
###Finding principal components with SVD
X = data
unscaled_cov = X.dot(X.T)
#V should be components in rows of returned matrix
U, S, V = npl.svd(unscaled_cov)


###Sum of squares and variance from PCA
unscaled_cov = X.dot(X.T)
print(unscaled_cov)

#When divided by N-1, should have same result as "np.cov"
N = X.shape[1]
np.allclose(unsaled_cov / (N-1), np.cov(X))

#Variances = Sum of squars divided by correction factor
variances = (S / (N-1))
print(variances)


#################################
#some plotting
fig = plt.figure()
ax = fig.add_subplot
colors = ['blue','red','black']
for i in xrange(samples):
ax.scatter(newdata2[i,0],newdata2[i,1], color= colors[int(data[i,-1])])
plt.xlabel('SVD1')
plt.ylabel('SVD2')
plt.show()