""" Script for PCA tests function. 
 Run with python PCA_script.py 
 """

 """Introducing the basics of Principal Component Analysis
Based off Stat 159 course notes on http://www.jarrodmillman.com/rcsds/lectures/pca_introduction.html
"""

#Libraries to use

import numpy as numpy
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

%matplotlib
%pylab

#Data paths (varies depending on where your file is)
pathtodata = ".../.../data/ds009/sub001/"
location_condition = pathtodata + "model/model001/onsets/task001_run001/"
location_images = ".../.../images/"

sys.path.append() ###incomplete

#Loading image for subject 1
img = nib.load(pathtodata + "BOLD/task001_run001/bold.nii.gz")
data = img.get_data()
data = data[...,6]

#
