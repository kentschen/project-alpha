"""
Run with:
    nosetests test_pca.py
"""

#Loading modules
import sys
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as npl
import nibabel as nib
import os
import sys
from PCA import PCA
from numpy.testing import assert_almost_equal, assert_array_equal


# Path to the subject 009 fMRI data used in class. 
pathtoclassdata = "data/ds114/"

# Add path to functions to the system path.
sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))


def test_pca():
    """PCA on dense arrays"""
    img = nib.load(pathtoclassdata + "ds114_sub009_t2r1.nii")
    data = img.get_data()[..., 4:]
     
    pca = PCA(n_components=2)
    X = pathtoclassdata
    X_r = pca.fit(X).transform(X)
    np.testing.assert_equal(X_r.shape[1], 2)

    X_r2 = pca.fit_transform(X)
    assert_array_almost_equal(X_r, X_r2)

    pca = PCA()
    pca.fit(X)
    assert_almost_equal(pca.explained_variance_ratio_.sum(), 1.0, 3)

    X_r = pca.transform(X)
    X_r2 = pca.fit_transform(X)

    assert_array_almost_equal(X_r, X_r2)
    

def test_pca_check_projection():
    """Test that the projection of data is correct"""
    rng = np.random.RandomState(0)
    n, p = 100, 3
    X = rng.randn(n, p) * .1
    X[:10] += np.array([3, 4, 5])
    Xt = 0.1 * rng.randn(1, p) + np.array([3, 4, 5])

    Yt = PCA(n_components=2).fit(X).transform(Xt)
    Yt /= np.sqrt((Yt ** 2).sum())

    assert_almost_equal(np.abs(Yt[0][0]), 1., 1)
    

if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])
    