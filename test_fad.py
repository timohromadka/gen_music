import numpy as np
from numpy import cov, trace, iscomplexobj, eye
from numpy.random import random
from scipy import linalg
from scipy.linalg import sqrtm

# Redefine the calculate_fid function to print intermediate values for diagnostic purposes
def calculate_fid(act1, act2, eps=1e-6):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # Regularize covariance matrices by adding a small value to the diagonal
    sigma1 += eye(sigma1.shape[0]) * eps
    sigma2 += eye(sigma2.shape[0]) * eps
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_fid2(act1, act2, eps=1e-6):
    mu1, cov1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, cov2 = act2.mean(axis=0), cov(act2, rowvar=False)
    
    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert cov1.shape == cov2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Eigenvalue decomposition of the product of covariance matrices
    eigvals, eigvecs = linalg.eigh(cov1.dot(cov2))
    print(f'eigvals: {eigvals}\n\n')
    print(f'eigvecs: {eigvecs}\n\n')
    # Clip negative eigenvalues to 0 to ensure the square root is real
    eigvals_clipped = np.clip(eigvals, a_min=0, a_max=None)
    print(f'eigvals_clipped: {eigvals_clipped}\n\n')
    # Compute the square root of the product matrix using clipped eigenvalues
    covmean_sqrt = eigvecs.dot(np.diag(np.sqrt(eigvals_clipped))).dot(eigvecs.T)
    print(f'covmean_sqrt1: {covmean_sqrt}\n\n')

    # Numerical errors can result in slight imaginary components, we ignore these.
    covmean_sqrt = np.real(covmean_sqrt)
    print(f'covmean_sqrt2: {covmean_sqrt}\n\n')

    # Frechet distance calculation
    tr_covmean_sqrt = np.trace(covmean_sqrt)
    print(f'tr_covmean_sqrt: {tr_covmean_sqrt}\n\n')
    print(f'diff.dot(diff): {diff.dot(diff)}\n\n')
    print(f'np.trace(cov1): {np.trace(cov1)}\n\n')
    print(f'np.trace(cov2): {np.trace(cov2)}\n\n')
    return (diff.dot(diff) + np.trace(cov1) + np.trace(cov2) - 2 * tr_covmean_sqrt)

# define two collections of activations
act1 = random(10*2048)
act1 = act1.reshape((10,2048))
act2 = random(10*2048)
act2 = act2.reshape((10,2048))

# Call calculate_fid with verbose=True to print intermediate values
# fid_same = calculate_fid(act1, act1)
# print('FID (same): %.3f' % fid_same)

# fid_different = calculate_fid(act1, act2)
# print('FID (different): %.3f' % fid_different)

fid_same = calculate_fid2(act1, act1)
print('FID2 (same): %.3f' % fid_same)

fid_different = calculate_fid2(act1, act2)
print('FID2 (different): %.3f' % fid_different)
