import numpy as np

import lin_alg as mp

def gaussian_density(x, u, sigma):
    xu = x - u
    sigma_inv = mp.get_inverse_2x2(sigma)
    det = mp.get_determinant_2x2(sigma)
    return np.exp(-(xu.T.dot(sigma_inv.dot(xu)).squeeze()) / 2) /  np.sqrt(det) # ignoring scaling denominator np.power(2*np.pi))
