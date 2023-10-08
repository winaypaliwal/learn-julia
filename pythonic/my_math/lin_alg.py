import numpy as np

def mean_1d(a):
    sum = 0
    for el in a: sum += el
    return sum/a.shape[0]

def mean_2d(a, axis=0):
    if axis == 1: a = a.T
    sum = np.zeros(a.shape[1])
    for el in a: sum += el
    return np.expand_dims(sum/a.shape[0], axis=1)

def l2_norm_sqaured(p1, p2): #Euclidean
    p = p1 - p2
    return p.T.dot(p).squeeze()

def mn_norm_sqaured(p1, mu, sigma_inv): #Mahalanobis
    p = p1 - mu
    return p.T.dot(sigma_inv.dot(p)).squeeze()

def variance(a):
    a = a - mean_1d(a)
    return a.dot(a.T) / (a.shape[0])

def covariance(a):
    a = a - mean_2d(a, axis=1)
    return a.dot(a.T) / (a.shape[1] - 1)

def get_determinant_2x2(a): return a[0][0] * a[1][1] - a[1][0] * a[0][1]

def get_adjoint_2x2(a):
    b = np.empty(a.shape)
    b[1][1] = a[0][0]
    b[0][0] = a[1][1]
    b[0][1] = -a[1][0]
    b[1][0] = -a[0][1]
    return b

def get_inverse_2x2(a):
    det = get_determinant_2x2(a)
    assert(det != 0)
    return get_adjoint_2x2(a) / det

def get_diag_sum(a):
    sum = 0
    for i in range(a.shape[0]): sum += a[i][i]
    return sum

def get_max_index(a):
    max_i = 0
    max = float('-inf')
    for i, el in enumerate(a): 
        if el > max:
            max = el
            max_i = i
    return max_i

def get_discriminant(x, u, sigma):
    xu = x - u
    sigma_inv = get_inverse_2x2(sigma)
    det = get_determinant_2x2(sigma)
    return -(xu.T.dot(sigma_inv.dot(xu)).squeeze() + np.log(det)) / 2 #ignoring ln(2*pi) and prior class prob.



    
        





    


