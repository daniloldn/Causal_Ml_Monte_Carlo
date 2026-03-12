import numpy as np


def generate_covariates(n:int, p:int, sigma:float = 1, seed:int = 42)-> np.ndarray:


    #variance covariance matrix
    vcov = np.identity(p) * sigma**2

    #mu
    mu = mu = np.zeros(p)

    #multivarite normal
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(mu, vcov, n)


    return X