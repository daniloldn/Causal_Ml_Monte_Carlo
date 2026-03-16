import numpy as np
from scipy.special import expit


def generate_covariates(n:int, p:int, sigma:float = 1, seed:int = 42)-> np.ndarray:


    #variance covariance matrix
    vcov = np.identity(p) * sigma**2

    #mu
    mu = mu = np.zeros(p)

    #multivarite normal
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(mu, vcov, n)


    return X


def f_linear(X: np.ndarray, weights: list[float]) -> np.ndarray:
    if X.shape[1] < len(weights):
        raise ValueError("X has fewer columns than the number of linear weights")

    return sum(weights[i] * X[:, i] for i in range(len(weights)))


def f_nonlinear(X: np.ndarray, terms: list[dict]) -> np.ndarray:
    n = X.shape[0]
    out = np.zeros(n)

    for term in terms:
        term_type = term["type"]
        weight = term["weight"]

        if term_type == "square":
            col = term["col"]
            center = term.get("center", 0.0)
            out += weight * (X[:, col] ** 2 - center)

        elif term_type == "interaction":
            c1, c2 = term["cols"]
            out += weight * (X[:, c1] * X[:, c2])

        elif term_type == "sin":
            col = term["col"]
            out += weight * np.sin(X[:, col])

        elif term_type == "abs":
            col = term["col"]
            center = term.get("center", 0.0)
            out += weight * (np.abs(X[:, col]) - center)

        else:
            raise ValueError(f"Unsupported nonlinear term type: {term_type}")

    return out
    

def f_alpha(X: np.ndarray, alpha_y: float, linear_weights: list[float], nonlinear_terms: list[dict]) -> np.ndarray:
    f_lin = f_linear(X, linear_weights)
    f_nonlin = f_nonlinear(X, nonlinear_terms)
    return (1 - alpha_y) * f_lin + alpha_y * f_nonlin


def g_raw(X: np.ndarray, terms: list[dict]) -> np.ndarray:
    
    n = X.shape[0]
    out = np.zeros(n)

    for term in terms:
        term_type = term["type"]
        weight = term["weight"]

        if term_type == "square":
            col = term["col"]
            center = term.get("center", 0.0)
            out += weight * (X[:, col] ** 2 - center)

        elif term_type == "linear":
            col = term["col"]
            out += weight * (X[:, col])

        elif term_type == "interaction":
            c1, c2 = term["cols"]
            out += weight * (X[:, c1] * X[:, c2])

        elif term_type == "sin":
            col = term["col"]
            out += weight * np.sin(X[:, col])

        elif term_type == "abs":
            col = term["col"]
            center = term.get("center", 0.0)
            out += weight * (np.abs(X[:, col]) - center)

        else:
            raise ValueError(f"Unsupported nonlinear term type: {term_type}")

    return out

def g_star(X: np.ndarray, terms: list[dict]) -> np.ndarray:
    g = g_raw(X, terms)
    return (g - np.mean(g)) / np.std(g)


def propensity_score(X:np.ndarray, terms: list[dict], alpha_d: float, s: float = 1, c=0) -> np.ndarray:
    return expit(c + alpha_d*s*g_star(X, terms))