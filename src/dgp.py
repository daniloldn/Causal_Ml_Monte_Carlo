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
    

def f_alpha(X: np.ndarray, alpha_y: float) -> np.ndarray:
    return (1 - alpha_y) * f_linear(X) + alpha_y * f_nonlinear(X)
