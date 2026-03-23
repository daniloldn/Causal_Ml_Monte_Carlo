import numpy as np
from scipy.special import expit


def generate_covariates(n:int, p:int, sigma:float = 1, seed:int = 42)-> np.ndarray:


    #variance covariance matrix
    vcov = np.identity(p) * sigma**2

    #mu
    mu = np.zeros(p)

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


def g_simple(X):
    return X[:, 0]

def g_complex(X):
    return np.sin(X[:, 0]) +  0.5 * (X[:, 1] ** 2 - 1)

def g_raw(X: np.ndarray, alpha_d: float) -> np.ndarray:
    return (1 - alpha_d) * g_simple(X) + alpha_d * g_complex(X)


def g_star(X: np.ndarray, alpha_d: float) -> np.ndarray:
    g = g_raw(X, alpha_d)
    return (g - np.mean(g)) / np.std(g)


def propensity_score(X:np.ndarray, alpha_d: float, k: float = 1, c=0) -> np.ndarray:
    return expit(c + k*g_star(X, alpha_d))

def generate_treatment(
    X: np.ndarray,
    alpha_d: float,
    k: float = 1.0,
    c: float = 0.0,
    seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    
    e = propensity_score(X, alpha_d=alpha_d, k=k, c=c)

    rng = np.random.default_rng(seed)
    D = rng.binomial(1, e)

    return D, e


def generate_dataset(config: dict, alpha_y: float, alpha_d: float,kappa:float,  seed: int):

    #setting seed
    rng = np.random.default_rng(seed)

    #generating X
    X = generate_covariates(config["sample_size"], config["num_covariates"], config["X_std"],seed=seed+1)

    #generating f(x)
    f_x = f_alpha(X, alpha_y, config["linear_weights"], config["nonlinear_terms"] )

    #generating treatment
    treatment, e = generate_treatment(X,
                                   alpha_d, kappa,
                                     config["intercept"], seed=seed+2)
    
    #generating epislion
    eps = rng.normal(0, config["noise_std"], config["sample_size"])

    #tau
    tau = config["treatment_effect"]

    #generating y
    y = tau * treatment + f_x + eps

    return {
    "X": X,
    "D": treatment,
    "Y": y,
    "tau_true": tau,
    "e": e,
    "f_alpha": f_x,
    "epsilon": eps
}