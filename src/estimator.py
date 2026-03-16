import numpy as np
import statsmodels.api as sm

def estimate_ols(X: np.ndarray, D: np.ndarray, Y: np.ndarray) -> dict:
    X_reg = np.column_stack([D, X])
    X_reg = sm.add_constant(X_reg)

    model = sm.OLS(Y, X_reg)
    results = model.fit(cov_type="HC1")

    tau_hat = results.params[1]
    se = results.bse[1]
    ci_lower = tau_hat - 1.96 * se
    ci_upper = tau_hat + 1.96 * se

    return {
        "tau_hat": tau_hat,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }