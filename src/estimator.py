import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from econml.dml import LinearDML

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

def estimate_dml(
    X: np.ndarray,
    D: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 2,
    random_state: int = 42
) -> dict:

    est = LinearDML(
        model_y=RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=5,
            random_state=random_state
        ),
        model_t=RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=5,
            random_state=random_state
        ),
        discrete_treatment=True,
        cv=n_splits,
        random_state=random_state
    )

    est.fit(Y, D, X=X)

    # ATE for binary treatment: effect of going from 0 to 1
    tau_hat = float(est.ate(X=X))
    ci = est.ate_interval(X=X, alpha=0.05)

    ci_lower = float(ci[0])
    ci_upper = float(ci[1])

    # Approximate standard error from CI width
    se = (ci_upper - ci_lower) / (2 * 1.96)

    return {
        "tau_hat": tau_hat,
        "se": se,
        "ci_lower": ci_lower,
    }