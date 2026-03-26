import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from econml.dml import LinearDML
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from scipy.stats import norm




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
            random_state=random_state, 
            n_jobs=1
        ),
        model_t=RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=5,
            random_state=random_state, 
            n_jobs=1
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
        "ci_upper": ci_upper
    }

def estimate_dml_manual(
    X,
    D,
    Y,
    tau_true=None,
    f_x=None,
    e_true=None,
    n_splits=5,
    random_state=42
):
    X = np.asarray(X)
    D = np.asarray(D).ravel().astype(float)
    Y = np.asarray(Y).ravel().astype(float)

    n = len(Y)

    model_y = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=1
    )
    model_t = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=1
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    m_hat = np.zeros(n)
    e_hat = np.zeros(n)

    for train_idx, test_idx in cv.split(X, D):
        X_tr, X_te = X[train_idx], X[test_idx]
        D_tr = D[train_idx]
        Y_tr = Y[train_idx]

        my = clone(model_y)
        mt = clone(model_t)

        my.fit(X_tr, Y_tr)
        mt.fit(X_tr, D_tr)

        m_hat[test_idx] = my.predict(X_te)
        e_hat[test_idx] = mt.predict_proba(X_te)[:, 1]

    Y_res = Y - m_hat
    D_res = D - e_hat

    # Final partialling-out regression without intercept
    reg = LinearRegression(fit_intercept=False)
    reg.fit(D_res.reshape(-1, 1), Y_res)
    tau_hat = float(reg.coef_[0])

    # Simple HC0-style variance estimate
    u = Y_res - tau_hat * D_res
    denom = np.mean(D_res ** 2) ** 2
    var_hat = np.mean((D_res ** 2) * (u ** 2)) / (n * denom)
    se = float(np.sqrt(var_hat))

    z = norm.ppf(0.975)
    ci_lower = tau_hat - z * se
    ci_upper = tau_hat + z * se

    out = {
        "tau_hat": tau_hat,
        "se": se,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "m_hat_oof": m_hat,
        "e_hat_oof": e_hat,
        "resid_d_var": np.var(D_res),
        "resid_d_sq_mean": float(np.mean(D_res**2))
    }

    if tau_true is not None:
        out["tau_bias"] = float(tau_hat - tau_true)

    if f_x is not None and e_true is not None:
        f_x = np.asarray(f_x).ravel()
        e_true = np.asarray(e_true).ravel()
        m_true = f_x + (tau_true * e_true if tau_true is not None else 0.0)

        out["m_mse"] = float(mean_squared_error(m_true, m_hat))
        out["e_mse"] = float(mean_squared_error(e_true, e_hat))
        out["m_rmse"] = float(np.sqrt(out["m_mse"]))
        out["e_rmse"] = float(np.sqrt(out["e_mse"]))

    return out