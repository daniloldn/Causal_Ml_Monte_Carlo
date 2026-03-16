from src.estimator import estimate_ols, estimate_dml
from src.dgp import generate_dataset


def one_replication(config, alpha_y, alpha_d, seed):
    data = generate_dataset(config, alpha_y=alpha_y, alpha_d=alpha_d, seed=seed)

    ols_res = estimate_ols(data["X"], data["D"], data["Y"])
    
    dml_res = estimate_dml(data["X"], data["D"], data["Y"])

    return {
        "alpha_y": alpha_y,
        "alpha_d": alpha_d,
        "seed": seed,
        "tau_true": data["tau_true"],
        "ols_tau_hat": ols_res["tau_hat"],
        "ols_se": ols_res["se"],
        "ols_ci_lower": ols_res["ci_lower"],
        "ols_ci_upper": ols_res["ci_upper"],
        "dml_tau_hat": dml_res["tau_hat"],
        "dml_se": dml_res["se"],
        "dml_ci_lower": dml_res["ci_lower"],
        "dml_ci_upper": dml_res["ci_upper"],
    }