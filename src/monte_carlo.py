from src.estimator import estimate_ols, estimate_dml
from src.dgp import generate_dataset

import pandas as pd
from tqdm import tqdm
from pathlib import Path


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



def run_scenario(config: dict, alpha_y: float, alpha_d: float) -> pd.DataFrame:
    results = []

    base_seed = config["random_seed"]
    R = config["num_replications"]

    for r in tqdm(range(R), desc=f"alpha_y={alpha_y}, alpha_d={alpha_d}"):
        seed = base_seed + r
        row = one_replication(config, alpha_y=alpha_y, alpha_d=alpha_d, seed=seed)
        row["replication"] = r
        results.append(row)

    return pd.DataFrame(results)

def run_simulation_grid(config: dict, save_each: bool = True) -> pd.DataFrame:
    all_results = []

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for alpha_y in config["alpha_y_grid"]:
        for alpha_d in config["alpha_d_grid"]:
            scenario_df = run_scenario(config, alpha_y=alpha_y, alpha_d=alpha_d)
            all_results.append(scenario_df)

            if save_each:
                filename = output_dir / f"alpha_y_{alpha_y}_alpha_d_{alpha_d}.csv"
                scenario_df.to_csv(filename, index=False)

    full_df = pd.concat(all_results, ignore_index=True)
    return full_df