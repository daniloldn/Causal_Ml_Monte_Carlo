from src.estimator import estimate_ols, estimate_dml, estimate_dml_manual
from src.dgp import generate_dataset
from src.utils import load_config

import pandas as pd
import numpy as np
import pyarrow
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os



def one_replication(config, alpha_y, alpha_d, kappa, seed, replication):
    data = generate_dataset(config, alpha_y=alpha_y, alpha_d=alpha_d, kappa=kappa, seed=seed)

    ols_res = estimate_ols(data["X"], data["D"], data["Y"])

    #dml_res = estimate_dml(data["X"], data["D"], data["Y"])
    dml_res = estimate_dml_manual(data["X"], data["D"], data["Y"], data["tau_true"], data["f_alpha"], data["e"])


    return {
        "alpha_y": alpha_y,
        "alpha_d": alpha_d,
        "kappa": kappa,
        "seed": seed,
        "overlap": np.mean(data["e"] * (1 - data["e"])),
        "residual_d_var": np.var(data["D"] - data["e"]),
        "replication": replication,
        "tau_true": data["tau_true"],
        "ols_tau_hat": ols_res["tau_hat"],
        "ols_se": ols_res["se"],
        "ols_ci_lower": ols_res["ci_lower"],
        "ols_ci_upper": ols_res["ci_upper"],
        "dml_tau_hat": dml_res["tau_hat"],
        "dml_se": dml_res["se"],
        "dml_ci_lower": dml_res["ci_lower"],
        "dml_ci_upper": dml_res["ci_upper"],
        "m_mse": dml_res["m_rmse"],
        "e_mse": dml_res["e_rmse"],
        "estimated_resid_var": dml_res["resid_d_var"], 
        "treatmeant_var": dml_res["resid_d_sq_mean"]
    }



def run_scenario(
    config: dict,
    alpha_y: float,
    alpha_d: float,
    kappa: float,
    n_jobs: int | None = None
) -> pd.DataFrame:
    base_seed = config["random_seed"]
    R = config["num_replications"]
    n_jobs = n_jobs or os.cpu_count()

    futures = []
    results = []

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for r in range(R):
            seed = base_seed + r
            futures.append(
                executor.submit(
                    one_replication,
                    config,
                    alpha_y,
                    alpha_d,
                    kappa,
                    seed,
                    r
                )
            )

        for fut in tqdm(
            as_completed(futures),
            total=R,
            desc=f"alpha_y={alpha_y}, alpha_d={alpha_d}, kappa={kappa}"
        ):
            results.append(fut.result())

    return pd.DataFrame(results).sort_values("replication").reset_index(drop=True)


def run_simulation_grid(config: dict, save_each: bool = True, n_jobs: int | None = None) -> pd.DataFrame:
    all_results = []

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for alpha_y in config["alpha_y_grid"]:
        for alpha_d in config["alpha_d_grid"]:
            for kappa in config["kappa"]:
                scenario_df = run_scenario(
                    config=config,
                    alpha_y=alpha_y,
                    alpha_d=alpha_d,
                    kappa=kappa,
                    n_jobs=n_jobs
                )
                all_results.append(scenario_df)

                if save_each:
                    filename = output_dir / f"alpha_y_{alpha_y}_alpha_d_{alpha_d}_kappa_{kappa}.csv"
                    scenario_df.to_parquet(filename.with_suffix(".parquet"), index=False)

    return pd.concat(all_results, ignore_index=True)



if __name__ == "__main__":
    config = load_config("baseline")
    df = run_simulation_grid(config, save_each=True, n_jobs=8)